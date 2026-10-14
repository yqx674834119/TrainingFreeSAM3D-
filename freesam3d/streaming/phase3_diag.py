"""Stage-3 ranking diagnosis: what did the node-quality head actually learn?

Question 1 of the stage-3 brief: the quality head only captured 41% of the perfect-ranking space
(AP50 0.422 of 0.634 on the 16 val scenes). Is that overfitting (good on training scenes, bad on
holdout), or a lack of signal (bad on both)? Two measurements:

* training scenes: the collected graphs of the model's own training set (each object node carries
  its *final* voxel IoU label), scored frame by frame; pooled over all object nodes and,
  separately, over the last frame of each scene (the "final objects" the output ranks);
* holdout scenes: stream each scene with the checkpoint, take the memory's final objects, and
  compare the stored quality against the true voxel best IoU (``window_data.final_object_labels``,
  the same label the trainer regresses).

Both report the Spearman correlation between quality and IoU and the AUC of "IoU >= 0.5 or not".
The holdout side additionally classifies the *wrong* answers (IoU < 0.5) that outrank the first
correct answer: how many of them are duplicates or parts of a better mask (>= 0.8 of their voxels
inside a higher-quality object), which is what the stage-3 containment suppression removes.

GT enters through labels and this diagnosis only - never through a merge decision.
"""
import numpy as np
import torch

from .phase3_eval import stream_scene
from .unified_dedup import voxel_containment
from .unified_train import GraphSet, _auc, _rank


def ranking_stats(quality, iou, min_n=3):
    """(spearman, auc_iou50, pos_rate, n) of one (quality, iou) population; NaN when undefined."""
    quality, iou = np.asarray(quality, np.float64), np.asarray(iou, np.float64)
    n = len(quality)
    out = dict(n=int(n), pos_rate=float((iou >= 0.5).mean()) if n else None,
               auc50=_auc(quality, iou >= 0.5) if n else float('nan'),
               spearman=float('nan'))
    if n >= min_n and quality.std() > 0 and iou.std() > 0:
        out['spearman'] = float(np.corrcoef(_rank(quality), _rank(iou))[0, 1])
    return out


def training_ranking_diag(dirs, checkpoint, device='cpu', scenes=200, seed=0, log=print):
    """Quality-vs-IoU on the model's own training graphs (pooled and last-frame objects)."""
    from .unified_train import load_unified_checkpoint
    graphs = GraphSet(dirs, log=log)
    model, std, ck = load_unified_checkpoint(checkpoint, device)
    idx = np.random.default_rng(seed).permutation(len(graphs.files))[:int(scenes)]
    pooled_q, pooled_iou, last_q, last_iou = [], [], [], []
    for fi in idx:
        refs = [r for r in graphs.refs if r.file == int(fi) and r.tag == 'c' and r.n_node]
        if not refs:
            continue
        frames = []
        for start in range(0, len(refs), 32):
            b = graphs.stack(refs[start:start + 32], device)
            geom, ef = std.apply(b['geom'], b['pf'])
            with torch.no_grad():
                _, q = model(geom, b['feat'], b['edges'], ef, b['kind'], b['n_frames'], b['age'],
                             b['etype'], b['rel'], b['frame_id'])
            frames.append((q.cpu().numpy(), b['target'].cpu().numpy(), b['nvalid'].cpu().numpy(),
                           b['kind'].cpu().numpy()))
        q_all = np.concatenate([f[0] for f in frames])
        t_all = np.concatenate([f[1] for f in frames])
        v_all = np.concatenate([f[2] for f in frames])
        k_all = np.concatenate([f[3] for f in frames])
        obj = (k_all == 2) & v_all
        pooled_q.append(q_all[obj]); pooled_iou.append(t_all[obj])
        last_q.append(frames[-1][0][(frames[-1][3] == 2) & frames[-1][2]])
        last_iou.append(frames[-1][1][(frames[-1][3] == 2) & frames[-1][2]])
    out = dict(scenes_used=int(len(pooled_q)),
               pooled=ranking_stats(np.concatenate(pooled_q), np.concatenate(pooled_iou))
               if pooled_q else None,
               last_frame=ranking_stats(np.concatenate(last_q), np.concatenate(last_iou))
               if last_q else None)
    log(f'training-side: {out["scenes_used"]} scenes | pooled {out["pooled"]} | '
        f'last-frame {out["last_frame"]}')
    return out


def holdout_ranking_diag(scenes, cfg, model, std, device, cache_dir, stuff, mem_overrides=None,
                         voxel=None, min_gt_voxels=20, dup_contain=0.8, log=print):
    """Quality-vs-IoU on the holdout scenes' final objects + the duplicate composition of the
    top-ranked wrong answers."""
    from .window_data import final_object_labels
    wc = cfg['stream_v2']
    voxel = float(wc['voxel'] if voxel is None else voxel)
    pooled_q, pooled_iou = [], []
    per_scene = []
    wrong_total = wrong_top = wrong_top_dup_any = wrong_top_dup_correct = 0
    wrong_all_dup_any = 0
    for scene in scenes:
        frames, mem, obj_vox, attrs, merges = stream_scene(scene, cfg, model, std, device, cache_dir,
                                                           mem_overrides)
        _, iou_of = final_object_labels(mem, frames, voxel, stuff, min_gt_voxels, merges, device)
        ids = np.asarray(attrs['ids'], np.int64)
        quality = np.asarray(attrs['quality'], np.float32)
        iou = np.asarray([float(iou_of.get(int(o), 0.0)) for o in ids], np.float32)
        if len(ids):
            pooled_q.append(quality); pooled_iou.append(iou)
        st = ranking_stats(quality, iou)
        st['scene'] = scene
        correct = iou >= 0.5
        order = np.argsort(-quality, kind='stable')
        first_correct = int(np.argmax(correct[order])) if correct.any() else len(order)
        top_wrong = order[:first_correct]
        contain = voxel_containment(obj_vox, device) if len(ids) else np.zeros((0, 0), np.float32)
        dup_any = np.zeros(len(ids), bool)
        dup_correct = np.zeros(len(ids), bool)
        for pos, i in enumerate(order[1:], 1):
            prev = order[:pos]
            dup_any[i] = bool((contain[i, prev] >= dup_contain).any())
            pc = prev[correct[prev]]
            dup_correct[i] = bool(len(pc) and (contain[i, pc] >= dup_contain).any())
        wrong = ~correct
        wrong_total += int(wrong.sum())
        wrong_all_dup_any += int((wrong & dup_any).sum())
        wrong_top += int(len(top_wrong))
        wrong_top_dup_any += int(dup_any[top_wrong].sum())
        wrong_top_dup_correct += int(dup_correct[top_wrong].sum())
        st.update(objects=int(len(ids)), correct=int(correct.sum()),
                  top_wrong=int(len(top_wrong)),
                  top_wrong_dup_any=int(dup_any[top_wrong].sum()),
                  top_wrong_dup_correct=int(dup_correct[top_wrong].sum()))
        per_scene.append(st)
        log(f'  {scene}: {st["objects"]} objects, correct {st["correct"]}, '
            f'spearman {st["spearman"]:.3f}, auc {st["auc50"]:.3f}')
    out = dict(scenes=len(per_scene),
               pooled=ranking_stats(np.concatenate(pooled_q), np.concatenate(pooled_iou))
               if pooled_q else None,
               scene_mean_spearman=float(np.nanmean([s['spearman'] for s in per_scene])),
               scene_mean_auc=float(np.nanmean([s['auc50'] for s in per_scene])),
               wrong_total=wrong_total, wrong_dup_any=wrong_all_dup_any,
               top_wrong=wrong_top, top_wrong_dup_any=wrong_top_dup_any,
               top_wrong_dup_correct=wrong_top_dup_correct,
               dup_contain=float(dup_contain), per_scene=per_scene)
    log(f'holdout-side: {out["scenes"]} scenes | pooled {out["pooled"]} | '
        f'top-wrong {wrong_top}: dup-any {wrong_top_dup_any} ({wrong_top_dup_any / max(wrong_top, 1):.3f}), '
        f'dup-correct {wrong_top_dup_correct} ({wrong_top_dup_correct / max(wrong_top, 1):.3f})')
    return out
