"""Stage 1 final: the holdout-selected merge configuration, run **once** on the 16 val scenes.

    source tools/env.sh
    python tools/window_phase1_eval.py                                  # variant from phase1/selected.json
    python tools/window_phase1_eval.py --mode two_step --rule p2a --cross-threshold 0.4

Streams every frame of a val scene (``work_dirs/merge_cache/val``, read-only) through the selected
``WindowMemory`` (window 1), renders the objects onto ESAM's mesh vertices, applies the stage-0b/0c
free finishing (rank by observation count, scene-level NMS, optionally drop single-frame objects) and
reports the mesh protocol: thing GT >= 100 vertices, mesh-segment smoothing, scene-averaged AP.
Nothing here is tuned on these scenes; GT enters only the S_oracle ranking row and the metrics.

Writes work_dirs/stream_v2/phase1/val16/{summary.md,summary.json,per_scene.json}.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.evaluation.granularity import stuff_ids  # noqa: E402
from freesam3d.evaluation.instance import association_stats  # noqa: E402
from freesam3d.streaming.evaluate import frames_from_merge_cache, mesh_gtset, scene_metrics, segment_smooth  # noqa: E402
from freesam3d.streaming.holdout import Variant, run_scene_masks  # noqa: E402
from freesam3d.streaming.postprocess import best_iou, filter_single, greedy_nms, iou_pair_matrix, rank_scores  # noqa: E402
from freesam3d.streaming.window_scores import load_group_merge_net  # noqa: E402

SCORES = ('conf', 'size', 'nobs', 'oracle')
CLEANS = ('none', 'nms', 'nms_single')
SETS = ('smooth', 'raw')
NMS_IOU = 0.5
METRIC_KEYS = ('masks', 'iou', 'r50', 'ap50', 'ap25', 'dup_rate', 'overmerge', 'frags')


def metrics_of(masks, scores, gts, device='cpu'):
    met = scene_metrics(masks, scores, gts, device=device)
    st = association_stats(masks, gts, device=device)
    met['dup_rate'], met['overmerge'], met['frags'] = st['dup_rate'], st['overmerge_rate'], st['frags_per_gt']
    return met


def load_mesh(ws, scene):
    mesh6 = np.fromfile(ws / 'data/scannet200/points' / f'{scene}.bin', np.float32).reshape(-1, 6)
    mesh_gt = np.fromfile(ws / 'data/scannet200/instance_mask' / f'{scene}.bin', np.int64)
    mesh_sem = np.fromfile(ws / 'data/scannet200/semantic_mask' / f'{scene}.bin', np.int64)
    sp = np.fromfile(ws / 'data/scannet200/super_points' / f'{scene}.bin', np.int64)
    return mesh6[:, :3], mesh_gt, mesh_sem, sp


def selected_variant(cfg, a, dev):
    """The variant to run: the holdout selection, or an explicit override."""
    if a.mode:
        return Variant(a.mode, a.rule, a.intra_threshold, a.cross_threshold)
    sel = resolve_path(cfg['stream_v2']['out_root']) / 'phase1' / 'selected.json'
    if not sel.exists():
        raise SystemExit(f'{sel} missing: run tools/window_holdout.py --stage report first, '
                         'or pass --mode/--rule/--cross-threshold')
    d = json.loads(sel.read_text())['variant']
    return Variant(d['mode'], d['rule'], d['intra_threshold'], d['cross_threshold'])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--mode', default=None, choices=['linkage', 'calib', 'two_step'])
    p.add_argument('--rule', default='p2a', choices=['p2a', 'p2b'])
    p.add_argument('--intra-threshold', type=float, default=0.65)
    p.add_argument('--cross-threshold', type=float, default=0.5)
    p.add_argument('--scenes', nargs='*')
    p.add_argument('--device', default='cuda')
    p.add_argument('--tag', default='val16')
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    dev = a.device if torch.cuda.is_available() else 'cpu'
    ws = resolve_path('work_dirs/esam_eval')
    out = resolve_path(cfg['stream_v2']['out_root']) / 'phase1' / a.tag
    out.mkdir(parents=True, exist_ok=True)
    variant = selected_variant(cfg, a, dev)
    stuff = stuff_ids('/home/yqx/SegmentAnything3D/ESAM/data/scannet200/meta_data/scannetv2-labels.combined.tsv')
    model, std, ck = load_group_merge_net(resolve_path('work_dirs/merge_net/full/model.pt'), dev)
    print(f'variant {variant.as_dict()} | checkpoint intra threshold {ck["threshold"]}', flush=True)

    files = sorted((resolve_path(cfg['cache_root']) / 'val').glob('*.npz'))
    by_scene = {}
    for f in files:
        by_scene.setdefault(f.stem.rsplit('_', 1)[0], []).append(f)
    scenes = a.scenes or sorted(by_scene)
    cache = out / 'cache'
    rows = []
    for scene in scenes:
        cpath = cache / f'{scene}.npz'
        if cpath.exists() and not a.overwrite:
            d = dict(np.load(cpath))
        else:
            mesh, mesh_gt, mesh_sem, sp = load_mesh(ws, scene)
            fs = sorted(by_scene[scene], key=lambda q: int(q.stem.rsplit('_', 1)[1]))
            frames = frames_from_merge_cache(fs, gt=cfg['eval']['gt'])
            t0 = time.perf_counter()
            masks_raw, scores, attrs = run_scene_masks(frames, variant, model, std, cfg, dev, mesh)
            masks_smooth = segment_smooth(masks_raw, sp, device=dev)
            cache.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cpath, masks_raw=np.packbits(masks_raw, axis=1),
                                masks_smooth=np.packbits(masks_smooth, axis=1),
                                num_vertices=np.int64(masks_raw.shape[1]),
                                scores=np.asarray(scores, np.float32), n_obs=attrs['n_obs'].astype(np.int32),
                                vc_raw=masks_raw.sum(1).astype(np.int32),
                                vc_smooth=masks_smooth.sum(1).astype(np.int32))
            d = dict(np.load(cpath))
            print(f'{scene}: {len(fs)} frames, {len(scores)} objects, {time.perf_counter() - t0:.1f}s', flush=True)
        V = int(d['num_vertices'])
        masks_raw = np.unpackbits(d['masks_raw'], axis=1, count=V).astype(bool)
        masks_smooth = np.unpackbits(d['masks_smooth'], axis=1, count=V).astype(bool)
        mesh, mesh_gt, mesh_sem, sp = load_mesh(ws, scene)
        gts = mesh_gtset(mesh_gt, mesh_sem, stuff, cfg['eval']['min_gt_points'])
        if gts is None:
            continue
        row = dict(scene=scene, num_gt=len(gts.ids), objects=int(len(d['scores'])), combo={})
        for set_name, mk, vk in (('smooth', masks_smooth, d['vc_smooth']), ('raw', masks_raw, d['vc_raw'])):
            base = np.where(vk >= cfg['eval']['min_gt_points'])[0]
            masks, conf, n_obs = mk[base], d['scores'][base], d['n_obs'][base]
            iou_gt = best_iou(masks, gts, dev)
            iou_pair = iou_pair_matrix(masks, dev)
            for sm in SCORES:
                s = rank_scores(sm, masks=masks, conf=conf, n_obs=n_obs, iou_gt=iou_gt)
                for cl in CLEANS:
                    keep = (greedy_nms(masks, s, NMS_IOU, iou=iou_pair) if cl != 'none'
                            else np.argsort(-s, kind='stable'))
                    if cl == 'nms_single':
                        keep = filter_single(keep, n_obs)
                    row['combo'][f'{sm}|{cl}|{set_name}'] = metrics_of(masks[keep], s[keep], gts, dev)
        rows.append(row)
        print(f"{scene}: {len(gts.ids)} GT | nobs|nms|smooth AP50 "
              f"{row['combo']['nobs|nms|smooth']['ap50']:.3f}", flush=True)
    write_report(out, rows, variant, cfg, a.tag)


def write_report(out, rows, variant, cfg, tag):
    if not rows:
        raise SystemExit('no scenes evaluated')
    n = len(rows)
    have = [f'{sm}|{cl}|{st}' for st in SETS for sm in SCORES for cl in CLEANS
            if all(f'{sm}|{cl}|{st}' in r['combo'] for r in rows)]
    mean = {k: {m: float(np.mean([r['combo'][k][m] for r in rows])) for m in METRIC_KEYS} for k in have}
    p0c = resolve_path(cfg['stream_v2']['out_root']) / 'phase0c' / 'summary.json'
    p0b = resolve_path(cfg['stream_v2']['out_root']) / 'phase0b' / 'summary.json'
    ref = {}
    if p0c.exists():
        m0c = json.loads(p0c.read_text())['mean']
        for label, k in (('P1：S_nobs+nms', 'p1_default|nobs|nms|smooth'),
                         ('P1：S_nobs+nms+丢单帧', 'p1_default|nobs|nms_single|smooth'),
                         ('P1：stage 0c 最好无 GT（S_conf+nms+丢单帧）', 'p1_default|conf|nms_single|smooth')):
            if k in m0c:
                ref[label] = m0c[k]
    if p0b.exists():
        ref['ESAM-E 官方导出'] = json.loads(p0b.read_text())['refs']['esam']
    key = 'nobs|nms|smooth'
    L = ['# 第 1 阶段：留出场景选中的方案在 16 个验证场景上（网格协议，跑一次）', '',
         f'方案 `{variant.tag}`：mode={variant.mode}、跨帧规则={variant.rule}、'
         f'帧内阈值={variant.intra_threshold}、跨帧阈值={variant.cross_threshold}，window=1。'
         f'{n} 个验证场景，ESAM 网格协议（thing GT ≥ {cfg["eval"]["min_gt_points"]} 顶点，'
         '段平滑 thr 0.5，场景平均）。**这些场景上没有做任何调参**；方案来自 60 个留出场景。', '',
         '## 1. 主表（段平滑后）', '',
         '| 排序 | 清理 | masks | 平均最佳 IoU | 召回@0.5 | AP50 | AP25 | dup_rate | overmerge | frags/GT |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for k in have:
        if not k.endswith('|smooth'):
            continue
        sm, cl, _ = k.split('|')
        r = mean[k]
        L.append(f"| S_{sm} | {cl} | {r['masks']:.1f} | {r['iou']:.3f} | {r['r50']:.3f} | {r['ap50']:.3f} | "
                 f"{r['ap25']:.3f} | {r['dup_rate']:.3f} | {r['overmerge']:.3f} | {r['frags']:.2f} |")
    L += ['', '## 2. 未平滑（同表）', '',
          '| 排序 | 清理 | masks | 平均最佳 IoU | 召回@0.5 | AP50 | AP25 |', '|---|---|---|---|---|---|---|']
    for k in have:
        if not k.endswith('|raw'):
            continue
        sm, cl, _ = k.split('|')
        r = mean[k]
        L.append(f"| S_{sm} | {cl} | {r['masks']:.1f} | {r['iou']:.3f} | {r['r50']:.3f} | {r['ap50']:.3f} | {r['ap25']:.3f} |")
    L += ['', '## 3. 与参考并排（段平滑 AP50）', '', '| 口径 | AP50 | 平均最佳 IoU | 召回@0.5 | masks/场景 |', '|---|---|---|---|---|']
    r = mean[key]
    L.append(f"| **本阶段方案 `{variant.tag}`（S_nobs+nms）** | **{r['ap50']:.3f}** | {r['iou']:.3f} | {r['r50']:.3f} | {r['masks']:.1f} |")
    for label, v in ref.items():
        L.append(f"| {label} | {v['ap50']:.3f} | {v['iou']:.3f} | {v['r50']:.3f} | {v['masks']:.1f} |")
    L.append('')
    md = '\n'.join(L) + '\n'
    (out / 'summary.md').write_text(md)
    (out / 'summary.json').write_text(json.dumps(dict(variant=variant.as_dict(), mean=mean, refs=ref,
                                                      scenes=n, tag=tag), indent=2))
    (out / 'per_scene.json').write_text(json.dumps(
        [dict(scene=r['scene'], num_gt=r['num_gt'], objects=r['objects'],
              combo={k: v for k, v in r['combo'].items() if k.endswith('|smooth')}) for r in rows], indent=1))
    print(md)


if __name__ == '__main__':
    main()
