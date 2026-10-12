"""Stream the 16 val scenes through QueryMemory and evaluate on ESAM's mesh.

    source tools/env.sh
    python tools/stream_eval.py [--scorer scorer] [--voxel 0.05] [--matcher rule|<tag>] [--scenes ...]
Reads proposals from work_dirs/merge_cache/proposals/val (no FastSAM re-run), renders the memory onto
work_dirs/esam_eval/data/scannet200/points/<scene>.bin and scores against instance_mask on the same vertices.
Writes work_dirs/stream/<tag>/{summary.md,summary.json,per_scene.json}.
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
from freesam3d.merging.proposal_data import list_proposal_cache, load_proposals  # noqa: E402
from freesam3d.streaming.evaluate import (frame_input_from_proposal_cache, mesh_gtset, scene_metrics,  # noqa: E402
                                          segment_smooth, select_frame_proposals, stack_frames_baseline)
from freesam3d.streaming.memory import MemoryConfig, QueryMemory  # noqa: E402


def scorer_scores(d, scorer, std, feat_dim, dev):
    X = std.apply(torch.as_tensor(d['scalars'], device=dev))
    Fe = torch.as_tensor(d['feat'], device=dev)
    return scorer.score(X, Fe if feat_dim else Fe[:, :0]).cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--scorer', default='scorer', help='mask-scorer tag; "oracle" uses the cached IoU target')
    p.add_argument('--matcher', default='rule', help='"rule" (voxel IoU) or a MatchScorer tag under work_dirs/stream/matcher/')
    p.add_argument('--voxel', type=float, default=0.05)
    p.add_argument('--match-threshold', type=float, default=0.3)
    p.add_argument('--admit-score', type=float, default=0.3)
    p.add_argument('--assignment', default='hungarian', choices=['many_to_one', 'hungarian'])
    p.add_argument('--match-score', default='max_contain',
                   choices=['iou', 'max_contain', 'iou_or_contain', 'contain_x_feat', 'feat_x_bbox', 'feat_gate'])
    p.add_argument('--merge-feat-cos', type=float, default=-1.0, help='geometric query merge also needs feature cos >= this (-1 off)')
    p.add_argument('--merge-touch-cos', type=float, default=-1.0, help='merge touching queries with feature cos >= this (-1 off)')
    p.add_argument('--merge-dilation', type=int, default=0)
    p.add_argument('--nms-contain', type=float, default=None,
                   help='per-frame nested NMS before the memory (iou --nms-iou, this containment), as GroupMerger does; default: none')
    p.add_argument('--nms-iou', type=float, default=0.5)
    p.add_argument('--segment-smooth', action='store_true',
                   help='also report the memory masks after ESAM mesh-segment smoothing (super_points of the scene)')
    p.add_argument('--projection', default=None, help='ProjectionHead tag under work_dirs/stream/projection/')
    p.add_argument('--feat-weight', type=float, default=1.0)
    p.add_argument('--match-dilation', type=int, default=1)
    p.add_argument('--render-min-votes', type=int, default=1)
    p.add_argument('--render-min-vote-frac', type=float, default=0.0)
    p.add_argument('--top-k', type=int, default=20, help='proposals per frame fed to the memory (ESAM topk_insts)')
    p.add_argument('--scenes', nargs='*')
    p.add_argument('--no-baseline', action='store_true')
    p.add_argument('--tag', default=None)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    dev = a.device if torch.cuda.is_available() else 'cpu'
    ws = resolve_path('work_dirs/esam_eval')
    stuff = stuff_ids(cfg['esam_root'] + '/data/scannet200/meta_data/scannetv2-labels.combined.tsv') if 'esam_root' in cfg \
        else stuff_ids('/home/yqx/SegmentAnything3D/ESAM/data/scannet200/meta_data/scannetv2-labels.combined.tsv')
    scorer = None
    if a.scorer != 'oracle':
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from scorer_eval import load_scorer
        scorer = load_scorer(resolve_path('work_dirs/mask_scorer') / a.scorer / 'model.pt', dev)
    projection = None
    if a.projection:
        from freesam3d.streaming.projection import load_projection
        projection = load_projection(resolve_path('work_dirs/stream/projection') / a.projection / 'model.pt', dev)
    matcher = None
    if a.matcher != 'rule':
        from freesam3d.streaming.matcher import load_matcher
        matcher = load_matcher(resolve_path('work_dirs/stream/matcher') / a.matcher / 'model.pt', dev)
    files = list_proposal_cache(resolve_path(cfg['cache_root']) / 'proposals' / 'val')
    by_scene = {}
    for f in files:
        by_scene.setdefault(f.stem.rsplit('_', 1)[0], []).append(f)
    scenes = a.scenes or sorted(by_scene)
    tag = a.tag or f'{a.matcher}_{a.match_score}_{a.assignment}_v{int(a.voxel * 100)}cm_{a.scorer}' + (f'_{a.projection}' if a.projection else '')
    out = resolve_path('work_dirs/stream') / tag
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene in scenes:
        fs = sorted(by_scene[scene], key=lambda q: int(q.stem.rsplit('_', 1)[1]))
        mesh = np.fromfile(ws / 'data/scannet200/points' / f'{scene}.bin', np.float32).reshape(-1, 6)[:, :3]
        mesh_gt = np.fromfile(ws / 'data/scannet200/instance_mask' / f'{scene}.bin', np.int64)
        mesh_sem = np.fromfile(ws / 'data/scannet200/semantic_mask' / f'{scene}.bin', np.int64)
        gts = mesh_gtset(mesh_gt, mesh_sem, stuff, cfg['eval']['min_gt_points'])
        if gts is None:
            continue
        mem = QueryMemory(MemoryConfig(voxel=a.voxel, match_threshold=a.match_threshold, admit_score=a.admit_score,
                                       assignment=a.assignment, match_score=a.match_score, match_dilation=a.match_dilation,
                                       render_min_votes=a.render_min_votes, render_min_vote_frac=a.render_min_vote_frac,
                                       feat_weight=a.feat_weight, merge_feat_cos=a.merge_feat_cos,
                                       merge_touch_cos=a.merge_touch_cos, merge_dilation=a.merge_dilation),
                          device=dev, scorer=matcher, projection=projection)
        frames, t_upd = [], []
        for f in fs:
            d = load_proposals(f, with_masks=False)
            d['masks'] = np.load(f)['masks']
            s = scorer_scores(d, *scorer, dev) if scorer else None
            fr = frame_input_from_proposal_cache(d, scores=s, top_k=None if a.nms_contain is not None else a.top_k)
            if a.nms_contain is not None:
                fr = select_frame_proposals(fr, a.nms_iou, a.nms_contain, a.top_k, dev)
            fr.frame_id = int(f.stem.rsplit('_', 1)[1])
            frames.append(fr)
            t0 = time.perf_counter()
            mem.update(fr)
            if dev != 'cpu':
                torch.cuda.synchronize()
            t_upd.append(time.perf_counter() - t0)
        masks, scores = mem.render(mesh)
        row = dict(scene=scene, frames=len(fs), num_gt=len(gts.ids), queries=mem.num_queries, records=mem.num_records,
                   scene_voxels=int(len(mem.scene_voxels)), update_ms=1000 * float(np.mean(t_upd)),
                   memory=scene_metrics(masks, scores, gts))
        if a.segment_smooth:
            sp = np.fromfile(ws / 'data/scannet200/super_points' / f'{scene}.bin', np.int64)
            assert len(sp) == len(mesh), (scene, len(sp), len(mesh))
            row['memory_smooth'] = scene_metrics(segment_smooth(masks, sp, device=dev), scores, gts)
        if not a.no_baseline:
            bm, bs = stack_frames_baseline(frames, mesh, a.voxel)
            row['stack_frames'] = scene_metrics(bm, bs, gts)
        esam = ws / 'predictions' / f'{scene}.npz'
        if esam.exists():
            z = np.load(esam)
            row['esam'] = scene_metrics(z['scene_masks'], z['scene_scores'], gts)
        rows.append(row)
        print(f"{scene}: {len(fs)} frames, {mem.num_queries} queries, {mem.num_records} records, update {row['update_ms']:.1f} ms | "
              f"memory AP50 {row['memory']['ap50']:.3f} IoU {row['memory']['iou']:.3f}"
              + (f" | stack AP50 {row['stack_frames']['ap50']:.3f}" if 'stack_frames' in row else '')
              + (f" | ESAM AP50 {row['esam']['ap50']:.3f}" if 'esam' in row else ''), flush=True)
    methods = [m for m in ('stack_frames', 'memory', 'memory_smooth', 'esam') if m in rows[0]]
    means = {m: {k: float(np.mean([r[m][k] for r in rows])) for k in ('ap50', 'ap25', 'iou', 'r50', 'masks')} for m in methods}
    mem_stats = {k: float(np.mean([r[k] for r in rows])) for k in ('queries', 'records', 'scene_voxels', 'update_ms')}
    lines = [f'# Streaming scene-level evaluation: {tag}', '',
             f'{len(rows)} val scenes on ESAM mesh vertices, thing GT >= {cfg["eval"]["min_gt_points"]} vertices, scene-averaged.',
             f'memory: voxel {a.voxel} m, match {a.matcher}/{a.match_score}/{a.assignment} (thr {a.match_threshold}), admit score {a.admit_score}, '
             f'top-{a.top_k} proposals/frame (nms contain {a.nms_contain}), proposal scorer {a.scorer}, projection {a.projection}, '
             f'merge feat cos {a.merge_feat_cos} / touch cos {a.merge_touch_cos}', '',
             '| method | masks/scene | mean best IoU | recall@0.5 | AP50 | AP25 |', '|---|---|---|---|---|---|']
    for m in methods:
        r = means[m]
        lines.append(f"| {m} | {r['masks']:.1f} | {r['iou']:.3f} | {r['r50']:.3f} | {r['ap50']:.3f} | {r['ap25']:.3f} |")
    lines += ['', f"memory per scene: {mem_stats['queries']:.1f} queries, {mem_stats['records']:.0f} (voxel,query) records, "
              f"{mem_stats['scene_voxels']:.0f} scene voxels, update {mem_stats['update_ms']:.1f} ms/frame"]
    md = '\n'.join(lines) + '\n'
    (out / 'summary.md').write_text(md)
    (out / 'summary.json').write_text(json.dumps(dict(means=means, memory=mem_stats, args=vars(a)), indent=2))
    (out / 'per_scene.json').write_text(json.dumps(rows, indent=1))
    print(md)


if __name__ == '__main__':
    main()
