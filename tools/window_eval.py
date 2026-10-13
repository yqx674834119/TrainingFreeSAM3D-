"""Scene-level evaluation of the v2 window memory and the reference baselines (stage 0: GT oracle).

    source tools/env.sh
    python tools/window_eval.py --memory window --score oracle --tag win_w5 --segment-smooth
    python tools/window_eval.py --memory esam_rule --score oracle --tag esam_rule --segment-smooth

Reads the merge cache (no FastSAM re-run), streams every frame of a scene through the memory, renders
it on ESAM's mesh vertices and scores against the mesh instance GT (same protocol as tools/stream_eval.py).
Writes work_dirs/stream_v2/phase0/<tag>/{summary.md,summary.json,per_scene.json}.
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
from freesam3d.streaming.esam_merge import EsamOnlineMerge  # noqa: E402
from freesam3d.streaming.evaluate import (frames_from_merge_cache, mesh_gtset, oracle_frame_instances,  # noqa: E402
                                          scene_metrics, segment_smooth)
from freesam3d.streaming.memory import MemoryConfig, QueryMemory  # noqa: E402
from freesam3d.streaming.window_memory import (GtAssocMemory, WindowConfig, WindowMemory,  # noqa: E402
                                               oracle_scores)

METRIC_KEYS = ('ap50', 'ap25', 'iou', 'r50', 'masks')


def build_memory(a, cfg, dev, num_feat):
    if a.memory == 'window':
        if a.score != 'oracle':
            raise SystemExit('stage 0 only implements --score oracle (the model scorer lands in stage 2)')
        wc = cfg['stream_v2']
        cfg_w = WindowConfig(window=a.window or wc['window'], past_unit=a.past_unit or wc['past_unit'],
                             voxel=a.voxel or wc['voxel'], touch_dilation=wc['touch_dilation'],
                             threshold=a.threshold if a.threshold is not None else wc['threshold'],
                             linkage=wc['linkage'],
                             admit_min_points=a.admit_min_points if a.admit_min_points is not None else wc['admit_min_points'],
                             object_merge=not a.no_object_merge, purity=wc['purity'])
        return WindowMemory(cfg_w, score_fn=lambda g: oracle_scores(g, cfg_w.purity), device=dev),False
    if a.memory == 'gt_assoc':
        return GtAssocMemory(purity=cfg['stream_v2']['purity'], voxel=a.voxel or cfg['stream_v2']['voxel'], device=dev),False
    if a.memory == 'esam_rule':
        return EsamOnlineMerge(topk=a.top_k, voxel=a.voxel or cfg['stream_v2']['voxel'], device=dev), True
    if a.memory == 'query_memory':
        sc = cfg['streaming']
        projection = None
        if a.projection and a.projection != 'none':
            from freesam3d.streaming.projection import load_projection
            path = Path(a.projection) if '/' in a.projection else resolve_path('work_dirs/stream/projection') / a.projection / 'model.pt'
            projection = load_projection(path, dev)
        mc = MemoryConfig(voxel=a.voxel or sc['voxel'], match_score=a.match_score or 'max_contain',
                          match_threshold=sc['match_threshold'], match_dilation=sc['match_dilation'],
                          admit_score=sc['admit_score'])
        return QueryMemory(mc, device=dev, scorer=None, projection=projection), True
    raise SystemExit(f'unknown --memory {a.memory}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--memory', default='window', choices=['window', 'esam_rule', 'query_memory', 'gt_assoc'])
    p.add_argument('--score', default='oracle', choices=['oracle', 'model'])
    p.add_argument('--checkpoint', default=None, help='model scorer checkpoint (stage 2)')
    p.add_argument('--window', type=int, default=None)
    p.add_argument('--past-unit', default=None, choices=['block', 'fragment'])
    p.add_argument('--no-object-merge', action='store_true')
    p.add_argument('--voxel', type=float, default=None)
    p.add_argument('--threshold', type=float, default=None)
    p.add_argument('--admit-min-points', type=int, default=None)
    p.add_argument('--match-score', default=None, help='query_memory rule (default max_contain)')
    p.add_argument('--projection', default=None, help='query_memory projection tag, or "none"')
    p.add_argument('--top-k', type=int, default=200, help='esam_rule output truncation (ESAM inscat_topk_insts)')
    p.add_argument('--segment-smooth', action='store_true')
    p.add_argument('--scenes', nargs='*')
    p.add_argument('--tag', default=None)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    dev = a.device if torch.cuda.is_available() else 'cpu'
    ws = resolve_path('work_dirs/esam_eval')
    stuff = stuff_ids('/home/yqx/SegmentAnything3D/ESAM/data/scannet200/meta_data/scannetv2-labels.combined.tsv')

    files = sorted((resolve_path(cfg['cache_root']) / 'val').glob('*.npz'))
    by_scene = {}
    for f in files:
        by_scene.setdefault(f.stem.rsplit('_', 1)[0], []).append(f)
    scenes = a.scenes or sorted(by_scene)
    wc = cfg['stream_v2']
    tag = a.tag or (f'{a.memory}_{a.score}' + (f'_w{a.window}' if a.memory == 'window' and a.window else '')
                    + (f'_{a.past_unit}' if a.memory == 'window' and a.past_unit else '')
                    + ('_noobjmerge' if a.no_object_merge else ''))
    out = resolve_path(wc['out_root']) / 'phase0' / tag
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene in scenes:
        fs = sorted(by_scene[scene], key=lambda q: int(q.stem.rsplit('_', 1)[1]))
        mesh6 = np.fromfile(ws / 'data/scannet200/points' / f'{scene}.bin', np.float32).reshape(-1, 6)
        mesh = mesh6[:, :3]
        mesh_gt = np.fromfile(ws / 'data/scannet200/instance_mask' / f'{scene}.bin', np.int64)
        mesh_sem = np.fromfile(ws / 'data/scannet200/semantic_mask' / f'{scene}.bin', np.int64)
        gts = mesh_gtset(mesh_gt, mesh_sem, stuff, cfg['eval']['min_gt_points'])
        if gts is None:
            continue
        frames = frames_from_merge_cache(fs, gt=cfg['eval']['gt'])
        mem, needs_instances = build_memory(a, cfg, dev, len(frames[0].node_feat))
        t_upd = []
        for fr in frames:
            inp = oracle_frame_instances(fr) if needs_instances else fr
            t0 = time.perf_counter()
            mem.update(inp)
            if dev != 'cpu':
                torch.cuda.synchronize()
            t_upd.append(time.perf_counter() - t0)
        masks, scores = mem.render(mesh)
        row = dict(scene=scene, frames=len(fs), num_gt=len(gts.ids), objects=int(len(scores)),
                   update_ms=1000 * float(np.mean(t_upd)),
                   memory=scene_metrics(masks, scores, gts), assoc=association_stats(masks, gts))
        if a.segment_smooth:
            sp = np.fromfile(ws / 'data/scannet200/super_points' / f'{scene}.bin', np.int64)
            assert len(sp) == len(mesh), (scene, len(sp), len(mesh))
            smooth = segment_smooth(masks, sp, device=dev)
            row['memory_smooth'] = scene_metrics(smooth, scores, gts)
            row['assoc_smooth'] = association_stats(smooth, gts)
        esam = ws / 'predictions' / f'{scene}.npz'
        if esam.exists():
            z = np.load(esam)
            row['esam'] = scene_metrics(z['scene_masks'], z['scene_scores'], gts)
        rows.append(row)
        print(f"{scene}: {len(fs)} frames, {row['objects']} objects, update {row['update_ms']:.1f} ms | "
              f"raw AP50 {row['memory']['ap50']:.3f} IoU {row['memory']['iou']:.3f}"
              + (f" | smooth AP50 {row['memory_smooth']['ap50']:.3f}" if 'memory_smooth' in row else '')
              + (f" | ESAM AP50 {row['esam']['ap50']:.3f}" if 'esam' in row else ''), flush=True)

    def mean_of(key):
        return {k: float(np.mean([r[key][k] for r in rows])) for k in METRIC_KEYS}

    means = {m: mean_of(m) for m in ('memory', 'memory_smooth', 'esam') if m in rows[0]}
    assoc = {m: {k: float(np.nanmean([r[m][k] for r in rows])) for k in ('dup_rate', 'overmerge_rate', 'frags_per_gt')}
             for m in ('assoc', 'assoc_smooth') if m in rows[0]}
    stats = {k: float(np.mean([r[k] for r in rows])) for k in ('objects', 'update_ms')}
    lines = [f'# Window-memory evaluation (stage 0, GT oracle): {tag}', '',
             f'{len(rows)} val scenes on ESAM mesh vertices, thing GT >= {cfg["eval"]["min_gt_points"]} vertices, '
             f'scene-averaged. memory={a.memory}, scorer={a.score}, '
             f'window={a.window or wc["window"]} ({a.past_unit or wc["past_unit"]}), voxel={a.voxel or wc["voxel"]} m, '
             f'threshold={a.threshold if a.threshold is not None else wc["threshold"]}, object_merge={not a.no_object_merge}', '',
             '| method | masks/scene | mean best IoU | recall@0.5 | AP50 | AP25 |', '|---|---|---|---|---|---|']
    for m, r in means.items():
        lines.append(f"| {m} | {r['masks']:.1f} | {r['iou']:.3f} | {r['r50']:.3f} | {r['ap50']:.3f} | {r['ap25']:.3f} |")
    lines += ['', f"objects/scene {stats['objects']:.1f}, update {stats['update_ms']:.1f} ms/frame"]
    for m, r in assoc.items():
        lines.append(f"{m}: dup_rate {r['dup_rate']:.3f}, overmerge_rate {r['overmerge_rate']:.3f}, "
                     f"frags_per_gt {r['frags_per_gt']:.2f}")
    md = '\n'.join(lines) + '\n'
    (out / 'summary.md').write_text(md)
    (out / 'summary.json').write_text(json.dumps(dict(means=means, assoc=assoc, stats=stats, args=vars(a)), indent=2))
    (out / 'per_scene.json').write_text(json.dumps(rows, indent=1))
    print(md)


if __name__ == '__main__':
    main()
