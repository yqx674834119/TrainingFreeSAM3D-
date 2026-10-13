"""Stage 1: pick the cross-frame merge configuration on the 60 holdout scenes (no training).

    source tools/env.sh
    for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python tools/window_holdout.py --stage run --shard $i/4 & done
    python tools/window_holdout.py --stage report

Streams each holdout scene's 12 consecutive frames (``work_dirs/merge_cache/seq/train``) through
``WindowMemory`` (window 1) once per configuration and scores the result on the voxel proxy GT
(``evaluate.voxel_scene_eval``) — the holdout scenes have no reconstructed mesh, so the mesh
protocol of the val-16 report is not available for them. Thresholds are chosen here and nowhere
else; the 16 val scenes are only run once, with the winning configuration.

Writes work_dirs/stream_v2/phase1/sweep/<variant>/<scene>.json and, on ``--stage report``,
work_dirs/stream_v2/phase1/{sweep_summary.md,sweep.json,selected.json}.
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
from freesam3d.streaming.holdout import (MODES, RULES, Variant, default_variants, evaluate_scene,  # noqa: E402
                                        load_scene_frames, run_scene)
from freesam3d.streaming.window_scores import load_group_merge_net  # noqa: E402

METRIC_KEYS = ('ap50', 'ap25', 'iou', 'r50', 'masks', 'objects', 'dup_rate', 'overmerge_rate',
               'frags_per_gt', 'obs_frames_mean', 'single_frame_objects', 'voxels', 'gt_voxels')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--stage', default='all', choices=['run', 'report', 'all'])
    p.add_argument('--scenes', nargs='*', default=None, help='default: stream_v2.holdout_scenes')
    p.add_argument('--shard', default='0/1')
    p.add_argument('--modes', nargs='*', default=list(MODES), choices=list(MODES))
    p.add_argument('--rules', nargs='*', default=list(RULES), choices=list(RULES))
    p.add_argument('--thresholds', nargs='*', type=float, default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8])
    p.add_argument('--cache-dir', default=None, help='default: <cache_root>/seq/train')
    p.add_argument('--out', default=None, help='default: <stream_v2.out_root>/phase1')
    p.add_argument('--gt', default='2d', choices=['2d', '3d'])
    p.add_argument('--min-gt-voxels', type=int, default=20)
    p.add_argument('--min-pred-voxels', type=int, default=20)
    p.add_argument('--device', default='cuda')
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    dev = a.device if torch.cuda.is_available() else 'cpu'
    cache_dir = Path(a.cache_dir) if a.cache_dir else resolve_path(cfg['cache_root']) / 'seq' / 'train'
    out = Path(a.out) if a.out else resolve_path(cfg['stream_v2']['out_root']) / 'phase1'
    sweep_dir = out / 'sweep'
    scenes = a.scenes if a.scenes else list(cfg['stream_v2']['holdout_scenes'])
    stuff = set(cfg['eval']['stuff_nyu40']) if a.gt == '2d' else set(cfg['eval']['stuff_raw'])
    voxel = float(cfg['stream_v2']['voxel'])

    model = std = None
    intra = 0.65
    if a.stage in ('run', 'all'):
        model, std, ck = load_group_merge_net(resolve_path('work_dirs/merge_net/full/model.pt'), dev)
        intra = float(ck['threshold'])
        print(f'GroupMergeNet intra-frame threshold from the checkpoint: {intra}', flush=True)
    variants = default_variants(a.thresholds, a.modes, a.rules, intra)
    print(f'{len(variants)} variants, {len(scenes)} holdout scenes, cache {cache_dir}', flush=True)

    if a.stage in ('run', 'all'):
        i, n = map(int, a.shard.split('/'))
        mine = [s for k, s in enumerate(scenes) if k % n == i]
        t0 = time.time()
        for si, scene in enumerate(mine):
            files = sorted(cache_dir.glob(f'{scene}_*.npz'))
            if len(files) < 2:
                print(f'{scene}: only {len(files)} cached frames, skip', flush=True)
                continue
            frames = load_scene_frames(cache_dir, scene, gt=a.gt)
            for v in variants:
                dst = sweep_dir / v.tag / f'{scene}.json'
                if dst.exists() and not a.overwrite:
                    continue
                obj, attrs, times = run_scene(frames, v, model, std, cfg, dev)
                met = evaluate_scene(obj, attrs, frames, stuff, voxel=voxel,
                                     min_gt_voxels=a.min_gt_voxels, min_pred_voxels=a.min_pred_voxels,
                                     device=dev)
                met.update(dict(scene=scene, frames=len(frames), variant=v.as_dict(),
                                stage_ms={k: float(val) for k, val in times.items()}))
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(json.dumps(met, indent=1))
            print(f'[{a.shard}] {si + 1}/{len(mine)} {scene}: {len(frames)} frames x {len(variants)} variants, '
                  f'{(time.time() - t0) / 60:.1f} min', flush=True)
    if a.stage == 'run':
        return
    write_report(out, sweep_dir, variants, scenes, cfg, dict(gt=a.gt, voxel=voxel,
                                                            min_gt_voxels=a.min_gt_voxels,
                                                            min_pred_voxels=a.min_pred_voxels,
                                                            intra_threshold=intra))


def write_report(out, sweep_dir, variants, scenes, cfg, meta):
    rows = {}
    for v in variants:
        per = []
        for scene in scenes:
            f = sweep_dir / v.tag / f'{scene}.json'
            if f.exists():
                per.append(json.loads(f.read_text()))
        if per:
            rows[v.tag] = dict(
                variant=v.as_dict(), scenes=len(per),
                mean={k: float(np.nanmean([r[k] for r in per if k in r])) for k in METRIC_KEYS if k in per[0]},
                ms_per_frame=float(np.mean([sum(r['stage_ms'].values()) for r in per])),
                per_scene={r['scene']: r['ap50'] for r in per})
    if not rows:
        raise SystemExit(f'no sweep results under {sweep_dir}; run --stage run first')
    n = max(r['scenes'] for r in rows.values())
    ranked = sorted(rows, key=lambda t: -rows[t]['mean']['ap50'])
    best = ranked[0]
    by_mode = {}
    for mode in MODES:
        cand = [t for t in ranked if rows[t]['variant']['mode'] == mode]
        if cand:
            by_mode[mode] = cand[0]
    ref = [t for t in ranked if rows[t]['variant']['mode'] == 'linkage'
           and abs(rows[t]['variant']['cross_threshold'] - meta['intra_threshold']) < 1e-9]

    L = ['# 第 1 阶段：留出场景上的跨帧合并方案选择（不训练）', '',
         f'{n} 个留出场景（训练列表排序后最后 60 个），每场景 12 帧连续序列（40 帧间隔）。'
         f'评估用**体素级代理指标**（{meta["voxel"]} m 体素，GT = 跨帧点投票的多数实例，'
         f'thing 且 ≥ {meta["min_gt_voxels"]} 体素）：留出场景没有 ESAM 重建网格，'
         '验证集那套网格协议在这里不可用。GT 只进指标，不进合并。', '',
         f'帧内边沿用 GroupMergeNet checkpoint 阈值 {meta["intra_threshold"]}；'
         f'跨帧阈值在本表内扫描。共 {len(rows)} 个配置。', '',
         '## 1. 三个方案族各自最好的一行', '',
         '| 方案 | 跨帧规则 | 跨帧阈值 | 体素 AP50 | AP25 | 平均最佳 IoU | 召回@0.5 | masks/场景 | dup_rate | overmerge | frags/GT | ms/帧 |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for mode in MODES:
        t = by_mode.get(mode)
        if not t:
            continue
        m = rows[t]['mean']
        L.append(f"| {mode} | {rows[t]['variant']['rule']} | {rows[t]['variant']['cross_threshold']:g} | "
                 f"{m['ap50']:.4f} | {m['ap25']:.4f} | {m['iou']:.4f} | {m['r50']:.4f} | {m['masks']:.1f} | "
                 f"{m['dup_rate']:.3f} | {m['overmerge_rate']:.3f} | {m['frags_per_gt']:.2f} | {rows[t]['ms_per_frame']:.0f} |")
    L += ['', '## 2. 全部配置（按体素 AP50 降序）', '',
          '| 方案 | 规则 | 跨帧阈值 | 体素 AP50 | AP25 | 平均最佳 IoU | 召回@0.5 | masks/场景 | dup_rate | overmerge | frags/GT | 单帧物体占比 |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for t in ranked:
        v, m = rows[t]['variant'], rows[t]['mean']
        single = m['single_frame_objects'] / max(m['objects'], 1)
        L.append(f"| {v['mode']} | {v['rule']} | {v['cross_threshold']:g} | {m['ap50']:.4f} | {m['ap25']:.4f} | "
                 f"{m['iou']:.4f} | {m['r50']:.4f} | {m['masks']:.1f} | {m['dup_rate']:.3f} | "
                 f"{m['overmerge_rate']:.3f} | {m['frags_per_gt']:.2f} | {single:.3f} |")
    if ref:
        L += ['', f'参考：stage 0c 的 P2 配置（linkage + 规则，阈值取 checkpoint 的 '
                  f'{meta["intra_threshold"]}）在留出场景上的体素 AP50 = ' +
              '、'.join(f"{rows[t]['variant']['rule']} {rows[t]['mean']['ap50']:.4f}" for t in ref) + '。']
    L += ['', f'**选中方案：`{best}`**（体素 AP50 {rows[best]["mean"]["ap50"]:.4f}）。'
              f'它被拿去 16 个验证场景上跑一次（网格协议，段平滑），见 `summary.md`。']
    md = '\n'.join(L) + '\n'
    (out / 'sweep_summary.md').write_text(md)
    (out / 'sweep.json').write_text(json.dumps(dict(meta=meta, rows=rows, ranked=ranked,
                                                    by_mode=by_mode, best=best), indent=2))
    (out / 'selected.json').write_text(json.dumps(dict(tag=best, variant=rows[best]['variant'],
                                                       holdout_ap50=rows[best]['mean']['ap50'],
                                                       meta=meta), indent=2))
    print(md)


if __name__ == '__main__':
    main()
