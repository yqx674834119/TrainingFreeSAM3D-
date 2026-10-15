"""阶段 5 · P3：掩码集合上界（不训练）。

    source tools/env.sh
    python tools/phase5_bounds.py --stage collect [--limit N] [--shard i/n]   # GPU：原始 mask 池
    python tools/phase5_bounds.py --stage report                              # CPU：汇总 bounds.md/json
    python tools/phase5_bounds.py --stage all
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.analysis import bounds as bd  # noqa: E402
from freesam3d.analysis.rank_probe import load_scene  # noqa: E402
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.models.fastsam import FastSAMRunner  # noqa: E402
from freesam3d.models.instance_head import FastSAMEncoder  # noqa: E402
from freesam3d.streaming.unified_train import load_unified_checkpoint  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--stage', default='all', choices=['collect', 'trim', 'report', 'all'])
    p.add_argument('--scenes', nargs='*', default=None)
    p.add_argument('--shard', default='0/1')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    out = resolve_path(cfg['phase5']['bounds']['out'])
    p1 = resolve_path(cfg['phase5']['out_root']) / 'rank_probe' / 'cache'
    dedup_thr = float(cfg['phase5']['rank_probe'].get('dedup', 0.7))
    if a.stage in ('collect', 'all'):
        dev = a.device if torch.cuda.is_available() else 'cpu'
        fast = cfg.get('fastsam') or load_config(resolve_path(cfg['lifting_config']))['fastsam']
        runner = FastSAMRunner(fast['checkpoint'], dev, fast.get('imgsz', 640),
                               fast.get('conf', 0.1), fast.get('iou', 0.9),
                               fast.get('fallback_iou', 0.7))
        scenes = a.scenes or list(cfg['stream_v2']['holdout_scenes'])
        i, n = map(int, a.shard.split('/'))
        scenes = [s for k, s in enumerate(scenes) if k % n == i]
        if a.limit:
            scenes = scenes[:a.limit]
        cache_dir = resolve_path(cfg['cache_root']) / 'seq' / 'train'
        image_root = cfg['dataset']['root']
        (out / 'raw').mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for si, scene in enumerate(scenes):
            dst = out / 'raw' / f'{scene}.json'
            if dst.exists() and not a.overwrite:
                print(f'  {scene}: cached', flush=True)
                continue
            d = bd.raw_pool_scene(scene, cfg, runner, cache_dir, image_root,
                                  voxel=float(cfg['stream_v2']['voxel']))
            c = load_scene(p1 / f'{scene}.npz')
            gt_vox, keep = c['gt_vox'], c['keep']
            rec = dict(scene=scene, masks_per_frame=float(np.mean(d['per_frame'])),
                       pool_n=int(len(d['pool'])))
            if len(keep) and len(d['pool']):
                rec['raw_ap50'], _ = bd.pool_ap50(d['pool'], gt_vox, keep, dev)
                rec['raw_ap50_dedup'], rec['raw_n_dedup'] = bd.pool_ap50(d['pool'], gt_vox, keep, dev,
                                                                         dedup=dedup_thr)
                union = np.concatenate([d['pool'], c['masks']], 0)
                rec['union_ap50'], _ = bd.pool_ap50(union, gt_vox, keep, dev)
                rec['union_ap50_dedup'], _ = bd.pool_ap50(union, gt_vox, keep, dev, dedup=dedup_thr)
                rec['recall_raw'] = list(bd.recall_at_50(bd.iou_matrix_np(d['pool'], gt_vox, keep)))
            iou_cur = bd.iou_matrix_np(c['masks'], gt_vox, keep) if len(keep) else np.zeros((0, 0))
            rec['recall_cur'] = list(bd.recall_at_50(iou_cur))
            dst.write_text(json.dumps(rec, ensure_ascii=False))
            print(f'[{a.shard}] {si + 1}/{len(scenes)} {scene}: raw {rec.get("raw_ap50", float("nan")):.4f}'
                  f' | union {rec.get("union_ap50", float("nan")):.4f}'
                  f' | pool {rec["pool_n"]} ({(time.time() - t0) / 60:.1f} min)', flush=True)
    if a.stage == 'trim':
        dev = a.device if torch.cuda.is_available() else 'cpu'
        model, std, ck = load_unified_checkpoint(
            resolve_path(cfg['phase5']['rank_probe']['checkpoint']), dev)
        fast = cfg.get('fastsam') or load_config(resolve_path(cfg['lifting_config']))['fastsam']
        enc = FastSAMEncoder(fast['checkpoint'], dev, fast.get('imgsz', 640), raw=True)
        thr = [float(t) for t in cfg['phase5']['rank_probe']['thresholds']]
        over = dict(threshold=thr[0], cross_threshold=thr[1], bo_threshold=thr[1], oo_threshold=thr[2])
        scenes = a.scenes or list(cfg['stream_v2']['holdout_scenes'])
        i, n = map(int, a.shard.split('/'))
        scenes = [s for k, s in enumerate(scenes) if k % n == i]
        if a.limit:
            scenes = scenes[:a.limit]
        cache_dir = resolve_path(cfg['cache_root']) / 'seq' / 'train'
        (out / 'trim').mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for si, scene in enumerate(scenes):
            dst = out / 'trim' / f'{scene}.json'
            if dst.exists() and not a.overwrite:
                print(f'  {scene}: cached', flush=True)
                continue
            c = load_scene(p1 / f'{scene}.npz')
            d = bd.feature_trim_scene(scene, cfg, model, std, enc, dev, cache_dir,
                                      cfg['dataset']['root'], c['gt_vox'], c['keep'],
                                      taus=(-1, 0.3, 0.5, 0.7), mem_overrides=over)
            dst.write_text(json.dumps(d, ensure_ascii=False))
            print(f'[trim] {si + 1}/{len(scenes)} {scene}: base {d["trim_-1_ap50"]:.4f} | '
                  f't0.3 {d["trim_0.3_ap50"]:.4f} t0.5 {d["trim_0.5_ap50"]:.4f} '
                  f't0.7 {d["trim_0.7_ap50"]:.4f} ({(time.time() - t0) / 60:.1f} min)', flush=True)
    if a.stage in ('report', 'all'):
        bd.write_report(out, cfg)
    print('done', flush=True)


if __name__ == '__main__':
    main()
