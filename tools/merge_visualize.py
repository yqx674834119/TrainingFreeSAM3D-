"""Render qualitative GroupMergeNet panels on val frames (worst / median / best by model IoU).

    source tools/env.sh
    python tools/merge_visualize.py --checkpoint full --eval-tag eval_3d
Writes work_dirs/merge_net/<eval-tag>/vis/*.png and 00_comparison.png.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.evaluation.instance import oracle_merge_adjacent  # noqa: E402
from freesam3d.merging.cluster import merge_groups  # noqa: E402
from freesam3d.merging.data import FrameSet, list_cache  # noqa: E402
from freesam3d.merging.train import load_checkpoint, predict_frame  # noqa: E402
from freesam3d.vis import merging as V  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--checkpoint', default='full')
    p.add_argument('--eval-tag', default='eval_3d')
    p.add_argument('--per-bucket', type=int, default=4)
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    ev_dir = resolve_path('work_dirs/merge_net') / a.eval_tag
    summary = json.loads((ev_dir / 'summary.json').read_text())
    rows = json.loads((ev_dir / 'per_frame.json').read_text())
    gt_source = summary.get('gt_source', '2d')
    out = ev_dir / 'vis'
    out.mkdir(exist_ok=True)
    V.comparison_bars(summary['means'], out / '00_comparison.png', summary.get('esam_reference'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, std, ck = load_checkpoint(resolve_path('work_dirs/merge_net') / a.checkpoint / 'model.pt', device)
    thr = summary['thresholds'][a.checkpoint]
    by_name = {f.stem: f for f in list_cache(resolve_path(cfg['cache_root']) / 'val')}
    rows = [r for r in rows if a.checkpoint in r and r['num_gt'] >= 3]
    rows.sort(key=lambda r: r[a.checkpoint]['mean_best_iou'] - r['FastSAM']['mean_best_iou'])
    picks = {'worst_delta': rows[:a.per_bucket], 'median': rows[len(rows) // 2 - a.per_bucket // 2:len(rows) // 2 + a.per_bucket // 2],
             'best_delta': rows[-a.per_bucket:]}
    root = Path(cfg['dataset']['root'])
    for bucket, rs in picks.items():
        for i, r in enumerate(rs):
            fr = FrameSet([by_name[r['name']]], keep_points=True).frames[0]
            gt = fr['gt_instance_3d'] if gt_source == '3d' else fr['gt_instance']
            prob = predict_frame(model, std, fr, device)
            merged = merge_groups(fr['groups'], fr['edges'], prob, thr, cfg['merge']['linkage'])
            oracle = oracle_merge_adjacent(fr['groups'], gt, fr['xyz'], 0.8, cfg['graph']['adjacency_eps'])
            scene, fid = fr['name'].rsplit('_', 1)
            extra = f"  IoU FastSAM {r['FastSAM']['mean_best_iou']:.2f} -> model {r[a.checkpoint]['mean_best_iou']:.2f} (oracle-adj {r['Oracle-adj']['mean_best_iou']:.2f})"
            V.merge_panel(fr, merged, oracle, gt, root / scene / 'color' / f'{fid}.jpg',
                          out / f'{bucket}_{i}_{fr["name"]}.png', extra, prob, thr)
    print(f'wrote panels to {out}')


if __name__ == '__main__':
    main()
