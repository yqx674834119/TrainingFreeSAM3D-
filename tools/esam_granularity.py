"""Granularity study: FastSAM groups vs ESAM decoder masks vs merge-only oracle.

    source tools/env.sh
    python tools/esam_prepare.py && python tools/esam_export.py   # once
    python tools/esam_granularity.py [--scenes ...]
Outputs work_dirs/esam_eval/granularity/{summary.md,summary.json,*.csv,*.png,frames/}.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.evaluation import granularity as G  # noqa: E402
from freesam3d.evaluation.instance import oracle_merge  # noqa: E402
from freesam3d.vis import granularity as V  # noqa: E402


def write_csv(rows, path):
    if rows:
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def to_markdown(s, official, scenes):
    lines = [f'# ESAM 粒度对齐验证（{len(scenes)} 个 ScanNet 验证场景，逐帧，20k 点）', '',
             f'官方 ESAM-E 场景级指标（同一批场景，重建网格上）：AP {official.get("all_ap", float("nan")):.4f} / '
             f'AP50 {official.get("all_ap_50%", float("nan")):.4f} / AP25 {official.get("all_ap_25%", float("nan")):.4f}', '']
    for subset in ('thing', 'all'):
        lines += [f'## 逐帧对比（GT 子集：{subset}，帧平均）', '',
                  '| 方法 | mask 数/帧 | 平均最佳 IoU | 召回@0.5 | 召回@0.25 | 精度@0.5 | AP50 | AP25 | 碎片数/GT |',
                  '|---|---|---|---|---|---|---|---|---|']
        for m in G.METHODS:
            r = s.get(f'{subset}/{m}')
            if r:
                lines.append(f'| {m} | {r["num_pred"]:.1f} | {r["mean_best_iou"]:.3f} | {r["recall@0.5"]:.3f} | '
                             f'{r["recall@0.25"]:.3f} | {r["precision@0.5"]:.3f} | {r["AP50"]:.3f} | {r["AP25"]:.3f} | '
                             f'{r["mean_fragments"]:.2f} |')
        lines.append('')
    p = s.get('per_gt', {})
    if p:
        lines += ['## 逐 GT 实例（thing）', '', '| 指标 | 值 |', '|---|---|']
        lines += [f'| {k} | {v:.3f} |' if isinstance(v, float) else f'| {k} | {v} |' for k, v in p.items()]
        lines.append('')
    e = s.get('esam_masks', {})
    if e:
        lines += ['## ESAM mask 由哪些 FastSAM 组构成', '', '| 指标 | 值 |', '|---|---|']
        lines += [f'| {k} | {v:.3f} |' if isinstance(v, float) else f'| {k} | {v} |' for k, v in e.items()]
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/esam_eval.yaml')
    p.add_argument('--scenes', nargs='*')
    p.add_argument('--panels-per-scene', type=int, default=2)
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    ws = resolve_path(cfg['workspace'])
    mv = ws / 'data/scannet200-mv_fast'
    out = ws / 'granularity'
    (out / 'frames').mkdir(parents=True, exist_ok=True)
    ev = cfg['eval']
    stuff = G.stuff_ids(cfg['esam_root'] + '/data/scannet200/meta_data/scannetv2-labels.combined.tsv')
    scenes = a.scenes or [s for s in cfg['scenes'] if (ws / 'predictions' / f'{s}.npz').exists()]
    acc = G.Accumulator()
    for scene in scenes:
        frames = list(G.load_scene_frames(mv, ws / 'predictions' / f'{scene}.npz', scene))
        for fd in frames:
            G.evaluate_frame(fd, stuff, acc, ev['purity'], ev['min_gt_points'], tuple(ev['iou_thresholds']),
                             ev['adj_eps'])
        for fd in frames[len(frames) // 4::max(1, len(frames) // a.panels_per_scene)][:a.panels_per_scene]:
            V.frame_panel(fd, oracle_merge(fd.groups, fd.gt, ev['purity']), mv / '2D' / scene,
                          out / 'frames' / f'{scene}_{fd.frame}.png')
        print(f'{scene}: {len(frames)} frames', flush=True)

    summary = G.summarize(acc)
    official_path = ws / 'predictions' / 'esam_official_metrics.json'
    official = json.loads(official_path.read_text()) if official_path.exists() else {}
    (out / 'summary.json').write_text(json.dumps(dict(scenes=scenes, official_scene_level=official,
                                                      summary=summary), indent=2))
    (out / 'summary.md').write_text(to_markdown(summary, official, scenes))
    write_csv(acc.frame_rows, out / 'per_frame.csv')
    write_csv(acc.gt_rows, out / 'per_gt.csv')
    write_csv(acc.esam_mask_rows, out / 'esam_masks.csv')
    V.summary_bars(summary, out / '01_summary_thing.png', 'thing')
    V.summary_bars(summary, out / '01_summary_all.png', 'all')
    V.composition(acc, out / '02_esam_composition.png')
    V.per_gt_scatter(acc, out / '03_per_gt_scatter.png')
    print((out / 'summary.md').read_text())


if __name__ == '__main__':
    main()
