"""Evaluate GroupMergeNet checkpoints on the val cache (16 scenes, same protocol as the ESAM study).

    source tools/env.sh
    python tools/merge_eval.py --checkpoints full geom_only edge_only
Writes work_dirs/merge_net/eval/{summary.md,summary.json,per_frame.json}.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.merging.data import FrameSet, list_cache  # noqa: E402
from freesam3d.merging.evaluate import evaluate, to_markdown  # noqa: E402
from freesam3d.merging.train import load_checkpoint  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--checkpoints', nargs='*', default=['full'], help='tags under work_dirs/merge_net/')
    p.add_argument('--threshold', type=float, help='override calibrated thresholds')
    p.add_argument('--cosine-threshold', type=float, default=0.5)
    p.add_argument('--device', default='cuda')
    p.add_argument('--tag', default='eval')
    p.add_argument('--gt', choices=['2d', '3d'], help='GT source (default: config eval.gt)')
    p.add_argument('--min-points', type=int, help='drop model masks smaller than this (default: config)')
    p.add_argument('--min-score', type=float, help='drop model masks below this objectness (default: config)')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    device = a.device if torch.cuda.is_available() else 'cpu'
    frames = FrameSet(list_cache(resolve_path(cfg['cache_root']) / 'val'), keep_points=True).frames
    cks = {}
    for tag in a.checkpoints:
        model, std, ck = load_checkpoint(resolve_path('work_dirs/merge_net') / tag / 'model.pt', device)
        thr = a.threshold if a.threshold is not None else (cfg['merge']['threshold'] or ck['threshold'] or 0.5)
        cks[tag] = (model, std, float(thr))
        print(f'{tag}: params {model.num_params()} threshold {thr}')
    ev = cfg['eval']
    gt_source = a.gt or ev.get('gt', '2d')
    stuff = set(ev['stuff_raw'] if gt_source == '3d' else ev['stuff_nyu40'])
    mp = a.min_points if a.min_points is not None else cfg['merge'].get('min_points', 0)
    msc = a.min_score if a.min_score is not None else cfg['merge'].get('min_score', 0.0)
    rows, means, ms = evaluate(frames, cks, stuff, ev['min_gt_points'], a.cosine_threshold,
                               cfg['graph']['adjacency_eps'], cfg['merge']['linkage'], device, gt_source, mp, msc)
    print(f'model-mask filter: min_points={mp} min_score={msc}')
    ref_path = resolve_path('work_dirs/esam_eval/granularity/summary.json')
    esam_ref = None
    if ref_path.exists():
        s = json.loads(ref_path.read_text())['summary']
        esam_ref = {m: s[f'thing/{m}'] for m in ('ESAM-snap', 'ESAM') if f'thing/{m}' in s}
    out = resolve_path('work_dirs/merge_net') / a.tag
    out.mkdir(parents=True, exist_ok=True)
    md = to_markdown(means, ms, len(rows), esam_ref, gt_source=gt_source)
    (out / 'summary.md').write_text(md)
    (out / 'summary.json').write_text(json.dumps(dict(means=means, timing=ms, thresholds={k: v[2] for k, v in cks.items()},
                                                      esam_reference=esam_ref, frames=len(rows), gt_source=gt_source,
                                                      min_points=mp, min_score=msc), indent=2))
    (out / 'per_frame.json').write_text(json.dumps(rows, indent=1))
    print(md)


if __name__ == '__main__':
    main()
