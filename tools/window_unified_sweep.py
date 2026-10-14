"""Select the stage-2 thresholds on the 60 holdout scenes (voxel proxy AP50, quality ranking).

    source tools/env.sh
    CK=work_dirs/stream_v2/phase2/model/A/seed0/model.pt
    # one shared threshold for all three edge types
    python tools/window_unified_sweep.py --checkpoint $CK --tag shared --intra 0.3 0.4 0.5 0.6 0.7 --bo 0.3 0.4 0.5 0.6 0.7
    # then one per type, around the shared optimum
    python tools/window_unified_sweep.py --checkpoint $CK --tag pertype --intra 0.5 0.6 --bo 0.2 0.3 0.4 --oo 0.2 0.3 0.4
    python tools/window_unified_sweep.py --report

--intra/--bo/--oo are lists; the sweep runs their full product (--shard i/n splits the scenes).
Every triple is written to <out_root>/phase2/sweep/<tag>_<intra>_<bo>_<oo>.json; --report picks the
best AP50 overall and writes selected.json next to them.
"""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.streaming.unified_sweep import run_triple, save_entry, select, triples  # noqa: E402
from freesam3d.streaming.unified_train import load_unified_checkpoint  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--tag', default='shared')
    p.add_argument('--intra', nargs='+', type=float, default=[0.3, 0.4, 0.5, 0.6, 0.7])
    p.add_argument('--bo', nargs='+', type=float, default=None, help='defaults to --intra')
    p.add_argument('--oo', nargs='+', type=float, default=None, help='defaults to --intra')
    p.add_argument('--shared', action='store_true',
                   help='one shared threshold: use --intra as the grid and ignore --bo/--oo')
    p.add_argument('--shard', default='0/1')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--report', action='store_true', help='only aggregate the sweep files and select')
    p.add_argument('--out', default=None)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    sweep_dir = Path(a.out) if a.out else resolve_path(cfg['stream_v2']['out_root']) / 'phase2' / 'sweep'
    if a.report:
        best = select(sweep_dir, checkpoint=a.checkpoint)
        print(f'best {best["tag"]} intra {best["intra"]} bo {best["bo"]} oo {best["oo"]} '
              f'-> AP50 {best["metrics"]["ap50"]:.4f} AP25 {best["metrics"]["ap25"]:.4f} '
              f'IoU {best["metrics"]["iou"]:.4f} masks {best["metrics"]["masks"]:.1f}')
        return
    if not a.checkpoint:
        raise SystemExit('--checkpoint is required for a sweep (or use --report)')
    dev = a.device if torch.cuda.is_available() else 'cpu'
    if str(dev).startswith('cuda'):
        print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    model, std, ck = load_unified_checkpoint(resolve_path(a.checkpoint), dev)
    print(f'checkpoint {a.checkpoint} (seed {ck.get("seed")})', flush=True)
    cache = resolve_path(cfg['cache_root']) / 'seq' / 'train'
    scenes = list(cfg['stream_v2']['holdout_scenes'])
    i, n = map(int, a.shard.split('/'))
    scenes = [s for k, s in enumerate(scenes) if k % n == i]
    if a.limit:
        scenes = scenes[:a.limit]
    grid = (triples(a.intra, a.intra, a.intra) if a.shared
            else triples(a.intra, a.bo or a.intra, a.oo or a.intra))
    print(f'{len(grid)} triples x {len(scenes)} holdout scenes -> {sweep_dir}', flush=True)
    for t in grid:
        tag = f'{a.tag}_{t[0]:.2f}_{t[1]:.2f}_{t[2]:.2f}'
        dst = sweep_dir / f'{tag}.json'
        if dst.exists():
            print(f'  {tag}: already done', flush=True)
            continue
        rows = run_triple(t, scenes, cfg, model, std, dev, cache, set(cfg['eval']['stuff_nyu40']))
        save_entry(dst, a.tag, t, rows, extra=dict(checkpoint=str(a.checkpoint), shard=a.shard))


if __name__ == '__main__':
    main()
