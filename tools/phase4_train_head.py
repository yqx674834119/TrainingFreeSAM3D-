"""Train the phase-4 2D instance head (PartField-style contrastive, frozen FastSAM).

Data: ESAM 2D export train scenes, the same consecutive frames as ``merge_cache/seq/train``;
supervision is the cached 2D instance id per sampled point. The 60 holdout scenes are excluded
by default. Writes one checkpoint + history JSON + curves per seed.

    python tools/phase4_train_head.py --seeds 0 1 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from freesam3d.config import load_config, resolve_path
from freesam3d.models.instance_head import save_head, train_instance_head
from freesam3d.vis.phase4 import training_curves


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/merge_net.yaml')
    ap.add_argument('--seeds', type=int, nargs='+', default=None, help='默认取配置 phase4.train.seeds')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--steps', type=int, default=None, help='每 epoch 的步数')
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--image-root', default=None)
    ap.add_argument('--out', default=None, help='默认 <phase4.out_root>/head')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--include-holdout', action='store_true', help='调试用：不排除 60 个留出场景')
    ap.add_argument('--max-scenes', type=int, default=0, help='调试用：只用前 N 个场景')
    args = ap.parse_args()

    cfg = load_config(args.config)
    p4 = cfg['phase4']
    out = Path(args.out) if args.out else resolve_path(p4['out_root']) / 'head'
    cache_dir = Path(args.cache_dir) if args.cache_dir \
        else resolve_path(cfg['cache_root']) / 'seq' / 'train'
    image_root = args.image_root or cfg['dataset']['root']

    scenes = sorted({p.stem.rsplit('_', 1)[0] for p in cache_dir.glob('*.npz')})
    if not args.include_holdout:
        holdout = set(cfg['stream_v2']['holdout_scenes'])
        scenes = [s for s in scenes if s not in holdout]
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    seeds = args.seeds if args.seeds is not None else list(p4['train']['seeds'])
    print(f'scenes {len(scenes)} | holdout excluded {not args.include_holdout} | seeds {seeds}',
          flush=True)

    for seed in seeds:
        t0 = time.time()
        head, log_tau, info = train_instance_head(
            cfg, device=args.device, seed=seed, scenes=scenes, cache_dir=cache_dir,
            image_root=image_root, epochs=args.epochs, steps_per_epoch=args.steps)
        dst = out / f'seed{seed}'
        save_head(dst / 'head.pt', head, log_tau,
                  extra=dict(seed=seed, scenes=len(scenes), history=info['history']))
        (dst / 'history.json').write_text(json.dumps(info, indent=1))
        training_curves(info['history'], dst / 'curves.png', title=f'phase4 instance head, seed {seed}')
        print(f'seed {seed} done in {time.time() - t0:.0f}s -> {dst}', flush=True)


if __name__ == '__main__':
    main()
