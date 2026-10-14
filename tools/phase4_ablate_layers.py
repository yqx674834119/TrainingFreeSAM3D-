"""Train drop-one-input-layer ablation heads for the phase-4 instance head.

Each run drops one (or several) of the six FastSAM feature maps and otherwise uses the phase-4
training recipe (same scenes, same epochs/steps, one seed by default). Diagnostic only: which
layers the learned feature actually depends on (probe AUC per drop).

    python tools/phase4_ablate_layers.py --drops backbone --seeds 0
    python tools/phase4_ablate_layers.py --drops backbone neck_p4 neck_p5 proto_hidden   # 只留 p3+proto
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.models.instance_head import HEAD_INPUTS, save_head, train_instance_head  # noqa: E402
from freesam3d.vis.phase4 import training_curves  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/merge_net.yaml')
    ap.add_argument('--drops', nargs='+', required=True, help='要丢掉的层（可多个）')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--out', default=None, help='默认 <phase4.out_root>/head_ablate')
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--image-root', default=None)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-scenes', type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out) if args.out else resolve_path(cfg['phase4']['out_root']) / 'head_ablate'
    cache_dir = Path(args.cache_dir) if args.cache_dir \
        else resolve_path(cfg['cache_root']) / 'seq' / 'train'
    image_root = args.image_root or cfg['dataset']['root']
    holdout = set(cfg['stream_v2']['holdout_scenes'])
    scenes = [s for s in sorted({p.stem.rsplit('_', 1)[0] for p in cache_dir.glob('*.npz')})
              if s not in holdout]
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]

    known = [k for k, _ in HEAD_INPUTS]
    bad = [d for d in args.drops if d not in known]
    if bad:
        raise SystemExit(f'unknown layers {bad}; choose from {known}')
    chans = {k: c for k, c in HEAD_INPUTS if k not in args.drops}
    tag = 'drop_' + '_'.join(args.drops)
    for seed in args.seeds:
        t0 = time.time()
        head, log_tau, info = train_instance_head(
            cfg, device=args.device, seed=seed, scenes=scenes, cache_dir=cache_dir,
            image_root=image_root, epochs=args.epochs, steps_per_epoch=args.steps,
            in_channels=chans)
        dst = out / tag / f'seed{seed}'
        save_head(dst / 'head.pt', head, log_tau,
                  extra=dict(seed=seed, scenes=len(scenes), drops=args.drops, history=info['history']))
        (dst / 'history.json').write_text(json.dumps(info, indent=1))
        training_curves(info['history'], dst / 'curves.png', title=f'phase4 head {tag} seed {seed}')
        print(f'{tag} seed {seed}: {time.time() - t0:.0f}s -> {dst}', flush=True)


if __name__ == '__main__':
    main()
