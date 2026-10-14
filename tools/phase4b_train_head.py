"""Train the phase-4b 2D instance head — track A, restricted to the low-cost fixes 0/1/3.

The §一 raw-layer probe put the best frozen layer (``backbone``) at 0.7208 on the hardest pair,
i.e. in the 0.70-0.78 band, so track A is restricted to the three low-cost fixes:

* ``--recipe fusion``  fix 0 only: concat fusion instead of the phase-4 sum (recipe otherwise
                       unchanged), to isolate that change;
* ``--recipe center``  fixes 0 + 1 only (isolates centring in the loss);
* ``--recipe miner``   fixes 0 + 3 only (isolates the miner fixes);
* ``--recipe all``     fixes 0 + 1 + 3: concat fusion, centring in the loss, and the miner fixes
                       (GT erosion, hard-negative share <= 50%, 2-epoch warm-up);
* ``--recipe none``    the phase-4 recipe verbatim (control).

Fix 2 (object-level objective) and fix 4 (x10 budget) are out of scope per the §一 verdict.

    python tools/phase4b_train_head.py --recipe fusion --tag concat
    python tools/phase4b_train_head.py --recipe all    --tag fixed
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

RECIPES = ('none', 'fusion', 'center', 'miner', 'all')


def build_recipe(cfg, name):
    """The ``recipe`` dict for ``train_instance_head`` (``None`` = phase-4 behaviour)."""
    if name == 'none':
        return None
    h = cfg['phase4b']['head2d']
    out = {'fusion': str(h['fusion'])}
    if name in ('center', 'all'):
        out['center_loss'] = bool(h['center_loss'])
    if name in ('miner', 'all'):
        out.update(erode_px=int(h['erode_px']), hard_frac_max=float(h['hard_frac_max']),
                   hard_warmup_epochs=int(h['hard_warmup_epochs']))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/merge_net.yaml')
    ap.add_argument('--recipe', choices=RECIPES, default='all')
    ap.add_argument('--tag', default=None, help='默认用 --recipe')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--image-root', default=None)
    ap.add_argument('--out', default=None, help='默认 <phase4b.out_root>/head2d')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-scenes', type=int, default=0, help='调试用：只用前 N 个场景')
    args = ap.parse_args()

    cfg = load_config(args.config)
    p4b = cfg['phase4b']
    out = Path(args.out) if args.out else resolve_path(p4b['out_root']) / 'head2d'
    cache_dir = Path(args.cache_dir) if args.cache_dir \
        else resolve_path(cfg['cache_root']) / 'seq' / 'train'
    image_root = args.image_root or cfg['dataset']['root']
    holdout = set(cfg['stream_v2']['holdout_scenes'])
    scenes = [s for s in sorted({p.stem.rsplit('_', 1)[0] for p in cache_dir.glob('*.npz')})
              if s not in holdout]
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    recipe = build_recipe(cfg, args.recipe)
    tag = args.tag or args.recipe
    print(f'recipe {args.recipe} -> {recipe} | scenes {len(scenes)} | seeds {args.seeds}', flush=True)

    for seed in args.seeds:
        t0 = time.time()
        head, log_tau, info = train_instance_head(
            cfg, device=args.device, seed=seed, scenes=scenes, cache_dir=cache_dir,
            image_root=image_root, epochs=args.epochs, steps_per_epoch=args.steps, recipe=recipe)
        dst = out / tag / f'seed{seed}'
        save_head(dst / 'head.pt', head, log_tau,
                  extra=dict(seed=seed, scenes=len(scenes), recipe=recipe, history=info['history']))
        (dst / 'history.json').write_text(json.dumps(info, indent=1))
        training_curves(info['history'], dst / 'curves.png', title=f'phase4b head {tag} seed {seed}')
        print(f'{tag} seed {seed} done in {time.time() - t0:.0f}s -> {dst}', flush=True)


if __name__ == '__main__':
    main()
