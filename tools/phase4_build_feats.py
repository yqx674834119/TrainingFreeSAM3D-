"""Build the phase-4 fragment-level instance feature cache with a trained instance head.

One npz per frame: ``{feat (G,32) fp16}`` — the mean of the frame's member *point* instance
features per FastSAM fragment. Pixels/groups come from the merge cache, so the fragment
partition is identical to every other stage. Resumable.

    python tools/phase4_build_feats.py --head work_dirs/stream_v2/phase4/head/seed0/head.pt --split holdout
    python tools/phase4_build_feats.py --head ... --split val
    python tools/phase4_build_feats.py --head ... --split train
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from freesam3d.config import load_config, resolve_path
from freesam3d.models.instance_head import FastSAMEncoder, build_feature_cache, load_head

SPLITS = ('train', 'holdout', 'val')


def scene_of(path):
    return Path(path).stem.rsplit('_', 1)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/merge_net.yaml')
    ap.add_argument('--head', required=True, help='训练好的 head.pt')
    ap.add_argument('--split', choices=SPLITS, default='holdout')
    ap.add_argument('--cache-dir', default=None,
                    help='默认 train/holdout 用 merge_cache/seq/train，val 用 merge_cache/val')
    ap.add_argument('--out', default=None, help='默认 <phase4.feat_cache>/<split>')
    ap.add_argument('--image-root', default=None)
    ap.add_argument('--limit', type=int, default=0, help='调试用：只建前 N 帧')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = resolve_path(cfg['cache_root'])
    holdout = set(cfg['stream_v2']['holdout_scenes'])
    cache_dir = Path(args.cache_dir) if args.cache_dir \
        else (root / 'val' if args.split == 'val' else root / 'seq' / 'train')
    if args.split == 'train':
        keep = lambda s: s not in holdout          # noqa: E731
    elif args.split == 'holdout':
        keep = lambda s: s in holdout              # noqa: E731
    else:
        val_scenes = set(cfg['dataset']['val_scenes'])
        keep = lambda s: s in val_scenes           # noqa: E731
    files = [p for p in sorted(cache_dir.glob('*.npz')) if keep(scene_of(p))]
    out = Path(args.out) if args.out else resolve_path(cfg['phase4']['feat_cache']) / args.split
    print(f'split {args.split}: {len(files)} frames from {cache_dir} -> {out}', flush=True)

    head, ck = load_head(args.head, device=args.device)
    fast = cfg.get('fastsam') or load_config(resolve_path(cfg['lifting_config']))['fastsam']
    enc = FastSAMEncoder(fast['checkpoint'], args.device, fast.get('imgsz', 640), raw=True)
    stats = build_feature_cache(head, enc, files, args.image_root or cfg['dataset']['root'], out,
                                limit=args.limit, overwrite=args.overwrite, device=args.device,
                                center=bool(cfg['phase4'].get('feat_center', True)))
    stats.update(split=args.split, head=args.head, seed=ck.get('seed'))
    (out / 'stats.json').write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
