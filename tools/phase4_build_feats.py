"""Build the phase-4b fragment-level feature caches: trained head, raw frozen layer, or geometry descriptors.

Exactly one mode is required:

* ``--head <ckpt>``    a trained instance head -> mean of the member points' unit features (G,32);
* ``--layer <name>``   P2D.0 raw-layer probe: one frozen map's per-point features (frame-mean
                       subtracted, L2) averaged per fragment — the appearance-information ceiling;
* ``--descriptor``     P3D.0 geometry probe: per-fragment shape descriptors (16-d), z-scored on the
                       holdout rows then L2-normalised — the geometry-information ceiling.

Pixels/groups come from the merge cache, so the fragment partition is identical to every other stage.

    python tools/phase4_build_feats.py --head work_dirs/stream_v2/phase4/head/seed0/head.pt --split holdout
    python tools/phase4_build_feats.py --layer proto_hidden --split holdout
    python tools/phase4_build_feats.py --descriptor --split holdout
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from freesam3d.config import load_config, resolve_path
from freesam3d.models.instance_head import (HEAD_NAMES, FastSAMEncoder, build_feature_cache,
                                            build_raw_layer_cache, load_head)
from freesam3d.models.instance_head3d import build_descriptor_cache

SPLITS = ('train', 'holdout', 'val')


def scene_of(path):
    return Path(path).stem.rsplit('_', 1)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/merge_net.yaml')
    ap.add_argument('--head', default=None, help='训练好的 head.pt')
    ap.add_argument('--layer', default=None, choices=list(HEAD_NAMES),
                    help='P2D.0：不给 head，缓存该冻结层的原始逐点特征（减帧均值 + L2）')
    ap.add_argument('--descriptor', action='store_true', help='P3D.0：逐碎片几何描述子（不用模型）')
    ap.add_argument('--split', choices=SPLITS, default='holdout')
    ap.add_argument('--cache-dir', default=None,
                    help='默认 train/holdout 用 merge_cache/seq/train，val 用 merge_cache/val')
    ap.add_argument('--out', default=None)
    ap.add_argument('--image-root', default=None)
    ap.add_argument('--limit', type=int, default=0, help='调试用：只建前 N 帧')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--no-center', action='store_true',
                    help='head 模式：不建中心化缓存（默认按 phase4.feat_center；用于检验双重中心化）')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    if sum([bool(args.head), bool(args.layer), args.descriptor]) != 1:
        raise SystemExit('pass exactly one of --head / --layer / --descriptor')

    cfg = load_config(args.config)
    p4b = cfg['phase4b']
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

    if args.descriptor:
        out = Path(args.out) if args.out else resolve_path(p4b['feat_cache']) / f'{args.split}_desc'
        print(f'split {args.split}: {len(files)} frames from {cache_dir} -> {out} (descriptor)', flush=True)
        stats = build_descriptor_cache(files, out, limit=args.limit, overwrite=args.overwrite,
                                       device=args.device,
                                       normal_radius=float(p4b['desc']['normal_radius']),
                                       normal_k=int(p4b['desc']['normal_k']))
        stats.update(split=args.split, mode='descriptor')
    else:
        tag = f'{args.split}_raw_{args.layer}' if args.layer else args.split
        out = Path(args.out) if args.out else resolve_path(p4b['feat_cache']) / tag
        print(f'split {args.split}: {len(files)} frames from {cache_dir} -> {out}', flush=True)
        fast = cfg.get('fastsam') or load_config(resolve_path(cfg['lifting_config']))['fastsam']
        enc = FastSAMEncoder(fast['checkpoint'], args.device, fast.get('imgsz', 640), raw=True)
        if args.layer:
            stats = build_raw_layer_cache(enc, args.layer, files,
                                          args.image_root or cfg['dataset']['root'], out,
                                          limit=args.limit, overwrite=args.overwrite)
            stats.update(split=args.split, mode='raw_layer', layer=args.layer)
        else:
            head, ck = load_head(args.head, device=args.device)
            center = False if args.no_center else bool(cfg['phase4'].get('feat_center', True))
            stats = build_feature_cache(head, enc, files, args.image_root or cfg['dataset']['root'], out,
                                        limit=args.limit, overwrite=args.overwrite, device=args.device,
                                        center=center)
            stats.update(split=args.split, mode='head', head=args.head, seed=ck.get('seed'))
    (out / 'stats.json').write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
