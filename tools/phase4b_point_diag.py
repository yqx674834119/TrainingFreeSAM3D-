"""Point-level diagnostic (F4.3 口径): is the trained head weaker than its own input?

F4.3 measured point-pair cosine AUC (positive = same instance) on intra-frame pairs and on
cross-frame pairs 1 and 3 cached frames apart, per holdout scene, with per-point features centred
(frame mean) and L2-normalised. This reproduces that reading for any feature source — a raw frozen
layer or a trained head — so track A's intermediate acceptance can be checked: the fixed head must
reach at least (best raw layer - 0.02) on the point level before the full probe is worth running.

    python tools/phase4b_point_diag.py --layer backbone
    python tools/phase4b_point_diag.py --head work_dirs/stream_v2/phase4b/head2d/all/seed0/head.pt --tag fixed

Diagnostic only: GT labels the pairs, nothing here feeds a merge.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from freesam3d.config import load_config, resolve_path
from freesam3d.models.instance_head import (HEAD_NAMES, FastSAMEncoder, load_cached_frame, load_head,
                                            load_rgb, sample_features)
from freesam3d.streaming.window_probe import column_auc


def point_features(rgb, enc, layer=None, head=None, pixels=None, center=True):
    """(N,C) per-point features at ``pixels``: frozen layer or head output, centred then L2."""
    maps, geom = enc.encode(rgb)
    with torch.inference_mode():
        dense = maps[layer] if layer else head(maps)
        pf = sample_features(dense, pixels, geom)
        if center:
            pf = pf - pf.mean(0, keepdim=True)
            pf = pf / pf.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return pf.cpu().numpy()


def pair_auc(feat_a, ids_a, feat_b, ids_b, drop_diagonal=False):
    """AUC over all cross pairs: positive = the two points share an instance id."""
    sim = (feat_a @ feat_b.T)
    same = (ids_a[:, None] == ids_b[None, :])
    if drop_diagonal:
        off = ~np.eye(len(ids_a), dtype=bool)
        sim, same = sim[off], same[off]
    return column_auc(sim.reshape(-1), same.reshape(-1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/merge_net.yaml')
    ap.add_argument('--layer', default=None, choices=list(HEAD_NAMES))
    ap.add_argument('--head', default=None)
    ap.add_argument('--scenes', nargs='*', default=None, help='默认留出场景里帧数最多的前 3 个（F4.3 用 3 个）')
    ap.add_argument('--points', type=int, default=2000, help='每帧采样的标注点数')
    ap.add_argument('--gaps', type=int, nargs='+', default=[1, 3])
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--image-root', default=None)
    ap.add_argument('--out', default=None, help='默认 <phase4b.out_root>/point_diag')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    if bool(args.layer) == bool(args.head):
        raise SystemExit('pass exactly one of --layer / --head')

    cfg = load_config(args.config)
    cache_dir = Path(args.cache_dir) if args.cache_dir \
        else resolve_path(cfg['cache_root']) / 'seq' / 'train'
    image_root = args.image_root or cfg['dataset']['root']
    if args.scenes:
        scenes = args.scenes
    else:
        have = [s for s in cfg['stream_v2']['holdout_scenes'] if sorted(cache_dir.glob(f'{s}_*.npz'))]
        have.sort(key=lambda s: -len(sorted(cache_dir.glob(f'{s}_*.npz'))))
        scenes = have[:3]
    fast = cfg.get('fastsam') or load_config(resolve_path(cfg['lifting_config']))['fastsam']
    enc = FastSAMEncoder(fast['checkpoint'], args.device, fast.get('imgsz', 640), raw=True)
    head = load_head(args.head, device=args.device)[0] if args.head else None
    rng = np.random.default_rng(args.seed)
    tag = args.tag or (f'raw_{args.layer}' if args.layer else Path(args.head).parent.name)
    print(f'point diag {tag}: {"layer " + args.layer if args.layer else "head " + args.head} | '
          f'{len(scenes)} scenes | {args.points} pts/frame | gaps {args.gaps}', flush=True)

    t0 = time.time()
    per_scene = []
    for scene in scenes:
        files = sorted(cache_dir.glob(f'{scene}_*.npz'), key=lambda q: int(q.stem.rsplit('_', 1)[1]))
        # CachedFrame carries the same 2D labels as load_scene_frames(gt='2d') *and* the pixel coords
        frames = [load_cached_frame(p) for p in files]
        if len(frames) < 2:
            continue
        feats, gts = [], []
        for fr in frames:
            rgb = load_rgb(image_root, fr.scene, fr.frame_id)
            lab = np.flatnonzero(fr.gt > 0)
            sel = np.sort(rng.choice(lab, size=min(args.points, len(lab)), replace=False))
            feats.append(point_features(rgb, enc, args.layer, head, fr.pixels[sel], True))
            gts.append(fr.gt[sel])
        row = dict(scene=scene, frames=len(frames),
                   intra=float(np.mean([pair_auc(f, g, f, g, drop_diagonal=True)
                                        for f, g in zip(feats, gts)])))
        for gap in args.gaps:
            vals = [pair_auc(feats[i], gts[i], feats[i + gap], gts[i + gap])
                    for i in range(len(feats) - gap)]
            row[f'gap{gap}'] = float(np.mean(vals)) if vals else float('nan')
        per_scene.append(row)
        print(f"  {scene} ({row['frames']} frames): intra {row['intra']:.4f} " +
              ' '.join(f"gap{g} {row[f'gap{g}']:.4f}" for g in args.gaps), flush=True)

    keys = ['intra'] + [f'gap{g}' for g in args.gaps]
    agg = {k: dict(mean=float(np.mean([r[k] for r in per_scene])),
                   std=float(np.std([r[k] for r in per_scene])),
                   min=float(np.nanmin([r[k] for r in per_scene])),
                   max=float(np.nanmax([r[k] for r in per_scene]))) for k in keys}
    out = Path(args.out) if args.out else resolve_path(cfg['phase4b']['out_root']) / 'point_diag'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{tag}.json').write_text(json.dumps(
        dict(tag=tag, source=dict(layer=args.layer, head=args.head), scenes=scenes,
             points=args.points, gaps=args.gaps, per_scene=per_scene, aggregate=agg,
             seconds=time.time() - t0), indent=1, default=float))
    print(json.dumps(agg, indent=1))
    print(f'wrote {out / (tag + ".json")}')


if __name__ == '__main__':
    main()
