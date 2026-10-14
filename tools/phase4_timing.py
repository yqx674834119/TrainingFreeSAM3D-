"""Phase-4 per-frame timing split: jpeg -> frozen FastSAM raw forward -> instance head -> sample+pool.

CUDA-synced at every stage boundary, so each number is real wall time. Run on an idle GPU.

    python tools/phase4_timing.py --frames 50
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.models.instance_head import (FastSAMEncoder, fragment_mean_features,  # noqa: E402
                                            load_cached_frame, load_head, load_rgb, sample_features)

NAMES = ('jpeg_load', 'fastsam_raw', 'head_fwd', 'sample_pool', 'total')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/merge_net.yaml')
    ap.add_argument('--head', default=None, help='默认 <phase4.out_root>/head/seed0/head.pt')
    ap.add_argument('--frames', type=int, default=50)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=None, help='默认 <phase4.out_root>/timing.json')
    a = ap.parse_args()
    cfg = load_config(a.config)
    out_root = resolve_path(cfg['phase4']['out_root'])
    head_path = Path(a.head) if a.head else out_root / 'head' / 'seed0' / 'head.pt'
    files = sorted((resolve_path(cfg['cache_root']) / 'seq' / 'train').glob('*.npz'))[:a.frames]
    head, _ = load_head(head_path, device=a.device)
    fast = cfg.get('fastsam') or load_config(resolve_path(cfg['lifting_config']))['fastsam']
    enc = FastSAMEncoder(fast['checkpoint'], a.device, fast.get('imgsz', 640), raw=True)
    root = cfg['dataset']['root']

    def sync():
        if str(a.device).startswith('cuda'):
            torch.cuda.synchronize()

    rows = []
    for i, path in enumerate(files):
        fr = load_cached_frame(path)
        t0 = time.perf_counter(); rgb = load_rgb(root, fr.scene, fr.frame_id); sync(); t1 = time.perf_counter()
        maps, geom = enc.encode(rgb); sync(); t2 = time.perf_counter()
        with torch.no_grad():
            dense = head(maps); sync(); t3 = time.perf_counter()
            pf = sample_features(dense, fr.pixels, geom)
            fragment_mean_features(pf, fr.groups, int(fr.groups.max()) + 1); sync(); t4 = time.perf_counter()
        if i >= a.warmup:
            rows.append((t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0))
    arr = np.array(rows) * 1000
    stats = {n: dict(mean=float(arr[:, k].mean()), std=float(arr[:, k].std()))
             for k, n in enumerate(NAMES)}
    stats['frames'] = len(rows)
    out = Path(a.out) if a.out else out_root / 'timing.json'
    out.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
