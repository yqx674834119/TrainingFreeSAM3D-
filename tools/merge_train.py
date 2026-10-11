"""Train GroupMergeNet on the cached train split.

    source tools/env.sh
    python tools/merge_train.py                              # config model.kind
    python tools/merge_train.py --kind geom_only --tag geom   # ablation
Writes work_dirs/merge_net/<tag>/{model.pt,train.json,train.log}.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.merging.data import FrameSet, list_cache, scene_of  # noqa: E402
from freesam3d.merging.train import save_checkpoint, train  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--kind', choices=['full', 'geom_only', 'edge_only'])
    p.add_argument('--tag', default=None)
    p.add_argument('--opts', nargs='*', default=[])
    p.add_argument('--calib-scenes', type=int, default=60, help='held-out train scenes for threshold calibration')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config), a.opts)
    if a.kind:
        cfg['model']['kind'] = a.kind
    tag = a.tag or cfg['model']['kind']
    out = resolve_path('work_dirs/merge_net') / tag
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / 'train.log', 'w')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    files = list_cache(resolve_path(cfg['cache_root']) / 'train')
    scenes = sorted({scene_of(f) for f in files})
    rng = np.random.default_rng(cfg['training']['seed'])
    calib = set(rng.choice(scenes, min(a.calib_scenes, len(scenes)), replace=False).tolist())
    train_files = [f for f in files if scene_of(f) not in calib]
    calib_files = [f for f in files if scene_of(f) in calib]
    t0 = time.time()
    lb = cfg['labels']
    train_set = FrameSet(train_files, purity=lb['purity'], min_labeled=lb['min_labeled'])
    calib_set = FrameSet(calib_files, keep_points=True, purity=lb['purity'], min_labeled=lb['min_labeled'])
    log(f'loaded {len(train_set)} train frames ({len(scenes) - len(calib)} scenes) + {len(calib_set)} calibration '
        f'frames ({len(calib)} scenes) in {time.time() - t0:.0f}s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    model, std, extra = train(train_set, cfg, device, log, calib_set.frames)
    extra['train_seconds'] = time.time() - t0
    extra['train_scenes'] = len(scenes) - len(calib)
    extra['calib_scenes'] = sorted(calib)
    save_checkpoint(out / 'model.pt', model, std, cfg, extra)
    (out / 'train.json').write_text(json.dumps(dict(tag=tag, params=model.num_params(), **extra), indent=2))
    log(f'saved {out / "model.pt"}  ({extra["train_seconds"]:.0f}s)')


if __name__ == '__main__':
    main()
