"""Stage-3 ranking diagnosis (question 1 of the brief): overfitting or under-informed?

    source tools/env.sh
    CK=work_dirs/stream_v2/phase2/model/AB/seed0/model.pt        # the stage-2 final model
    python tools/window_phase3_diag.py --stage train --checkpoint $CK
    python tools/window_phase3_diag.py --stage holdout --checkpoint $CK --thresholds 0.80 0.60 1.01

Training side: the model's own training graphs (A / B / intra_extra by default), object nodes
only, pooled over frames and over each scene's last frame. Holdout side: the 60 holdout scenes
streamed with the checkpoint, final objects, quality vs true voxel IoU (Spearman + AUC of
IoU >= 0.5), plus the duplicate composition of the wrong answers ranked above the first correct
one. Writes <out>/training.json and <out>/holdout.json.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.streaming.phase3_diag import holdout_ranking_diag, training_ranking_diag  # noqa: E402
from freesam3d.streaming.unified_train import load_unified_checkpoint  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--stage', default='all', choices=['train', 'holdout', 'all'])
    p.add_argument('--thresholds', nargs=3, type=float, default=[0.80, 0.60, 1.01],
                   metavar=('INTRA', 'BO', 'OO'))
    p.add_argument('--train-graphs', nargs='*', default=None,
                   help='training graph dirs (default: phase2 A B intra_extra)')
    p.add_argument('--train-scenes', type=int, default=200)
    p.add_argument('--scenes', nargs='*', default=None)
    p.add_argument('--shard', default='0/1')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--dup-contain', type=float, default=0.8)
    p.add_argument('--tag', default='diag')
    p.add_argument('--out', default=None)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    out = Path(a.out) if a.out else resolve_path(cfg['phase3']['out_root']) / 'diag' / a.tag
    out.mkdir(parents=True, exist_ok=True)
    dev = a.device if torch.cuda.is_available() else 'cpu'
    if str(dev).startswith('cuda'):
        print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    if not a.checkpoint:
        raise SystemExit('--checkpoint is required')
    if a.stage in ('train', 'all'):
        dirs = [Path(d) for d in a.train_graphs] if a.train_graphs else [
            resolve_path(cfg['phase2']['graphs_root']) / r for r in ('A', 'B', 'intra_extra')]
        res = training_ranking_diag(dirs, resolve_path(a.checkpoint), dev, scenes=a.train_scenes)
        (out / 'training.json').write_text(json.dumps(res, indent=2, default=float))
    if a.stage in ('holdout', 'all'):
        model, std, ck = load_unified_checkpoint(resolve_path(a.checkpoint), dev)
        cache = resolve_path(cfg['cache_root']) / 'seq' / 'train'
        scenes = a.scenes or list(cfg['stream_v2']['holdout_scenes'])
        i, n = map(int, a.shard.split('/'))
        scenes = [s for k, s in enumerate(scenes) if k % n == i]
        if a.limit:
            scenes = scenes[:a.limit]
        over = dict(threshold=a.thresholds[0], cross_threshold=a.thresholds[1],
                    bo_threshold=a.thresholds[1], oo_threshold=a.thresholds[2])
        res = holdout_ranking_diag(scenes, cfg, model, std, dev, cache,
                                   set(cfg['eval']['stuff_nyu40']), mem_overrides=over,
                                   dup_contain=a.dup_contain)
        (out / 'holdout.json').write_text(json.dumps(res, indent=2, default=float))
        print('holdout side:', json.dumps({k: v for k, v in res.items() if k != 'per_scene'},
                                          indent=2, default=float), flush=True)
    print(f'wrote {out}', flush=True)


if __name__ == '__main__':
    main()
