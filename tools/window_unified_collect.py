"""Stage-2 data collection: run the per-frame flow online over the training cache and save graphs.

    source tools/env.sh
    # round A: stage 1's rule drives the memory (GroupMergeNet intra + containment, no object merge)
    for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python tools/window_unified_collect.py --round A --shard $i/4 & done
    # extra per-frame graphs (intra only, no memory) from the older per-frame cache
    for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python tools/window_unified_collect.py --round intra_extra --shard $i/4 & done
    # round B: the model trained on round A drives the memory
    python tools/window_unified_collect.py --round B --checkpoint <model.pt> --thresholds 0.4 0.5 0.5

Writes <out_root>/phase2/graphs/{A,B,intra_extra}/<scene>.npz. The 60 holdout scenes are excluded:
they calibrate thresholds, never train. Round A's thresholds are read from
phase1/selected.json (the holdout-selected stage-1 configuration); round B's come from
--thresholds or the sweep file written by tools/window_unified_sweep.py.
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
from freesam3d.streaming.evaluate import frames_from_merge_cache  # noqa: E402
from freesam3d.streaming.unified_flow import IdentityStandardizer, RuleAdapter  # noqa: E402
from freesam3d.streaming.window_data import collect_intra_frame, collect_scene, save  # noqa: E402
from freesam3d.streaming.window_scores import load_group_merge_net  # noqa: E402

ROUNDS = ('A', 'B', 'intra_extra')


def round_a_setup(cfg, dev):
    """Stage 1's rule as the memory driver: the trained GroupMergeNet + its checkpoint threshold."""
    model, std, ck = load_group_merge_net(resolve_path('work_dirs/merge_net/full/model.pt'), dev)
    sel_path = resolve_path(cfg['stream_v2']['out_root']) / 'phase1' / 'selected.json'
    if sel_path.exists():
        sel = json.loads(sel_path.read_text())['variant']
        intra, cross = float(sel['intra_threshold']), float(sel['cross_threshold'])
    else:
        intra, cross = float(ck['threshold']), 0.2
    print(f'round A driver: GroupMergeNet (checkpoint threshold {ck["threshold"]}) | '
          f'intra {intra} cross {cross} (phase1/selected.json)', flush=True)
    return RuleAdapter(model, std, dev), IdentityStandardizer(), dict(threshold=intra, cross_threshold=cross)


def round_b_setup(path, thresholds, dev):
    """The model trained on round A (or A+B) drives the memory."""
    from freesam3d.streaming.unified_train import load_unified_checkpoint
    model, std, ck = load_unified_checkpoint(resolve_path(path), dev)
    if thresholds is None:
        raise SystemExit('--round B needs --thresholds INTRA BO OO (or a sweep file via --sweep)')
    intra, bo, oo = (float(t) for t in thresholds)
    print(f'round B driver: {path} | intra {intra} bo {bo} oo {oo}', flush=True)
    return model, std, dict(threshold=intra, cross_threshold=bo, bo_threshold=bo, oo_threshold=oo)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--round', default='A', choices=list(ROUNDS))
    p.add_argument('--checkpoint', default=None, help='round B: unified checkpoint (.pt)')
    p.add_argument('--thresholds', nargs=3, type=float, default=None, metavar=('INTRA', 'BO', 'OO'))
    p.add_argument('--sweep', default=None, help='round B: JSON written by tools/window_unified_sweep.py')
    p.add_argument('--shard', default='0/1')
    p.add_argument('--scenes', nargs='*', default=None)
    p.add_argument('--limit', type=int, default=0, help='debug: only the first N scenes of this shard')
    p.add_argument('--cache-dir', default=None)
    p.add_argument('--out', default=None)
    p.add_argument('--device', default='cuda')
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    dev = a.device if torch.cuda.is_available() else 'cpu'
    if str(dev).startswith('cuda'):
        print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    out = Path(a.out) if a.out else resolve_path(cfg['stream_v2']['out_root']) / 'phase2' / 'graphs' / a.round
    root = resolve_path(cfg['cache_root'])
    cache_dir = Path(a.cache_dir) if a.cache_dir else (root / 'seq' / 'train' if a.round != 'intra_extra' else root / 'train')
    stuff = set(cfg['eval']['stuff_nyu40'])          # the labels use the 2D instance PNGs
    holdout = set(cfg['stream_v2']['holdout_scenes'])

    if a.round == 'A':
        model, std, overrides = round_a_setup(cfg, dev)
    elif a.round == 'B':
        thr = a.thresholds
        if thr is None and a.sweep:
            sw = json.loads(resolve_path(a.sweep).read_text())
            thr = [sw['intra'], sw['bo'], sw['oo']]
        if a.checkpoint is None:
            raise SystemExit('--round B needs --checkpoint')
        model, std, overrides = round_b_setup(a.checkpoint, thr, dev)
    else:
        model = std = overrides = None

    files = sorted(cache_dir.glob('*.npz'))
    by_scene = {}
    for f in files:
        scene = f.stem.rsplit('_', 1)[0]
        if scene not in holdout:
            by_scene.setdefault(scene, []).append(f)
    scenes = a.scenes if a.scenes else sorted(by_scene)
    scenes = [s for s in scenes if s in by_scene]
    i, n = map(int, a.shard.split('/'))
    mine = [s for k, s in enumerate(scenes) if k % n == i]
    if a.limit:
        mine = mine[:a.limit]
    print(f'round {a.round}: {len(mine)}/{len(scenes)} scenes, cache {cache_dir}, out {out}', flush=True)

    t0 = time.time()
    done = 0
    for si, scene in enumerate(mine):
        dst = out / f'{scene}.npz'
        if dst.exists() and not a.overwrite:
            continue
        fs = sorted(by_scene[scene], key=lambda q: int(q.stem.rsplit('_', 1)[1]))
        frames = frames_from_merge_cache(fs, gt='2d')
        if a.round == 'intra_extra':
            from freesam3d.streaming.window_data import pack
            rows = [collect_intra_frame(fr, stuff, voxel=float(cfg['stream_v2']['voxel']),
                                        dilation=int(cfg['stream_v2']['touch_dilation']), device=dev)
                    for fr in frames]
            data = pack(rows, [], np.zeros(0, np.float32), np.zeros(0, bool), scene=scene,
                        feat_dim=int(frames[0].node_feat.shape[1]))
        else:
            data = collect_scene(frames, cfg, model, std, dev, stuff, overrides=overrides)
        save(dst, data)
        done += 1
        if done % 20 == 0 or si + 1 == len(mine):
            print(f'[{a.shard}] {si + 1}/{len(mine)} {scene}: {len(frames)} frames, '
                  f'{(time.time() - t0) / 60:.1f} min', flush=True)
    print(f'round {a.round} shard {a.shard}: wrote {done} scenes in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
