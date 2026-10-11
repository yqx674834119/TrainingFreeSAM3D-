"""Build GroupMergeNet frame caches from the ESAM ScanNet 2D export.

    source tools/env.sh
    python tools/merge_build_cache.py --split val      # 16 val scenes, every 40th frame
    python tools/merge_build_cache.py --split train    # 1201 train scenes, N frames each
Resumable: existing <cache>/<split>/<scene>_<frame>.npz are skipped.
Several workers can run in parallel on different GPUs with --shard i/n.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.merging.cache import CacheBuilder, attach_esam_gt, frame_ids, load_frame, read_scene_list, save  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--split', choices=['train', 'val'], required=True)
    p.add_argument('--scenes', nargs='*')
    p.add_argument('--shard', default='0/1', help='i/n: process scenes with index %% n == i')
    p.add_argument('--device', default='cuda')
    p.add_argument('--attach-3d-gt', action='store_true',
                   help="attach ESAM's 3D-transferred GT from work_dirs/esam_eval to existing val caches and exit")
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    if a.attach_3d_gt:
        n = attach_esam_gt(resolve_path(cfg['cache_root']) / a.split, resolve_path('work_dirs/esam_eval/data/scannet200-mv_fast'))
        print(f'{n} {a.split} frames carry ESAM 3D GT', flush=True)
        return
    ds = cfg['dataset']
    lifting_cfg = load_config(resolve_path(cfg['lifting_config']))
    lifting_cfg['fastsam']['device'] = a.device
    root = Path(ds['root'])
    if a.scenes:
        scenes = a.scenes
    elif a.split == 'val':
        scenes = ds['val_scenes']
    else:
        val = set(read_scene_list(ds['val_list']))
        scenes = [s for s in read_scene_list(ds['train_list']) if s not in val]
    i, n = map(int, a.shard.split('/'))
    scenes = [s for k, s in enumerate(scenes) if k % n == i]
    per_scene = ds['train_frames_per_scene'] if a.split == 'train' else ds['val_frames_per_scene']
    out = resolve_path(cfg['cache_root']) / a.split
    out.mkdir(parents=True, exist_ok=True)
    builder = CacheBuilder(lifting_cfg, cfg['graph'], cfg['labels'], cfg['feature'], cfg['boundary_feature'])
    print(f'split={a.split} scenes={len(scenes)} frames/scene={per_scene or "all"} device={builder.lifting.runner.device}', flush=True)
    done = skipped = 0
    t0 = time.time()
    for si, scene in enumerate(scenes):
        for fid in frame_ids(root / scene, ds['frame_stride'], per_scene):
            path = out / f'{scene}_{fid}.npz'
            if path.exists():
                skipped += 1
                continue
            frame, gt = load_frame(root, scene, fid, ds['intrinsic'])
            if frame is None:          # invalid pose, as ESAM skips
                continue
            save(builder.build(frame, *gt, seed=si * 100003 + fid), path)
            done += 1
        if (si + 1) % 20 == 0 or si + 1 == len(scenes):
            el = time.time() - t0
            print(f'[{a.shard}] {si + 1}/{len(scenes)} scenes, {done} built, {skipped} skipped, '
                  f'{el / max(done, 1):.1f}s/frame, {el / 60:.1f} min', flush=True)
    (out / f'manifest_{a.shard.replace("/", "of")}.json').write_text(json.dumps(dict(
        split=a.split, scenes=scenes, built=done, skipped=skipped, frames_per_scene=per_scene,
        frame_stride=ds['frame_stride'], feature=cfg['feature'], boundary_feature=cfg['boundary_feature'],
        graph=cfg['graph'], labels=cfg['labels']), indent=2))


if __name__ == '__main__':
    main()
