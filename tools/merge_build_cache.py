"""Build GroupMergeNet frame caches from the ESAM ScanNet 2D export.

    source tools/env.sh
    python tools/merge_build_cache.py --split val      # 16 val scenes, every 40th frame
    python tools/merge_build_cache.py --split train    # 1201 train scenes, N frames each
Resumable: existing <cache>/<split>/<scene>_<frame>.npz are skipped.
Several workers can run in parallel on different GPUs with --shard i/n.

Cross-frame training data (stage 1): ``--consecutive K --out-subdir seq`` builds the first K
*consecutive* 40-frame-strided frames of every training scene into <cache_root>/seq/train, so a
scene's frames form a real sequence instead of the ``train_frames_per_scene`` evenly spaced
samples. Labels: the ESAM 3D-transferred GT is attached where it exists (``attach_esam_gt``);
ScanNet's own per-frame instance PNG (scene-consistent ids) is always in the cache as
``gt_instance`` and is the fallback the cross-frame labels come from.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.merging.cache import CacheBuilder, attach_esam_gt, frame_ids, load_frame, read_scene_list, save  # noqa: E402

MV_FAST = 'work_dirs/esam_eval/data/scannet200-mv_fast'


def out_dir(cfg, split, subdir=None):
    """<cache_root>/<subdir>/<split> when ``subdir`` is given, else <cache_root>/<split> (unchanged)."""
    root = resolve_path(cfg['cache_root'])
    return (root / subdir / split) if subdir else (root / split)


def scene_list(cfg, split, scenes=None):
    """Explicit --scenes, else the val scenes, else the training list minus ESAM's val list."""
    ds = cfg['dataset']
    if scenes:
        return list(scenes)
    if split == 'val':
        return list(ds['val_scenes'])
    val = set(read_scene_list(ds['val_list']))
    return [s for s in read_scene_list(ds['train_list']) if s not in val]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--split', choices=['train', 'val'], required=True)
    p.add_argument('--scenes', nargs='*')
    p.add_argument('--shard', default='0/1', help='i/n: process scenes with index %% n == i')
    p.add_argument('--device', default='cuda')
    p.add_argument('--consecutive', type=int, default=0,
                   help='K > 0: the first K consecutive 40-frame-strided frames of each scene '
                        '(cross-frame sequences); 0 = the existing even subsample')
    p.add_argument('--out-subdir', default=None,
                   help='write to <cache_root>/<out_subdir>/<split> instead of <cache_root>/<split>')
    p.add_argument('--attach-3d-gt', action='store_true',
                   help="attach ESAM's 3D-transferred GT from work_dirs/esam_eval to existing caches and exit")
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    if a.attach_3d_gt:
        n = attach_esam_gt(out_dir(cfg, a.split, a.out_subdir), resolve_path(MV_FAST))
        print(f'{n} {a.split} frames carry ESAM 3D GT', flush=True)
        return
    ds = cfg['dataset']
    lifting_cfg = load_config(resolve_path(cfg['lifting_config']))
    lifting_cfg['fastsam']['device'] = a.device
    root = Path(ds['root'])
    scenes = scene_list(cfg, a.split, a.scenes)
    i, n = map(int, a.shard.split('/'))
    scenes = [s for k, s in enumerate(scenes) if k % n == i]
    per_scene = ds['train_frames_per_scene'] if a.split == 'train' else ds['val_frames_per_scene']
    out = out_dir(cfg, a.split, a.out_subdir)
    out.mkdir(parents=True, exist_ok=True)
    builder = CacheBuilder(lifting_cfg, cfg['graph'], cfg['labels'], cfg['feature'], cfg['boundary_feature'])
    print(f'split={a.split} scenes={len(scenes)} frames/scene={a.consecutive or per_scene or "all"} '
          f'device={builder.lifting.runner.device} out={out}', flush=True)
    done = skipped = 0
    t0 = time.time()
    for si, scene in enumerate(scenes):
        # --consecutive needs the *strided* list itself, not the even subsample of it
        fids = frame_ids(root / scene, ds['frame_stride'], 0 if a.consecutive else per_scene)
        if a.consecutive:
            fids = fids[:a.consecutive]
        for fid in fids:
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
    elapsed = time.time() - t0
    disk = sum(f.stat().st_size for f in out.glob('*.npz'))
    (out / f'manifest_{a.shard.replace("/", "of")}.json').write_text(json.dumps(dict(
        split=a.split, scenes=scenes, built=done, skipped=skipped, frames_per_scene=per_scene,
        consecutive=a.consecutive, out_subdir=a.out_subdir, elapsed_s=elapsed, disk_bytes=disk,
        frame_stride=ds['frame_stride'], feature=cfg['feature'], boundary_feature=cfg['boundary_feature'],
        graph=cfg['graph'], labels=cfg['labels']), indent=2))
    print(f'[{a.shard}] done {done} built, {skipped} skipped, {elapsed / 60:.1f} min, '
          f'{disk / 2**30:.2f} GiB', flush=True)


if __name__ == '__main__':
    main()
