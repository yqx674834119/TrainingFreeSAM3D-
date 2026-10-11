"""Benchmark the all-GPU inference path (RGB-D -> merged instances) and check it against the cached CPU path.

    source tools/env.sh
    python tools/merge_benchmark.py [--frames 50]
Writes work_dirs/merge_net/benchmark.json.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.evaluation.instance import GTSet, coverage_stats, groups_to_masks  # noqa: E402
from freesam3d.merging.cache import load_frame  # noqa: E402
from freesam3d.merging.data import list_cache, FrameSet  # noqa: E402
from freesam3d.merging.infer import GroupMerger  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--checkpoint', default='full')
    p.add_argument('--frames', type=int, default=50)
    p.add_argument('--device', default='cuda')
    p.add_argument('--scorer', default=None, help='mask-scorer tag under work_dirs/mask_scorer/ (adds proposals + NMS stage)')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    lc = load_config(resolve_path(cfg['lifting_config']))
    sc = resolve_path('work_dirs/mask_scorer') / a.scorer / 'model.pt' if a.scorer else None
    scfg = cfg.get('scorer', {})
    merger = GroupMerger(lc, cfg, resolve_path('work_dirs/merge_net') / a.checkpoint / 'model.pt', a.device, sc,
                         scfg.get('nms_iou', 0.5), scfg.get('nms_contain', 0.8), scfg.get('top_k'))
    files = list_cache(resolve_path(cfg['cache_root']) / 'val')
    files = files[:: max(1, len(files) // a.frames)][:a.frames]
    cached = FrameSet(files, keep_points=True).frames
    root = Path(cfg['dataset']['root'])
    stuff = set(cfg['eval']['stuff_raw'])
    times, ious = [], {'gpu path': [], 'cached cpu path (same model)': []}
    from scipy.spatial import cKDTree
    from freesam3d.merging.train import predict_frame
    from freesam3d.merging.cluster import merge_groups
    for f, fr in zip(files, cached):
        scene, fid = f.stem.rsplit('_', 1)
        frame, _ = load_frame(root, scene, int(fid), cfg['dataset']['intrinsic'])
        res = merger(frame.rgb, frame.depth, frame.intrinsic, frame.pose_c2w)
        times.append(res.timing_ms)
        # GT for the new sample: nearest cached point (same frame, same pose)
        _, idx = cKDTree(fr['xyz']).query(res.xyz, k=1)
        gt, sem = fr['gt_instance_3d'][idx], fr['gt_sem_3d'][idx]
        ids, c = np.unique(gt[gt > 0], return_counts=True)
        keep = [k for k in ids[c >= 100] if np.bincount(sem[gt == k]).argmax() not in stuff]
        if keep:
            gts = GTSet.from_labels(gt, keep)
            m, _ = groups_to_masks(res.instances)
            ious['gpu path'].append(coverage_stats(m, gts)['mean_best_iou'])
            if res.masks is not None and len(res.masks):
                ious.setdefault('gpu path + scorer (proposals)', []).append(coverage_stats(res.masks, gts)['mean_best_iou'])
            prob = predict_frame(merger.model, merger.std, fr, a.device)
            m2, _ = groups_to_masks(merge_groups(fr['groups'], fr['edges'], prob, merger.threshold, merger.linkage))
            gt2 = fr['gt_instance_3d']
            ious['cached cpu path (same model)'].append(coverage_stats(m2, GTSet.from_labels(gt2, keep))['mean_best_iou'])
    keys = list(times[0])
    mean = {k: float(np.mean([t[k] for t in times[1:]])) for k in keys}      # drop warm-up frame
    mean['total'] = float(sum(mean.values()))
    report = dict(device=a.device, frames=len(times), timing_ms=mean, params=merger.model.num_params(),
                  mean_best_iou={k: float(np.mean(v)) for k, v in ious.items()})
    out = resolve_path('work_dirs/merge_net/benchmark.json')
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
