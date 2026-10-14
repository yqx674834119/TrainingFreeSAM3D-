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
    p.add_argument('--memory', default=None,
                   help='ProjectionHead tag under work_dirs/stream/projection/: attach a QueryMemory and time its update '
                        '(runs --frames consecutive frames of one scene so the memory grows; needs --scorer)')
    p.add_argument('--unified', default=None,
                   help='stage-2 checkpoint (.pt): time the unified flow (WindowMemory + UnifiedGrouping) on '
                        'consecutive frames of one scene instead of the GroupMerger path')
    p.add_argument('--unified-thresholds', nargs=3, type=float, default=[0.6, 0.3, 0.3],
                   metavar=('INTRA', 'BO', 'OO'))
    p.add_argument('--scene', default=None, help='scene for --unified (default: the first val scene)')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    lc = load_config(resolve_path(cfg['lifting_config']))
    sc = resolve_path('work_dirs/mask_scorer') / a.scorer / 'model.pt' if a.scorer else None
    scfg = cfg.get('scorer', {})
    mem = None
    nms_contain = scfg.get('nms_contain', 0.8)
    if a.memory:
        from freesam3d.streaming.memory import MemoryConfig, QueryMemory
        from freesam3d.streaming.projection import load_projection
        st = cfg.get('streaming', {})
        proj = load_projection(resolve_path('work_dirs/stream/projection') / a.memory / 'model.pt', a.device)
        mem = QueryMemory(MemoryConfig(voxel=st.get('voxel', 0.05), match_score=st.get('match_score', 'contain_x_feat'),
                                       match_threshold=st.get('match_threshold', 0.3), match_dilation=st.get('match_dilation', 1),
                                       admit_score=st.get('admit_score', 0.3)), device=a.device, projection=proj)
        nms_contain = st.get('nms_contain', nms_contain)      # the memory wants the stricter per-frame NMS
    if a.unified:                                          # stage 2: one model, per-frame stages of the unified flow
        import torch
        from freesam3d.streaming.holdout import load_scene_frames
        from freesam3d.streaming.unified_flow import make_unified_memory
        from freesam3d.streaming.unified_train import load_unified_checkpoint
        model, std, ck = load_unified_checkpoint(resolve_path(a.unified), a.device)
        cache = resolve_path(cfg['cache_root']) / 'val'
        scene = a.scene or sorted({f.stem.rsplit('_', 1)[0] for f in cache.glob('*.npz')})[0]
        frames = load_scene_frames(cache, scene, gt='3d')[:a.frames]
        thr = [float(t) for t in a.unified_thresholds]
        mem = make_unified_memory(cfg, model, std, a.device, threshold=thr[0], cross_threshold=thr[1],
                                  bo_threshold=thr[1], oo_threshold=thr[2])
        stages = []
        for fr in frames:
            mem.update(fr)
            if str(a.device).startswith('cuda'):
                torch.cuda.synchronize()
            stages.append(dict(mem.times))
        keys = list(stages[0])
        warm = stages[1:] or stages
        timing = {k: float(np.mean([s[k] for s in warm])) for k in keys}
        timing['total'] = float(sum(timing.values()))
        report = dict(device=a.device, scene=scene, frames=len(stages), thresholds=thr,
                      params=model.num_params(), timing_ms=timing, objects=len(mem.objects))
        out = resolve_path('work_dirs/merge_net/benchmark_unified.json')
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return

    merger = GroupMerger(lc, cfg, resolve_path('work_dirs/merge_net') / a.checkpoint / 'model.pt', a.device, sc,
                         scfg.get('nms_iou', 0.5), nms_contain, scfg.get('top_k'), memory=mem)
    files = list_cache(resolve_path(cfg['cache_root']) / 'val')
    if mem is not None:                                   # one scene in frame order, so the memory actually grows
        scene0 = files[0].stem.rsplit('_', 1)[0]
        files = sorted([f for f in files if f.stem.rsplit('_', 1)[0] == scene0], key=lambda q: int(q.stem.rsplit('_', 1)[1]))[:a.frames]
    else:
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
        res = merger(frame.rgb, frame.depth, frame.intrinsic, frame.pose_c2w, frame_id=int(fid))
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
    if mem is not None:
        report['memory'] = dict(projection=a.memory, queries=mem.num_queries, records=mem.num_records,
                                scene_voxels=int(len(mem.scene_voxels)), projection_params=proj.num_params())
    out = resolve_path('work_dirs/merge_net/benchmark_memory.json' if mem is not None else 'work_dirs/merge_net/benchmark.json')
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
