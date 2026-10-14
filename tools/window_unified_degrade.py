"""Degradation check: the stage-2 model on single frames only (history off), stage-1 protocol.

Stage 2 replaced GroupMergeNet, so the per-frame behaviour it inherited is the baseline it must not
lose. This runs the per-frame protocol of ``merging/evaluate.py`` - one frame at a time, 3D GT, thing
instances with >= ``eval.min_gt_points`` points, the same FastSAM/cosine/Oracle reference rows - with
the unified model used *without* its cross-frame half.

The unified path builds each frame's graph through ``window_data.intra_frame_graph``, i.e. exactly
what ``UnifiedGrouping`` step 1 feeds the model: node geometry from the frame's fragments and the
29-column pair features (14 cached intra-frame columns + the 15 cross-frame columns, which exist on
intra-frame edges too). ``SingleFrameAdapter`` then fills the remaining node/edge inputs with their
single-frame constants (node type 0, one observation, age 0, intra-frame edges). GroupMergeNet keeps
its own 14-column path, so both models are scored on identical frames.

    source tools/env.sh
    python tools/window_unified_degrade.py --checkpoint work_dirs/stream_v2/phase2/model/A/seed0/model.pt --threshold 0.5

Writes work_dirs/stream_v2/phase2/degrade/{summary.md,summary.json,per_frame.json}.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.merging.cluster import merge_groups  # noqa: E402
from freesam3d.merging.data import FrameSet, list_cache  # noqa: E402
from freesam3d.merging.evaluate import (METRIC_KEYS, merged_scores, score_masks,  # noqa: E402
                                        thing_gtset)
from freesam3d.merging.train import predict_frame  # noqa: E402
from freesam3d.streaming.evaluate import frames_from_merge_cache  # noqa: E402
from freesam3d.streaming.unified_flow import SingleFrameAdapter  # noqa: E402
from freesam3d.streaming.unified_train import load_unified_checkpoint  # noqa: E402
from freesam3d.streaming.window_data import intra_frame_graph  # noqa: E402
from freesam3d.streaming.window_scores import load_group_merge_net  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--checkpoint', default=None, help='stage-2 checkpoint (omit for the baseline only)')
    p.add_argument('--thresholds', nargs='+', type=float, default=[0.5],
                   help='stage-2 intra-frame thresholds to report (a curve: the model merges more eagerly '
                        'than GroupMergeNet, so its own operating point has to be seen, not assumed)')
    p.add_argument('--group-thresholds', nargs='+', type=float, default=[0.65])
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--out', default=None)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    dev = a.device if torch.cuda.is_available() else 'cpu'
    if str(dev).startswith('cuda'):
        print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    ev = cfg['eval']
    stuff = set(ev['stuff_raw'])                       # the 3D GT semantic ids are ScanNet raw ids
    min_gt_points = int(ev['min_gt_points'])
    linkage = cfg['merge']['linkage']
    voxel = float(cfg['stream_v2']['voxel'])
    dilation = int(cfg['stream_v2']['touch_dilation'])

    files = list_cache(resolve_path(cfg['cache_root']) / 'val')
    if a.limit:
        files = files[:a.limit]
    key = lambda fr: (fr['scene'], int(fr['name'].rsplit('_', 1)[1]))
    dicts = {key(d): d for d in FrameSet(files, keep_points=True).frames}
    ffs = frames_from_merge_cache(files, gt='3d')
    pairs = [(dicts[(f.scene, f.frame_id)], f) for f in ffs if (f.scene, f.frame_id) in dicts]

    gm, gs, _ = load_group_merge_net(resolve_path('work_dirs/merge_net/full/model.pt'), dev)
    adapter = None
    if a.checkpoint:
        um, us, ck = load_unified_checkpoint(resolve_path(a.checkpoint), dev)
        adapter = SingleFrameAdapter(um)
        print(f'unified checkpoint {a.checkpoint} (seed {ck.get("seed")}) '
              f'thresholds {a.thresholds}', flush=True)

    gthr, uthr = [float(t) for t in a.group_thresholds], [float(t) for t in a.thresholds]
    rows = {f'group@{t:.2f}': [] for t in gthr}
    rows.update({f'unified@{t:.2f}': [] for t in uthr})
    for fd, ff in pairs:
        gts = thing_gtset(fd, stuff, min_gt_points, '3d')
        if gts is None:
            continue
        g, e = fd['groups'], fd['edges']
        base = dict(name=fd['name'], scene=fd['scene'], num_gt=len(gts.ids), FastSAM=score_masks(g, gts))
        prob = predict_frame(gm, gs, fd, dev, with_objectness=True)
        for t in gthr:
            merged = merge_groups(g, e, prob[0], t, linkage)
            rows[f'group@{t:.2f}'].append(dict(base, group=score_masks(
                merged, gts, merged_scores(fd, merged, prob[0], prob[1]), 0, 0.0)))
        if adapter is not None:
            graph = intra_frame_graph(ff, voxel=voxel, dilation=dilation, device=dev)
            geom = torch.as_tensor(graph['node_geom'], device=dev)
            feat = torch.as_tensor(graph['node_feat'], device=dev)
            edges = torch.as_tensor(graph['edges'], dtype=torch.int64, device=dev)
            pf = torch.as_tensor(graph['pair_feat'], device=dev)
            geom_s, pf_s = us.apply(geom, pf)
            with torch.no_grad():
                logit, q = adapter(geom_s, feat, edges, pf_s, return_objectness=True)
            pu, qu = torch.sigmoid(logit).cpu().numpy(), torch.sigmoid(q).cpu().numpy()
            for t in uthr:
                merged = merge_groups(g, e, pu, t, linkage)
                rows[f'unified@{t:.2f}'].append(dict(base, unified=score_masks(
                    merged, gts, merged_scores(fd, merged, pu, qu), 0, 0.0)))

    def means_of(key):
        field = key.split('@')[0]                      # rows carry metrics under 'group' / 'unified'
        rs = rows.get(key, next(iter(rows.values())))  # FastSAM lives on every row list
        return {k: float(np.nanmean([r[field][k] for r in rs])) for k in METRIC_KEYS}

    means = {'FastSAM': means_of('FastSAM')}
    means.update({m: means_of(m) for m in rows})
    n_frames = len(next(iter(rows.values())))
    out = Path(a.out) if a.out else resolve_path(cfg['stream_v2']['out_root']) / 'phase2' / 'degrade'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(
        dict(means=means, frames=n_frames, gt='3d', min_gt_points=min_gt_points,
             group_thresholds=gthr, unified_thresholds=uthr, checkpoint=str(a.checkpoint)),
        indent=2, default=float))
    (out / 'per_frame.json').write_text(json.dumps({k: [{m: r[m] for m in r if m != 'FastSAM'}
                                                       for r in v] for k, v in rows.items()},
                                                  indent=1, default=float))
    L = ['# 第 2 阶段退化检查：关掉历史、只用单帧（逐帧协议，3D GT）', '',
         f'{n_frames} 个验证帧，thing 且 ≥ {min_gt_points} 点。GroupMergeNet 用它的 14 列路径，'
         '统一模型用推理时同样的单帧图（29 列 pair 特征 + 单帧常量）。', '',
         '| 方法 | 阈值 | masks/帧 | mean best IoU | recall@0.5 | recall@0.25 | AP50 | AP25 | fragments/GT |',
         '|---|---|---|---|---|---|---|---|---|']
    for key, v in means.items():
        meth, thr = key.split('@') if '@' in key else (key, '-')
        L.append(f'| {meth} | {thr} | {v["num_masks"]:.1f} | {v["mean_best_iou"]:.4f} | {v["recall@0.5"]:.4f} | '
                 f'{v["recall@0.25"]:.4f} | {v["AP50"]:.4f} | {v["AP25"]:.4f} | {v["fragments"]:.2f} |')
    best_u = max((k for k in means if k.startswith('unified')), key=lambda k: means[k]['mean_best_iou'],
                 default=None)
    best_g = max((k for k in means if k.startswith('group@')), key=lambda k: means[k]['mean_best_iou'],
                 default=None)
    if best_u and best_g:
        L += ['', f'统一模型最好的一档是 {best_u}（mean best IoU {means[best_u]["mean_best_iou"]:.4f}），'
                  f'GroupMergeNet 最好的一档是 {best_g}（{means[best_g]["mean_best_iou"]:.4f}，门槛 ≥ 0.62）。']
    (out / 'summary.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L), flush=True)


if __name__ == '__main__':
    main()
