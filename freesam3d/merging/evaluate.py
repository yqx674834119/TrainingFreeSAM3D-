"""Evaluate merged groups on the val cache with the same protocol as the ESAM study.

Per frame, thing GT instances (NYU40 wall/floor/ceiling excluded) with >= min_gt_points
points; metrics: mean best IoU, recall@0.5/0.25, AP50/AP25, fragments per GT.
Methods: FastSAM (no merge), cosine threshold (training-free), each checkpoint,
Oracle-adj and Oracle (GT merge upper bounds). Also times graph build + model.
"""
import time

import numpy as np
import torch

from ..evaluation.instance import (GTSet, class_agnostic_ap, coverage_stats, fragments_per_gt, groups_to_masks,
                                   oracle_merge, oracle_merge_adjacent)
from .cluster import merge_groups
from .graph import edge_feature_names
from .train import predict_frame

METRIC_KEYS = ('mean_best_iou', 'recall@0.5', 'recall@0.25', 'AP50', 'AP25', 'fragments', 'num_masks')


def frame_gt(fr, source):
    """(instance, semantic) arrays for ``source`` in {'2d', '3d'}."""
    if source == '3d':
        if 'gt_instance_3d' not in fr:
            raise KeyError(f"{fr['name']}: cache has no ESAM 3D GT; run tools/merge_build_cache.py --attach-3d-gt")
        return fr['gt_instance_3d'], fr['gt_sem_3d']
    return fr['gt_instance'], fr['gt_nyu40']


def thing_gtset(fr, stuff, min_gt_points, source='2d'):
    gt, sem_arr = frame_gt(fr, source)
    ids, counts = np.unique(gt[gt > 0], return_counts=True)
    ids = ids[counts >= min_gt_points]
    keep = []
    for k in ids:
        sem = np.bincount(sem_arr[gt == k]).argmax()
        if sem not in stuff:
            keep.append(k)
    return GTSet.from_labels(gt, keep) if keep else None


def score_masks(merged, gts, scores=None, min_points=0, min_score=0.0):
    masks, _ = groups_to_masks(merged)
    sc = masks.sum(1).astype(float) if scores is None else scores
    if min_points or min_score:
        masks, sc = filter_masks(masks, sc, min_points, min_score)
    cov = coverage_stats(masks, gts)
    return {'mean_best_iou': cov['mean_best_iou'], 'recall@0.5': cov['recall@0.5'], 'recall@0.25': cov['recall@0.25'],
            'AP50': class_agnostic_ap(masks, sc, gts, 0.5), 'AP25': class_agnostic_ap(masks, sc, gts, 0.25),
            'fragments': float(fragments_per_gt(masks, gts).mean()), 'num_masks': float(len(masks))}


def node_clusters(fr, merged):
    """Cluster id of each original group under ``merged`` (point-level ids)."""
    first = np.zeros(len(fr['sizes']), np.int64)
    first[fr['groups'][::-1]] = np.arange(len(fr['groups']))[::-1]      # first point index per group
    return merged[first]


def merged_scores(fr, merged, prob=None, objectness=None):
    """Score per merged mask.

    objectness given : size-weighted mean of the per-group objectness (the learned "is a thing" head)
    else             : mean FastSAM confidence x mean internal edge probability (training-free fallback)
    """
    masks, ids = groups_to_masks(merged)
    conf = np.zeros(len(fr['sizes']), np.float32)
    has = fr['group_mask_index'] >= 0
    conf[has] = fr['fastsam_scores'][fr['group_mask_index'][has]]
    nc = node_clusters(fr, merged)
    out = np.zeros(len(ids))
    for i, k in enumerate(ids):
        nodes = np.flatnonzero(nc == k)
        w = fr['sizes'][nodes].astype(float)
        if objectness is not None:
            out[i] = float((objectness[nodes] * w).sum() / w.sum())
        else:
            c = float((conf[nodes] * w).sum() / w.sum())
            if prob is not None and len(nodes) > 1:
                inner = np.isin(fr['edges'][:, 0], nodes) & np.isin(fr['edges'][:, 1], nodes)
                c *= float(prob[inner].mean()) if inner.any() else 1.0
            out[i] = c
    return out


def filter_masks(masks, scores, min_points=0, min_score=0.0):
    keep = (masks.sum(1) >= min_points) & (scores >= min_score)
    return masks[keep], scores[keep]


def evaluate(frames, checkpoints, stuff, min_gt_points=100, cosine_threshold=0.5, adj_eps=0.05,
             linkage='average', device='cpu', gt_source='2d', min_points=0, min_score=0.0):
    """``checkpoints``: {name: (model, std, threshold)}. Returns per-frame rows and means.

    ``min_points`` / ``min_score`` filter the model's output masks (ESAM's npoint_thr / inst_score_thr
    counterpart); FastSAM / cosine / oracles are reported unfiltered.
    """
    rows, timing = [], {k: [] for k in checkpoints}
    cos_col = edge_feature_names().index('group_cos')
    for fr in frames:
        gts = thing_gtset(fr, stuff, min_gt_points, gt_source)
        if gts is None:
            continue
        g, e = fr['groups'], fr['edges']
        gt_inst = frame_gt(fr, gt_source)[0]
        row = dict(name=fr['name'], scene=fr['scene'], num_gt=len(gts.ids), num_groups=len(fr['sizes']), num_edges=len(e))
        row['FastSAM'] = score_masks(g, gts)
        cos = merge_groups(g, e, fr['edge_feat'][:, cos_col], cosine_threshold, linkage)
        row['cosine'] = score_masks(cos, gts)
        for name, (model, std, thr) in checkpoints.items():
            t0 = time.perf_counter()
            prob, obj = predict_frame(model, std, fr, device, with_objectness=True)
            if device != 'cpu':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            merged = merge_groups(g, e, prob, thr, linkage)
            t2 = time.perf_counter()
            timing[name].append((t1 - t0, t2 - t1))
            sc = merged_scores(fr, merged, prob, obj if model.has_objectness else None)
            row[name] = score_masks(merged, gts, sc, min_points, min_score)
        row['Oracle-adj'] = score_masks(oracle_merge_adjacent(g, gt_inst, fr['xyz'], 0.8, adj_eps), gts)
        row['Oracle'] = score_masks(oracle_merge(g, gt_inst, 0.8), gts)
        rows.append(row)
    methods = ['FastSAM', 'cosine', *checkpoints, 'Oracle-adj', 'Oracle']
    means = {m: {k: float(np.nanmean([r[m][k] for r in rows])) for k in METRIC_KEYS} for m in methods}
    ms = {n: dict(model_ms=1000 * float(np.mean([a for a, _ in t])), cluster_ms=1000 * float(np.mean([b for _, b in t])))
          for n, t in timing.items() if t}
    return rows, means, ms


def to_markdown(means, ms, n_frames, esam_ref=None, title='GroupMergeNet evaluation', gt_source='2d'):
    src = {'2d': 'dataset instance PNGs', '3d': "ESAM's 3D-transferred per-frame GT (same protocol as the ESAM study)"}[gt_source]
    lines = [f'# {title}', '', f'{n_frames} val frames, GT = {src}, thing instances with >= 100 points, frame-averaged.', '',
             '| method | masks/frame | mean best IoU | recall@0.5 | recall@0.25 | AP50 | AP25 | fragments/GT |',
             '|---|---|---|---|---|---|---|---|']
    for m, r in means.items():
        lines.append(f"| {m} | {r['num_masks']:.1f} | {r['mean_best_iou']:.3f} | {r['recall@0.5']:.3f} | "
                     f"{r['recall@0.25']:.3f} | {r['AP50']:.3f} | {r['AP25']:.3f} | {r['fragments']:.2f} |")
    if esam_ref:
        for m, r in esam_ref.items():
            lines.append(f"| {m} (ESAM study) | {r['num_pred']:.1f} | {r['mean_best_iou']:.3f} | {r['recall@0.5']:.3f} | "
                         f"{r['recall@0.25']:.3f} | {r['AP50']:.3f} | {r['AP25']:.3f} | {r['mean_fragments']:.2f} |")
    if ms:
        lines += ['', '| checkpoint | model forward ms/frame | clustering ms/frame |', '|---|---|---|']
        lines += [f'| {n} | {t["model_ms"]:.2f} | {t["cluster_ms"]:.2f} |' for n, t in ms.items()]
    return '\n'.join(lines) + '\n'
