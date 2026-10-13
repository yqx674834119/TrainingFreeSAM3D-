"""Scene-level evaluation of a streamed QueryMemory on ESAM's reconstructed mesh.

Protocol = ESAM's (mixformer3d.py:762-782, unified_metric.py): final instance masks on mesh
vertices, class-agnostic AP against the mesh instance GT (thing instances >= min_gt_points
vertices), plus the per-frame-stacking lower bound (ESAM's ``ins_cat``) and, where available,
ESAM's own exported ``scene_masks`` on the same vertices.
"""
import os

import numpy as np
import torch

from ..evaluation.instance import (GTSet, class_agnostic_ap, coverage_stats, fragments_per_gt,
                                    groups_to_masks, oracle_merge_adjacent)
from .memory import FrameInput


def frame_input_from_proposal_cache(d, score_key='scorer', scores=None, feat_key='feat_raw', top_k=None):
    """Build a FrameInput from a proposal-cache dict (see tools/proposal_build_cache.py / stream_build_cache.py)."""
    masks = np.unpackbits(d['masks'], axis=1, count=int(d['num_points'])).astype(bool)
    s = scores if scores is not None else d['iou']          # oracle fallback when no scorer scores are given
    feat = d[feat_key].astype(np.float32) if feat_key in d else d['feat'].astype(np.float32)
    if top_k and len(s) > top_k:
        keep = np.argsort(-s)[:top_k]
        masks, s, feat = masks[keep], s[keep], feat[keep]
    return FrameInput(d['xyz'].astype(np.float32), masks, s.astype(np.float32), feat, int(d.get('frame_id', 0)))


def select_frame_proposals(fr, nms_iou=0.5, nms_contain=1.01, top_k=20, device='cuda'):
    """What ``GroupMerger`` emits per frame: nested NMS over the scored proposals, then the top_k by score."""
    from ..merging.proposals_gpu import nested_nms_gpu
    if len(fr.masks) == 0:
        return fr
    keep = nested_nms_gpu(fr.masks, fr.scores, nms_iou, nms_contain, device)
    if top_k:
        keep = keep[:top_k]
    return FrameInput(fr.xyz, fr.masks[keep], fr.scores[keep], fr.feat[keep], fr.frame_id)


def segment_smooth(masks, segment_ids, thr=0.5, device='cpu'):
    """ESAM's mesh-segment smoothing (mixformer3d.py::segment_smooth): a segment belongs to a mask when more than
    ``thr`` of its vertices do. (K,V) bool -> (K,V) bool; one index_add_ per mask row, no per-segment loop."""
    seg = torch.as_tensor(np.unique(np.asarray(segment_ids), return_inverse=True)[1], device=device)
    S = int(seg.max()) + 1
    m = torch.as_tensor(np.asarray(masks), device=device).float()
    if len(m) == 0:
        return np.asarray(masks, bool)
    cnt = torch.bincount(seg, minlength=S).float().clamp_min(1)
    frac = torch.zeros((len(m), S), device=device).index_add_(1, seg, m) / cnt
    return (frac > thr)[:, seg].cpu().numpy()


def mesh_gtset(mesh_gt, mesh_sem, stuff, min_gt_points=100):
    ids, c = np.unique(mesh_gt[mesh_gt > 0], return_counts=True)
    keep = [k for k in ids[c >= min_gt_points] if np.bincount(mesh_sem[mesh_gt == k]).argmax() not in stuff]
    return GTSet.from_labels(mesh_gt, keep) if keep else None


def scene_metrics(masks, scores, gts, min_points=100, device='cpu'):
    keep = masks.sum(1) >= min_points
    masks, scores = masks[keep], scores[keep]
    if len(masks) == 0:
        return dict(ap50=0.0, ap25=0.0, iou=0.0, r50=0.0, masks=0)
    cov = coverage_stats(masks, gts, device=device)
    return dict(ap50=class_agnostic_ap(masks, scores, gts, .5, device=device),
                ap25=class_agnostic_ap(masks, scores, gts, .25, device=device),
                iou=cov['mean_best_iou'], r50=cov['recall@0.5'],
                frags=float(fragments_per_gt(masks, gts, device=device).mean()),
                masks=int(len(masks)))


def stack_frames_baseline(frames, mesh_xyz, voxel, top_k_total=200):
    """ESAM ``ins_cat`` analogue: every per-frame proposal becomes its own scene instance, rendered through the
    same voxel grid, capped to the top-k by score. The no-memory lower bound."""
    from .memory import MemoryConfig, QueryMemory
    all_masks, all_scores = [], []
    for fr in frames:
        mem = QueryMemory(MemoryConfig(voxel=voxel, admit_score=0.0, admit_min_voxels=1, admit_support=0.0,
                                       max_queries=10 ** 6), device='cpu')
        mem.update(fr)
        m, s = mem.render(mesh_xyz)
        all_masks.append(m); all_scores.append(s)
    masks = np.concatenate(all_masks) if all_masks else np.zeros((0, len(mesh_xyz)), bool)
    scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
    if len(scores) > top_k_total:
        keep = np.argsort(-scores)[:top_k_total]
        masks, scores = masks[keep], scores[keep]
    return masks, scores


def frames_from_merge_cache(files, gt='3d'):
    """merge_cache npz files -> FragmentFrames sorted by (scene, frame id).

    Rebuilds exactly what the cache stores for a frame: the 20k points, the compact
    fragment ids, the cached intra-frame graph, and the FastSAM confidence per fragment
    (0 for the KMeans fill groups, which have no proposal mask).
    """
    from ..merging import cache as merge_cache
    from .window_memory import FragmentFrame

    def key(path):
        scene, fid = os.path.basename(path)[:-4].rsplit('_', 1)
        return scene, int(fid)

    out = []
    for path in sorted(files, key=key):
        d = merge_cache.load(path)
        scene, fid = key(path)
        groups = np.asarray(d['groups'], np.int64)
        num_groups = int(len(d['sizes']))
        conf = np.zeros(num_groups, np.float32)
        gmi = np.asarray(d['group_mask_index'])
        fs = np.asarray(d['fastsam_scores'], np.float32)
        has = gmi >= 0
        conf[has] = fs[gmi[has]]
        gt_arr = np.asarray(d['gt_instance_3d' if gt == '3d' else 'gt_instance'], np.int64)
        out.append(FragmentFrame(scene=scene, frame_id=fid, xyz=np.asarray(d['xyz'], np.float32),
                                 rgb=np.asarray(d['rgb'], np.float32), groups=groups,
                                 node_feat=np.asarray(d['node_feat'], np.float32),
                                 edges=np.asarray(d['edges'], np.int64),
                                 edge_feat=np.asarray(d['edge_feat'], np.float32),
                                 conf=conf, gt=gt_arr))
    return out


def oracle_frame_instances(fr, purity=0.8, eps=0.05):
    """Per-frame oracle instances (GT merge of the frame's fragments) as a FrameInput.

    Feature = point-weighted mean of the members' ``node_feat``; score = point-weighted mean
    of the members' FastSAM confidence. This is what feeds the reference memories
    (ESAM rule / QueryMemory), which take per-frame *instances* rather than fragments.
    """
    merged = oracle_merge_adjacent(fr.groups, fr.gt, fr.xyz, purity, eps)
    masks, ids = groups_to_masks(merged)
    num_groups = len(fr.node_feat)
    row = np.searchsorted(ids, merged)
    w = np.zeros((len(ids), num_groups), np.float32)
    np.add.at(w, (row, fr.groups), 1.0)
    wsum = np.maximum(w.sum(1, keepdims=True), 1)
    feat = (w @ fr.node_feat) / wsum
    score = ((w @ fr.conf) / wsum.ravel()).astype(np.float32)
    return FrameInput(fr.xyz, masks, score, feat.astype(np.float32), fr.frame_id)
