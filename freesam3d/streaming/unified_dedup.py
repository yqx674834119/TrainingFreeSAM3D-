"""Output-side containment suppression (stage 3): after NMS, drop a mask that lies mostly inside a
higher-scoring one.

NMS only compares IoU: a fragment and the whole object that contains it overlap with an IoU well
below 0.5, so both survive. Stage 2's output is full of such pairs (dup_rate 0.239, 78.9 masks per
scene against ESAM's 40.2), so stage 3 adds a second cleaning step: walk the masks in score order
and drop every mask whose *containment* in an already-kept, higher-scoring mask reaches ``thr``
(containment = |a & b| / |a|, so a part is suppressed by its whole but not the other way around).

Everything is matrix-based: one (K, K) overlap matrix per scene (voxel sets through
``voxel.overlap_matrix``, mesh masks through one float matmul), then one greedy pass in score
order - the same shape as ``postprocess.greedy_nms``.
"""
import numpy as np
import torch

from .voxel import overlap_matrix


def containment_suppress(scores, contain, thr):
    """Greedy keep in score order: drop i when contain[i, j] >= thr for a kept j with a higher score.

    ``contain`` is (K, K) with contain[i, j] = share of i inside j. Marks only ever come from kept
    masks (suppressed ones are skipped before they can mark anything), so a mask survives exactly
    when no higher-scoring *survivor* contains it. Returns the kept indices in score order.
    """
    scores = np.asarray(scores, np.float64)
    contain = np.asarray(contain)
    k = len(scores)
    if k == 0 or thr is None or float(thr) > 1.0:
        return np.arange(k, dtype=np.int64)
    order = np.argsort(-scores, kind='stable')
    suppressed = np.zeros(k, bool)
    keep = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        suppressed |= contain[:, i] >= float(thr)
    return np.asarray(keep, np.int64)


def voxel_containment(vox_sets, device='cpu'):
    """(K, K) containment |i & j| / |i| of the object voxel sets (deduplicated cell ids)."""
    k = len(vox_sets)
    if k == 0:
        return np.zeros((0, 0), np.float32)
    owner = torch.repeat_interleave(
        torch.arange(k, device=device),
        torch.tensor([len(v) for v in vox_sets], dtype=torch.int64, device=device))
    vox = torch.cat([torch.as_tensor(v, dtype=torch.int64, device=device) for v in vox_sets])
    inter, size, _ = overlap_matrix(owner, vox, owner, vox, k, k)
    return (inter / size[:, None].clamp_min(1)).cpu().numpy().astype(np.float32)


def mask_containment(masks, device='cpu'):
    """(K, K) containment |i & j| / |i| of boolean masks (one float matmul, like ``iou_pair_matrix``)."""
    m = np.asarray(masks)
    if len(m) == 0:
        return np.zeros((0, 0), np.float32)
    if not (isinstance(device, str) and device == 'cpu'):
        t = torch.from_numpy(np.ascontiguousarray(m)).to(device).to(torch.float32)
        inter = t @ t.t()
        return (inter / t.sum(1)[:, None].clamp_min(1)).cpu().numpy().astype(np.float32)
    m = m.astype(np.float32)
    inter = m @ m.T
    return (inter / np.maximum(m.sum(1)[:, None], 1)).astype(np.float32)


def dedup_objects(obj_vox, scores, thr, device='cpu'):
    """Kept indices (score order) of a scene's object voxel sets after containment suppression."""
    if thr is None or float(thr) > 1.0 or len(obj_vox) == 0:
        return np.arange(len(obj_vox), dtype=np.int64)
    return containment_suppress(scores, voxel_containment(obj_vox, device), thr)


def dedup_masks(masks, scores, thr, device='cpu'):
    """Kept indices (score order) of boolean masks after containment suppression (post-NMS step)."""
    if thr is None or float(thr) > 1.0 or len(masks) == 0:
        return np.arange(len(masks), dtype=np.int64)
    return containment_suppress(scores, mask_containment(masks, device), thr)
