"""Class-agnostic instance evaluation on a fixed point set.

Conventions
-----------
* ``gt``: (N,) int instance id per point, ``<= 0`` = unlabeled. IoU is computed
  on labeled points only (prediction points on unlabeled areas are ignored).
* predictions: (P, N) bool masks. They may overlap (ESAM) or be exclusive
  (FastSAM groups, oracle merges).
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass
class GTSet:
    ids: np.ndarray        # (K,) instance ids, sorted
    onehot: np.ndarray     # (N, K) bool membership
    sizes: np.ndarray      # (K,) point counts
    labeled: np.ndarray    # (N,) bool

    @classmethod
    def from_labels(cls, gt, keep_ids=None):
        labeled = gt > 0
        ids = np.unique(gt[labeled]) if keep_ids is None else np.asarray(sorted(keep_ids))
        onehot = gt[:, None] == ids[None, :]
        return cls(ids, onehot, onehot.sum(0), labeled)


def groups_to_masks(group_ids):
    """(N,) group id per point (negative = none) -> (G, N) bool masks and the ids."""
    ids = np.unique(group_ids[group_ids >= 0])
    return group_ids[None, :] == ids[:, None], ids


def _labeled_terms(pred, gts: GTSet, device='cpu'):
    """``pred`` (P, N) Boolean x ``gts`` -> (inter (P, K) float32, |pred & labeled| (P,) int).

    One float32 matmul. On CUDA the AND, the matmul and the reduction all run on the GPU and
    only the (P, K) / (P,) results come back: with P up to ~1500 masks over ~211k mesh vertices
    the numpy path costs minutes per scene. Numerically identical to the numpy reference
    (asserted in tests/test_iou_torch.py).
    """
    if not (isinstance(device, str) and device == 'cpu'):   # torch.device('cpu') -> torch path
        import torch
        t_pred = torch.from_numpy(np.ascontiguousarray(pred)).to(device)
        t_lab = torch.from_numpy(np.ascontiguousarray(gts.labeled)).to(device)
        t_oh = torch.from_numpy(np.ascontiguousarray(gts.onehot)).to(device).to(torch.float32)
        p = t_pred & t_lab
        pf = p.to(torch.float32)
        return (pf @ t_oh).to('cpu').numpy(), p.sum(1).to('cpu').numpy()
    p = pred & gts.labeled[None, :]
    return p.astype(np.float32) @ gts.onehot.astype(np.float32), p.sum(1)


def iou_matrix(pred, gts: GTSet, device='cpu'):
    """(P, K) IoU, counting only labeled points."""
    inter, size_p = _labeled_terms(pred, gts, device)
    union = size_p[:, None] + gts.sizes[None, :] - inter
    return inter / np.maximum(union, 1)


def coverage_stats(pred, gts: GTSet, thresholds=(0.25, 0.5), device='cpu'):
    """Best IoU per GT, recall@t (GT matched by any mask) and precision@t."""
    if len(gts.ids) == 0:
        return {}
    iou = iou_matrix(pred, gts, device) if len(pred) else np.zeros((0, len(gts.ids)))
    best_gt = iou.max(0) if len(pred) else np.zeros(len(gts.ids))
    out = dict(best_iou=best_gt, mean_best_iou=float(best_gt.mean()), num_pred=int(len(pred)))
    # precision: among predictions that touch labeled points
    touching = (pred & gts.labeled[None, :]).any(1) if len(pred) else np.zeros(0, bool)
    best_pred = iou.max(1) if len(gts.ids) else np.zeros(len(pred))
    for t in thresholds:
        out[f'recall@{t}'] = float((best_gt >= t).mean())
        out[f'precision@{t}'] = float((best_pred[touching] >= t).mean()) if touching.any() else float('nan')
    return out


def fragments_per_gt(pred, gts: GTSet, purity=0.8, min_abs=50, min_frac=0.05, device='cpu'):
    """Number of predicted masks that are a *fragment* of each GT instance.

    A mask counts for GT k if its labeled points are >= ``purity`` inside k
    and it contributes >= max(min_abs, min_frac*|G_k|) points to k.
    """
    if len(pred) == 0:
        return np.zeros(len(gts.ids), int)
    inter, size_p = _labeled_terms(pred, gts, device)                      # (P, K)
    pur = inter / np.maximum(size_p[:, None], 1)
    need = np.maximum(min_abs, min_frac * gts.sizes)[None, :]
    return ((pur >= purity) & (inter >= need)).sum(0)


def class_agnostic_ap(pred, scores, gts: GTSet, threshold, device='cpu'):
    """ScanNet-style AP at one IoU threshold: greedy matching by score, each GT once.

    Precision is made monotone (envelope) and integrated over recall.
    """
    k = len(gts.ids)
    if k == 0:
        return float('nan')
    if len(pred) == 0:
        return 0.0
    iou = iou_matrix(pred, gts, device)
    order = np.argsort(-np.asarray(scores), kind='stable')
    matched = np.zeros(k, bool)
    tp = np.zeros(len(order))
    for r, i in enumerate(order):
        cand = np.where(~matched & (iou[i] >= threshold), iou[i], -1)
        j = cand.argmax()
        if cand[j] >= 0:
            matched[j] = True
            tp[r] = 1
    ctp = np.cumsum(tp)
    recall = ctp / k
    precision = ctp / np.arange(1, len(order) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall = np.concatenate([[0.0], recall])
    return float(np.sum((recall[1:] - recall[:-1]) * precision))


def oracle_merge(group_ids, gt, purity=0.8, min_labeled=0.3):
    """Merge-only GT oracle: every group with purity >= θ joins its majority GT.

    Purity is measured on the group's labeled points; groups with less than
    ``min_labeled`` labeled points are left alone. Groups are never split;
    impure / unlabeled groups stay as they are. This is the best any *merging*
    method can do on top of the given groups. Returns a new exclusive (N,) assignment.
    """
    out = np.asarray(group_ids).copy()
    labeled = gt > 0
    next_id = out.max() + 1
    target_of = {}
    for g in np.unique(out[out >= 0]):
        sel = (out == g) & labeled
        if not sel.any():
            continue
        if sel.sum() < min_labeled * (out == g).sum():
            continue
        ids, counts = np.unique(gt[sel], return_counts=True)
        if counts.max() / sel.sum() >= purity:
            target_of[g] = ids[counts.argmax()]
    merged = {}
    for g, k in target_of.items():
        if k not in merged:
            merged[k] = next_id
            next_id += 1
        out[np.asarray(group_ids) == g] = merged[k]
    return out


def group_inclusion(pred, group_ids):
    """(P, G) fraction of each group's points that lie inside each predicted mask."""
    masks, _ = groups_to_masks(group_ids)
    inter = pred.astype(np.float32) @ masks.T.astype(np.float32)
    return inter / np.maximum(masks.sum(1)[None, :], 1)


def group_adjacency(xyz, group_ids, eps=0.05):
    """Set of (g, h), g < h, of groups with at least one point pair closer than ``eps``."""
    pairs = cKDTree(xyz).query_pairs(eps, output_type='ndarray')
    g, h = group_ids[pairs[:, 0]], group_ids[pairs[:, 1]]
    keep = (g != h) & (g >= 0) & (h >= 0)
    a, b = np.minimum(g[keep], h[keep]), np.maximum(g[keep], h[keep])
    return set(zip(a.tolist(), b.tolist()))


def oracle_merge_adjacent(group_ids, gt, xyz, purity=0.8, eps=0.05, min_labeled=0.3):
    """Like ``oracle_merge`` but groups of the same GT merge only through spatially
    adjacent groups (connected components of the adjacency graph)."""
    target = oracle_merge(group_ids, gt, purity, min_labeled)
    assigned = target != group_ids                       # groups that the oracle moved
    adj = group_adjacency(xyz, group_ids, eps)
    ids = np.unique(group_ids[group_ids >= 0])
    index = {g: i for i, g in enumerate(ids.tolist())}
    tgt_of = {g: int(target[group_ids == g][0]) for g in np.unique(group_ids[assigned])}
    rows = [index[a] for a, b in adj if a in tgt_of and b in tgt_of and tgt_of[a] == tgt_of[b]]
    cols = [index[b] for a, b in adj if a in tgt_of and b in tgt_of and tgt_of[a] == tgt_of[b]]
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids), len(ids)))
    _, comp = connected_components(graph, directed=False)
    out = np.asarray(group_ids).copy()
    base = out.max() + 1
    for g in tgt_of:
        out[group_ids == g] = base + comp[index[g]]
    return out


def association_stats(masks, gts: GTSet, thr=0.25, device='cpu'):
    """Duplicate / over-merge diagnostics for one scene's prediction set.

    * ``dup_rate``: fraction of GT instances covered (IoU >= ``thr``) by at least two
      predictions — "the same object opened several pages".
    * ``overmerge_rate``: fraction of predictions that cover >= 50% of at least two GT
      instances — "two objects glued into one page".
    * ``frags_per_gt``: mean number of fragment-like predictions per GT (same definition
      as ``fragments_per_gt``).
    """
    n_gt = len(gts.ids)
    if n_gt == 0:
        return dict(dup_rate=float('nan'), overmerge_rate=float('nan'), frags_per_gt=float('nan'))
    if len(masks) == 0:
        return dict(dup_rate=0.0, overmerge_rate=float('nan'), frags_per_gt=0.0)
    inter, size_p = _labeled_terms(masks, gts, device)
    iou = inter / np.maximum(size_p[:, None] + gts.sizes[None, :] - inter, 1)
    dup = (iou >= thr).sum(0) >= 2
    contain = inter / np.maximum(gts.sizes[None, :], 1)
    over = (contain >= 0.5).sum(1) >= 2
    return dict(dup_rate=float(dup.mean()), overmerge_rate=float(over.mean()),
                frags_per_gt=float(fragments_per_gt(masks, gts, device=device).mean()))
