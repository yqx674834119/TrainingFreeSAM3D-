"""Mask proposals from one frame: FastSAM's raw (overlapping) masks ∪ GroupMergeNet merged masks.

FastSAM's raw masks form a hierarchy (part ⊂ whole); flattening them into
exclusive groups — what ESAM does — throws that away and then re-learns it
with a 47M-param decoder. Here each raw mask is kept as a *proposal*, merged
masks are added as extra proposals, and a scorer ranks them for nested NMS.

Per proposal we keep GT-free features only:
  geometry (log size, extents, PCA eigenvalues, flatness, height),
  group composition (number of exclusive groups it spans, concentration,
  mean / min objectness of those groups, mean FastSAM confidence),
  hierarchy (raw confidence, #parents, #children, #raw masks inside),
  appearance (size-weighted mean of the pooled FastSAM backbone feature,
  proto-feature coherence = ||mean of unit proto vectors||).
"""
from dataclasses import dataclass

import numpy as np
import torch

NUM_SCALARS = 20


def scalar_names():
    return ['log_size', 'ext_x', 'ext_y', 'ext_z', 'ev1', 'ev2', 'ev3', 'flatness', 'height',
            'obj_mean', 'obj_min', 'n_groups', 'group_conc', 'conf_mean', 'is_raw', 'raw_conf',
            'proto_coh', 'n_parent', 'n_child', 'n_raw_inside']


@dataclass
class Proposals:
    masks: np.ndarray        # (P, N) bool on the 20k points
    scalars: np.ndarray      # (P, NUM_SCALARS) float32
    feat: np.ndarray         # (P, C) float32, size-weighted mean of pooled group features (L2-normalised)
    is_raw: np.ndarray       # (P,) bool
    source_index: np.ndarray  # (P,) raw mask index, or -1 for merged proposals


def _containment(masks_f):
    """contain[i, j] = fraction of mask j inside mask i."""
    inter = masks_f @ masks_f.T
    return inter / np.maximum(masks_f.sum(1)[None, :], 1)


def build_proposals(xyz, groups, group_feat, point_proto, raw_masks_pts, raw_scores, mask_index, objectness,
                    merged_masks=None, min_points=100, nest_thr=0.8):
    """
    xyz (N,3); groups (N,) exclusive group id; group_feat (G,C) L2-normalised; point_proto (N,Cp);
    raw_masks_pts (M,N) bool = raw FastSAM masks sampled at the points; raw_scores (M,);
    mask_index (G,) raw index per group or -1; objectness (G,) from GroupMergeNet;
    merged_masks (K,N) bool, optional.
    """
    xyz = np.asarray(xyz, np.float32)
    groups = np.asarray(groups)
    G = int(groups.max()) + 1
    sizes = np.bincount(groups, minlength=G).astype(np.float32)
    conf_g = np.zeros(G, np.float32)
    has = mask_index >= 0
    conf_g[has] = raw_scores[mask_index[has]]
    proto = np.asarray(point_proto, np.float32)
    proto = proto / np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), 1e-8)
    z0 = float(xyz[:, 2].min())

    keep_raw = raw_masks_pts.sum(1) >= min_points
    raw = raw_masks_pts[keep_raw]
    raw_idx = np.flatnonzero(keep_raw)
    cands = [(m, True, float(raw_scores[i]), int(i)) for m, i in zip(raw, raw_idx)]
    if merged_masks is not None:
        cands += [(m, False, 0.0, -1) for m in merged_masks if m.sum() >= min_points]
    if not cands:
        return Proposals(np.zeros((0, len(xyz)), bool), np.zeros((0, NUM_SCALARS), np.float32),
                         np.zeros((0, group_feat.shape[1]), np.float32), np.zeros(0, bool), np.zeros(0, int))
    masks = np.stack([c[0] for c in cands])
    mf = masks.astype(np.float32)
    raw_f = raw.astype(np.float32)
    # hierarchy among raw masks, and raw masks inside every candidate
    if len(raw):
        contain_rr = _containment(raw_f)                                  # (R, R)
        n_parent_raw = (contain_rr >= nest_thr).sum(0) - 1
        n_child_raw = (contain_rr >= nest_thr).sum(1) - 1
        inside = (mf @ raw_f.T) / np.maximum(raw_f.sum(1)[None, :], 1)   # (P, R) frac of raw j inside cand i
        n_raw_inside = (inside >= nest_thr).sum(1) - np.array([c[1] for c in cands])   # exclude itself
    else:
        n_parent_raw = n_child_raw = np.zeros(0, int)
        n_raw_inside = np.zeros(len(cands), int)

    # group composition via one matmul: (P, G) point counts per group
    onehot = np.zeros((len(xyz), G), np.float32)
    onehot[np.arange(len(xyz)), groups] = 1
    pg = mf @ onehot                                                      # (P, G) points of group g in cand p
    w = pg / np.maximum(pg.sum(1, keepdims=True), 1)
    obj_mean = w @ objectness
    obj_min = np.where(pg > 0, objectness[None, :], np.inf).min(1)
    n_groups = (pg > 0).sum(1)
    conc = (w ** 2).sum(1)
    conf_mean = w @ conf_g
    feat = w @ group_feat
    feat /= np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-8)
    coh = np.linalg.norm((mf @ proto) / np.maximum(mf.sum(1, keepdims=True), 1), axis=1)

    # per-mask geometry from first/second moments: (P,N) @ (N,·) matmuls, no Python loop
    cnt = mf.sum(1, keepdims=True)
    mean = (mf @ xyz) / np.maximum(cnt, 1)                                 # (P,3)
    sq = (mf @ (xyz[:, :, None] * xyz[:, None, :]).reshape(len(xyz), 9)).reshape(-1, 3, 3) / np.maximum(cnt, 1)[:, :, None]
    cov = sq - mean[:, :, None] * mean[:, None, :]
    cov *= (cnt / np.maximum(cnt - 1, 1))[:, :, None]                      # unbiased, as np.cov
    ev = np.sqrt(np.maximum(np.linalg.eigvalsh(cov), 0))[:, ::-1]          # (P,3) descending
    ev[cnt[:, 0] <= 3] = 0
    big = np.where(masks[:, :, None], xyz[None, :, :], np.nan)
    ext = np.nanmax(big, 1) - np.nanmin(big, 1)
    scal = np.zeros((len(cands), NUM_SCALARS), np.float32)
    scal[:, 0] = np.log(cnt[:, 0])
    scal[:, 1:4] = np.log1p(ext)
    scal[:, 4:7] = ev
    scal[:, 7] = ev[:, 2] / np.maximum(ev[:, 0], 1e-6)
    scal[:, 8] = mean[:, 2] - z0
    scal[:, 9], scal[:, 10], scal[:, 11], scal[:, 12], scal[:, 13] = obj_mean, obj_min, n_groups, conc, conf_mean
    scal[:, 16] = coh
    scal[:, 19] = n_raw_inside
    for i, (m, is_raw, rconf, ridx) in enumerate(cands):
        scal[i, 14], scal[i, 15] = float(is_raw), rconf
        if is_raw:
            r = np.searchsorted(raw_idx, ridx)
            scal[i, 17], scal[i, 18] = n_parent_raw[r], n_child_raw[r]
    return Proposals(masks, scal, feat.astype(np.float32), np.array([c[1] for c in cands]),
                     np.array([c[3] for c in cands]))


def proposal_targets(masks, gt, min_gt_points=100, keep_ids=None):
    """IoU of each proposal to its best GT instance (labeled points only), plus best GT id."""
    labeled = gt > 0
    ids = keep_ids if keep_ids is not None else np.unique(gt[labeled])
    if len(ids) == 0:
        return np.zeros(len(masks), np.float32), np.zeros(len(masks), np.int64)
    onehot = (gt[:, None] == np.asarray(ids)[None, :]).astype(np.float32)
    p = (masks & labeled[None, :]).astype(np.float32)
    inter = p @ onehot
    union = p.sum(1, keepdims=True) + onehot.sum(0)[None, :] - inter
    iou = inter / np.maximum(union, 1)
    return iou.max(1).astype(np.float32), np.asarray(ids)[iou.argmax(1)]


def nested_nms(masks, scores, iou_thr=0.5, contain_thr=0.8):
    """Greedy selection by score; a candidate is suppressed if its IoU with a kept
    mask exceeds ``iou_thr`` OR it is >= ``contain_thr`` inside a kept mask (child of a
    selected whole) OR it contains >= ``contain_thr`` of a kept mask (parent of a
    selected part). Returns kept indices in score order.
    """
    if len(masks) == 0:
        return np.zeros(0, int)
    mf = masks.astype(np.float32)
    inter = mf @ mf.T
    area = mf.sum(1)
    iou = inter / np.maximum(area[:, None] + area[None, :] - inter, 1)
    frac_in = inter / np.maximum(area[None, :], 1)          # frac_in[i,j] = frac of j inside i
    order = np.argsort(-scores, kind='stable')
    keep, sup = [], np.zeros(len(masks), bool)
    for i in order:
        if sup[i]:
            continue
        keep.append(i)
        sup |= (iou[i] > iou_thr) | (frac_in[i] >= contain_thr) | (frac_in[:, i] >= contain_thr)
    return np.array(keep, int)
