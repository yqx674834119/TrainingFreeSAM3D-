"""Group adjacency graph with compact node and edge features for merging.

Design constraints (see docs/merge_net_design.zh-CN.md):
* every edge feature is a scalar computed once while building the graph, so
  the model only ever sees 14 numbers per edge plus a 16-d projection of
  the group's FastSAM feature; inference cost is dominated by graph building,
  not by the network;
* the graph is built on a 2 cm voxel subsample of the 20k points (one point
  per voxel *and* group) — this keeps group boundaries intact, recovers
  >98% of the adjacency edges and is faster than the full KD-tree;
* boundary contrast uses a *fine* feature map (stride-4 ``proto``): the
  stride-32 backbone map puts two points 5 cm apart in the same cell, so it
  cannot see a boundary.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

NODE_GEOM = 11   # log size, height, extents(3), sqrt-eigvals(3), mean rgb(3)
EDGE_DIM = 14    # see edge_feature_names()


def edge_feature_names():
    return ['min_dist', 'contact_norm', 'centroid_dist', 'height_diff', 'extent_ratio', 'size_ratio',
            'color_dist', 'conf_diff', 'normal_angle', 'contact_depth_gap',
            'group_cos', 'boundary_cos', 'boundary_cos_min', 'nested']


@dataclass
class GroupGraph:
    group_ids: np.ndarray      # (G,) group id per node row
    edges: np.ndarray          # (E, 2) node indices, a < b
    node_geom: np.ndarray      # (G, NODE_GEOM)
    node_feat: np.ndarray      # (G, C) pooled FastSAM feature (L2-normalised)
    edge_feat: np.ndarray      # (E, EDGE_DIM)
    sizes: np.ndarray          # (G,) point counts

    @property
    def num_nodes(self):
        return len(self.group_ids)


def _voxel_subsample(xyz, groups, voxel):
    """One representative point per (voxel, group) cell."""
    if voxel <= 0:
        return np.arange(len(xyz))
    key = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(np.c_[key, groups], axis=0, return_index=True)
    return idx


def _contact_pairs(xyz, groups, eps, voxel):
    """Cross-group point pairs closer than ``eps``, sorted by (ga, gb).

    Returns edges (E,2), pair point indices pa/pb (P,), and ``starts`` (E+1,)
    delimiting each edge's pairs.
    """
    sub = _voxel_subsample(xyz, groups, voxel)
    pairs = cKDTree(xyz[sub]).query_pairs(eps, output_type='ndarray')
    if len(pairs) == 0:
        return np.empty((0, 2), np.int64), np.empty(0, np.int64), np.empty(0, np.int64), np.zeros(1, np.int64)
    pa, pb = sub[pairs[:, 0]], sub[pairs[:, 1]]
    ga, gb = groups[pa], groups[pb]
    keep = ga != gb
    pa, pb, ga, gb = pa[keep], pb[keep], ga[keep], gb[keep]
    if len(ga) == 0:      # every near pair was inside one group -> no contact edges
        return np.empty((0, 2), np.int64), np.empty(0, np.int64), np.empty(0, np.int64), np.zeros(1, np.int64)
    swap = ga > gb
    pa, pb = np.where(swap, pb, pa), np.where(swap, pa, pb)
    ga, gb = np.minimum(ga, gb), np.maximum(ga, gb)
    key = ga * (groups.max() + 1) + gb
    order = np.argsort(key, kind='stable')
    key, pa, pb, ga, gb = key[order], pa[order], pb[order], ga[order], gb[order]
    starts = np.r_[0, np.flatnonzero(np.diff(key)) + 1, len(key)]
    edges = np.stack([ga[starts[:-1]], gb[starts[:-1]]], 1)
    return edges, pa, pb, starts


def _node_stats(xyz, rgb, groups, num_groups):
    sizes = np.bincount(groups, minlength=num_groups).astype(np.float32)
    inv = 1.0 / np.maximum(sizes, 1)
    cent = np.zeros((num_groups, 3), np.float64)
    np.add.at(cent, groups, xyz)
    cent *= inv[:, None]
    col = np.zeros((num_groups, 3), np.float64)
    np.add.at(col, groups, rgb / 255.0)
    col *= inv[:, None]
    d = xyz - cent[groups]
    cov = np.zeros((num_groups, 3, 3), np.float64)
    np.add.at(cov, groups, d[:, :, None] * d[:, None, :])
    cov *= inv[:, None, None]
    eig, vec = np.linalg.eigh(cov)                  # ascending
    eig = np.sqrt(np.maximum(eig, 0))
    normals = vec[:, :, 0]                          # smallest-variance direction
    lo = np.full((num_groups, 3), np.inf)
    hi = np.full((num_groups, 3), -np.inf)
    np.minimum.at(lo, groups, xyz)
    np.maximum.at(hi, groups, xyz)
    ext = np.where(np.isfinite(hi - lo), hi - lo, 0)
    height = cent[:, 2] - np.median(xyz[:, 2])
    geom = np.concatenate([np.log1p(sizes)[:, None], height[:, None], ext, eig[:, ::-1], col], 1)
    return sizes, cent, ext, col, normals, geom.astype(np.float32)


def _normalise(f, centering):
    f = np.asarray(f, np.float32)
    if centering:
        f = f - f.mean(0, keepdims=True)
    return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)


def mask_nesting(raw_masks, stride=4):
    """(M,M) containment of the smaller raw mask in the larger: |a∩b| / min(|a|,|b|).

    Computed on a ``stride``-subsampled pixel grid: the ratio is a coarse cue
    (nested or not) and the full 307k-pixel product dominated graph build time.
    """
    m = raw_masks[:, ::stride, ::stride].reshape(len(raw_masks), -1).astype(np.float32)
    inter = m @ m.T
    area = m.sum(1)
    return inter / np.maximum(np.minimum(area[:, None], area[None, :]), 1)


def _segment_reduce(values, starts, fn):
    """Apply ``fn`` (mean/min/max with axis) over contiguous segments of ``values``."""
    return np.array([fn(values[s:e]) for s, e in zip(starts[:-1], starts[1:])], np.float32)


def build_graph(xyz, rgb, groups, group_feat, point_feat=None, scores=None, mask_index=None,
                raw_masks=None, eps=0.05, voxel=0.02, boundary_k=32, feature_centering=True):
    """Build the adjacency graph of one lifted frame.

    Parameters
    ----------
    xyz, rgb, groups : (N,3), (N,3) uint8, (N,) compact group id in 0..G-1
    group_feat : (G,C) pooled FastSAM feature per group (node feature + group_cos)
    point_feat : (N,C') per-point FastSAM feature for boundary contrast; use a
                 fine map (stride <= 8). Optional: falls back to group_cos.
    scores : (M,) FastSAM confidence per raw mask; mask_index : (G,) raw mask index or -1
    raw_masks : (M,H,W) bool raw (overlapping) FastSAM masks, for the nesting cue; optional
    """
    xyz = np.asarray(xyz, np.float64)
    groups = np.asarray(groups, np.int64)
    G = int(groups.max()) + 1
    sizes, cent, ext, col, normals, geom = _node_stats(xyz, np.asarray(rgb, np.float64), groups, G)
    conf = np.zeros(G, np.float32)
    if scores is not None and mask_index is not None:
        has = mask_index >= 0
        conf[has] = scores[mask_index[has]]
    feat = _normalise(group_feat, feature_centering)
    pf = _normalise(point_feat, feature_centering) if point_feat is not None else None
    nested = mask_nesting(raw_masks) if (raw_masks is not None and mask_index is not None and len(raw_masks)) else None

    edges, pa, pb, starts = _contact_pairs(xyz, groups, eps, voxel)
    E = len(edges)
    ef = np.zeros((E, EDGE_DIM), np.float32)
    if E == 0:
        return GroupGraph(np.arange(G), edges, geom, feat, ef, sizes)
    a, b = edges[:, 0], edges[:, 1]
    dist = np.linalg.norm(xyz[pa] - xyz[pb], axis=1)
    # keep the k closest contact pairs per edge for the local statistics
    seg = np.repeat(np.arange(E), np.diff(starts))
    order = np.lexsort((dist, seg))
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order)) - starts[seg[order]]
    near = rank < boundary_k
    n_starts = np.r_[0, np.cumsum(np.bincount(seg[near], minlength=E))]
    pa_n, pb_n = pa[near], pb[near]

    ef[:, 0] = _segment_reduce(dist, starts, np.min)
    touched = _segment_reduce(np.r_[pa, pb].reshape(2, -1).T, starts, lambda v: len(np.unique(v)))
    ef[:, 1] = touched / np.maximum(np.minimum(sizes[a], sizes[b]), 1)
    ef[:, 2] = np.linalg.norm(cent[a] - cent[b], axis=1)
    ef[:, 3] = np.abs(cent[a, 2] - cent[b, 2])
    ef[:, 4] = np.abs(np.log((ext[a].max(1) + 1e-3) / (ext[b].max(1) + 1e-3)))
    ef[:, 5] = np.abs(np.log(sizes[a] / sizes[b]))
    ef[:, 6] = np.linalg.norm(col[a] - col[b], axis=1)
    ef[:, 7] = np.abs(conf[a] - conf[b])
    ef[:, 8] = 1 - np.abs(np.einsum('ij,ij->i', normals[a], normals[b]))
    ef[:, 9] = _segment_reduce(np.abs(xyz[pa_n, 2] - xyz[pb_n, 2]), n_starts, np.mean)
    ef[:, 10] = np.einsum('ij,ij->i', feat[a], feat[b])
    if pf is not None:
        cs = np.einsum('ij,ij->i', pf[pa_n], pf[pb_n])
        ef[:, 11] = _segment_reduce(cs, n_starts, np.mean)
        ef[:, 12] = _segment_reduce(cs, n_starts, np.min)
    else:
        ef[:, 11] = ef[:, 12] = ef[:, 10]
    if nested is not None:
        ma, mb = mask_index[a], mask_index[b]
        ok = (ma >= 0) & (mb >= 0)
        ef[ok, 13] = nested[ma[ok], mb[ok]]
    return GroupGraph(np.arange(G), edges, geom, feat, ef, sizes)
