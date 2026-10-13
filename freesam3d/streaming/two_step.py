"""Two-step grouping for the window memory (stage-1 option): blocks first, then cross-frame association.

``WindowMemory``'s default grouping feeds *every* edge of the frame's graph — the cached intra-frame
edges and the cross-frame ones — into one average-linkage pass. Stage 0c diagnosed two failure modes
of that (F0c.7): the cross-frame score lives on a different scale than the checkpoint threshold, and
averaging over a cluster's *already existing* edges lets a couple of wrong intra-frame edges outvote
several correct cross-frame ones. This module implements the alternative the stage-1 brief asks for:

1. cluster the current frame's fragments using the intra-frame edges alone (the GroupMergeNet
   probabilities and the threshold the per-frame pipeline already uses) -> *blocks*;
2. score every block against the objects it touches by the containment of **all** the block's cells
   in the object (optionally times the clamped feature cosine), and let each block join its
   best-scoring object when that score passes a threshold calibrated on the holdout scenes.
   Several blocks may join the same object; objects are never merged with one another.

Step 2 is deliberately a *directed* comparison (block -> object) rather than a cluster average, so
one object's evidence cannot be diluted by another pair's mistake. All functions are pure
(numpy/torch in, numpy/torch out) and matrix-based; ``window_memory.WindowMemory`` owns the state.
"""
import numpy as np
import torch

from .voxel import overlap_matrix

RULES = ('p2a', 'p2b')


def as_bool_np(x):
    """A numpy bool copy of a flag that may live on the GPU (``PairGraph.has_intra`` does)."""
    x = x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return x.astype(bool, copy=False)


def intra_blocks(num_cur, edges, prob, threshold, linkage='average_sparse'):
    """(num_cur,) block id per current fragment, from the intra-frame edges alone."""
    from ..merging.cluster import LINKAGE
    return LINKAGE[linkage](int(num_cur), np.asarray(edges, np.int64).reshape(-1, 2),
                            np.asarray(prob, np.float64).reshape(-1), float(threshold))


def block_voxel_sets(block_labels, frag_vox_list, device='cpu'):
    """(b_own, b_vox): the (block, cell) pairs of every block = union of its members' cell sets.

    ``frag_vox_list[i]`` is fragment ``i``'s deduplicated cell ids; ``block_labels[i]`` its block.
    """
    counts = torch.tensor([len(v) for v in frag_vox_list], dtype=torch.int64, device=device)
    owner = torch.repeat_interleave(torch.arange(len(frag_vox_list), device=device), counts)
    blk = torch.as_tensor(np.asarray(block_labels, np.int64), dtype=torch.int64, device=device)[owner]
    pairs = torch.unique(torch.stack([blk, torch.cat(frag_vox_list)], 1), dim=0)
    return pairs[:, 0], pairs[:, 1]


def block_features(block_labels, num_blocks, frag_feat, frag_n, device='cpu'):
    """(B, C) point-weighted mean of the members' member-mean features, L2-normalised."""
    f = torch.as_tensor(np.asarray(frag_feat, np.float32), device=device)
    w = torch.as_tensor(np.asarray(frag_n, np.float32), device=device)
    blk = torch.as_tensor(np.asarray(block_labels, np.int64), dtype=torch.int64, device=device)
    acc = torch.zeros(int(num_blocks), f.shape[1], device=device).index_add_(0, blk, f * w[:, None])
    tot = torch.zeros(int(num_blocks), device=device).index_add_(0, blk, w)
    out = acc / tot.clamp_min(1)[:, None]
    return out / out.norm(dim=1, keepdim=True).clamp_min(1e-6)


def object_candidates(node_prev, n_own, n_vox, node_feat, node_n, device='cpu'):
    """The distinct objects the graph's past nodes belong to, as cell sets plus one feature each.

    ``node_prev[i] >= 0`` marks a node that already belongs to an object (a window node or a frozen
    object); current fragments carry -1 and are skipped. Returns ``(cand_ids, c_own, c_vox, c_feat)``
    with ``cand_ids`` sorted, so a column index maps straight back to an object id.
    """
    prev = np.asarray(node_prev, np.int64)
    cand_ids = np.unique(prev[prev >= 0])
    c_idx = np.searchsorted(cand_ids, prev) if len(cand_ids) else np.zeros(len(prev), np.int64)
    if len(cand_ids) == 0:
        return cand_ids, torch.zeros(0, dtype=torch.int64, device=device), \
            torch.zeros(0, dtype=torch.int64, device=device), torch.zeros(0, 0, device=device)
    has = torch.as_tensor(prev >= 0, device=device)
    keep = has[n_own]                                     # per (node, cell) record
    owner = torch.as_tensor(c_idx, dtype=torch.int64, device=device)[n_own[keep]]
    pairs = torch.unique(torch.stack([owner, n_vox[keep]], 1), dim=0)
    nf = torch.as_tensor(np.asarray(node_feat, np.float32), device=device)
    nw = torch.as_tensor(np.asarray(node_n, np.float32), device=device)
    ci = torch.as_tensor(c_idx, dtype=torch.int64, device=device)[has]
    w = nw[has]
    acc = torch.zeros(len(cand_ids), nf.shape[1], device=device).index_add_(0, ci, nf[has] * w[:, None])
    tot = torch.zeros(len(cand_ids), device=device).index_add_(0, ci, w)
    feat = acc / tot.clamp_min(1)[:, None]
    feat = feat / feat.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return cand_ids, pairs[:, 0], pairs[:, 1], feat


def block_object_scores(b_own, b_vox, b_feat, c_own, c_vox, c_feat, rule='p2a'):
    """(B, K) score = fraction of the block's cells inside the object (x clamp(cos,0,1) for ``p2b``).

    ``p2a`` is the geometry-only rule, ``p2b`` gates it on the feature cosine, mirroring the two
    cross-frame rules of stage 0c (``streaming/window_scores.py``) but computed on the block as a
    whole instead of per fragment.
    """
    if rule not in RULES:
        raise ValueError(f'unknown rule {rule!r}')
    num_b = int(b_feat.shape[0]) if b_feat.ndim == 2 else 0
    num_c = int(c_feat.shape[0]) if c_feat.ndim == 2 else 0
    if num_b == 0 or num_c == 0:
        return torch.zeros((num_b, num_c), device=b_feat.device if b_feat.ndim == 2 else 'cpu')
    inter, size_b, _ = overlap_matrix(b_own, b_vox, c_own, c_vox, num_b, num_c)
    score = inter / size_b[:, None].clamp_min(1)
    if rule == 'p2b':
        score = score * (b_feat @ c_feat.T).clamp(0.0, 1.0)
    return score


def assign_blocks(score, cand_ids, block_n, cross_threshold, admit_min_points, new_object):
    """(B,) object id per block: the best-scoring object above threshold, else a new one or -1.

    ``new_object`` is a zero-argument callable allocating one fresh object id, called once per block
    that matched nothing and holds at least ``admit_min_points`` points. Several blocks may pick the
    same object; blocks below the point floor stay unassigned (-1).
    """
    num_b = int(score.shape[0])
    out = np.full(num_b, -1, np.int64)
    if num_b == 0:
        return out
    if score.shape[1] == 0:
        accepted = np.zeros(num_b, bool)
    else:
        best_score, best = score.max(1)
        accepted = (best_score >= float(cross_threshold)).cpu().numpy()
        out[accepted] = np.asarray(cand_ids, np.int64)[best.cpu().numpy()[accepted]]
    for b in np.flatnonzero(~accepted):
        if float(block_n[b]) >= float(admit_min_points):
            out[b] = new_object()
    return out
