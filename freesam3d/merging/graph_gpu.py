"""GPU graph construction: the same 14 edge scalars as ``graph.build_graph`` with
no KD-tree and no Python loops, in the style of ESAM's OnlineMerge.

Everything is a dense (G, ·) or (G, G) tensor op; G ≈ 70 groups per frame, so
a G×G matrix is tiny. Steps:

1. Node statistics via one-hot scatter:  M (G,N) = one_hot(groups).T;
   centroid = M @ xyz / size, covariance from second moments, bbox via
   scatter_reduce(amin/amax), colour mean.
2. Adjacency via a *voxel* subsample (one point per voxel & group, as the CPU
   path) and ``torch.cdist`` on the subsample: a (S, S) distance matrix with
   S ≈ 5–10k is 25–100M floats — fine on a 5090, and a single kernel. Pairs
   closer than eps across different groups are reduced to (G, G) with
   scatter: min distance, pair count, mean |Δz|, mean / min boundary cosine.
3. Nesting: raw masks on a stride-4 grid, (M, P) @ (P, M).

Returns the same ``GroupGraph`` as the CPU builder so the model, labels,
clustering and evaluation are unchanged.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .graph import EDGE_DIM, GroupGraph


def _onehot(groups, G):
    return F.one_hot(groups, G).to(torch.float32)              # (N, G)


def _voxel_keys(xyz, groups, voxel):
    """Index of one representative point per (voxel, group)."""
    if voxel <= 0:
        return torch.arange(len(xyz), device=xyz.device)
    key = torch.floor(xyz / voxel).to(torch.int64)
    key = key - key.min(0, keepdim=True).values
    span = key.max(0).values + 1
    flat = ((key[:, 0] * span[1] + key[:, 1]) * span[2] + key[:, 2]) * int(groups.max() + 1) + groups
    _, inv = torch.unique(flat, return_inverse=True)
    first = torch.full((int(inv.max()) + 1,), len(xyz), dtype=torch.long, device=xyz.device)
    first.scatter_reduce_(0, inv, torch.arange(len(xyz), device=xyz.device), 'amin')
    return first


@torch.no_grad()
def build_graph_gpu(xyz, rgb, groups, group_feat, point_feat=None, scores=None, mask_index=None,
                    raw_masks=None, eps=0.05, voxel=0.02, boundary_k=32, feature_centering=True,
                    device='cuda', chunk=4096):
    dev = torch.device(device)

    def T(x, dt=torch.float32):
        return x.to(device=dev, dtype=dt) if torch.is_tensor(x) else torch.as_tensor(np.asarray(x), dtype=dt, device=dev)
    xyz = T(xyz)
    rgb = T(rgb) / 255.0
    groups = T(groups, torch.long)
    N, G = len(xyz), int(groups.max()) + 1
    M = _onehot(groups, G)                                     # (N, G)
    size = M.sum(0)                                            # (G,)
    inv = 1.0 / size.clamp_min(1)

    # ---- node statistics (all matmuls) -------------------------------------------------------
    cent = (M.T @ xyz) * inv[:, None]                          # (G, 3)
    col = (M.T @ rgb) * inv[:, None]
    d = xyz - cent[groups]
    cov = (M.T @ (d[:, :, None] * d[:, None, :]).reshape(N, 9)).reshape(G, 3, 3) * inv[:, None, None]
    eig, vec = torch.linalg.eigh(cov)                          # ascending
    eig = eig.clamp_min(0).sqrt()
    normals = vec[:, :, 0]
    lo = torch.full((G, 3), float('inf'), device=dev).scatter_reduce(0, groups[:, None].expand(-1, 3), xyz, 'amin')
    hi = torch.full((G, 3), float('-inf'), device=dev).scatter_reduce(0, groups[:, None].expand(-1, 3), xyz, 'amax')
    ext = (hi - lo).nan_to_num(0.0)
    height = cent[:, 2] - xyz[:, 2].median()
    geom = torch.cat([torch.log1p(size)[:, None], height[:, None], ext, eig.flip(1), col], 1)   # (G, 11)
    conf = torch.zeros(G, device=dev)
    if scores is not None and mask_index is not None:
        mi = torch.as_tensor(np.asarray(mask_index), dtype=torch.long, device=dev)
        has = mi >= 0
        conf[has] = torch.as_tensor(np.asarray(scores), dtype=torch.float32, device=dev)[mi[has]]

    feat = T(group_feat)
    if feature_centering:
        feat = feat - feat.mean(0, keepdim=True)
    feat = F.normalize(feat, dim=1)
    pf = None
    if point_feat is not None:
        pf = T(point_feat)
        if feature_centering:
            pf = pf - pf.mean(0, keepdim=True)
        pf = F.normalize(pf, dim=1)

    # ---- contacts on the voxel subsample: cdist in chunks -> (G, G) scatter reductions --------
    sub = _voxel_keys(xyz, groups, voxel)
    sx, sg = xyz[sub], groups[sub]
    S = len(sub)
    inf = torch.full((G, G), float('inf'), device=dev)
    min_d = inf.clone()
    cnt = torch.zeros((G, G), device=dev)
    sum_dz = torch.zeros((G, G), device=dev)
    sum_cos = torch.zeros((G, G), device=dev)
    min_cos = inf.clone()
    touched = torch.zeros((G, S), dtype=torch.bool, device=dev)   # group a touched by subpoint j
    spf = pf[sub] if pf is not None else None
    for s0 in range(0, S, chunk):
        sl = slice(s0, min(S, s0 + chunk))
        D = torch.cdist(sx[sl], sx)                             # (c, S)
        close = (D < eps) & (sg[sl][:, None] != sg[None, :])
        i, j = close.nonzero(as_tuple=True)                    # pair (sub i+s0, sub j)
        if len(i) == 0:
            continue
        ga, gb = sg[sl][i], sg[j]
        key = ga * G + gb
        dist = D[i, j]
        min_d.view(-1).scatter_reduce_(0, key, dist, 'amin')
        cnt.view(-1).index_add_(0, key, torch.ones_like(dist))
        sum_dz.view(-1).index_add_(0, key, (sx[sl][i, 2] - sx[j, 2]).abs())
        touched[ga, j] = True
        touched[gb, i + s0] = True
        if spf is not None:
            cs = (spf[sl][i] * spf[j]).sum(1)
            sum_cos.view(-1).index_add_(0, key, cs)
            min_cos.view(-1).scatter_reduce_(0, key, cs, 'amin')
    # symmetrise (pairs were counted in both orientations; keep upper triangle)
    adj = torch.isfinite(min_d)
    adj = adj | adj.T
    iu = torch.triu_indices(G, G, offset=1, device=dev)
    keep = adj[iu[0], iu[1]]
    a, b = iu[0][keep], iu[1][keep]
    E = len(a)
    ef = torch.zeros((E, EDGE_DIM), device=dev)
    if E:
        ab = (a, b)
        mind = torch.minimum(min_d[ab], min_d[b, a])
        n_ab = cnt[ab] + cnt[b, a]
        ef[:, 0] = mind
        # contact extent: distinct sub-points of either group involved in the contact, over the smaller group
        tcount = (touched[a] & (sg[None, :] == b[:, None])).sum(1) + (touched[b] & (sg[None, :] == a[:, None])).sum(1)
        ef[:, 1] = tcount.float() / torch.minimum(size[a], size[b]).clamp_min(1)
        ef[:, 2] = (cent[a] - cent[b]).norm(dim=1)
        ef[:, 3] = (cent[a, 2] - cent[b, 2]).abs()
        ef[:, 4] = torch.log((ext[a].max(1).values + 1e-3) / (ext[b].max(1).values + 1e-3)).abs()
        ef[:, 5] = torch.log(size[a] / size[b]).abs()
        ef[:, 6] = (col[a] - col[b]).norm(dim=1)
        ef[:, 7] = (conf[a] - conf[b]).abs()
        ef[:, 8] = 1 - (normals[a] * normals[b]).sum(1).abs()
        ef[:, 9] = (sum_dz[ab] + sum_dz[b, a]) / n_ab.clamp_min(1)
        ef[:, 10] = (feat[a] * feat[b]).sum(1)
        if spf is not None:
            ef[:, 11] = (sum_cos[ab] + sum_cos[b, a]) / n_ab.clamp_min(1)
            ef[:, 12] = torch.minimum(min_cos[ab], min_cos[b, a])
        else:
            ef[:, 11] = ef[:, 12] = ef[:, 10]
        if raw_masks is not None and mask_index is not None and len(raw_masks):
            rm = torch.as_tensor(np.asarray(raw_masks)[:, ::4, ::4].reshape(len(raw_masks), -1), dtype=torch.float32, device=dev)
            inter = rm @ rm.T
            area = rm.sum(1)
            nest = inter / torch.minimum(area[:, None], area[None, :]).clamp_min(1)
            mi = torch.as_tensor(np.asarray(mask_index), dtype=torch.long, device=dev)
            ok = (mi[a] >= 0) & (mi[b] >= 0)
            ef[ok, 13] = nest[mi[a][ok], mi[b][ok]]
    edges = torch.stack([a, b], 1)
    return GroupGraph(np.arange(G), edges.cpu().numpy().astype(np.int64), geom.cpu().numpy(), feat.cpu().numpy(),
                      ef.cpu().numpy(), size.cpu().numpy())
