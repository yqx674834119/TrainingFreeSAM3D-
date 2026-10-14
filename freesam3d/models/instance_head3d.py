"""Phase-4b 3D geometry primitives: voxel-bucket neighbourhoods, point normals, fragment descriptors.

Everything works on the *cached* merge frames (``merge_cache/seq/*``): the 20k sampled points'
world coordinates and their compact fragment ids. No KD-tree, no sklearn, no per-edge Python loop
(AGENTS.md §2): neighbours come from a voxel-bucket lookup whose buckets are contiguous ranges of a
voxel-id-sorted array, and every per-point / per-fragment aggregate is a ``scatter``/``index_add_``.
The one exception is ``neighbor_pairs_brute``, a deliberately quadratic reference used only by the
unit tests (the same role ``cKDTree`` plays for ``merging/graph.py``).

The cache carries **no camera pose** (checked against the npz keys), so a single-frame PCA normal
has an undefined sign. Every consumer here is therefore sign-invariant: the descriptors use
``|n_z|`` and the resultant length of the unit normals, and the 3D head's point input uses ``|n_z|``
plus the three independent components of ``n ⊗ n`` (see ``instance_head3d``). This is the
"no pose -> sign-free form" branch the phase-4b brief asks to record in the summary.
"""
import time
from pathlib import Path

import numpy as np
import torch

from ..streaming.voxel import BITS, voxelize

# per-fragment shape descriptors (P3D.0): the columns are z-scored then L2-normalised at build time
DESC_NAMES = ('log_n', 'ext_x', 'ext_y', 'ext_z', 'log_diag', 'linearity', 'planarity', 'scatter',
              'sqrt_l1', 'sqrt_l2', 'sqrt_l3', 'height_mean', 'height_std', 'mean_abs_nz',
              'normal_coherence', 'log_density')
DESC_DIM = len(DESC_NAMES)

# per-point local geometry (3D head input, §三.1): linearity / planarity / scatter / local scale
LOCAL_GEOM_NAMES = ('lin', 'plan', 'scatter', 'scale')


def _as_tensor(x, dtype, device):
    """Accept numpy arrays or tensors alike; return a tensor of ``dtype`` on ``device``."""
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def _offset_shifts(device):
    """(27,) packed voxel-id offsets of the Chebyshev-1 neighbourhood (matches ``voxel.dilate_ids``)."""
    r = torch.arange(-1, 2, device=device)
    g = torch.stack(torch.meshgrid(r, r, r, indexing='ij'), -1).reshape(-1, 3)
    return (g[:, 0] << (2 * BITS)) + (g[:, 1] << BITS) + g[:, 2]


def neighbor_pairs(xyz, radius, voxel=None, device='cpu'):
    """(P,2) int64 pairs ``(i, j)``, ``i != j``, both orientations, with ``||xyz_i - xyz_j|| < radius``.

    Voxel buckets: ``voxel`` must be >= ``radius`` so that two points within ``radius`` always share
    a voxel or a face/edge/corner neighbour (the 27-cell Chebyshev neighbourhood). Each bucket is a
    contiguous range of the id-sorted array, so point ``i``'s candidates are the union of 27 ranges
    found by ``searchsorted`` — 27 vectorised gathers, no per-pair loop.
    """
    xyz = _as_tensor(xyz, torch.float32, device)
    voxel = float(radius if voxel is None else voxel)
    if voxel < float(radius) - 1e-9:
        raise ValueError(f'voxel ({voxel}) must be >= radius ({radius})')
    if len(xyz) == 0:
        return torch.zeros((0, 2), dtype=torch.int64, device=device)
    ids = voxelize(xyz, voxel)
    order = torch.argsort(ids)
    ids_sorted = ids[order]
    shift = _offset_shifts(device)
    n_off = len(shift)
    target = (ids[:, None] + shift[None, :]).reshape(-1)
    lo = torch.searchsorted(ids_sorted, target, right=False)
    hi = torch.searchsorted(ids_sorted, target, right=True)
    counts = (hi - lo).reshape(-1)
    has = counts > 0
    if not bool(has.any()):
        return torch.zeros((0, 2), dtype=torch.int64, device=device)
    flat = torch.nonzero(has).squeeze(1)
    point = torch.div(flat, n_off, rounding_mode='floor')      # source point of each non-empty bucket
    starts, lens = lo[flat], counts[flat]
    offs = torch.arange(int(lens.sum()), device=device) \
        - torch.repeat_interleave(torch.cumsum(lens, 0) - lens, lens)
    pos = torch.repeat_interleave(starts, lens) + offs
    src = torch.repeat_interleave(point, lens)
    dst = order[pos]
    keep = src != dst
    src, dst = src[keep], dst[keep]
    d = (xyz[src] - xyz[dst]).norm(dim=1)
    keep = d < float(radius)
    return torch.stack([src[keep], dst[keep]], 1)


def neighbor_pairs_brute(xyz, radius):
    """Quadratic reference for :func:`neighbor_pairs` (test only): all ordered pairs within ``radius``."""
    xyz = np.asarray(xyz, np.float64)
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    i, j = np.nonzero((d < float(radius)) & (d > 0))
    return np.stack([i, j], 1).astype(np.int64)


def _knn_within(xyz, radius, k, voxel=None, device='cpu'):
    """(N, k) int64 indices of each point's ``k`` nearest neighbours within ``radius``; -1 = none.

    Neighbours are ordered by distance (ties by index). Two stable sorts give the (src, dist)
    lexicographic order without a ``lexsort`` (torch has none).
    """
    xyz = _as_tensor(xyz, torch.float32, device)
    n = len(xyz)
    k = int(k)
    out = torch.full((n, k), -1, dtype=torch.int64, device=device)
    pairs = neighbor_pairs(xyz, radius, voxel, device)
    if len(pairs) == 0 or k == 0:
        return out
    src, dst = pairs[:, 0], pairs[:, 1]
    d = (xyz[src] - xyz[dst]).norm(dim=1)
    o = torch.argsort(d, stable=True)
    src, dst = src[o], dst[o]
    o = torch.argsort(src, stable=True)
    src, dst = src[o], dst[o]
    cnt = torch.bincount(src, minlength=n)
    starts = torch.cumsum(cnt, 0) - cnt
    rank = torch.arange(len(src), device=device) - torch.repeat_interleave(starts, cnt)
    sel = rank < k
    out[src[sel], rank[sel]] = dst[sel]
    return out


def _local_pca(xyz, radius, k, voxel=None, device='cpu'):
    """(N,3,3) local covariance of each point with its k nearest neighbours (the point included)."""
    xyz = _as_tensor(xyz, torch.float32, device)
    n = len(xyz)
    idx = _knn_within(xyz, radius, k, voxel, device)
    valid = idx >= 0
    nb = xyz[idx.clamp_min(0)]                                  # (N,k,3)
    nb = torch.where(valid[:, :, None], nb, xyz[:, None, :])     # pad with the point itself
    pts = torch.cat([xyz[:, None, :], nb], 1)                   # (N, k+1, 3)
    d = pts - pts.mean(1, keepdim=True)
    return (d[:, :, :, None] * d[:, :, None, :]).mean(1), pts


def point_normals(xyz, radius=0.08, k=16, voxel=None, device='cpu'):
    """(N,3) unit PCA normals: the smallest-variance direction of each point's local neighbourhood.

    Sign is arbitrary (the cache has no pose), so callers must use sign-invariant forms.
    """
    cov, _ = _local_pca(xyz, radius, k, voxel, device)
    _, vec = torch.linalg.eigh(cov)                             # ascending eigenvalues
    nrm = vec[:, :, 0]
    return nrm / nrm.norm(dim=1, keepdim=True).clamp_min(1e-9)


def point_local_geometry(xyz, radius=0.08, k=16, voxel=None, device='cpu'):
    """(N,4) per-point shape scalars: linearity, planarity, scatter, local scale (sqrt of l1).

    ``linearity = (s1-s2)/s1``, ``planarity = (s2-s3)/s1``, ``scatter = s3/s1`` with
    ``s1 >= s2 >= s3`` the sqrt eigenvalues of the local covariance; ``scale`` is ``s1``.
    """
    cov, _ = _local_pca(xyz, radius, k, voxel, device)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0).flip(1)       # descending l1 >= l2 >= l3
    s1, s2, s3 = eig[:, 0].sqrt(), eig[:, 1].sqrt(), eig[:, 2].sqrt()
    denom = s1.clamp_min(1e-9)
    return torch.stack([(s1 - s2) / denom, (s2 - s3) / denom, s3 / denom, s1], 1)


def fragment_descriptors(xyz, groups, num_groups, z_ref=None, normals=None, device='cpu'):
    """(G, DESC_DIM) per-fragment shape descriptors from the cached xyz + compact fragment ids.

    ``z_ref`` defaults to the frame's 5th percentile of z (this frame's "ground"); the two
    orientation columns use ``normals`` (N,3) when given, else stay 0. Pure one-hot matmul plus
    ``scatter_reduce``; no loop over fragments.
    """
    xyz = _as_tensor(xyz, torch.float32, device)
    g = _as_tensor(groups, torch.int64, device)
    G = int(num_groups)
    if z_ref is None:
        z_ref = float(torch.quantile(xyz[:, 2].float(), 0.05))
    m = torch.zeros((len(xyz), G), dtype=torch.float32, device=device)
    m[torch.arange(len(xyz), device=device), g] = 1
    n = m.sum(0)
    inv = 1.0 / n.clamp_min(1)
    cent = (m.T @ xyz) * inv[:, None]
    d = xyz - cent[g]
    cov = (m.T @ (d[:, :, None] * d[:, None, :]).reshape(len(xyz), 9)).reshape(G, 3, 3) * inv[:, None, None]
    eig = torch.linalg.eigvalsh(cov).clamp_min(0).flip(1)       # descending l1 >= l2 >= l3
    s1, s2, s3 = eig[:, 0].sqrt(), eig[:, 1].sqrt(), eig[:, 2].sqrt()
    denom = s1.clamp_min(1e-9)
    lo = torch.full((G, 3), float('inf'), device=device).scatter_reduce(
        0, g[:, None].expand(-1, 3), xyz, 'amin')
    hi = torch.full((G, 3), float('-inf'), device=device).scatter_reduce(
        0, g[:, None].expand(-1, 3), xyz, 'amax')
    ext = (hi - lo).nan_to_num(0.0).clamp_min(0)
    zmean = cent[:, 2] - z_ref
    zstd = (((m.T @ (xyz[:, 2] ** 2)) * inv) - cent[:, 2] ** 2).clamp_min(0).sqrt()
    if normals is not None:
        nrm = _as_tensor(normals, torch.float32, device)
        mean_abs_nz = (m.T @ nrm[:, 2].abs()) * inv
        coherence = (m.T @ nrm).norm(dim=1) * inv
    else:
        mean_abs_nz = torch.zeros(G, device=device)
        coherence = torch.zeros(G, device=device)
    density = torch.log1p(n / ext.prod(1).clamp_min(1e-6))
    return torch.stack([torch.log1p(n), ext[:, 0], ext[:, 1], ext[:, 2], torch.log1p(ext.norm(dim=1)),
                        (s1 - s2) / denom, (s2 - s3) / denom, s3 / denom,
                        s1, s2, s3, zmean, zstd, mean_abs_nz, coherence, density], 1)


def descriptor_stats(feats, eps=1e-6):
    """Per-column mean and std over a stack of (G, DESC_DIM) blocks (holdout scenes only)."""
    f = np.concatenate([np.asarray(x, np.float64) for x in feats], 0)
    return f.mean(0), f.std(0) + eps


def standardize_descriptors(feat, mean, std):
    """Per-column z-score (holdout stats) then L2-normalise — the P3D.0 cache 口径.

    Without the z-score the cosine is dominated by the large-magnitude columns (point count,
    extents), which would understate the geometry signal; the L2 keeps the probe's cosine reading
    directly comparable with the learned features.
    """
    f = (np.asarray(feat, np.float64) - np.asarray(mean, np.float64)) / np.asarray(std, np.float64)
    f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)
    return f.astype(np.float32)


def build_descriptor_cache(files, out_dir, log=print, overwrite=False, limit=0, device='cpu',
                           normal_radius=0.08, normal_k=16):
    """P3D.0: per-frame fragment geometry descriptors -> ``<out_dir>/<scene>_<fid>.npz {feat (G,16) fp16}``.

    Two passes: raw descriptors for every frame, then the per-column mean/std over *those rows*
    (only the holdout scenes are passed in, so the statistics are holdout-only), then z-score + L2
    and write. Resumable.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(files)[:limit or None]
    t0 = time.perf_counter()
    raws = []
    for f in files:
        with np.load(f) as z:
            xyz = np.asarray(z['xyz'], np.float32)
            groups = np.asarray(z['groups'], np.int64)
            num = int(len(z['sizes']))
        nrm = point_normals(xyz, normal_radius, normal_k, device=device)
        raws.append(fragment_descriptors(xyz, groups, num, normals=nrm, device=device).cpu().numpy())
    mean, std = descriptor_stats(raws) if raws else (np.zeros(DESC_DIM), np.ones(DESC_DIM))
    done = skipped = 0
    for f, raw in zip(files, raws):
        dst = out_dir / Path(f).name
        if dst.exists() and not overwrite:
            skipped += 1
            continue
        np.savez_compressed(dst, feat=standardize_descriptors(raw, mean, std).astype(np.float16))
        done += 1
    stats = dict(frames=done, skipped=skipped, seconds=time.perf_counter() - t0, desc_dim=DESC_DIM,
                 names=list(DESC_NAMES), mean=mean.tolist(), std=std.tolist(),
                 normal_radius=normal_radius, normal_k=normal_k,
                 disk_bytes=sum(q.stat().st_size for q in out_dir.glob('*.npz')))
    log(f'descriptor cache: {done} built, {skipped} skipped, {stats["seconds"] / 60:.1f} min')
    return stats
