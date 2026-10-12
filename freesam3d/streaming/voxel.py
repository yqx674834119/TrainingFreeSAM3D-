"""Voxel hashing and sparse (P x K) voxel-overlap matrices — the geometric core of the query memory.

A voxel id packs the integer grid coordinate into one int64 (21 bits per axis, offset so that
negative coordinates work: |coord| < 2^20 voxels = 52 km at 5 cm). Proposals and memory
queries are both sets of voxel ids; every overlap statistic is a sparse join on those ids:
``torch.unique`` over the concatenated (voxel, owner) pairs, then ``index_add_`` into a
(P, K) matrix. No KD-tree, no per-pair Python loop (AGENTS.md §2).
"""
import numpy as np
import torch

BITS = 21
OFFSET = 1 << (BITS - 1)
MASK = (1 << BITS) - 1


def voxelize(xyz, voxel):
    """(N,3) float -> (N,) int64 voxel ids (numpy or torch, same type back)."""
    if torch.is_tensor(xyz):
        g = torch.floor(xyz / voxel).to(torch.int64) + OFFSET
        return (g[:, 0] << (2 * BITS)) | (g[:, 1] << BITS) | g[:, 2]
    g = np.floor(np.asarray(xyz, np.float64) / voxel).astype(np.int64) + OFFSET
    return (g[:, 0] << (2 * BITS)) | (g[:, 1] << BITS) | g[:, 2]


def devoxelize(ids, voxel):
    """int64 voxel ids -> (N,3) voxel-centre coordinates."""
    if torch.is_tensor(ids):
        g = torch.stack([(ids >> (2 * BITS)) & MASK, (ids >> BITS) & MASK, ids & MASK], 1) - OFFSET
        return (g.to(torch.float32) + 0.5) * voxel
    ids = np.asarray(ids, np.int64)
    g = np.stack([(ids >> (2 * BITS)) & MASK, (ids >> BITS) & MASK, ids & MASK], 1) - OFFSET
    return (g.astype(np.float32) + 0.5) * voxel


def mask_voxel_sets(masks, vox_ids):
    """(P,N) bool masks over points with voxel ids (N,) -> flat (owner, voxel) pairs, deduplicated per owner.

    Returns ``owner`` (M,) and ``voxel`` (M,) int64 tensors sorted by owner.
    """
    masks = torch.as_tensor(masks)
    vox_ids = torch.as_tensor(vox_ids, dtype=torch.int64, device=masks.device)
    p, n = torch.nonzero(masks, as_tuple=True)
    pairs = torch.unique(torch.stack([p, vox_ids[n]], 1), dim=0)      # dedupe (owner, voxel), sorted by owner
    return pairs[:, 0], pairs[:, 1]


def overlap_matrix(owner_a, vox_a, owner_b, vox_b, na, nb):
    """Sparse join of two (owner, voxel) sets -> dense (na, nb) intersection counts and the size vectors.

    Both inputs must be deduplicated per owner (``mask_voxel_sets`` / memory invariant).
    Matrix ops only: sort-merge via ``torch.unique`` on the union of voxel ids, then
    ``index_add_`` over matching (a, b) owner pairs.
    """
    dev = vox_a.device
    size_a = torch.bincount(owner_a, minlength=na).float()
    size_b = torch.bincount(owner_b, minlength=nb).float()
    inter = torch.zeros((na, nb), device=dev)
    if len(vox_a) == 0 or len(vox_b) == 0:
        return inter, size_a, size_b
    # map every voxel id to a compact index shared by both sides
    allv, inv = torch.unique(torch.cat([vox_a, vox_b]), return_inverse=True)
    ia, ib = inv[:len(vox_a)], inv[len(vox_a):]
    V = len(allv)
    # (V, na) and (V, nb) occupancy as sparse COO -> intersection = A^T B. With V ~ 1e4-1e5 and
    # na, nb <= 200 the dense product is cheap on GPU and avoids any Python loop.
    A = torch.zeros((V, na), device=dev)
    A[ia, owner_a] = 1
    B = torch.zeros((V, nb), device=dev)
    B[ib, owner_b] = 1
    inter = A.T @ B
    return inter, size_a, size_b


def iou_from_counts(inter, size_a, size_b):
    union = size_a[:, None] + size_b[None, :] - inter
    return inter / union.clamp_min(1)


def containment_from_counts(inter, size_a, size_b):
    """(frac of a inside b, frac of b inside a)."""
    return inter / size_a[:, None].clamp_min(1), inter / size_b[None, :].clamp_min(1)


def dilate_ids(vox, radius=1):
    """Voxel ids dilated by ``radius`` (Chebyshev) -> unique ids. Pure integer offset arithmetic on the packed id."""
    if radius <= 0:
        return torch.unique(vox)
    r = torch.arange(-radius, radius + 1, device=vox.device)
    offs = torch.stack(torch.meshgrid(r, r, r, indexing='ij'), -1).reshape(-1, 3)
    shift = (offs[:, 0] << (2 * BITS)) + (offs[:, 1] << BITS) + offs[:, 2]
    return torch.unique((vox[:, None] + shift[None]).reshape(-1))


def dilate_sets(owner, vox, radius=1):
    """Per-owner dilation of (owner, voxel) pairs; returns deduplicated (owner, voxel) sorted by owner."""
    if radius <= 0:
        return owner, vox
    r = torch.arange(-radius, radius + 1, device=vox.device)
    offs = torch.stack(torch.meshgrid(r, r, r, indexing='ij'), -1).reshape(-1, 3)
    shift = (offs[:, 0] << (2 * BITS)) + (offs[:, 1] << BITS) + offs[:, 2]
    o = owner[:, None].expand(-1, len(shift)).reshape(-1)
    v = (vox[:, None] + shift[None]).reshape(-1)
    pairs = torch.unique(torch.stack([o, v], 1), dim=0)
    return pairs[:, 0], pairs[:, 1]
