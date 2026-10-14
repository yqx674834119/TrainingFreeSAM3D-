"""CPU unit tests for the phase-4b 3D geometry primitives (no GPU, no FastSAM weights)."""
import numpy as np
import pytest
import torch

from freesam3d.models.instance_head3d import (
    DESC_DIM, DESC_NAMES, LOCAL_GEOM_NAMES, descriptor_stats, fragment_descriptors,
    neighbor_pairs, neighbor_pairs_brute, point_local_geometry, point_normals,
    standardize_descriptors,
)


def _grid(spacing, n=6):
    g = np.arange(n) * spacing
    return np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3).astype(np.float32)


def test_neighbor_pairs_matches_brute_force_on_a_grid():
    xyz = _grid(0.05, 6)
    radius = 0.075                       # 0.05 and 0.0707 inside, 0.0866 outside: no boundary ties
    got = neighbor_pairs(xyz, radius).cpu().numpy()
    ref = neighbor_pairs_brute(xyz, radius)
    key = lambda a: sorted(map(tuple, a.tolist()))          # noqa: E731
    assert key(got) == key(ref) and len(ref) > 0


def test_neighbor_pairs_is_symmetric_and_excludes_self():
    xyz = np.random.default_rng(0).uniform(0, 0.3, (200, 3)).astype(np.float32)
    p = neighbor_pairs(xyz, 0.05).cpu().numpy()
    assert (p[:, 0] != p[:, 1]).all()
    fwd = {tuple(x) for x in p.tolist()}
    assert all((b, a) in fwd for a, b in fwd)               # both orientations present


def test_neighbor_pairs_requires_voxel_at_least_radius():
    with pytest.raises(ValueError):
        neighbor_pairs(np.zeros((3, 3), np.float32), 0.1, voxel=0.05)


def test_neighbor_pairs_empty_input():
    p = neighbor_pairs(np.zeros((0, 3), np.float32), 0.05)
    assert p.shape == (0, 2) and p.dtype == torch.int64


def test_point_normals_are_perpendicular_to_a_plane():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 1, (400, 2))
    xyz = np.c_[xy, np.zeros(400)].astype(np.float32)       # all on z = 0
    n = point_normals(xyz, radius=0.2, k=16).cpu().numpy()
    assert np.allclose(np.abs(n[:, 2]), 1.0, atol=1e-4)     # sign is arbitrary -> |n_z| ~ 1
    assert np.allclose(n[:, :2], 0.0, atol=1e-4)


def test_point_local_geometry_reads_a_plane_and_a_line():
    # a dense regular grid plane: no sampling noise, so planarity reads cleanly (~0.99)
    sp = 0.005
    ax = np.arange(0.0, 1.0 + sp, sp, dtype=np.float32)
    gx, gy = np.meshgrid(ax, ax, indexing='ij')
    plane = np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size, np.float32)]
    g = point_local_geometry(plane, radius=0.1, k=64).cpu().numpy()
    # interior points only: near the edge the neighbourhood is a half-disc and the local PCA is
    # anisotropic for a reason unrelated to the surface being a plane
    inb = (plane[:, 0] > 0.05) & (plane[:, 0] < 0.95) & (plane[:, 1] > 0.05) & (plane[:, 1] < 0.95)
    assert g[inb, 2].mean() < 1e-5                          # all on z = 0 -> scatter exactly 0
    assert g[inb, 1].mean() > 0.95                          # planarity dominates on a plane
    assert g[inb, 1].mean() > g[inb, 0].mean()              # ... and beats linearity
    t = np.linspace(0, 1, 3000, dtype=np.float32)
    line = np.stack([t, np.zeros_like(t), np.zeros_like(t)], 1)
    gl = point_local_geometry(line, radius=0.2, k=64).cpu().numpy()
    assert gl[:, 0].mean() > 0.99                           # collinear -> linearity exactly 1
    assert gl[:, 2].mean() < 1e-4


def test_fragment_descriptors_shapes_box_and_slab():
    rng = np.random.default_rng(3)
    cube = rng.uniform(0, 1, (600, 3))
    slab = np.c_[rng.uniform(0, 1, (600, 2)), rng.uniform(0, 0.02, 600)]
    xyz = np.r_[cube, slab + np.array([3.0, 0.0, 0.0])].astype(np.float32)
    groups = np.r_[np.zeros(600, np.int64), np.ones(600, np.int64)]
    d = fragment_descriptors(xyz, groups, 2).cpu().numpy()
    assert d.shape == (2, DESC_DIM) and len(DESC_NAMES) == DESC_DIM
    log_n, ext, lin, plan, scat = d[:, 0], d[:, 1:4], d[:, 5], d[:, 6], d[:, 7]
    assert np.allclose(log_n, np.log1p(600))
    assert ext[0].min() > 0.8 and ext[0].max() < 1.2        # cube spans ~1 in every axis
    assert scat[0] > 0.5 and plan[0] < 0.3                  # isotropic -> scatter high, planarity low
    assert plan[1] > 0.9 and scat[1] < 0.1                  # slab -> planarity high, scatter low
    assert lin[0] < 0.3


def test_fragment_descriptors_use_the_frame_fifth_percentile_as_ground():
    z = np.linspace(0, 10, 1000, dtype=np.float32)
    xyz = np.c_[np.zeros(1000, np.float32), np.zeros(1000, np.float32), z]
    d = fragment_descriptors(xyz, np.zeros(1000, np.int64), 1).cpu().numpy()
    assert d[0, 11] == pytest.approx(z.mean() - np.quantile(z, 0.05), rel=1e-5)   # height_mean


def test_fragment_descriptors_use_normals_when_given():
    xyz = np.zeros((10, 3), np.float32)
    groups = np.zeros(10, np.int64)
    nrm = np.tile(np.array([[0.0, 0.0, 1.0]], np.float32), (10, 1))
    d = fragment_descriptors(xyz, groups, 1, normals=nrm).cpu().numpy()
    assert d[0, 13] == pytest.approx(1.0)                   # mean |n_z|
    assert d[0, 14] == pytest.approx(1.0)                   # coherence


def test_standardize_descriptors_zscore_then_unit_norm():
    scale = np.ones(DESC_DIM)
    scale[0], scale[1] = 5.0, 0.1
    f = (np.random.default_rng(4).normal(size=(50, DESC_DIM)) * scale + 3).astype(np.float32)
    mean, std = descriptor_stats([f])
    out = standardize_descriptors(f, mean, std)
    z = (f - mean) / std
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    assert np.allclose(out, z, atol=1e-5)                   # z-score on holdout stats, then L2
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert out.shape == f.shape and out.dtype == np.float32


def test_local_geom_names():
    assert len(LOCAL_GEOM_NAMES) == 4
