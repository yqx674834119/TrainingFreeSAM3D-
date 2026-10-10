import numpy as np
import torch

from freesam3d.lifting.geometry import backproject, sample_indices
from freesam3d.lifting.grouping import compact_ids, exclusive_group_map, fill_uncovered
from freesam3d.lifting.pooling import center_mean_l2, mean_l2


def test_smaller_mask_wins_overlap_and_hidden_mask_vanishes():
    m = np.zeros((3, 4, 4), bool)
    m[0] = True              # big
    m[1, :2, :2] = True      # small, on top of big
    m[2, :2, :2] = True      # same area as m1: stable sort paints m2 after m1
    g = exclusive_group_map(m)
    assert g[3, 3] == 0      # big mask keeps the pixels nobody else claims
    assert g[0, 0] == 2
    assert 1 not in g        # fully hidden mask disappears


def test_uncovered_pixels_stay_minus_one():
    m = np.zeros((1, 3, 3), bool)
    m[0, 0, 0] = True
    g = exclusive_group_map(m)
    assert (g == -1).sum() == 8


def test_fill_uncovered_base_and_small_case():
    ids = np.array([0, 0, -1, -1, 3])
    out = fill_uncovered(ids, np.zeros((5, 3)), n_clusters=20, min_points=20, base=10)
    assert out.tolist() == [0, 0, 10, 10, 3]
    out = fill_uncovered(ids, np.zeros((5, 3)), n_clusters=20, min_points=20)   # ESAM: max+1
    assert out.tolist() == [0, 0, 4, 4, 3]


def test_fill_uncovered_kmeans_separates_clusters():
    x = np.concatenate([np.zeros((30, 3)), np.full((30, 3), 10.0)])
    out = fill_uncovered(np.full(60, -1), x, n_clusters=2, min_points=20, seed=0)
    assert len(set(out[:30])) == 1 and len(set(out[30:])) == 1 and out[0] != out[-1]


def test_compact_ids():
    c, orig = compact_ids(np.array([7, 3, 7, 50]))
    assert c.tolist() == [1, 0, 1, 2] and orig.tolist() == [3, 7, 50]


def test_backproject_identity_pose():
    depth = np.zeros((4, 6), np.uint16)
    depth[2, 3] = 2000
    k = np.array([[100.0, 0, 3], [0, 100.0, 2], [0, 0, 1]])
    xyz, px = backproject(depth, k, np.eye(4))
    assert px.tolist() == [[3, 2]]
    np.testing.assert_allclose(xyz, [[0, 0, 2]], atol=1e-6)
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    xyz, _ = backproject(depth, k, pose)
    np.testing.assert_allclose(xyz, [[1, 2, 5]], atol=1e-6)


def test_sampling_without_replacement_when_enough():
    idx = sample_indices(50000, 20000, seed=0)
    assert len(idx) == 20000 and len(np.unique(idx)) == 20000
    assert len(sample_indices(10, 20000)) == 20000


def test_mean_l2_pooling():
    f = torch.tensor([[1.0, 0], [3.0, 0], [0, 2.0]])
    g = mean_l2(f, np.array([0, 0, 1]), 2)
    np.testing.assert_allclose(g.numpy(), [[1, 0], [0, 1]], atol=1e-6)
    c = center_mean_l2(f, np.array([0, 0, 1]), 2)
    np.testing.assert_allclose(c.norm(dim=1).numpy(), [1, 1], atol=1e-6)
