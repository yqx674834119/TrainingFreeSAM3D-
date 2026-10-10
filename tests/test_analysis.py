import numpy as np

from freesam3d.analysis.similarity import majority_gt, pair_metrics


def test_majority_and_purity():
    groups = np.array([0, 0, 0, 1, 1])
    gt = np.array([5, 5, 2, 0, 0])
    major, purity, sizes = majority_gt(groups, gt, 2)
    assert major.tolist() == [5, 0]
    np.testing.assert_allclose(purity, [2 / 3, 1.0])
    assert sizes.tolist() == [3, 2]


def test_pair_metrics_perfect_separation():
    gt = np.array([1, 1, 2, 2, 0])
    sim = np.full((5, 5), 0.1)
    sim[:2, :2] = sim[2:4, 2:4] = 0.9
    m, vals, same = pair_metrics(sim, gt)
    assert m['n_masks'] == 4 and m['n_pairs'] == 6 and m['n_same_pairs'] == 2
    assert m['auc'] == 1.0 and m['nn_hit_rate'] == 1.0
    np.testing.assert_allclose(m['gap'], 0.8)
