import numpy as np

from freesam3d.evaluation.instance import (GTSet, class_agnostic_ap, coverage_stats, fragments_per_gt,
                                           group_inclusion, groups_to_masks, oracle_merge)


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    gt = np.repeat([1, 2, 3, 0], [300, 200, 100, 50])
    return gt[rng.permutation(len(gt))]


def test_gt_as_prediction_is_perfect():
    gt = _scene()
    masks, _ = groups_to_masks(np.where(gt > 0, gt, -1))
    gts = GTSet.from_labels(gt)
    for t in (0.25, 0.5, 0.95):
        assert class_agnostic_ap(masks, np.ones(len(masks)), gts, t) == 1.0
    cov = coverage_stats(masks, gts)
    assert cov['mean_best_iou'] == 1.0 and cov['recall@0.5'] == 1.0 and cov['precision@0.5'] == 1.0
    assert fragments_per_gt(masks, gts).tolist() == [1, 1, 1]


def test_corrupted_gt_drops_ap_more_at_high_iou():
    gt = _scene()
    rng = np.random.default_rng(1)
    noisy = gt.copy()
    for k in (1, 2, 3):
        idx = np.flatnonzero(gt == k)
        swap = rng.choice(idx, len(idx) // 3, replace=False)
        noisy[swap] = rng.choice([x for x in (1, 2, 3) if x != k], len(swap))
    masks, _ = groups_to_masks(np.where(noisy > 0, noisy, -1))
    gts = GTSet.from_labels(gt)
    ap = [class_agnostic_ap(masks, np.ones(len(masks)), gts, t) for t in (0.25, 0.5, 0.75)]
    assert ap[0] == 1.0 and ap[2] < ap[0]


def test_fragments_and_oracle_merge():
    gt = np.repeat([1, 2, 0], [100, 100, 20])
    groups = np.repeat([0, 1, 2, 3, 4], [50, 50, 100, 10, 10])   # GT1 split in two; group 3/4 unlabeled
    gts = GTSet.from_labels(gt)
    masks, _ = groups_to_masks(groups)
    assert fragments_per_gt(masks, gts).tolist() == [2, 1]
    merged = oracle_merge(groups, gt, 0.8)
    assert len(np.unique(merged)) == 4                      # 0+1 merged, 2 kept, 3/4 untouched
    assert len(np.unique(merged[:100])) == 1 and len(merged) == len(groups)
    assert np.array_equal(merged[200:] , groups[200:])
    mm, _ = groups_to_masks(merged)
    assert coverage_stats(mm, gts)['mean_best_iou'] == 1.0
    assert (mm.sum(0) == 1).all()                            # still exclusive, no point lost


def test_oracle_keeps_impure_groups():
    gt = np.repeat([1, 2], [60, 40])
    groups = np.zeros(100, int)                              # one group, purity 0.6
    assert np.array_equal(oracle_merge(groups, gt, 0.8), groups)


def test_group_inclusion():
    groups = np.repeat([0, 1], [10, 10])
    pred = np.zeros((1, 20), bool)
    pred[0, :15] = True
    np.testing.assert_allclose(group_inclusion(pred, groups), [[1.0, 0.5]])


def test_oracle_adjacent_keeps_disconnected_fragments_apart():
    from freesam3d.evaluation.instance import oracle_merge_adjacent
    x = np.concatenate([np.linspace(0, 1, 50), np.linspace(1.01, 2, 50), np.linspace(5, 6, 50)])
    xyz = np.stack([x, np.zeros_like(x), np.zeros_like(x)], 1)
    groups = np.repeat([0, 1, 2], 50)
    gt = np.ones(150, int)                       # one GT instance, three fragments; 2 is far away
    out = oracle_merge_adjacent(groups, gt, xyz, 0.8, eps=0.05)
    assert out[0] == out[50] and out[0] != out[100]
    assert len(np.unique(oracle_merge(groups, gt, 0.8))) == 1
