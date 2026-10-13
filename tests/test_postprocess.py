"""CPU tests for the stage-0b output post-processing (ranking / NMS / decomposition)."""
import numpy as np

from freesam3d.evaluation.instance import GTSet
from freesam3d.streaming.postprocess import (best_iou, classify_masks, filter_impure, filter_single,
                                             greedy_nms, iou_pair_matrix, rank_scores)


def naive_nms(masks, scores, thr):
    order = np.argsort(-np.asarray(scores), kind='stable')
    keep = []
    for i in order:
        ok = True
        for j in keep:
            inter = np.logical_and(masks[i], masks[j]).sum()
            union = np.logical_or(masks[i], masks[j]).sum()
            if union and inter / union > thr:
                ok = False
                break
        if ok:
            keep.append(int(i))
    return np.asarray(keep)


def test_iou_pair_matrix_matches_loop():
    rng = np.random.default_rng(1)
    masks = rng.random((8, 50)) < 0.3
    m = iou_pair_matrix(masks)
    for i in range(len(masks)):
        for j in range(len(masks)):
            inter = np.logical_and(masks[i], masks[j]).sum()
            union = np.logical_or(masks[i], masks[j]).sum()
            assert abs(m[i, j] - (inter / union if union else 0.0)) < 1e-5


def test_greedy_nms_matches_naive():
    rng = np.random.default_rng(0)
    for density, thr in ((0.25, 0.5), (0.5, 0.3)):
        masks = rng.random((15, 40)) < density
        scores = rng.random(15).astype(np.float32)
        assert np.array_equal(greedy_nms(masks, scores, thr), naive_nms(masks, scores, thr))
    assert len(greedy_nms(np.zeros((0, 5), bool), np.zeros(0))) == 0


def test_classify_masks_handmade():
    labels = np.zeros(40, np.int64)
    labels[:20] = 1
    labels[20:] = 2
    gts = GTSet.from_labels(labels)
    masks = np.zeros((6, 40), bool)
    masks[0, 0:20] = True     # IoU 1.0 with gt1 -> correct, gt1 consumed
    masks[1, 0:15] = True     # IoU .75 with the consumed gt1 -> duplicate
    masks[2, 20:32] = True    # IoU .6 with gt2 -> correct
    masks[3, 10:30] = True    # IoU 1/3 with both -> partial
    masks[4, 30:38] = True    # IoU .4 with gt2 -> partial
    masks[5, 24:27] = True    # IoU .15 -> junk
    scores = np.array([0.9, 0.8, 0.7, 0.65, 0.6, 0.5], np.float32)
    assert classify_masks(masks, scores, gts).tolist() == [0, 1, 0, 2, 2, 3]


def test_rank_scores_and_filters():
    masks = np.zeros((3, 10), bool)
    masks[0, :5] = True
    masks[1, :2] = True
    masks[2, :10] = True
    s = rank_scores('nobs', masks=masks, n_obs=np.array([2, 2, 5], np.float32))
    assert int(s.argmax()) == 2 and s[0] > s[1]          # tie on n_obs -> larger mask first
    assert rank_scores('size', masks=masks).tolist() == [5, 2, 10]
    assert np.allclose(rank_scores('conf', conf=[.1, .9, .5]), [.1, .9, .5])
    assert np.allclose(rank_scores('oracle', iou_gt=[.2, .8, .4]), [.2, .8, .4])
    keep = np.arange(3)
    assert filter_impure(keep, [0.6, 0.5, 0.1]).tolist() == [1, 2]   # drop strictly-above-thr only
    assert filter_single(keep, [1, 2, 1]).tolist() == [1]
    gts = GTSet.from_labels(np.r_[np.ones(5), 2 * np.ones(5)].astype(np.int64))
    assert np.allclose(best_iou(masks, gts), [1.0, 0.4, 0.5])
