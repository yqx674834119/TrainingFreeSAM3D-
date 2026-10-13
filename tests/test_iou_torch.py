"""The torch branch of the IoU primitives must reproduce the numpy reference exactly.

Exercised on CPU via ``torch.device('cpu')`` so the suite still runs without a GPU; the CUDA
path (``device='cuda'``) runs the identical code against a different target device.
"""
import numpy as np
import pytest

from freesam3d.evaluation.instance import (GTSet, association_stats, class_agnostic_ap,
                                           coverage_stats, fragments_per_gt, iou_matrix)
from freesam3d.streaming.postprocess import best_iou, greedy_nms, iou_pair_matrix

torch = pytest.importorskip('torch')
TCPU = torch.device('cpu')


def _fixture(seed=0, n=96, k=6, p=11):
    rng = np.random.default_rng(seed)
    gt = rng.integers(0, k + 1, n)              # 0 = unlabeled point
    gts = GTSet.from_labels(gt)
    pred = rng.random((p, n)) < 0.3             # includes all-empty masks
    return pred, gts, rng


def test_iou_matrix_torch_matches_numpy():
    pred, gts, _ = _fixture()
    np.testing.assert_allclose(iou_matrix(pred, gts, TCPU), iou_matrix(pred, gts), rtol=1e-6, atol=1e-7)


def test_coverage_stats_torch_matches_numpy():
    pred, gts, _ = _fixture()
    a, b = coverage_stats(pred, gts, device=TCPU), coverage_stats(pred, gts)
    assert a.keys() == b.keys()
    for k in b:
        if k == 'best_iou':
            np.testing.assert_allclose(a[k], b[k], rtol=1e-6, atol=1e-7)
        else:
            assert a[k] == pytest.approx(b[k], rel=1e-6, abs=1e-9)


def test_fragments_per_gt_and_association_stats_torch_match_numpy():
    pred, gts, _ = _fixture()
    np.testing.assert_array_equal(fragments_per_gt(pred, gts, device=TCPU), fragments_per_gt(pred, gts))
    a, b = association_stats(pred, gts, device=TCPU), association_stats(pred, gts)
    for k in b:
        assert a[k] == pytest.approx(b[k], rel=1e-6, abs=1e-9)


def test_class_agnostic_ap_and_best_iou_torch_match_numpy():
    pred, gts, rng = _fixture()
    scores = rng.random(len(pred))
    assert class_agnostic_ap(pred, scores, gts, .5, device=TCPU) == pytest.approx(
        class_agnostic_ap(pred, scores, gts, .5), rel=1e-6)
    np.testing.assert_allclose(best_iou(pred, gts, TCPU), best_iou(pred, gts), rtol=1e-6, atol=1e-7)


def test_iou_pair_matrix_torch_matches_numpy():
    _, _, rng = _fixture()
    masks = rng.random((13, 64)) < 0.2
    np.testing.assert_allclose(iou_pair_matrix(masks, TCPU), iou_pair_matrix(masks), rtol=1e-6, atol=1e-7)


def test_greedy_nms_torch_and_precomputed_matrix_match_numpy():
    _, _, rng = _fixture()
    masks = rng.random((40, 128)) < 0.15
    scores = rng.random(40)
    ref = greedy_nms(masks, scores, 0.3)
    np.testing.assert_array_equal(greedy_nms(masks, scores, 0.3, device=TCPU), ref)
    np.testing.assert_array_equal(greedy_nms(masks, scores, 0.3, iou=iou_pair_matrix(masks, TCPU)), ref)


def test_empty_inputs_stay_empty_on_the_torch_path():
    _, gts, _ = _fixture()
    assert iou_pair_matrix(np.zeros((0, 5), bool), TCPU).shape == (0, 0)
    assert greedy_nms(np.zeros((0, 5), bool), np.zeros(0), device=TCPU).size == 0
    assert best_iou(np.zeros((0, 5), bool), gts, TCPU).size == 0
