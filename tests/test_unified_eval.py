"""Stage-2 mesh reporting rows (CPU): every ranking mode plus the oracle ceiling."""
import numpy as np

from freesam3d.evaluation.instance import GTSet
from freesam3d.streaming.evaluate import mesh_rows
from freesam3d.streaming.postprocess import best_iou


def _scene():
    labels = np.zeros(20, np.int64)
    labels[:10], labels[10:] = 1, 2
    gts = GTSet.from_labels(labels, [1, 2])
    masks = np.zeros((3, 20), bool)
    masks[0, :10] = True                       # exactly GT 1
    masks[1, 10:] = True                       # exactly GT 2
    masks[2, :] = True                         # both glued together
    return masks, np.array([0.9, 0.8, 0.5], np.float32), np.array([3, 2, 1]), gts


def test_mesh_rows_reports_every_ranking_and_the_oracle_ceiling():
    masks, scores, n_obs, gts = _scene()
    quality = np.array([0.1, 0.9, 0.2], np.float32)
    rows = mesh_rows(masks, masks, scores, n_obs, gts, 'cpu', min_points=5, quality=quality,
                     iou_gt=best_iou(masks, gts, 'cpu'))
    assert set(rows) == {'conf', 'nobs', 'quality', 'oracle'}
    for r in rows.values():
        assert set(r) >= {'ap50', 'ap25', 'iou', 'masks', 'dup_rate', 'overmerge', 'frags_gt'}
    assert rows['oracle']['ap50'] >= rows['quality']['ap50'] - 1e-9
    assert rows['oracle']['masks'] <= 3


def test_mesh_rows_filters_small_masks_and_handles_nothing_left():
    masks, scores, n_obs, gts = _scene()
    assert mesh_rows(masks, masks, scores, n_obs, gts, 'cpu', min_points=50) == {}
    rows = mesh_rows(masks, masks, scores, n_obs, gts, 'cpu', min_points=5, quality=np.ones(3, np.float32))
    assert set(rows) == {'conf', 'nobs', 'quality'}          # no oracle without GT IoU
    assert rows['nobs']['masks'] >= 1
