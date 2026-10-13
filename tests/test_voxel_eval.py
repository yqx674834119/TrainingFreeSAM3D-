"""Voxel proxy GT and voxel-level scene AP (CPU, offline: no GPU, no weights).

The stage-1 holdout scenes have no reconstructed mesh, so their scene-level number is this
proxy: 5 cm cells, GT = cross-frame majority vote of the per-frame instance PNG.
"""
import numpy as np
import torch

from freesam3d.streaming.evaluate import voxel_majority_gt, voxel_scene_eval
from freesam3d.streaming.voxel import voxelize
from freesam3d.streaming.window_memory import FragmentFrame


def _frame(xyz, gt, sem, fid=0):
    n = len(xyz)
    return FragmentFrame('s', fid, np.asarray(xyz, np.float32), np.zeros((n, 3), np.float32),
                         np.zeros(n, np.int64), np.zeros((1, 4), np.float32), np.zeros((0, 2), np.int64),
                         np.zeros((0, 14), np.float32), np.zeros(1, np.float32),
                         np.asarray(gt, np.int64), np.asarray(sem, np.int64))


def _cells(ids):
    """Point at the centre of each cell id along x (voxel = 1 m), so ids map 1:1 to cells."""
    return [[float(i) + 0.5, 0.5, 0.5] for i in ids]


def test_voxel_majority_gt_takes_the_majority_instance_and_semantic():
    f0 = _frame(_cells([5]), [7], [5])
    f1 = _frame(_cells([5, 5, 5]), [7, 9, 7], [5, 5, 3])
    cells, gt, sem = voxel_majority_gt([f0, f1], voxel=1.0)
    assert cells.tolist() == voxelize(np.asarray(_cells([5]), np.float32), 1.0).tolist()
    assert gt.tolist() == [7]                     # 7 twice, 9 once
    assert sem.tolist() == [5]                    # 5 twice, 3 once


def test_voxel_scene_eval_matches_the_voxel_gt_and_drops_stuff_and_tiny_instances():
    thing, stuff_id, tiny = _cells(range(0, 10)), _cells(range(20, 30)), _cells(range(40, 43))
    frames = [_frame(thing + stuff_id + tiny, [7] * 10 + [9] * 10 + [11] * 3,
                     [5] * 10 + [1] * 10 + [5] * 3),
              _frame(thing, [7] * 10, [5] * 10)]
    obj = [torch.as_tensor(voxelize(np.asarray(thing, np.float32), 1.0), dtype=torch.int64)]
    met = voxel_scene_eval(obj, np.asarray([0.9], np.float32), frames, stuff={1}, voxel=1.0,
                           min_gt_voxels=5, min_pred_voxels=5)
    assert met['gt_voxels'] == 1                  # 9 is stuff, 11 has only 3 cells
    assert met['voxels'] == 23
    assert abs(met['ap50'] - 1.0) < 1e-9 and abs(met['iou'] - 1.0) < 1e-9 and abs(met['r50'] - 1.0) < 1e-9


def test_voxel_scene_eval_penalises_a_mask_that_misses_the_gt():
    thing = _cells(range(0, 10))
    frames = [_frame(thing, [7] * 10, [5] * 10)]
    wrong = [torch.as_tensor(voxelize(np.asarray(_cells(range(50, 60)), np.float32), 1.0), dtype=torch.int64)]
    met = voxel_scene_eval(wrong, np.asarray([0.9], np.float32), frames, stuff={1}, voxel=1.0,
                           min_gt_voxels=5, min_pred_voxels=5)
    assert met['gt_voxels'] == 1 and met['ap50'] == 0.0 and met['iou'] == 0.0


def test_voxel_scene_eval_ignores_predictions_below_min_pred_voxels():
    thing = _cells(range(0, 10))
    frames = [_frame(thing, [7] * 10, [5] * 10)]
    obj = [torch.as_tensor(voxelize(np.asarray(thing, np.float32), 1.0), dtype=torch.int64)]
    met = voxel_scene_eval(obj, np.asarray([0.9], np.float32), frames, stuff={1}, voxel=1.0,
                           min_gt_voxels=5, min_pred_voxels=100)
    assert met['masks'] == 0 and met['ap50'] == 0.0
