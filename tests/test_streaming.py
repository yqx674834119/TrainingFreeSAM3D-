import numpy as np
import pytest
import torch

from freesam3d.streaming.voxel import (containment_from_counts, devoxelize, iou_from_counts, mask_voxel_sets,
                                       overlap_matrix, voxelize)

HAS_CUDA = torch.cuda.is_available()


def test_voxel_hash_roundtrip_and_negative_coords():
    xyz = np.array([[0.01, 0.02, 0.03], [-1.23, 4.56, -7.89], [100.0, -50.0, 0.0]], np.float32)
    ids = voxelize(xyz, 0.05)
    back = devoxelize(ids, 0.05)
    assert np.all(np.abs(back - xyz) <= 0.05)                  # within one voxel
    assert len(np.unique(ids)) == 3
    assert np.array_equal(voxelize(torch.as_tensor(xyz), 0.05).numpy(), ids)


def _toy():
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 1, (500, 3)).astype(np.float32)
    vox = voxelize(xyz, 0.1)
    a = np.zeros((2, 500), bool); a[0, :250] = True; a[1, 250:] = True
    b = np.zeros((3, 500), bool); b[0, :125] = True; b[1, 125:375] = True; b[2, 375:] = True
    return xyz, vox, a, b


def test_overlap_matrix_matches_python_sets():
    xyz, vox, a, b = _toy()
    oa, va = mask_voxel_sets(torch.as_tensor(a), vox)
    ob, vb = mask_voxel_sets(torch.as_tensor(b), vox)
    inter, sa, sb = overlap_matrix(oa, va, ob, vb, 2, 3)
    ref_sets_a = [set(vox[m].tolist()) for m in a]
    ref_sets_b = [set(vox[m].tolist()) for m in b]
    for i in range(2):
        assert sa[i] == len(ref_sets_a[i])
        for j in range(3):
            assert inter[i, j] == len(ref_sets_a[i] & ref_sets_b[j])
    iou = iou_from_counts(inter, sa, sb)
    ref = [[len(ref_sets_a[i] & ref_sets_b[j]) / len(ref_sets_a[i] | ref_sets_b[j]) for j in range(3)] for i in range(2)]
    np.testing.assert_allclose(iou.numpy(), ref, atol=1e-6)
    ca, cb = containment_from_counts(inter, sa, sb)
    assert torch.all(ca <= 1 + 1e-6) and torch.all(cb <= 1 + 1e-6)
    assert iou[0, 0] > 0
    # genuinely disjoint regions -> exactly 0
    far = xyz.copy(); far[250:] += 10.0
    vf = voxelize(far, 0.1)
    oa2, va2 = mask_voxel_sets(torch.as_tensor(a), vf)
    ob2, vb2 = mask_voxel_sets(torch.as_tensor(b), vf)
    inter2, sa2, sb2 = overlap_matrix(oa2, va2, ob2, vb2, 2, 3)
    assert inter2[1, 0] == 0 and inter2[0, 2] == 0


def test_overlap_matrix_gpu_matches_cpu():
    if not HAS_CUDA:
        pytest.skip('no CUDA')
    xyz, vox, a, b = _toy()
    oa, va = mask_voxel_sets(torch.as_tensor(a), vox)
    ob, vb = mask_voxel_sets(torch.as_tensor(b), vox)
    c = overlap_matrix(oa, va, ob, vb, 2, 3)[0]
    g = overlap_matrix(oa.cuda(), va.cuda(), ob.cuda(), vb.cuda(), 2, 3)[0].cpu()
    assert torch.equal(c, g)


def _synthetic_scene(rng, n_obj=3, frames=5, pts=300, dev='cpu'):
    """3 cubes 1 m apart; each frame samples random points from each (plus the floor) and returns a FrameInput."""
    from freesam3d.streaming.memory import FrameInput
    centres = np.array([[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0]], np.float32)[:n_obj]
    out = []
    for f in range(frames):
        xyz, masks, gt = [], [], []
        for k, c in enumerate(centres):
            p = c + rng.uniform(-0.2, 0.2, (pts, 3)).astype(np.float32)
            xyz.append(p); gt.append(np.full(pts, k + 1))
        xyz = np.concatenate(xyz); gt = np.concatenate(gt)
        masks = np.stack([gt == k + 1 for k in range(n_obj)])
        feat = np.eye(n_obj, 640, dtype=np.float32)[np.arange(n_obj)] + 0.01 * rng.normal(size=(n_obj, 640)).astype(np.float32)
        feat /= np.linalg.norm(feat, axis=1, keepdims=True)
        out.append((FrameInput(xyz, masks, np.full(n_obj, 0.9, np.float32), feat, f), gt))
    return out


def test_memory_recovers_objects_and_is_idempotent():
    from freesam3d.streaming.memory import MemoryConfig, QueryMemory
    rng = np.random.default_rng(0)
    seq = _synthetic_scene(rng)
    mem = QueryMemory(MemoryConfig(voxel=0.05, admit_support=0.0), device='cpu')
    for fr, gt in seq:
        matched = mem.update(fr)
    assert mem.num_queries == 3                                   # one query per object, no duplicates
    assert (matched >= 0).all()                                   # last frame fully matched
    # feeding the last frame again must not create queries or voxels
    q, r = mem.num_queries, mem.num_records
    mem.update(seq[-1][0])
    assert mem.num_queries == q and mem.num_records == r
    # render on the last frame's points: each point belongs to the query of its object
    masks, scores = mem.render(seq[-1][0].xyz)
    gt = seq[-1][1]
    for k in range(3):
        best = masks[:, gt == k + 1].mean(1).argmax()
        assert masks[best, gt == k + 1].mean() > 0.95 and masks[best, gt != k + 1].mean() < 0.05
    assert all(s['records'] <= 3 * 500 for s in mem.stats)        # bounded by scene voxels, not frames


def test_memory_admission_gate_rejects_floating_junk():
    from freesam3d.streaming.memory import FrameInput, MemoryConfig, QueryMemory
    rng = np.random.default_rng(1)
    seq = _synthetic_scene(rng, frames=2)
    mem = QueryMemory(MemoryConfig(voxel=0.05, admit_support=0.8, support_radius_voxels=2), device='cpu')
    mem.update(seq[0][0])
    # a proposal floating 5 m away from everything seen so far must be rejected
    junk = rng.uniform(4.8, 5.2, (200, 3)).astype(np.float32)
    fr = FrameInput(junk, np.ones((1, 200), bool), np.array([0.9], np.float32), np.ones((1, 640), np.float32) / 640 ** 0.5, 1)
    mem.update(fr)
    assert mem.num_queries == 3


def test_feature_gated_touch_merge_joins_fragments_of_one_object():
    from freesam3d.streaming.memory import FrameInput, MemoryConfig, QueryMemory
    rng = np.random.default_rng(2)
    p = rng.uniform(-0.2, 0.2, (600, 3)).astype(np.float32)
    masks = np.stack([p[:, 0] < 0, p[:, 0] >= 0])                 # two halves of one cube, touching at x = 0
    same = np.tile(np.eye(1, 640, dtype=np.float32), (2, 1))      # identical appearance
    diff = np.eye(2, 640, dtype=np.float32)                       # orthogonal appearance
    s = np.array([0.9, 0.9], np.float32)
    mem = QueryMemory(MemoryConfig(voxel=0.05, admit_support=0.0), device='cpu')
    mem.update(FrameInput(p, masks, s, same, 0))
    assert mem.num_queries == 2                                   # geometry alone (IoU ~ 0) keeps them apart
    mem = QueryMemory(MemoryConfig(voxel=0.05, admit_support=0.0, merge_touch_cos=0.9, merge_dilation=1), device='cpu')
    mem.update(FrameInput(p, masks, s, same, 0))
    assert mem.num_queries == 1                                   # touching + same appearance -> one query
    mem = QueryMemory(MemoryConfig(voxel=0.05, admit_support=0.0, merge_touch_cos=0.9, merge_dilation=1), device='cpu')
    mem.update(FrameInput(p, masks, s, diff, 0))
    assert mem.num_queries == 2                                   # touching but different appearance -> stay apart


def test_segment_smooth_is_per_segment_majority():
    from freesam3d.streaming.evaluate import segment_smooth
    seg = np.array([7, 7, 7, 3, 3, 3, 9, 9])                      # non-compact ids, as in ScanNet super_points
    masks = np.array([[1, 1, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 1, 1]], bool)
    ref = np.array([[1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1]], bool)   # 2/3 -> whole segment, 1/3 -> none
    assert np.array_equal(segment_smooth(masks, seg), ref)
    assert segment_smooth(np.zeros((0, 8), bool), seg).shape == (0, 8)


def test_vis_pred_colors_follow_the_matched_gt():
    from freesam3d.evaluation.instance import GTSet
    from freesam3d.vis.streaming import BACKGROUND, gt_panel, paint_masks, pred_colors
    gts = GTSet.from_labels(np.array([1, 1, 1, 1, 2, 2, 2, 2, 0, 0]))
    gt_col, img = gt_panel(gts)
    assert img.shape == (10, 3) and np.allclose(img[8], BACKGROUND)            # unlabeled -> background grey
    masks = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],                           # GT 1, IoU 0.75
                      [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],                           # GT 2, IoU 1
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]], bool)                    # only unlabeled points -> no match
    col, ok = pred_colors(masks, gts, gt_col)
    assert ok.tolist() == [True, True, False]
    assert np.allclose(col[0], gt_col[0]) and np.allclose(col[1], gt_col[1])
    painted = paint_masks(masks, np.array([0.9, 0.5, 0.1]), col, 10)
    assert np.allclose(painted[0], gt_col[0]) and np.allclose(painted[4], gt_col[1]) and np.allclose(painted[3], BACKGROUND)
