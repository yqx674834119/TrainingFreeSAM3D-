"""阶段 5 · P3（CPU）：拆分池、oracle recall、池 AP50 与体素键渲染。"""
import numpy as np

from freesam3d.analysis.bounds import (iou_matrix_np, pieces_pool, pool_ap50, recall_at_50,
                                       render_keys, share_matrix)


def _scene():
    V = 260
    gt = np.zeros(V, np.int64)
    for k in range(4):
        gt[k * 50:(k + 1) * 50] = k + 1
    return gt, np.arange(1, 5, dtype=np.int32)


def test_shares_and_pieces():
    gt, keep = _scene()
    masks = np.zeros((3, 260), bool)
    masks[0, 0:50] = True
    masks[1, 0:30] = True
    masks[1, 50:70] = True          # 60% gt1 + 40% gt2
    masks[2, 100:150] = True
    sh = share_matrix(masks, gt, keep)
    assert abs(sh[0, 0] - 1.0) < 1e-6 and abs(sh[1, 0] - 0.6) < 1e-6 and abs(sh[1, 1] - 0.4) < 1e-6
    p = pieces_pool(masks, gt, keep, 0.2)
    assert p.shape[0] == 4           # 两个纯候选 + 混合候选切成 2 块
    assert (iou_matrix_np(p, gt, keep).max(1) >= 0.3).all()


def test_recall_at_50():
    iou = np.array([[0.9, 0.0], [0.55, 0.0], [0.0, 0.8]], np.float32)
    assert recall_at_50(iou) == (2, 2)
    assert recall_at_50(np.zeros((0, 0), np.float32)) == (0, 0)


def test_pool_ap50_gt_render_is_one_and_dedup():
    gt, keep = _scene()
    g = np.stack([gt == k for k in keep])
    ap, n = pool_ap50(g, gt, keep)
    assert abs(ap - 1.0) < 1e-6 and n == 4
    nested = np.zeros((2, 260), bool)
    nested[0, 0:50] = True
    nested[1, 0:45] = True           # 45/50 = 0.9 被包含 → 去重留 1
    _, n_d = pool_ap50(nested, gt, keep, dedup=0.7)
    assert n_d == 1


def test_split_improves_missed_mixed_mask():
    gt, keep = _scene()
    mixed = np.zeros((1, 260), bool)
    mixed[0, 0:36] = True
    mixed[0, 50:74] = True           # 60% gt1 + 40% gt2，与 gt1 的 IoU = 36/74 < 0.5
    ap_m, _ = pool_ap50(mixed, gt, keep)
    p = pieces_pool(mixed, gt, keep, 0.2)
    ap_p, _ = pool_ap50(p, gt, keep)
    assert ap_m == 0.0 and ap_p > ap_m


def test_render_keys():
    cells = np.array([1, 5, 9], np.int64)
    m = render_keys([np.array([5, 9, 100], np.int64)], cells)
    assert m.shape == (1, 3) and m[0].tolist() == [False, True, True]


def test_cos_keep():
    from freesam3d.analysis.bounds import cos_keep
    pf = np.array([[1, 0], [0.6, 0.8], [0, 1], [-1, 0]], np.float32)
    m = np.array([1, 0], np.float32)
    assert cos_keep(pf, m, 0.5).tolist() == [True, True, False, False]
    assert cos_keep(pf, m, -1).all()
