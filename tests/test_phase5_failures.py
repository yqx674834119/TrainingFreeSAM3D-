"""阶段 5 失败案例诊断的纯函数单测（CPU、无权重）。"""
import numpy as np

from freesam3d.vis import failures as fv


GT = np.array([0, 0, 1, 1, 1, 2, 2, 2])          # 1、2 是 thing 实例
SEM = np.array([1, 0, 0, 0, 0, 0, 0, 0])         # 0 号体素是墙
PRED = np.array([False, True, True, True, False, True, False, False])
KEEP, STUFF = np.array([1, 2]), (1,)


def test_categories_split_a_prediction_into_its_sources():
    cat, best = fv.object_categories(PRED, GT, SEM, KEEP, STUFF)
    assert best == 1                                    # 与 GT 1 的 IoU 0.4 > 与 GT 2 的 1/6
    assert cat[1] == fv.CAT_UNLABELED                   # 预测吃到的无标注体素
    assert list(cat[2:4]) == [fv.CAT_CORE] * 2          # 落在最好 GT 里的核心
    assert cat[4] == fv.CAT_MISSING                     # GT 有、预测没覆盖
    assert list(cat[5:8]) == [fv.CAT_OTHER_GT] * 3      # 落在另一个 thing 物体里
    assert cat[0] == fv.CAT_STUFF                       # 墙


def test_stats_match_the_categories_and_shares():
    st = fv.object_stats(PRED, GT, SEM, KEEP, STUFF)
    assert (st['best'], st['n_pred'], st['n_gt']) == (1, 4, 3)
    assert (st['core'], st['other_gt'], st['stuff'], st['unlabeled'], st['missing']) == (2, 1, 0, 1, 1)
    assert abs(st['core_share'] - 0.5) < 1e-9 and abs(st['other_gt_share'] - 0.25) < 1e-9
    assert st['mixed'] == 2                             # GT1 占 0.5、GT2 占 0.25，都过 0.2
    assert abs(st['size_ratio'] - 4 / 3) < 1e-9


def test_missing_gt_case_is_all_missing():
    st = fv.object_stats(np.zeros(8, bool), GT, SEM, KEEP, STUFF, best=2)
    assert (st['best'], st['n_pred'], st['n_gt']) == (2, 0, 3)
    assert st['missing'] == 3 and st['core'] == 0


def test_prediction_with_no_thing_gt_has_best_minus_one():
    pred = np.array([True, True, False, False, False, False, False, False])
    st = fv.object_stats(pred, GT, SEM, KEEP, STUFF)
    assert st['best'] == -1 and st['core'] == 0 and st['unlabeled'] == 1 and st['stuff'] == 1


def test_subsample_never_drops_core_voxels():
    cat = np.full(100, fv.CAT_UNLABELED, np.int8)
    cat[:7] = fv.CAT_CORE
    idx = fv.select_and_subsample(cat, np.ones(100, bool), max_points=20, seed=0)
    assert len(idx) == 20 and np.count_nonzero(cat[idx] == fv.CAT_CORE) == 7
    assert list(cat[idx][-7:]) == [fv.CAT_CORE] * 7     # 核心画在最上面


def test_colors_and_names_stay_aligned_with_the_constants():
    assert len(fv.CAT_COLORS) == len(fv.CAT_NAMES) == 5
    assert fv.CAT_COLORS.shape == (5, 3) and fv.CAT_COLORS.max() <= 1.0
