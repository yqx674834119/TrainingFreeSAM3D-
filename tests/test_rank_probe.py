"""阶段 5 · P1（CPU）：外观可加统计量的恒等式、关系特征、留一逻辑回归与 AP50 口径、npz 往返。"""
import numpy as np

from freesam3d.analysis.rank_probe import (aggregate_ap50, appearance_from_sums, decide,
                                           feature_layout, loocv_logistic_scores, load_scene,
                                           relation_features, save_scene, spearman_table, _spearman)
from freesam3d.analysis.rank_probe import R_MOSTLY_TRAINING


def test_appearance_from_sums_matches_bruteforce():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(37, 8))
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    pair, var, cons = appearance_from_sums(f.sum(0)[None], [float((f ** 2).sum())], [len(f)])
    cos = f @ f.T
    want_pair = (cos.sum() - np.trace(cos)) / (len(f) * (len(f) - 1))
    assert abs(pair[0] - want_pair) < 1e-9
    assert abs(var[0] - f.var(0).mean()) < 1e-9
    assert abs(cons[0] - np.linalg.norm(f.mean(0))) < 1e-9


def test_appearance_from_sums_singleton():
    pair, var, cons = appearance_from_sums(np.array([[1.0, 0.0, 0.0]]), [1.0], [1.0])
    assert np.isnan(pair[0]) and abs(var[0]) < 1e-12 and abs(cons[0] - 1.0) < 1e-9


def test_relation_features_on_nested_sets():
    size = np.array([10, 9, 10], float)
    contain = np.eye(3)
    contain[1, 0] = 0.9      # B 的 90% 落在 A 内
    contain[0, 2] = 0.5      # A 的一半落在 C 内
    rel = relation_features(size, contain)
    assert abs(rel['rel_contain_in_max'][1] - 0.9) < 1e-9
    assert rel['rel_contain_out_max'][0] >= 0.9
    assert rel['rel_is_subset'][1] == 1.0 and rel['rel_is_subset'][0] == 0.0
    assert rel['rel_n_overlap'].tolist() == [1.0, 1.0, 0.0]
    assert abs(rel['rel_iou_max'][1] - 8.1 / 10.9) < 1e-9   # inter = 0.9×9，并 = 9+10−inter
    single = relation_features(np.array([5.0]), np.eye(1))
    assert single['rel_iou_max'][0] == 0.0 and single['rel_n_overlap'][0] == 0.0


def test_spearman_layout_and_decide():
    names, slices = feature_layout()
    assert names[slices['self']][0] == 'n_pts' and names[slices['control']][0] == 'quality'
    assert len(set(names)) == len(names)
    x = np.arange(10.0)
    assert abs(_spearman(x, x ** 2) - 1.0) < 1e-9
    assert abs(_spearman(x, -(x ** 2)) + 1.0) < 1e-9
    tab = spearman_table(np.c_[x, -x], x ** 2, np.zeros(10, int), ['a', 'b'])
    assert tab['a']['pooled'] > 0.99 and tab['b']['pooled'] < -0.99
    r, v = decide(0.422 + R_MOSTLY_TRAINING * (0.635 - 0.422), 0.422, 0.635)
    assert r >= R_MOSTLY_TRAINING and v.startswith('A 档')
    assert decide(0.43, 0.422, 0.635)[1].startswith('C 档')


def _synth_scenes(n=4):
    """每个场景：4 个 GT 实例 × 50 体素；5 个候选（全部 ≥ 口径的 20 体素），已知最佳 IoU。"""
    out = []
    for _ in range(n):
        V = 260
        gt = np.zeros(V, np.int64)
        for k in range(4):
            gt[k * 50:(k + 1) * 50] = k + 1
        masks = np.zeros((5, V), bool)
        masks[0, 0:50] = True      # gt1 完美    -> IoU 1
        masks[1, 0:25] = True      # gt1 一半    -> IoU 0.5
        masks[2, 50:100] = True    # gt2 完美    -> IoU 1
        masks[3, 200:224] = True   # 未标注区    -> IoU 0
        masks[4, 100:145] = True   # gt3 的 45/50 -> IoU 0.9
        out.append(dict(gt_vox=gt, keep=np.arange(1, 5, dtype=np.int32), masks=masks,
                        iou=np.array([1.0, 0.5, 1.0, 0.0, 0.9], np.float32)))
    return out


def test_loocv_ap50_recovers_the_oracle_ranking():
    sd = _synth_scenes()
    X = np.concatenate([s['iou'][:, None] for s in sd])
    y = (X[:, 0] >= 0.5).astype(int)
    scene = np.concatenate([np.full(len(s['iou']), i) for i, s in enumerate(sd)])
    scores = loocv_logistic_scores(X, y, scene)
    order = np.argsort(-X[:, 0], kind='stable')
    assert (np.argsort(-scores, kind='stable') == order).all()   # 与标签特征同序
    ap, _ = aggregate_ap50(sd, scores)
    oracle, _ = aggregate_ap50(sd, np.concatenate([s['iou'] for s in sd]))
    assert abs(ap - oracle) < 1e-6
    neg, _ = aggregate_ap50(sd, -np.concatenate([s['iou'] for s in sd]))
    assert neg < oracle - 0.05


def test_scene_npz_roundtrip(tmp_path):
    d = dict(scene='s', names=['a', 'b'], feat=np.ones((3, 2), np.float32),
             iou=np.zeros(3, np.float32), quality=np.zeros(3, np.float32),
             n_obs=np.ones(3, np.int32), source=np.zeros(3, np.int8),
             n_pts=np.ones(3, np.int32), n_vox=np.ones(3, np.int32), n_frag=np.ones(3, np.int32),
             masks_packed=np.packbits(np.eye(3, dtype=bool), axis=1), ncells=3,
             gt_vox=np.array([1, 1, 0], np.int32), keep=np.array([1], np.int32),
             pt_final=np.zeros((2, 4), np.int16), frame_ids=np.array([0, 40], np.int32),
             voxel=0.05, metrics={'ap50': 0.1}, seconds=1.0)
    save_scene(tmp_path / 'x.npz', d)
    e = load_scene(tmp_path / 'x.npz')
    assert e['names'] == ['a', 'b'] and e['masks'].shape == (3, 3) and bool(e['masks'][1, 1])
    assert e['metrics']['ap50'] == 0.1 and e['scene'] == 's'


def test_gbm_loocv_scores_recovers_the_oracle_ranking():
    from freesam3d.analysis.rank_probe import gbm_loocv_scores
    sd = _synth_scenes()
    X = np.concatenate([s['iou'][:, None] for s in sd])
    y = (X[:, 0] >= 0.5).astype(int)
    scene = np.concatenate([np.full(len(s['iou']), i) for i, s in enumerate(sd)])
    s = gbm_loocv_scores(X, y, scene, min_samples_leaf=2)   # 合成集很小，放宽叶节点下限
    assert np.isfinite(s).all() and len(s) == len(y)
    ap, _ = aggregate_ap50(sd, s)
    # 单特征 + 取值很少的阶梯函数会把正类并列（并列序按稳定排序），AP 达不到 oracle；
    # 本测试只验证管线（分数有限、可评 AP、显著优于全错排序），估计器质量由真实数据报告回答。
    assert ap > 0.4
