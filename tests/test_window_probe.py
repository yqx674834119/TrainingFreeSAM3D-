"""Stage-1 probe primitives: AUC of a column, logistic CV AUC, relations, and the top-1 ranking (CPU)."""
import numpy as np

from freesam3d.streaming.holdout import Variant
from freesam3d.streaming.pair_features import cross_feature_names, pair_feature_names
from freesam3d.streaming.window_probe import (RELATIONS, collect_cross_edges, column_auc, cross_columns,
                                              logistic_auc, relation_of, rule_scores_of,
                                              single_column_auc, top1_stats)
from tests.test_holdout import CFG, _frame

NAMES = pair_feature_names()


def test_column_auc_handles_separation_ties_and_constants():
    y = np.array([1, 1, 0, 0], bool)
    assert column_auc([3, 2, 1, 0], y) == 1.0
    assert column_auc([0, 1, 2, 3], y) == 0.0
    assert column_auc([1, 1, 1, 1], y) == 0.5                                   # constant -> no discrimination
    assert column_auc([1, 1, 0, 0], y) == 1.0                                 # ties, still separating
    assert abs(column_auc([1, 1, 0, 0], np.array([1, 0, 1, 0], bool)) - 0.5) < 1e-9   # ties, uninformative
    assert np.isnan(column_auc([1, 2, 3, 4], np.ones(4, bool)))                 # one class only


def test_relation_names_the_three_cases():
    assert relation_of(False, 0.99) == 'adjacent'
    assert relation_of(True, 0.95) == 'part_whole'
    assert relation_of(True, 0.2) == 'reobserve'
    assert set(RELATIONS) == {'reobserve', 'part_whole', 'adjacent'}


def test_cross_columns_and_rule_scores_read_the_cached_layout():
    pf = np.zeros((1, len(NAMES)), np.float32)
    assert cross_columns(pf).shape == (1, len(cross_feature_names()))
    pf[0, NAMES.index('contain_a_in_b')] = 0.9
    pf[0, NAMES.index('contain_b_in_a')] = 0.2
    pf[0, NAMES.index('feat_cos')] = 0.5
    assert abs(float(rule_scores_of(pf, 'p2a')[0]) - 0.9) < 1e-6
    assert abs(float(rule_scores_of(pf, 'p2b')[0]) - 0.45) < 1e-6
    pf[0, NAMES.index('feat_cos')] = -0.5
    assert float(rule_scores_of(pf, 'p2b')[0]) == 0.0


def test_single_column_auc_reports_every_column():
    pf = np.zeros((4, len(NAMES)), np.float32)
    pf[:, NAMES.index('contain_a_in_b')] = [0.9, 0.8, 0.2, 0.1]
    rows = single_column_auc(pf, np.array([1, 1, 0, 0], bool))
    assert set(rows) == set(NAMES)
    assert rows['contain_a_in_b']['auc'] == 1.0 and rows['contain_a_in_b']['nonzero'] == 1.0
    assert rows['min_dist']['nonzero'] == 0.0 and rows['min_dist']['auc'] == 0.5


def test_logistic_auc_is_out_of_fold_over_scenes():
    rng = np.random.default_rng(0)
    n = 300
    groups = np.repeat([f's{i}' for i in range(10)], n // 10)
    y = np.tile([1, 0], n // 2)
    pf = np.zeros((n, len(NAMES)), np.float32)
    pf[:, NAMES.index('contain_a_in_b')] = np.where(y == 1, 0.8, 0.2) + rng.normal(0, 0.01, n)
    assert logistic_auc(pf, y, groups) > 0.98
    assert np.isnan(logistic_auc(pf, np.ones(n, int), groups))        # one class -> undefined


def test_top1_stats_ranks_the_correct_object():
    # one frame, one current fragment (node 0), two frozen objects (nodes 1, 2)
    edges = np.array([[0, 1], [0, 2]], np.int64)
    types = np.array([[0, 2], [0, 2]], np.int64)
    fidx = np.zeros(2, np.int64)
    nodes = [dict(major=np.array([5, 5, 9]), purity=np.ones(3, np.float32))]
    good = top1_stats(edges, types, fidx, nodes, np.array([0.9, 0.1]))
    assert good['n_fragments'] == 1 and good['n_with_correct'] == 1
    assert good['accuracy'] == 1.0 and good['mean_best_rank'] == 1.0
    bad = top1_stats(edges, types, fidx, nodes, np.array([0.1, 0.9]))
    assert bad['accuracy'] == 0.0 and bad['mean_best_rank'] == 2.0
    assert bad['mean_distractors_above'] == 1.0
    # a fragment whose candidates are all wrong is counted separately, not as a miss
    nodes = [dict(major=np.array([5, 9, 9]), purity=np.ones(3, np.float32))]
    none = top1_stats(edges, types, fidx, nodes, np.array([0.9, 0.1]))
    assert none['n_fragments'] == 1 and none['n_with_correct'] == 0
    assert np.isnan(none['accuracy']) and none['n_no_correct_candidate'] == 1


def test_collect_cross_edges_end_to_end():
    """Two frames of the same box: one cross-frame edge, labeled positive, with its own frame's nodes."""
    rng = np.random.default_rng(0)
    box = np.stack([rng.uniform(0, 0.2, 400), rng.uniform(0, 0.2, 400), rng.uniform(0, 0.2, 400)], 1)
    frames = [_frame([box], fid=0), _frame([box], fid=1)]
    for f in frames:
        f.gt = np.ones(400, np.int64)
    got = collect_cross_edges(frames, Variant('linkage', 'p2a', 0.65, 0.3), None, None, CFG, 'cpu')
    assert len(got['pair_feat']) == 1 and got['y'].tolist() == [1] and got['valid'].tolist() == [True]
    assert got['relation'].tolist() == ['part_whole']          # same GT, containment ~1
    assert len(got['nodes']) == 2 and got['nodes'][0]['num_cur'] == 1
    assert got['pair_feat'][0, NAMES.index('min_dist')] == 0.0  # intra columns are zero on cross edges
    assert got['pair_feat'][0, NAMES.index('has_intra')] == 0.0
    assert got['types'].tolist() == [[0, 2]]
