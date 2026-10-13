"""Stage 0c scoring and mixed-fragment fate helpers (CPU, offline: no GPU, no checkpoints)."""
import numpy as np
import torch

from freesam3d.merging.graph import edge_feature_names
from freesam3d.streaming.fragment_fate import (FATES, classify_fates, frag_object_rows,
                                               frame_fragment_purity, impure_fragments)
from freesam3d.streaming.pair_features import cross_feature_names
from freesam3d.streaming.voxel import voxelize
from freesam3d.streaming.window_memory import FragmentFrame, PairGraph
from freesam3d.streaming.window_scores import (CONTAIN_A_IN_B, CONTAIN_B_IN_A, CROSS_OFFSET, FEAT_COS,
                                               num_current_fragments, rule_scores, window_score_fn)


def _graph(pair_feat, edges, has_intra, node_type, node_n=None):
    edges = np.asarray(edges, np.int64).reshape(-1, 2)
    num = int(np.max(node_type)) + 1
    return PairGraph(
        edges=edges, pair_feat=torch.as_tensor(np.asarray(pair_feat), dtype=torch.float32),
        has_intra=torch.as_tensor(np.asarray(has_intra), dtype=torch.bool),
        node_major=np.zeros(num, np.int64), node_purity=np.ones(num, np.float32),
        node_n=np.ones(num, np.int64) if node_n is None else np.asarray(node_n, np.int64),
        node_type=np.asarray(node_type, np.int64), node_prev_obj=np.full(num, -1, np.int64),
        node_geom=np.zeros((num, 11), np.float32), node_feat=np.zeros((num, 4), np.float32),
        num_nodes=num, z_ref=0.0, frame_pos=0, scene='s', frame_id=0)


def _blanks(n):
    return np.zeros((n, CROSS_OFFSET + len(cross_feature_names())), np.float32)


def _frame(groups, gt, xyz=None):
    groups = np.asarray(groups, np.int64)
    num = int(groups.max()) + 1
    xyz = np.zeros((len(groups), 3), np.float32) if xyz is None else np.asarray(xyz, np.float32)
    return FragmentFrame('s', 0, xyz, np.zeros((len(groups), 3), np.float32), groups,
                         np.zeros((num, 4), np.float32), np.zeros((0, 2), np.int64),
                         np.zeros((0, 14), np.float32), np.zeros(num, np.float32),
                         np.asarray(gt, np.int64))


def test_pair_feat_columns_are_where_window_scores_thinks():
    """The cross-frame columns are addressed by the layout itself, not by hard-coded offsets."""
    names = edge_feature_names() + cross_feature_names()
    assert CROSS_OFFSET == len(edge_feature_names())
    assert names[CONTAIN_A_IN_B] == 'contain_a_in_b'
    assert names[CONTAIN_B_IN_A] == 'contain_b_in_a'
    assert names[FEAT_COS] == 'feat_cos'


def test_rule_scores_p2a_and_p2b():
    pf = _blanks(3)
    pf[0, CONTAIN_A_IN_B], pf[0, CONTAIN_B_IN_A], pf[0, FEAT_COS] = 0.9, 0.2, 0.5
    pf[1, CONTAIN_A_IN_B], pf[1, CONTAIN_B_IN_A], pf[1, FEAT_COS] = 0.3, 0.7, -0.4
    pf[2, CONTAIN_A_IN_B], pf[2, CONTAIN_B_IN_A], pf[2, FEAT_COS] = 0.4, 0.4, 0.25
    g = _graph(pf, [[0, 2], [1, 2], [0, 1]], [False] * 3, [0, 0, 2])
    assert np.allclose(rule_scores(g, 'p2a').numpy(), [0.9, 0.7, 0.4])       # max of the two containments
    assert np.allclose(rule_scores(g, 'p2b').numpy(), [0.45, 0.0, 0.1])     # times clamp(cos, 0, 1)
    try:
        rule_scores(g, 'p2c')
    except ValueError:
        pass
    else:
        raise AssertionError('an unknown rule must be rejected')


def test_intra_edges_get_the_model_and_the_others_get_the_rule():
    class _Model:
        def __call__(self, geom, feat, edges, ef):
            return torch.full((len(edges),), 2.0)                            # sigmoid(2)
    class _Std:
        def apply(self, geom, ef):
            return geom, ef
    pf = _blanks(2)
    pf[0, CONTAIN_A_IN_B] = 0.1              # the intra edge's own rule value must be ignored
    pf[1, CONTAIN_A_IN_B] = 0.8
    g = _graph(pf, [[0, 1], [0, 2]], [True, False], [0, 0, 2])
    prob = window_score_fn(_Model(), _Std(), 'p2a', 'cpu')(g).numpy()
    assert abs(prob[0] - 1 / (1 + np.exp(-2))) < 1e-5
    assert abs(prob[1] - 0.8) < 1e-5
    assert np.allclose(window_score_fn(None, None, 'p2a')(g).numpy(), [0.1, 0.8])


def test_num_current_fragments():
    g = _graph(_blanks(1), [[0, 3]], [False], [0, 0, 1, 2])
    assert num_current_fragments(g) == 2


def test_frame_fragment_purity_and_impure_selection():
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    gt = np.array([5, 5, 5, 7, 9, 9, 0, 0, 0, 0])
    major, purity, labeled_n = frame_fragment_purity(groups, gt, 3)
    assert major.tolist() == [5, 9, 0]
    assert abs(purity[0] - 0.75) < 1e-6                                      # 3 of 4 labeled points are GT 5
    assert abs(purity[1] - 1.0) < 1e-6
    assert labeled_n.tolist() == [4, 2, 0]
    got = impure_fragments([_frame(groups, gt)], purity=0.8)
    assert [g[1] for g in got] == [0]                                        # group 2 is unlabeled, not mixed
    assert abs(got[0][3] - 0.75) < 1e-6 and got[0][4] == 4


def test_frag_object_rows_maps_fragments_to_their_object():
    xyz = np.array([[0, 0, 0], [0.01, 0, 0], [2, 0, 0], [2.01, 0, 0], [5, 0, 0], [5.01, 0, 0]], np.float32)
    f = _frame([0, 0, 1, 1, 2, 2], np.zeros(6, np.int64), xyz)
    v0 = torch.unique(voxelize(torch.as_tensor(xyz[:2]), 0.05))
    v1 = torch.unique(voxelize(torch.as_tensor(xyz[2:4]), 0.05))
    assert frag_object_rows(f, [0, 1, 2], [v0, v1], voxel=0.05).tolist() == [0, 1, -1]
    assert frag_object_rows(f, [], [v0, v1]).tolist() == []
    assert frag_object_rows(f, [0], []).tolist() == [-1]


def test_classify_fates_totals_and_shares():
    shares, totals = classify_fates([0, 1, 2, -1], [10, 10, 20, 60],
                                    [False, True, False], [1, 5, 7])
    assert totals == {'own_single': 10.0, 'glued_overmerge': 10.0, 'glued_clean': 20.0, 'dropped': 60.0}
    assert abs(sum(shares.values()) - 1.0) < 1e-9
    assert abs(shares['dropped'] - 0.6) < 1e-9
    assert list(FATES) == ['own_single', 'glued_overmerge', 'glued_clean', 'dropped']
