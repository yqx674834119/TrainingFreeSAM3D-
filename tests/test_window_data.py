"""Stage-2 data collection (CPU, offline): labels, the intra-graph replica, and one scene end to end."""
import numpy as np
import torch

from freesam3d.streaming.pair_features import pair_feature_names
from freesam3d.streaming.unified_flow import UnifiedGrouping
from freesam3d.streaming.window_data import (POS_PURITY, collect_scene, edge_labels, fragment_stats,
                                             intra_frame_graph, pack, votes_stats)
from freesam3d.streaming.window_memory import FragmentFrame, WindowConfig, WindowMemory

VOXEL = 0.05
STUFF = {1, 2, 22}                                       # NYU40 wall/floor/ceiling
_CA = pair_feature_names().index('contain_a_in_b')


class _StubModel:
    """Edge logit = 10 * (contain_a_in_b - 0.5); quality logit = log1p(n_frames)."""

    def __call__(self, geom, feat, edges, edge_feat, node_type, n_frames, age, edge_type):
        x = edge_feat[:, _CA] if len(edges) else edge_feat.new_zeros(0)
        return 10.0 * (x - 0.5), torch.log1p(n_frames)


class _Std:
    def apply(self, geom, ef):
        return geom, ef


CFG = dict(stream_v2=dict(past_unit='block', voxel=VOXEL, touch_dilation=1, linkage='average_sparse',
                          admit_min_points=100, object_merge=True, purity=0.8, threshold=0.5))


def _frame(frag_pts, edges=(), fid=0, gt=None, sem=None):
    xyz = np.concatenate([np.asarray(p, np.float32) for p in frag_pts])
    groups = np.concatenate([np.full(len(p), i, np.int64) for i, p in enumerate(frag_pts)])
    num = len(frag_pts)
    e = np.asarray(edges, np.int64).reshape(-1, 2)
    gt = np.zeros(len(xyz), np.int64) if gt is None else np.asarray(gt, np.int64)
    sem = np.full(len(xyz), 5, np.int64) if sem is None else np.asarray(sem, np.int64)   # 5 = a thing
    return FragmentFrame('scene', fid, xyz, np.full_like(xyz, 128), groups, np.zeros((num, 4), np.float32),
                         e, np.zeros((len(e), 14), np.float32), np.full(num, 0.9, np.float32), gt, sem)


def _scene():
    """Two objects 1.5 m apart, each cut into two heavily overlapping fragments, over three frames.

    The overlap matters: the stub's block rule is containment, so fragments that merely touch stay
    separate objects and each then covers half its GT instance (IoU 0.5 by construction).
    """
    rng = np.random.default_rng(0)
    frames = []
    for t in range(3):
        pts, gt = [], []
        for k in range(2):
            for h in range(2):
                base = np.array([1.5 * k + 0.05 * h - 0.05, -0.2, -0.2])
                pts.append(base + rng.uniform(0, 0.2, (200, 3)))
                gt.append(np.full(200, k + 1))
        frames.append(_frame(pts, [[0, 1], [2, 3]], fid=t, gt=np.concatenate(gt)))
    return frames


# ------------------------------------------------------------------ pure pieces
def test_votes_stats_purity_and_ties():
    major, purity, labeled = votes_stats([{3: 7, 5: 2}, {4: 5}, {2: 3, 9: 3}, {}])
    assert major.tolist() == [3, 4, 2, 0]                  # ties go to the smaller id
    assert abs(purity[0] - 7 / 9) < 1e-6 and purity[1] == 1.0 and purity[3] == 0.0
    assert labeled.tolist() == [9, 5, 6, 0]


def test_edge_labels_mask_impure_and_unlabeled():
    major_a = np.array([1, 1, 2, 0, 3])
    pur_a = np.array([0.9, 0.9, 0.9, 0.9, 0.4])
    major_b = np.array([1, 2, 2, 1, 3])
    pur_b = np.array([0.8, 0.8, 0.3, 0.9, 0.9])
    y, valid = edge_labels(major_a, pur_a, major_b, pur_b)
    assert valid.tolist() == [True, True, False, False, False]
    assert y[:2].tolist() == [1.0, 0.0]                    # same instance -> 1, different -> 0


def test_intra_frame_graph_matches_the_flow():
    """The standalone intra-graph builder must reproduce exactly what the flow feeds the model."""
    frames = _scene()
    fr = frames[0]
    mem = WindowMemory(WindowConfig(window=1, grouping='unified', threshold=0.5, cross_threshold=0.5,
                                    admit_min_points=100, voxel=VOXEL),
                       unified=UnifiedGrouping(_StubModel(), _Std(), 'cpu'), device='cpu')
    mem.update(fr)
    got, want = mem.last_unified['intra'], intra_frame_graph(fr, voxel=VOXEL, dilation=1)
    assert np.allclose(got['node_geom'], want['node_geom'], atol=1e-5)
    assert np.allclose(got['node_feat'], want['node_feat'], atol=1e-5)
    assert np.array_equal(got['edges'], want['edges'])
    assert np.allclose(got['pair_feat'], want['pair_feat'], atol=1e-5)


def test_fragment_stats_and_labels_agree_with_the_cache_conventions():
    frames = _scene()
    fr = frames[0]
    major, pur, labfrac, sem = fragment_stats(fr)
    assert major.tolist() == [1, 1, 2, 2] and np.allclose(pur, 1.0) and np.allclose(labfrac, 1.0)
    assert sem.tolist() == [5, 5, 5, 5]


def test_pack_pointers_and_dtypes():
    frames = _scene()
    rows = [intra_frame_graph(fr, voxel=VOXEL) | dict(node_type=np.zeros(4, np.int64),
                                                      edge_type=np.zeros(2, np.int64),
                                                      edge_y=np.ones(2, np.float32),
                                                      edge_valid=np.ones(2, bool),
                                                      node_y=np.ones(4, np.float32),
                                                      node_valid=np.ones(4, bool)) for fr in frames]
    data = pack(rows, [], np.zeros(0, np.float32), np.zeros(0, bool), scene='s', feat_dim=4)
    assert data['node_ptr_i'].tolist() == [0, 4, 8, 12] and data['edge_ptr_i'].tolist() == [0, 2, 4, 6]
    assert data['feat_i'].dtype == np.float16 and data['feat_i'].shape == (12, 4)
    assert data['pf_i'].shape == (6, len(pair_feature_names()))
    assert data['edges_c'].shape == (0, 2) and data['geom_c'].shape == (0, 11)   # empty cross side
    assert len(data['node_ptr_c']) == data['frames'] + 1                        # one pointer per frame


# ------------------------------------------------------------------ one scene end to end
def test_collect_scene_labels_objects_edges_and_nodes():
    frames = _scene()
    data = collect_scene(frames, CFG, _StubModel(), _Std(), 'cpu', STUFF, voxel=VOXEL, min_gt_voxels=5)
    assert data['frames'] == 3
    # --- intra graph: two blocks per frame (the cached edges glue each pair), all labels valid
    assert data['node_ptr_i'][-1] == 12 and data['edge_ptr_i'][-1] == 6
    assert data['ev_i'].all() and (data['ey_i'] == 1).all()          # both ends pure and equal
    assert data['nv_i'].all() and (data['ny_i'] == 1).all()          # thing objectness
    assert data['kind_i'].tolist() == [0] * 12 and data['etype_i'].tolist() == [0] * 6
    # --- cross graph: blocks join the objects of the previous frame, edges are labeled
    assert data['etype_c'].max() <= 2 and data['ev_c'].sum() > 0
    assert data['kind_c'].max() == 2
    # --- object nodes carry a valid final voxel IoU in [0, 1] (the ranking target)
    obj = data['kind_c'] == 2
    assert obj.sum() > 0 and data['niou_valid_c'][obj].all()
    assert float(data['niou_c'][obj].min()) >= 0.0 and float(data['niou_c'][obj].max()) <= 1.0
    assert obj.sum() == 4                                            # two objects, in frames 1 and 2
    assert float(data['niou_c'][obj].max()) > 0.9                    # the synthetic objects are matched
    # --- block nodes keep the objectness label, never the IoU one
    blk = data['kind_c'] == 1
    assert blk.sum() > 0 and not data['niou_valid_c'][blk].any()
    assert data['nv_c'][blk].all() and (data['ny_c'][blk] == 1).all()
    assert np.isnan(data['ny_c'][obj]).all()
