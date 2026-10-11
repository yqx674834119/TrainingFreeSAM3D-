import numpy as np
import torch

from freesam3d.merging.cluster import average_linkage, connected_components, merge_groups
from freesam3d.merging.graph import EDGE_DIM, NODE_GEOM, build_graph, edge_feature_names
from freesam3d.merging.labels import edge_labels, group_majority
from freesam3d.merging.model import build_model


def _toy_frame():
    # three groups on a line: 0 and 1 touch (<5cm), 2 is 1 m away
    rng = np.random.default_rng(0)
    g0 = rng.uniform([0, 0, 0], [0.2, 0.2, 0.2], (80, 3))
    g1 = rng.uniform([0.21, 0, 0], [0.4, 0.2, 0.2], (80, 3))
    g2 = rng.uniform([1.5, 0, 0], [1.7, 0.2, 0.2], (80, 3))
    xyz = np.concatenate([g0, g1, g2])
    groups = np.repeat([0, 1, 2], 80)
    rgb = np.repeat([[200, 50, 50], [200, 60, 60], [20, 20, 200]], 80, axis=0).astype(np.uint8)
    feat = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 0, 1]], np.float32)
    pf = feat[groups] + rng.normal(0, 0.01, (240, 3)).astype(np.float32)
    return xyz, rgb, groups, feat, pf


def test_graph_edges_and_feature_layout():
    xyz, rgb, groups, feat, pf = _toy_frame()
    g = build_graph(xyz, rgb, groups, feat, pf, eps=0.05, voxel=0.02, feature_centering=False)
    assert g.edges.tolist() == [[0, 1]]
    assert g.node_geom.shape == (3, NODE_GEOM) and g.edge_feat.shape == (1, EDGE_DIM)
    assert len(edge_feature_names()) == EDGE_DIM
    names = edge_feature_names()
    e = dict(zip(names, g.edge_feat[0]))
    assert 0 <= e['min_dist'] < 0.05 and e['group_cos'] > 0.9 and e['boundary_cos'] > 0.9
    assert e['color_dist'] < 0.1 and e['nested'] == 0


def test_voxel_subsample_matches_full_graph():
    xyz, rgb, groups, feat, pf = _toy_frame()
    full = build_graph(xyz, rgb, groups, feat, pf, eps=0.05, voxel=0.0)
    vox = build_graph(xyz, rgb, groups, feat, pf, eps=0.05, voxel=0.02)
    assert full.edges.tolist() == vox.edges.tolist()


def test_labels_pure_and_impure():
    groups = np.repeat([0, 1, 2, 3], 50)
    gt = np.concatenate([np.full(50, 7), np.full(50, 7), np.full(50, 9), np.r_[np.full(25, 7), np.full(25, 9)]])
    major, pure = group_majority(groups, gt, 4)
    assert major.tolist() == [7, 7, 9, 7] and pure.tolist() == [True, True, True, False]
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    y, valid = edge_labels(groups, gt, edges)
    assert y.tolist() == [1, 0, 0] and valid.tolist() == [True, True, False]
    gt[:50] = 0                                    # group 0 unlabeled -> not pure
    _, valid = edge_labels(groups, gt, edges)
    assert valid.tolist() == [False, True, False]


def test_models_forward_and_symmetry():
    xyz, rgb, groups, feat, pf = _toy_frame()
    g = build_graph(xyz, rgb, groups, feat, pf, eps=2.0, voxel=0.02)    # all pairs adjacent
    cfg = dict(hidden=16, feature_proj=4, message_passing=1, dropout=0.0)
    for kind in ('full', 'geom_only', 'edge_only'):
        m = build_model(kind, NODE_GEOM, feat.shape[1], EDGE_DIM, cfg).eval()
        out = m(torch.from_numpy(g.node_geom), torch.from_numpy(g.node_feat), torch.from_numpy(g.edges),
                torch.from_numpy(g.edge_feat))
        assert out.shape == (len(g.edges),)
        flipped = m(torch.from_numpy(g.node_geom), torch.from_numpy(g.node_feat),
                    torch.from_numpy(g.edges[:, ::-1].copy()), torch.from_numpy(g.edge_feat))
        assert torch.allclose(out, flipped, atol=1e-5), kind     # orientation-free
        assert m.num_params() < 60_000


def test_average_linkage_blocks_weak_bridge():
    edges = np.array([[0, 1], [1, 2], [0, 2]])
    prob = np.array([0.95, 0.95, 0.05])
    assert len(np.unique(connected_components(3, edges, prob, 0.5))) == 1
    avg = average_linkage(3, edges, prob, 0.6)
    assert avg[0] == avg[1] and avg[2] != avg[0]
    out = merge_groups(np.repeat([0, 1, 2], 3), edges, prob, 0.6)
    assert len(np.unique(out)) == 2 and out.shape == (9,)


def test_relabel_matches_edge_labels():
    from freesam3d.merging.labels import group_purity, relabel
    rng = np.random.default_rng(3)
    groups = rng.integers(0, 6, 300)
    gt = np.where(rng.random(300) < 0.85, groups // 2 + 1, rng.integers(0, 4, 300))
    edges = np.array([[a, b] for a in range(6) for b in range(a + 1, 6)])
    for purity in (0.5, 0.7, 0.9):
        y1, v1 = edge_labels(groups, gt, edges, purity, 0.3)
        y2, v2 = relabel(*group_purity(groups, gt, 6), edges, purity, 0.3)
        assert np.array_equal(y1, y2) and np.array_equal(v1, v2)


def test_objectness_labels_and_head():
    from freesam3d.merging.labels import objectness_labels
    major = np.array([0, 5, 6, 7, 8])          # 0 unlabeled
    purity = np.array([0.0, 0.95, 0.95, 0.5, 0.9])
    labeled = np.array([0.0, 1.0, 1.0, 1.0, 0.1])
    sem = np.array([0, 4, 1, 4, 4])            # nyu40: 1 = wall (stuff), 4 = bed (thing)
    y, valid = objectness_labels(major, purity, labeled, sem, {1, 2, 22})
    assert y.tolist() == [0, 1, 0, 1, 1]
    assert valid.tolist() == [False, True, True, False, False]     # mixed and barely-labeled masked out
    xyz, rgb, groups, feat, pf = _toy_frame()
    g = build_graph(xyz, rgb, groups, feat, pf, eps=2.0, voxel=0.02)
    m = build_model('full', NODE_GEOM, feat.shape[1], EDGE_DIM, dict(hidden=16, feature_proj=4, message_passing=1, dropout=0.0)).eval()
    e, o = m(torch.from_numpy(g.node_geom), torch.from_numpy(g.node_feat), torch.from_numpy(g.edges),
             torch.from_numpy(g.edge_feat), return_objectness=True)
    assert e.shape == (len(g.edges),) and o.shape == (3,)
