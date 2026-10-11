import numpy as np
import pytest
import torch

HAS_CUDA = torch.cuda.is_available()

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


def test_gpu_graph_matches_cpu_on_toy():
    if not HAS_CUDA:
        pytest.skip('no CUDA')
    from freesam3d.merging.graph_gpu import build_graph_gpu
    xyz, rgb, groups, feat, pf = _toy_frame()
    c = build_graph(xyz, rgb, groups, feat, pf, eps=0.05, voxel=0.02, boundary_k=10 ** 9, feature_centering=False)
    g = build_graph_gpu(xyz, rgb, groups, feat, pf, eps=0.05, voxel=0.02, feature_centering=False)
    assert c.edges.tolist() == g.edges.tolist()
    # column 1 (height) uses the frame-median z: torch.median takes the lower middle, numpy averages
    np.testing.assert_allclose(np.delete(c.node_geom, 1, 1), np.delete(g.node_geom, 1, 1), atol=1e-4)
    np.testing.assert_allclose(c.node_geom[:, 1], g.node_geom[:, 1], atol=1e-2)
    # all columns except contact_norm (sub-point counting differs slightly) within tolerance
    names = edge_feature_names()
    for i, n in enumerate(names):
        if n == 'contact_norm':
            continue
        np.testing.assert_allclose(c.edge_feat[:, i], g.edge_feat[:, i], atol=2e-2, err_msg=n)


def test_torch_kmeans_fill_groups_uncovered_points():
    if not HAS_CUDA:
        pytest.skip('no CUDA')
    from freesam3d.lifting.grouping import fill_uncovered
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 0.05, (60, 3)), rng.normal(3, 0.05, (60, 3))])
    ids = np.full(120, -1)
    out = fill_uncovered(ids, x, n_clusters=2, min_points=20, seed=0, base=5, backend='torch')
    assert set(out.tolist()) == {5, 6}
    assert len(set(out[:60])) == 1 and len(set(out[60:])) == 1


def test_gpu_exclusive_map_and_backproject_match_numpy():
    if not HAS_CUDA:
        pytest.skip('no CUDA')
    from freesam3d.lifting.grouping import exclusive_group_map, exclusive_group_map_torch
    from freesam3d.lifting.geometry import backproject, backproject_torch
    rng = np.random.default_rng(0)
    masks = np.zeros((6, 40, 50), bool)
    for i in range(6):
        y, x = rng.integers(0, 30), rng.integers(0, 40)
        masks[i, y:y + rng.integers(3, 12), x:x + rng.integers(3, 12)] = True
    masks[3] = masks[2]                                  # exact tie in area -> later index wins
    ref = exclusive_group_map(masks)
    out = exclusive_group_map_torch(torch.as_tensor(masks, device='cuda')).cpu().numpy()
    assert np.array_equal(ref, out)
    depth = (rng.random((40, 50)) * 3000).astype(np.uint16)
    depth[::7] = 0
    K = np.array([[500.0, 0, 25], [0, 500.0, 20], [0, 0, 1]])
    pose = np.eye(4); pose[:3, 3] = [1, 2, 3]
    xyz, px = backproject(depth, K, pose)
    xyz_t, px_t = backproject_torch(depth, K, pose)
    np.testing.assert_allclose(xyz, xyz_t.cpu().numpy(), atol=1e-4)
    assert np.array_equal(px, px_t.cpu().numpy())


def test_proposals_and_nested_nms():
    from freesam3d.merging.proposals import NUM_SCALARS, build_proposals, nested_nms, proposal_targets, scalar_names
    assert len(scalar_names()) == NUM_SCALARS
    rng = np.random.default_rng(0)
    n = 600
    xyz = rng.uniform(0, 1, (n, 3)).astype(np.float32)
    groups = np.repeat([0, 1, 2], 200)
    gfeat = np.eye(3, dtype=np.float32)
    proto = rng.normal(size=(n, 4)).astype(np.float32)
    whole = np.zeros(n, bool); whole[:400] = True            # raw mask 0 = groups 0+1 (the "object")
    part = np.zeros(n, bool); part[:200] = True              # raw mask 1 = group 0 (a part, nested in 0)
    other = np.zeros(n, bool); other[400:] = True            # raw mask 2 = group 2
    raw = np.stack([whole, part, other])
    P = build_proposals(xyz, groups, gfeat, proto, raw, np.array([0.9, 0.8, 0.7], np.float32),
                        np.array([1, 2, -1]), np.array([0.9, 0.9, 0.1]), merged_masks=np.stack([whole]), min_points=50)
    assert P.masks.shape == (4, n) and P.scalars.shape == (4, NUM_SCALARS) and P.is_raw.tolist() == [True, True, True, False]
    names = scalar_names()
    s = {nm: P.scalars[:, i] for i, nm in enumerate(names)}
    assert s['n_groups'].tolist() == [2, 1, 1, 2]
    assert s['n_child'][0] == 1 and s['n_parent'][1] == 1 and s['n_raw_inside'][3] == 2   # merged copy contains raw 0 and 1
    gt = np.repeat([7, 7, 9], 200)
    iou, best = proposal_targets(P.masks, gt)
    np.testing.assert_allclose(iou, [1.0, 0.5, 1.0, 1.0]) and best.tolist() == [7, 7, 9, 7]
    # nested NMS: whole (score .9) suppresses its part (.8) and its duplicate (.95 wins as top) -> keep {dup, other}
    keep = nested_nms(P.masks, np.array([0.9, 0.8, 0.7, 0.95]))
    assert sorted(keep.tolist()) == [2, 3]


def test_mask_scorer_forward_shapes():
    from freesam3d.merging.scorer import MaskScorer
    from freesam3d.merging.proposals import NUM_SCALARS
    m = MaskScorer(NUM_SCALARS, 8, hidden=16).eval()
    s = m.score(torch.randn(5, NUM_SCALARS), torch.randn(5, 8))
    assert s.shape == (5,) and (s >= 0).all() and (s <= 1).all()
    m0 = MaskScorer(NUM_SCALARS, 0, hidden=16).eval()
    assert m0.score(torch.randn(5, NUM_SCALARS), torch.randn(5, 0)).shape == (5,)


def test_gpu_proposals_match_cpu():
    if not HAS_CUDA:
        pytest.skip('no CUDA')
    from freesam3d.merging.proposals import build_proposals, nested_nms
    from freesam3d.merging.proposals_gpu import build_proposals_gpu, nested_nms_gpu
    rng = np.random.default_rng(1)
    n, G = 2000, 6
    xyz = rng.uniform(0, 2, (n, 3)).astype(np.float32)
    groups = rng.integers(0, G, n)
    gfeat = rng.normal(size=(G, 8)).astype(np.float32)
    proto = rng.normal(size=(n, 4)).astype(np.float32)
    raw = np.stack([np.isin(groups, s) for s in ([0, 1], [0], [2, 3, 4], [3], [5], [1])])
    merged = np.stack([np.isin(groups, [0, 1, 2]), np.isin(groups, [5])])
    args = (xyz, groups, gfeat, proto, raw, rng.random(6).astype(np.float32), np.array([1, 0, 3, -1, 4, 5]), rng.random(G).astype(np.float32))
    c = build_proposals(*args, merged_masks=merged, min_points=50)
    g = build_proposals_gpu(*args, merged_masks=merged, min_points=50)
    assert np.array_equal(c.masks, g.masks) and np.array_equal(c.is_raw, g.is_raw) and np.array_equal(c.source_index, g.source_index)
    np.testing.assert_allclose(c.scalars, g.scalars, rtol=1e-3, atol=2e-3)
    np.testing.assert_allclose(c.feat, g.feat, atol=1e-4)
    s = rng.random(len(c.masks))
    assert nested_nms(c.masks, s).tolist() == nested_nms_gpu(g.masks, s).tolist()
