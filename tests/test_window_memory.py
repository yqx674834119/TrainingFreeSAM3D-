"""WindowMemory (unified intra + cross-frame merging), pair features, sparse linkage (CPU, offline)."""
import numpy as np
import torch

from freesam3d.merging.cluster import average_linkage, average_linkage_sparse
from freesam3d.merging.graph import build_graph
from freesam3d.streaming.esam_merge import EsamOnlineMerge
from freesam3d.streaming.memory import FrameInput
from freesam3d.streaming.pair_features import NodeStats, pair_feature_names, pair_features
from freesam3d.streaming.voxel import overlap_matrix, voxelize
from freesam3d.streaming.window_memory import (FragmentFrame, GtAssocMemory, WindowConfig, WindowMemory,
                                               oracle_scores, stats_from_frame)

VOXEL = 0.05


def _partition(x):
    return {frozenset(np.where(x == v)[0].tolist()) for v in np.unique(x)}


def test_sparse_linkage_matches_dense():
    rng = np.random.default_rng(0)
    for _ in range(50):
        g = int(rng.integers(3, 12))
        pairs = [(i, j) for i in range(g) for j in range(i + 1, g)]
        take = rng.random(len(pairs)) < rng.uniform(0.2, 0.9)
        edges = np.array([p for p, t in zip(pairs, take) if t], np.int64).reshape(-1, 2)
        prob = rng.integers(0, 3, len(edges)).astype(float) / 2       # ties on purpose
        thr = float(rng.choice([0.25, 0.5, 0.75]))
        assert _partition(average_linkage(g, edges, prob, thr)) == _partition(average_linkage_sparse(g, edges, prob, thr))


def test_pair_features_cubes():
    rng = np.random.default_rng(0)
    a_pts = rng.uniform(0.0, 0.4, (500, 3))
    b_pts = rng.uniform(0.1, 0.5, (500, 3))
    sa = NodeStats.from_points(a_pts, np.full((500, 3), 128), feat=np.array([1.0, 0, 0, 0]), conf=0.9)
    sb = NodeStats.from_points(b_pts, np.full((500, 3), 128), feat=np.array([0.0, 1, 0, 0]), conf=0.5)
    stats = NodeStats.cat([sa, sb])
    va = torch.unique(voxelize(torch.as_tensor(a_pts, dtype=torch.float32), VOXEL))
    vb = torch.unique(voxelize(torch.as_tensor(b_pts, dtype=torch.float32), VOXEL))
    own = torch.cat([torch.zeros(len(va), dtype=torch.int64), torch.ones(len(vb), dtype=torch.int64)])
    inter, size_a, size_b = overlap_matrix(own, torch.cat([va, vb]), own, torch.cat([va, vb]), 2, 2)
    a, b = torch.tensor([0]), torch.tensor([1])
    names = pair_feature_names()
    f = dict(zip(names, pair_features(stats, a, b, inter[a, b], inter[a, b], size_a, size_b,
                                      torch.tensor([1.0]), torch.tensor([False]), torch.tensor([True]),
                                      torch.tensor([0.0]))[0].tolist()))
    i0 = float(inter[0, 1])
    assert abs(f['iou0'] - i0 / (len(va) + len(vb) - i0)) < 1e-5
    assert abs(f['contain_a_in_b'] - i0 / len(va)) < 1e-5
    assert abs(f['contain_b_in_a'] - i0 / len(vb)) < 1e-5
    assert abs(f['inter0_frac_min'] - i0 / min(len(va), len(vb))) < 1e-5
    assert abs(f['centroid_dist'] - float(np.linalg.norm(a_pts.mean(0) - b_pts.mean(0)))) < 1e-4
    lo, hi = np.maximum(a_pts.min(0), b_pts.min(0)), np.minimum(a_pts.max(0), b_pts.max(0))
    vol_i = np.prod(np.maximum(hi - lo, 0))
    vol_u = np.prod(np.maximum(np.maximum(a_pts.max(0), b_pts.max(0)) - np.minimum(a_pts.min(0), b_pts.min(0)), 0))
    assert abs(f['bbox_iou'] - vol_i / vol_u) < 1e-4
    assert abs(f['feat_cos']) < 1e-6 and abs(f['frame_gap'] - 1.0) < 1e-6
    # swapping the endpoints leaves every symmetric feature unchanged and swaps containment
    g = dict(zip(names, pair_features(stats, b, a, inter[b, a], inter[b, a], size_a, size_b,
                                      torch.tensor([1.0]), torch.tensor([False]), torch.tensor([True]),
                                      torch.tensor([0.0]))[0].tolist()))
    for name in ['inter0_frac_min', 'inter1_frac_min', 'iou0', 'centroid_dist', 'height_diff',
                 'log_size_ratio', 'extent_ratio', 'bbox_iou', 'feat_cos', 'frame_gap', 'same_frame',
                 'has_intra', 'frozen_count']:
        assert abs(g[name] - f[name]) < 1e-6, name
    assert abs(g['contain_a_in_b'] - f['contain_b_in_a']) < 1e-6


def test_node_geom_matches_graph():
    rng = np.random.default_rng(1)
    n = 301                                                            # odd -> median unambiguous
    xyz = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, (n, 3)).astype(np.uint8)
    groups = np.zeros(n, np.int64)
    groups[n // 2:] = 1
    xyz[n // 2:] += 10.0                                               # two groups, no contact pairs
    feat = rng.normal(size=(2, 8)).astype(np.float32)
    g = build_graph(xyz, rgb, groups, feat)
    z_ref = float(np.median(xyz[:, 2]))
    fr = FragmentFrame('s', 0, xyz, rgb, groups, feat, np.zeros((0, 2), np.int64),
                       np.zeros((0, 14), np.float32), np.zeros(2, np.float32), np.zeros(n, np.int64))
    got = stats_from_frame(fr, 0, 'cpu').geom(z_ref).numpy()
    assert np.allclose(got, g.node_geom, atol=1e-4)
    # the direct single-node path agrees with the vectorised per-fragment path
    sel = groups == 0
    one = NodeStats.from_points(xyz[sel], rgb[sel], feat=feat[0]).geom(z_ref)[0].numpy()
    assert np.allclose(one, got[0], atol=1e-4)


def _synth_frames(num_objs=3, num_frames=8, points=200, seed=0):
    """3 objects 1.5 m apart, each split into two touching fragments, one frame per step."""
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(num_frames):
        xyz, groups, gt = [], [], []
        for k in range(num_objs):
            for h in range(2):
                corner = np.array([1.5 * k - 0.2 + 0.2 * h, -0.2, -0.2]) + rng.normal(0, 0.01, 3)
                xyz.append(corner + rng.uniform(0, 0.2, (points, 3)))
                groups.append(np.full(points, 2 * k + h))
                gt.append(np.full(points, k + 1))
        xyz, groups, gt = np.concatenate(xyz).astype(np.float32), np.concatenate(groups), np.concatenate(gt)
        num = 2 * num_objs
        feat = np.zeros((num, 4), np.float32)
        for k in range(num_objs):
            feat[2 * k, k] = feat[2 * k + 1, k] = 1.0
        edges = np.array([[2 * k, 2 * k + 1] for k in range(num_objs)], np.int64)
        frames.append(FragmentFrame('scene', t, xyz, np.full_like(xyz, 128), groups, feat, edges,
                                    np.zeros((len(edges), 14), np.float32),
                                    np.full(num, 0.9, np.float32), gt))
    return frames


def test_window_memory_objects_and_revision():
    frames = _synth_frames()
    mem = WindowMemory(WindowConfig(window=5, past_unit='fragment'),
                       score_fn=oracle_scores, device='cpu')
    for fr in frames[:5]:
        mem.update(fr)
    masks, _ = mem.render(frames[4].xyz)
    assert len(masks) == 3
    for k in range(3):
        sel = frames[4].gt == k + 1
        assert int((masks[:, sel].sum(1) > 0.9 * sel.sum()).sum()) == 1

    # wrong glue inside the window: fragment 0 of frame 3 is moved to another object and must be
    # corrected by the next update (its node is pure, the oracle re-glues it to its own object)
    ent = [e for e in mem.window if e['pos'] == 3][0]
    obj_a, obj_b = int(ent['frags'][0]['obj']), int(ent['frags'][2]['obj'])
    assert obj_a != obj_b
    ent['frags'][0]['obj'] = obj_b
    mem.update(frames[5])
    ent = [e for e in mem.window if e['pos'] == 3][0]
    assert int(ent['frags'][0]['obj']) == obj_a

    # wrong glue outside the window: frame 1 slides out before the next update -> frozen in obj_b
    ent = [e for e in mem.window if e['pos'] == 1][0]
    ent['frags'][0]['obj'] = obj_b
    frag_vox = torch.unique(voxelize(torch.as_tensor(frames[1].xyz[frames[1].groups == 0], dtype=torch.float32), VOXEL))
    mem.update(frames[6])
    mem.update(frames[7])
    assert len(mem.objects) == 3
    assert bool(torch.isin(frag_vox, mem.objects[obj_b]['vox']).all())


def test_window_sliding_votes_conserved():
    frames = _synth_frames(num_frames=6, seed=1)
    mem = WindowMemory(WindowConfig(window=2), score_fn=oracle_scores, device='cpu')
    expected = 0
    for fr in frames:
        labels = mem.update(fr)
        for rec in mem.objects.values():                               # no duplicate (object, voxel) records
            assert len(torch.unique(rec['vox'])) == len(rec['vox'])
        for f in range(len(fr.node_feat)):                             # one vote per fragment voxel
            if labels[fr.groups == f][0] >= 0:
                expected += len(np.unique(voxelize(fr.xyz[fr.groups == f].astype(np.float32), VOXEL)))
        total = int(sum(int(rec['votes'].sum()) for rec in mem.objects.values()))
        total += sum(len(f2['vox']) for e in mem.window for f2 in e['frags'])
        assert total == expected
    assert len(mem.objects) == 3 and len(mem.window) == 2


def test_gt_assoc_memory():
    frames = _synth_frames(num_frames=3)
    mem = GtAssocMemory(purity=0.8, voxel=VOXEL, device='cpu')
    for fr in frames:
        mem.update(fr)
    masks, _ = mem.render(frames[-1].xyz)
    assert len(masks) == 3


def test_esam_online_merge_rule():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 0.5, (300, 3)).astype(np.float32)
    b = (a + 0.02).astype(np.float32)
    c = rng.uniform(3, 3.5, (300, 3)).astype(np.float32)
    feat = np.array([[1.0, 0, 0, 0]], np.float32)

    def fr(points, score, fid):
        return FrameInput(points, np.ones((1, len(points)), bool), np.array([score], np.float32), feat, fid)

    m = EsamOnlineMerge(voxel=VOXEL, device='cpu')
    m.update(fr(a, 0.9, 0))
    m.update(fr(b, 0.8, 1))                                            # same object, shifted: merges
    masks, _ = m.render(a)
    assert len(masks) == 1 and masks[0].mean() > 0.9
    m2 = EsamOnlineMerge(voxel=VOXEL, device='cpu')
    m2.update(fr(a, 0.9, 0))
    m2.update(fr(c, 0.9, 1))                                           # disjoint: stays two instances
    assert len(m2.render(a)[0]) == 2
