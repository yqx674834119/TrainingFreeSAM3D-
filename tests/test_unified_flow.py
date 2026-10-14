"""Unified per-frame grouping (stage 2) on synthetic sequences: blocks, block -> object joins,
object <-> object merges, the point floor and the node-quality ranking.

CPU and offline. The model is a scripted stub (``contain_a_in_b`` decides the block/object edges,
``edge_type == 2`` decides the object-object edges), so every assertion is about the *flow*, not
about learned weights.
"""
import numpy as np
import torch

from freesam3d.streaming.pair_features import pair_feature_names
from freesam3d.streaming.unified_flow import UnifiedGrouping, block_stats, object_state
from freesam3d.streaming.window_memory import FragmentFrame, WindowConfig, WindowMemory, stats_from_frame

_CA = pair_feature_names().index('contain_a_in_b')


class _StubModel:
    """Edge logit = 10 * (contain_a_in_b - 0.5); object <-> object edges take ``oo_logit``.

    ``contain_a_in_b`` is the share of the first node's cells inside the second, so the stub
    reproduces "join the thing you are contained in", with a probability that varies with how
    contained the node is - which is what the threshold tests need. The node quality head returns
    log1p(n_frames), i.e. quality grows with the number of observations.
    """

    def __init__(self, oo_logit=None):
        self.oo_logit = oo_logit

    def __call__(self, geom, feat, edges, edge_feat, node_type, n_frames, age, edge_type):
        x = edge_feat[:, _CA] if len(edges) else edge_feat.new_zeros(0)
        logit = 10.0 * (x - 0.5)
        if self.oo_logit is not None:
            logit = torch.where(edge_type == 2, torch.full_like(logit, float(self.oo_logit)), logit)
        return logit, torch.log1p(n_frames)


class _Std:
    def apply(self, geom, ef):
        return geom, ef


def _frame(frag_pts, edges=(), feat=None, conf=0.9, fid=0):
    xyz = np.concatenate([np.asarray(p, np.float32) for p in frag_pts])
    groups = np.concatenate([np.full(len(p), i, np.int64) for i, p in enumerate(frag_pts)])
    num = len(frag_pts)
    feat = np.zeros((num, 4), np.float32) if feat is None else np.asarray(feat, np.float32)
    e = np.asarray(edges, np.int64).reshape(-1, 2)
    return FragmentFrame('scene', fid, xyz, np.full_like(xyz, 128), groups, feat, e,
                         np.zeros((len(e), 14), np.float32), np.full(num, conf, np.float32),
                         np.zeros(len(xyz), np.int64))


def _memory(model=None, threshold=0.5, cross_threshold=0.5, **kw):
    cfg = WindowConfig(window=1, grouping='unified', threshold=threshold,
                       cross_threshold=cross_threshold, admit_min_points=100, **kw)
    return WindowMemory(cfg, unified=UnifiedGrouping(model or _StubModel(), _Std(), 'cpu'), device='cpu')


def _two_frames():
    """Frame 0: A cut into two overlapping fragments (one intra edge) + a far fragment.
    Frame 1: A seen whole (one fragment) + another far fragment."""
    rng = np.random.default_rng(0)
    a1 = np.stack([rng.uniform(0.00, 0.15, 400), rng.uniform(0, 0.2, 400), rng.uniform(0, 0.2, 400)], 1)
    a2 = np.stack([rng.uniform(0.05, 0.20, 400), rng.uniform(0, 0.2, 400), rng.uniform(0, 0.2, 400)], 1)
    far = rng.uniform(5.0, 5.2, (200, 3))
    f0 = _frame([a1, a2, far], [[0, 1]], fid=0)
    whole = rng.uniform(0.0, 0.2, (400, 3))
    other = rng.uniform(7.0, 7.2, (200, 3))
    return f0, _frame([whole, other], [], fid=1)


def _adjacent_objects():
    """Frame 0: two adjacent fragments with no intra edge -> two touching objects.
    Frame 1: a fragment over the first one, so both objects are in the frame's graph."""
    rng = np.random.default_rng(1)
    left = np.stack([rng.uniform(0.00, 0.10, 400), rng.uniform(0, 0.2, 400), rng.uniform(0, 0.2, 400)], 1)
    right = np.stack([rng.uniform(0.10, 0.20, 400), rng.uniform(0, 0.2, 400), rng.uniform(0, 0.2, 400)], 1)
    near = np.stack([rng.uniform(0.00, 0.10, 300), rng.uniform(0, 0.2, 300), rng.uniform(0, 0.2, 300)], 1)
    return _frame([left, right], [], fid=0), _frame([near], [], fid=1)


# ------------------------------------------------------------------ pure pieces
def test_block_stats_sum_the_members():
    rng = np.random.default_rng(3)
    pts = [rng.uniform(0, 0.2, (n, 3)) for n in (50, 120, 300)]
    xyz = np.concatenate([p.astype(np.float32) for p in pts])
    groups = np.concatenate([np.full(len(p), i) for i, p in enumerate(pts)])
    fr = FragmentFrame('s', 0, xyz, np.full_like(xyz, 128), groups, np.zeros((3, 4), np.float32),
                       np.zeros((0, 2), np.int64), np.zeros((0, 14), np.float32), np.full(3, 0.5, np.float32))
    stats = stats_from_frame(fr, 3, 'cpu')
    out = block_stats(np.array([0, 0, 1]), stats, 2)
    manual = stats.select(0).add(stats.select(1))
    assert int(out.n[0]) == 170 and int(out.n[1]) == 300
    assert np.allclose(out.sum_xyz[0].numpy(), manual.sum_xyz[0].numpy())
    assert np.allclose(out.lo[0].numpy(), np.minimum(stats.lo[0], stats.lo[1]).numpy())
    assert np.allclose(out.hi[0].numpy(), np.maximum(stats.hi[0], stats.hi[1]).numpy())
    assert abs(float(out.n_frames[0]) - 2.0) < 1e-6 and abs(float(out.last[0]) - 3.0) < 1e-6


def test_object_state_covers_frozen_plus_window():
    f0, f1 = _two_frames()
    mem = _memory()
    mem.update(f0)
    ids, stats, vox = object_state(mem)
    assert len(ids) == 2 and int(stats.n.sum()) == 1000            # 800 (A) + 200 (far)
    mem.update(f1)
    ids, stats, vox = object_state(mem)
    assert int(stats.n.sum()) == 1600                              # + 400 (A again) + 200 (new)


# ------------------------------------------------------------------ the flow
def test_unified_forms_blocks_and_joins_the_touching_object():
    f0, f1 = _two_frames()
    mem = _memory()
    lab0 = mem.update(f0)
    assert len(mem.objects) == 2                                   # {a1,a2} -> one object, far -> another
    assert lab0[0] == lab0[400] and lab0[0] != lab0[800]
    info = mem.last_unified
    assert info['num_blocks'] == 2 and set(np.unique(info['edge_type']).tolist()) <= {1, 2}
    assert len(info['prob_intra']) == 1                            # the frame's single cached intra edge
    lab1 = mem.update(f1)
    assert len(mem.objects) == 3                                   # A again, and `other` opened one
    assert lab1[0] == lab0[0] and lab1[400] != lab0[0]


def test_unified_block_object_threshold_gates_the_join():
    f0, f1 = _two_frames()
    mem = _memory(cross_threshold=1.01)                            # nothing can pass
    lab0 = mem.update(f0)
    lab1 = mem.update(f1)
    assert len(mem.objects) == 4 and lab1[0] != lab0[0]


def test_unified_merges_two_touching_objects():
    f0, f1 = _adjacent_objects()
    mem = _memory(_StubModel(oo_logit=6.0))
    lab0 = mem.update(f0)
    assert len(mem.objects) == 2 and lab0[0] != lab0[400]
    keep = int(lab0[0])
    lab1 = mem.update(f1)
    assert len(mem.objects) == 1 and lab1[0] == keep               # merged into the smaller id
    assert mem.last_unified['merged'] and mem.last_unified['merged'][0]['keep'] == keep
    assert int(mem.object_attrs()['n_pts'][0]) == 1100


def test_unified_object_object_threshold_gates_the_merge():
    f0, f1 = _adjacent_objects()
    mem = _memory(_StubModel(oo_logit=6.0), oo_threshold=1.01)
    mem.update(f0)
    mem.update(f1)
    assert len(mem.objects) == 2


def test_unified_point_floor_blocks_a_new_object():
    rng = np.random.default_rng(2)
    big = rng.uniform(0, 0.2, (400, 3))
    tiny = rng.uniform(5.0, 5.2, (50, 3))                          # below admit_min_points
    mem = _memory()
    lab = mem.update(_frame([big, tiny], [], fid=0))
    assert len(mem.objects) == 1 and lab[0] >= 0 and lab[400] == -1


def test_unified_quality_grows_with_observations_and_is_exposed():
    f0, f1 = _two_frames()
    mem = _memory()
    mem.update(f0)
    mem.update(f1)
    attrs = mem.object_attrs()
    assert set(attrs) == {'ids', 'n_pts', 'imp_pts', 'n_obs', 'conf', 'quality'}
    q = attrs['quality']
    assert np.isfinite(q).all() and (q > 0).all() and (q < 1).all()
    a = int(mem.window[-1]['frags'][0]['obj'])                     # the object A came back to
    row_a = int(np.flatnonzero(attrs['ids'] == a)[0])
    others = [i for i, oid in enumerate(attrs['ids']) if oid != a]
    assert q[row_a] > max(q[i] for i in others)                    # seen twice > seen once
    assert int(attrs['n_obs'][row_a]) == 2


def test_unified_grouping_requires_the_grouping_object():
    mem = WindowMemory(WindowConfig(grouping='unified'), device='cpu')
    try:
        mem.update(_two_frames()[0])
    except RuntimeError as e:
        assert 'UnifiedGrouping' in str(e)
    else:
        raise AssertionError('unified grouping without a UnifiedGrouping must fail loudly')


def test_single_frame_adapter_fills_the_single_frame_inputs():
    """The adapter must hand the unified model exactly the single-frame constants."""
    from freesam3d.streaming.unified_flow import SingleFrameAdapter
    from freesam3d.merging.model import UnifiedMergeNet
    torch.manual_seed(0)
    net = UnifiedMergeNet(11, 4, 29, hidden=8, feature_proj=4, dropout=0.0)
    ad = SingleFrameAdapter(net)
    geom = torch.rand(3, 11)
    feat = torch.rand(3, 4)
    edges = torch.tensor([[0, 1], [1, 2]])
    ef = torch.rand(2, 29)
    logit, q = ad(geom, feat, edges, ef, return_objectness=True)
    want_logit, want_q = net(geom, feat, edges, ef, torch.zeros(3, dtype=torch.int64),
                             torch.ones(3), torch.zeros(3), torch.zeros(2, dtype=torch.int64))
    assert torch.allclose(logit, want_logit) and torch.allclose(q, want_q)
    assert ad.has_objectness and ad.num_params() == net.num_params()
    assert ad(geom, feat, edges, ef).shape == (2,)
    assert ad(geom, feat, edges[:0], ef[:0], return_objectness=True)[1].shape == (3,)
