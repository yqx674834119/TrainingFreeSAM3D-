"""Stage 3 (CPU, offline): containment suppression, relational node features, candidate expansion,
join statistics, the ranking loss, the object-attention mask, and the phase-3 eval's GT reuse."""
import numpy as np
import torch

from freesam3d.merging.data import Standardizer
from freesam3d.merging.graph import NODE_GEOM
from freesam3d.merging.model import UnifiedMergeNet
from freesam3d.streaming.pair_features import (NodeStats, node_relations, pair_feature_names,
                                               rel_feature_names)
from freesam3d.streaming.phase3_diag import ranking_stats
from freesam3d.streaming.phase3_eval import eval_objects, pred_masks, voxel_gt
from freesam3d.streaming.unified_dedup import (containment_suppress, dedup_masks, dedup_objects,
                                               mask_containment, voxel_containment)
from freesam3d.streaming.unified_flow import N_PAIR, UnifiedGrouping
from freesam3d.streaming.pair_features import geom_pair_features  # noqa: E402
from freesam3d.streaming.unified_train import GraphSet, rank_loss
from freesam3d.streaming.voxel import voxelize
from freesam3d.streaming.window_data import pack, save
from freesam3d.streaming.window_memory import FragmentFrame, WindowConfig, WindowMemory

VOXEL = 0.05
R = len(rel_feature_names())
_CA = pair_feature_names().index('contain_a_in_b')


class _StubModel:
    """Edge logit = 10 * (contain_a_in_b - 0.5); quality = log1p(n_frames)."""

    def __call__(self, geom, feat, edges, edge_feat, node_type, n_frames, age, edge_type):
        x = edge_feat[:, _CA] if len(edges) else edge_feat.new_zeros(0)
        return 10.0 * (x - 0.5), torch.log1p(n_frames)


class _Std:
    def apply(self, geom, ef):
        return geom, ef


def _frame(frag_pts, edges=(), fid=0, gt=None, sem=None, feat=None):
    xyz = np.concatenate([np.asarray(p, np.float32) for p in frag_pts])
    groups = np.concatenate([np.full(len(p), i, np.int64) for i, p in enumerate(frag_pts)])
    num = len(frag_pts)
    feat = np.zeros((num, 4), np.float32) if feat is None else np.asarray(feat, np.float32)
    e = np.asarray(edges, np.int64).reshape(-1, 2)
    gt = np.zeros(len(xyz), np.int64) if gt is None else np.asarray(gt, np.int64)
    sem = np.full(len(xyz), 5, np.int64) if sem is None else np.asarray(sem, np.int64)   # 5 = a thing
    return FragmentFrame('scene', fid, xyz, np.full_like(xyz, 128), groups, feat, e,
                         np.zeros((len(e), 14), np.float32), np.full(num, 0.9, np.float32), gt, sem)


def _memory(model=None, threshold=0.5, cross_threshold=0.5, **kw):
    cfg = WindowConfig(window=1, grouping='unified', threshold=threshold,
                       cross_threshold=cross_threshold, admit_min_points=100, voxel=VOXEL, **kw)
    return WindowMemory(cfg, unified=UnifiedGrouping(model or _StubModel(), _Std(), 'cpu'), device='cpu')


# ------------------------------------------------------------------ containment suppression
def test_containment_suppress_drops_parts_not_wholes():
    scores = np.array([0.9, 0.5, 0.8], np.float32)
    contain = np.eye(3, dtype=np.float32)
    contain[1, 0] = 0.9                      # mask 1 lies inside the better mask 0
    keep = containment_suppress(scores, contain, 0.8)
    assert keep.tolist() == [0, 2]           # score order, mask 1 gone
    contain2 = np.eye(3, dtype=np.float32)
    contain2[0, 1] = 0.9                     # the *worse* mask contains the better one: both stay
    assert containment_suppress(scores, contain2, 0.8).tolist() == [0, 2, 1]   # score order
    assert containment_suppress(scores, contain, None).tolist() == [0, 1, 2]
    assert containment_suppress(scores, contain, 1.01).tolist() == [0, 1, 2]


def test_voxel_and_mask_containment_agree():
    sets = [torch.arange(0, 10), torch.arange(0, 9), torch.arange(20, 30)]
    masks = np.zeros((3, 30), bool)
    for i, v in enumerate(sets):
        masks[i, v.numpy()] = True
    a = voxel_containment(sets)
    b = mask_containment(masks)
    assert np.allclose(a, b, atol=1e-6)
    assert abs(float(a[1, 0]) - 1.0) < 1e-6 and abs(float(a[0, 1]) - 0.9) < 1e-6   # |i&j| / |i|


def test_dedup_objects_and_masks_suppress_the_part():
    whole = torch.arange(0, 20)
    part = torch.arange(0, 18)               # 90% inside `whole`
    other = torch.arange(40, 60)
    keep = dedup_objects([whole, part, other], np.array([0.9, 0.5, 0.8], np.float32), 0.8)
    assert keep.tolist() == [0, 2]
    masks = np.zeros((3, 60), bool)
    for i, v in enumerate((whole, part, other)):
        masks[i, v.numpy()] = True
    assert dedup_masks(masks, np.array([0.9, 0.5, 0.8], np.float32), 0.8).tolist() == [0, 2]


# ------------------------------------------------------------------ relational node features
def test_node_relations_aggregates_the_intersecting_objects():
    stats = NodeStats.zeros(4, 1, 'cpu')
    stats.n[:] = torch.tensor([10.0, 20.0, 5.0, 8.0])
    size = torch.tensor([10.0, 20.0, 5.0, 8.0])
    a = torch.tensor([0, 2, 2])
    b = torch.tensor([1, 0, 1])
    inter0 = torch.tensor([5.0, 5.0, 0.0])           # pair (2, 1) touches but does not intersect
    node_type = torch.tensor([2, 2, 1, 2])           # 0/1 objects, 2 a block, 3 an object
    join = torch.tensor([[2.0, 1.5, 0.6], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    rel = node_relations(stats, a, b, inter0, size, node_type, join)
    assert rel.shape == (4, R)
    # node 0: one object neighbour (node 1), iou 5/25, 50% of itself inside it, 25% of it inside itself
    assert abs(float(rel[0, 0]) - 0.2) < 1e-6 and abs(float(rel[0, 1]) - 0.5) < 1e-6
    assert abs(float(rel[0, 2]) - 0.25) < 1e-6 and abs(float(rel[0, 3]) - np.log1p(1)) < 1e-6
    assert float(rel[0, 4]) == 0.0                   # 10 < 20: not the largest
    assert abs(float(rel[0, 5]) - np.log1p(2)) < 1e-6 and abs(float(rel[0, 6]) - 0.75) < 1e-6
    assert abs(float(rel[0, 7]) - 0.6) < 1e-6
    # node 1: the same edge seen from the other side; 20 >= 10 -> largest; no joins
    assert abs(float(rel[1, 1]) - 0.25) < 1e-6 and abs(float(rel[1, 2]) - 0.5) < 1e-6
    assert float(rel[1, 4]) == 1.0 and float(rel[1, 5]) == 0.0
    # node 2 (block): only the object neighbour counts (the non-intersecting pair is dropped)
    assert abs(float(rel[2, 0]) - 0.5) < 1e-6 and abs(float(rel[2, 1]) - 1.0) < 1e-6
    assert abs(float(rel[2, 3]) - np.log1p(1)) < 1e-6
    assert float(rel[3].abs().max()) == 0.0          # no neighbours at all


# ------------------------------------------------------------------ flow: candidates, joins, rel
def test_flow_candidate_expansion_adds_non_contact_pairs():
    rng = np.random.default_rng(5)
    a = np.stack([rng.uniform(0, 0.1, 400), rng.uniform(0, 0.1, 400), rng.uniform(0, 0.1, 400)], 1)
    b = np.stack([rng.uniform(0.25, 0.35, 400), rng.uniform(0, 0.1, 400), rng.uniform(0, 0.1, 400)], 1)
    base = _memory()
    base.update(_frame([a], [], fid=0))
    base.update(_frame([b], [], fid=1))
    assert base.last_unified['num_blocks'] == 1
    assert len(base.last_unified['edges']) == 0          # 0.25 m apart: no contact even after dilation
    exp = _memory(cand_dist=0.3, cand_cos=-1.0)
    exp.update(_frame([a], [], fid=0))
    exp.update(_frame([b], [], fid=1))
    info = exp.last_unified
    assert len(info['edges']) == 1 and int(info['edge_type'][0]) == 1     # one block <-> object candidate
    assert info['cross']['node_rel'].shape == (2, R)


def _joined_memory(**kw):
    rng = np.random.default_rng(6)
    a = rng.uniform(0.0, 0.2, (400, 3))
    whole = a + rng.uniform(-0.01, 0.01, a.shape)
    mem = _memory(**kw)
    mem.update(_frame([a], [], fid=0))
    mem.update(_frame([whole], [], fid=1))
    return mem


def test_flow_tracks_join_statistics():
    mem = _joined_memory()
    oid = int(mem.object_attrs()['ids'][0])
    rec = mem.objects[oid]
    assert rec['join_n'] == 1
    assert 0.5 < rec['join_min'] <= 1.0                          # one join, one probability
    assert abs(rec['join_sum'] - rec['join_min']) < 1e-9


def test_flow_stores_node_rel_and_object_rows_stay_empty():
    mem = _joined_memory()
    rel = mem.last_unified['cross']['node_rel']
    assert rel.shape == (2, R)
    assert rel[0, 1] > 0.5                               # the block is mostly inside the object
    assert abs(float(rel[0, 3]) - np.log1p(1)) < 1e-5    # one intersecting object
    assert float(np.abs(rel[1]).max()) == 0.0            # the only object has no *other* object


# ------------------------------------------------------------------ ranking loss
def _rank_loss_ref(q, t, valid, fid):
    tot, n = 0.0, 0
    for i in range(len(q)):
        for j in range(len(q)):
            if fid[i] != fid[j] or not valid[i] or not valid[j] or t[i] == t[j]:
                continue
            tot += np.log1p(np.exp(-(q[i] - q[j]) * np.sign(t[i] - t[j])))
            n += 1
    return tot / n if n else 0.0


def test_rank_loss_matches_the_pairwise_reference():
    q = torch.tensor([2.0, 1.0, 0.0, 0.5])
    t = torch.tensor([0.9, 0.5, 0.1, 0.5])
    v = torch.tensor([True, True, True, False])
    fid = torch.tensor([0, 0, 0, 0])
    got = float(rank_loss(q, t, v, fid))
    assert abs(got - _rank_loss_ref(q.numpy(), t.numpy(), v.numpy(), fid.numpy())) < 1e-5
    inv = float(rank_loss(-q, t, v, fid))
    assert inv > got                                     # the inverted order must cost more


def test_rank_loss_respects_frame_boundaries():
    q = torch.tensor([1.0, 0.0, 1.0, 0.0])
    t = torch.tensor([0.1, 0.9, 0.9, 0.1])
    v = torch.ones(4, dtype=torch.bool)
    fid = torch.tensor([0, 0, 1, 1])
    split = float(rank_loss(q, t, v, fid))
    merged = float(rank_loss(q, t, v, torch.zeros(4, dtype=torch.long)))
    assert abs(split - _rank_loss_ref(q.numpy(), t.numpy(), v.numpy(), fid.numpy())) < 1e-5
    assert abs(merged - _rank_loss_ref(q.numpy(), t.numpy(), v.numpy(), np.zeros(4))) < 1e-5
    assert abs(split - merged) > 1e-6                    # ignoring the frame ids would change the loss
    ties = float(rank_loss(q, torch.ones(4), v, fid))
    assert ties == 0.0


# ------------------------------------------------------------------ model: rel input + attention
def test_model_rel_input_reaches_the_quality_head():
    torch.manual_seed(0)
    model = UnifiedMergeNet(NODE_GEOM, 8, N_PAIR, hidden=16, feature_proj=4, dropout=0.0,
                            rel_dim=R).eval()
    geom, feat = torch.rand(4, NODE_GEOM), torch.rand(4, 8)
    edges, ef = torch.tensor([[0, 1]]), torch.rand(1, N_PAIR)
    nt, nf, age, et = torch.tensor([1, 1, 2, 2]), torch.ones(4), torch.zeros(4), torch.tensor([1])
    rel = torch.zeros(4, R)
    rel2 = rel.clone(); rel2[2, 0] = 5.0
    with torch.no_grad():
        e0, q0 = model(geom, feat, edges, ef, nt, nf, age, et, rel)
        e1, q1 = model(geom, feat, edges, ef, nt, nf, age, et, rel2)
    assert torch.allclose(e0, e1)                        # the cues never reach the edge path
    assert not torch.allclose(q0, q1)
    assert torch.allclose(q0[:2], q1[:2])                # only node 2's input changed


def test_object_attention_stays_within_the_frame():
    torch.manual_seed(0)
    model = UnifiedMergeNet(NODE_GEOM, 8, N_PAIR, hidden=16, feature_proj=4, dropout=0.0,
                            attn_layers=1, attn_inner=8).eval()
    plain = UnifiedMergeNet(NODE_GEOM, 8, N_PAIR, hidden=16, feature_proj=4, dropout=0.0)
    assert model.num_params() - plain.num_params() < 10_000
    geom, feat = torch.rand(4, NODE_GEOM), torch.rand(4, 8)
    edges = torch.zeros((0, 2), dtype=torch.int64)
    ef = torch.zeros((0, N_PAIR))
    nt, nf, age = torch.tensor([2, 2, 2, 2]), torch.ones(4), torch.zeros(4)
    et = torch.zeros(0, dtype=torch.int64)
    fid = torch.tensor([0, 0, 1, 1])
    with torch.no_grad():
        _, q0 = model(geom, feat, edges, ef, nt, nf, age, et, None, fid)
        g2 = geom.clone(); g2[3] += 3.0
        _, q1 = model(g2, feat, edges, ef, nt, nf, age, et, None, fid)
    assert torch.allclose(q0[:2], q1[:2], atol=1e-6)     # frame 0 is untouched by frame 1
    assert not torch.allclose(q0[2:], q1[2:])


# ------------------------------------------------------------------ standardizer, storage, eval
def test_standardizer_rel_roundtrip():
    rng = np.random.default_rng(0)
    geom = rng.normal(size=(50, NODE_GEOM)).astype(np.float32)
    ef = rng.normal(size=(30, len(pair_feature_names()))).astype(np.float32)
    rel = (rng.random((40, R)) * 5).astype(np.float32)
    std = Standardizer(geom, ef, rel)
    st = std.state()
    assert 'rm' in st and 'rs' in st
    std2 = Standardizer.from_state(st)
    out = std2.apply_rel(torch.as_tensor(rel[:3]))
    assert out.shape == (3, R) and not torch.allclose(out, torch.as_tensor(rel[:3]))
    plain = Standardizer.from_state(Standardizer(geom, ef).state())
    assert plain.rm is None
    assert torch.equal(plain.apply_rel(torch.as_tensor(rel[:3])), torch.as_tensor(rel[:3]))


def _train_rows(rng, n=2, with_rel=True):
    rows_i, rows_c = [], []
    for _ in range(n):
        ri = dict(node_geom=rng.random((4, NODE_GEOM), dtype=np.float32),
                  node_feat=rng.random((4, 6), np.float32), n_frames=np.ones(4, np.float32),
                  age=np.zeros(4, np.float32), edges=np.array([[0, 1]]),
                  pair_feat=rng.random((1, len(pair_feature_names())), np.float32),
                  node_type=np.zeros(4, np.int64), edge_type=np.zeros(1, np.int64),
                  edge_y=np.ones(1, np.float32), edge_valid=np.ones(1, bool),
                  node_y=np.ones(4, np.float32), node_valid=np.ones(4, bool))
        rc = dict(node_geom=rng.random((3, NODE_GEOM), dtype=np.float32),
                  node_feat=rng.random((3, 6), np.float32), n_frames=np.ones(3, np.float32),
                  age=np.zeros(3, np.float32), edges=np.array([[0, 2]]),
                  pair_feat=rng.random((1, len(pair_feature_names())), np.float32),
                  node_type=np.array([1, 1, 2]), edge_type=np.array([1], np.int64),
                  edge_y=np.ones(1, np.float32), edge_valid=np.ones(1, bool),
                  node_y=np.array([1.0, 0.0, np.nan], np.float32),
                  node_valid=np.array([True, True, False]))
        if with_rel:
            ri['node_rel'] = rng.random((4, R), np.float32)
            rc['node_rel'] = np.full((3, R), 0.25, np.float32)
        rows_i.append(ri); rows_c.append(rc)
    return rows_i, rows_c


def test_graphset_loads_rel_and_zero_fills_old_files(tmp_path):
    rng = np.random.default_rng(1)
    rows_i, rows_c = _train_rows(rng, with_rel=True)
    data = pack(rows_i, rows_c, np.full(6, 0.4, np.float32), np.ones(6, bool), scene='s', feat_dim=6)
    assert data['rel_c'].shape == (6, R) and float(data['rel_c'].max()) == 0.25
    save(tmp_path / 'a.npz', data)
    data.pop('rel_i'); data.pop('rel_c')                 # simulate a stage-2 file
    save(tmp_path / 'b.npz', data)
    g = GraphSet([tmp_path], log=lambda *a: None)
    by_file = {g.files[i].split('/')[-1]: i for i in range(len(g.files))}
    cross = [r for r in g.refs if r.tag == 'c']
    b_a = g.stack([r for r in cross if r.file == by_file['a.npz']], 'cpu')
    b_b = g.stack([r for r in cross if r.file == by_file['b.npz']], 'cpu')
    assert b_a['rel'].shape == (6, R) and float(b_a['rel'].max()) == 0.25
    assert b_b['rel'].shape == (6, R) and float(b_b['rel'].abs().max()) == 0.0
    assert b_a['frame_id'].tolist() == [0, 0, 0, 1, 1, 1]


def _gt_frames():
    rng = np.random.default_rng(7)
    frames = []
    for t in range(3):
        pts, gt = [], []
        for k in range(2):
            pts.append(np.array([1.5 * k, 0.0, 0.0]) + rng.uniform(0, 0.2, (200, 3)))
            gt.append(np.full(200, k + 1))
        frames.append(_frame(pts, [], fid=t, gt=np.concatenate(gt)))
    return frames


def test_phase3_eval_matches_voxel_scene_eval():
    from freesam3d.streaming.evaluate import voxel_scene_eval
    frames = _gt_frames()
    obj_vox = []
    for k in range(2):
        pts = np.concatenate([fr.xyz[fr.gt == k + 1] for fr in frames])
        obj_vox.append(torch.unique(torch.as_tensor(voxelize(pts, VOXEL))))
    scores = np.array([0.9, 0.8], np.float32)
    want = voxel_scene_eval(obj_vox, scores, frames, {1, 2, 22}, voxel=VOXEL, min_gt_voxels=5,
                            min_pred_voxels=5, device='cpu')
    cells, gts = voxel_gt(frames, {1, 2, 22}, voxel=VOXEL, min_gt_voxels=5)
    got = eval_objects(pred_masks(obj_vox, cells), scores, gts, min_pred_voxels=5, device='cpu')
    for k in ('ap50', 'ap25', 'iou', 'r50', 'masks', 'dup_rate'):
        assert abs(float(got[k]) - float(want[k])) < 1e-6, (k, got[k], want[k])


def test_ranking_stats_separates_and_inverts():
    q = np.array([0.9, 0.8, 0.1, 0.2])
    iou = np.array([0.8, 0.6, 0.1, 0.2])
    st = ranking_stats(q, iou)
    assert st['spearman'] > 0.99 and st['auc50'] == 1.0 and st['pos_rate'] == 0.5
    st2 = ranking_stats(-q, iou)
    assert st2['spearman'] < -0.99 and st2['auc50'] == 0.0


def test_attention_frame_id_none_matches_single_frame():
    """Inference passes no frame_id: None must mean "one frame" for the object rows only."""
    from freesam3d.merging.model import ObjectAttention
    torch.manual_seed(0)
    attn = ObjectAttention(8, inner=4, dropout=0.0)
    h = torch.randn(7, 8)
    obj = torch.tensor([False, True, True, False, True, False, True])
    a = attn(h, obj, None)
    b = attn(h, obj, torch.zeros(7, dtype=torch.int64))
    assert a.shape == h.shape and torch.allclose(a, b, atol=1e-6)
    split = attn(h, obj, torch.tensor([0, 0, 0, 1, 1, 1, 1]))
    assert not torch.allclose(a[obj], split[obj], atol=1e-6)     # the mask really splits the frames
    assert torch.allclose(a[~obj], h[~obj], atol=1e-6)           # non-object rows pass through


def test_geom_pair_extends_edge_features_and_keeps_the_first_29():
    """阶段 5 · P4：geom_pair=True 时每条边追加 44 列成对几何，前 29 列逐位不变。"""
    def blob(x0):
        g = np.stack(np.meshgrid(np.linspace(0, 0.09, 6), np.linspace(0, 0.09, 6),
                                 np.linspace(0, 0.09, 4)), -1).reshape(-1, 3)
        return (g + np.array([x0, 0.0, 0.0], np.float32)).astype(np.float32)

    frames = [_frame([blob(0.0), blob(0.09)], fid=0), _frame([blob(0.09)], fid=1)]
    base = _memory()
    for fr in frames:
        base.update(fr)
    g0 = base.last_graph
    assert g0.pair_feat.shape[1] == 29
    ext = _memory(geom_pair=True)
    for fr in frames:
        ext.update(fr)
    g1 = ext.last_graph
    assert g1.pair_feat.shape[1] == 73 and len(g1.edges) > 0
    assert torch.equal(g1.pair_feat[:, :29], g0.pair_feat)
    gm = torch.as_tensor(g1.node_geom, dtype=torch.float32)
    e = torch.as_tensor(g1.edges, dtype=torch.int64)
    assert torch.allclose(g1.pair_feat[:, 29:], geom_pair_features(gm[e[:, 0]], gm[e[:, 1]]),
                          atol=1e-6)
