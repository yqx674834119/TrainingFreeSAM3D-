"""Stage-2 training (CPU, offline): frame batching, the balanced edge loss, node targets, one epoch."""
from pathlib import Path

import numpy as np
import torch

from freesam3d.streaming.unified_train import (GraphSet, _auc, build_unified_model, edge_loss,
                                               fit_standardizer, load_unified_checkpoint, node_loss,
                                               save_unified_checkpoint, train)
from freesam3d.streaming.window_data import pack, save

PC = dict(hidden=8, feature_proj=4, dropout=0.0, lr=0.01, weight_decay=0.0, epochs=1,
          frames_per_step=2, fit_files=4, edge_weight=1.0, node_weight=1.0, seeds=[0], init_from=None)
CFG = dict(phase2=dict(PC))
FEAT = 6


def _rows(rng, n_frames=2):
    rows_i, rows_c = [], []
    for _ in range(n_frames):
        rows_i.append(dict(
            node_geom=rng.random((4, 11), dtype=np.float32), node_feat=rng.random((4, FEAT), np.float32),
            n_frames=np.ones(4, np.float32), age=np.zeros(4, np.float32),
            edges=np.array([[0, 1], [2, 3]]), pair_feat=rng.random((2, 29), np.float32),
            node_type=np.zeros(4, np.int64), edge_type=np.zeros(2, np.int64),
            edge_y=np.array([1, 0], np.float32), edge_valid=np.array([True, False]),
            node_y=np.array([1, 0, 1, 0], np.float32), node_valid=np.ones(4, bool)))
        rows_c.append(dict(
            node_geom=rng.random((3, 11), dtype=np.float32), node_feat=rng.random((3, FEAT), np.float32),
            n_frames=np.ones(3, np.float32), age=np.zeros(3, np.float32),
            edges=np.array([[0, 2]]), pair_feat=rng.random((1, 29), np.float32),
            node_type=np.array([1, 1, 2]), edge_type=np.array([1], np.int64),
            edge_y=np.array([1.0], np.float32), edge_valid=np.array([True]),
            node_y=np.array([1.0, 0.0, np.nan], np.float32),
            node_valid=np.array([True, True, False])))
    return rows_i, rows_c


def _graphs(tmp_path, names=('s0', 's1'), n_frames=2):
    rng = np.random.default_rng(0)
    for name in names:
        rows_i, rows_c = _rows(rng, n_frames)
        # niou_c covers *every* cross node: the object node of each frame is the third of its frame
        niou = np.full(len(rows_c) * 3, np.nan, np.float32)
        niouv = np.zeros(len(rows_c) * 3, bool)
        niou[2::3] = 0.4 + 0.3 * np.arange(len(rows_c))
        niouv[2::3] = True
        data = pack(rows_i, rows_c, niou, niouv, scene=name, feat_dim=FEAT)
        assert len(data['niou_c']) == int(data['node_ptr_c'][-1])
        save(Path(tmp_path) / f'{name}.npz', data)
    return GraphSet([Path(tmp_path)], log=lambda *a: None)


# ------------------------------------------------------------------ batching
def test_graphset_stacks_frames_into_one_disjoint_graph(tmp_path):
    g = _graphs(tmp_path)
    assert len(g.files) == 2 and g.feat_dim == FEAT
    intra = [r for r in g.refs if r.tag == 'i']
    assert len(intra) == 4 and all(r.n_node == 4 and r.n_edge == 2 for r in intra)
    b = g.stack(intra[:2], 'cpu')
    assert b['geom'].shape == (8, 11) and b['pf'].shape == (4, 29) and b['feat'].shape == (8, FEAT)
    assert b['node_off'].tolist() == [0, 4]
    # the second frame's edges are shifted onto its own nodes
    assert b['edges'].tolist() == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert b['ev'].tolist() == [True, False, True, False]
    assert b['etype'].tolist() == [0, 0, 0, 0]


def test_batches_walk_every_frame_once(tmp_path):
    g = _graphs(tmp_path, names=('s0', 's1', 's2'), n_frames=3)
    seen = [r for batch in g.batches(4, seed=0) for r in batch]
    assert len(seen) == len(g.refs)
    assert {(r.file, r.frame, r.tag) for r in seen} == {(r.file, r.frame, r.tag) for r in g.refs}


# ------------------------------------------------------------------ losses
def test_edge_loss_balances_the_three_types():
    logit = torch.tensor([6.0, 6.0, 6.0, -6.0, -6.0])
    y = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])            # type 2's single edge is wrong
    valid = torch.ones(5, dtype=torch.bool)
    etype = torch.tensor([0, 0, 0, 1, 2])
    loss = edge_loss(logit, y, valid, etype)
    per_type = [float(edge_loss(logit[etype == t], y[etype == t], valid[etype == t], etype[etype == t]))
                for t in range(3)]
    assert abs(float(loss) - float(np.mean(per_type))) < 1e-6
    assert per_type[0] < 0.01 and per_type[2] > 5.0          # the rare type is not outvoted
    # masked edges and types without valid edges drop out instead of contributing 0
    loss2 = edge_loss(logit, y, torch.tensor([True, True, True, False, True]), etype)
    assert abs(float(loss2) - float(np.mean([per_type[0], per_type[2]]))) < 1e-6
    assert float(edge_loss(logit.new_zeros(0), y.new_zeros(0), valid.new_zeros(0), etype.new_zeros(0))) == 0.0


def test_node_loss_uses_the_sigmoid_and_the_valid_mask():
    q = torch.tensor([0.0, 3.0, 10.0])
    t = torch.tensor([0.5, 0.0, 1.0])
    v = torch.tensor([True, True, False])
    want = np.mean([(0.5 - 0.5) ** 2, (1 / (1 + np.exp(-3)) - 0.0) ** 2])
    assert abs(float(node_loss(q, t, v)) - want) < 1e-6


def test_auc_is_rank_based_and_handles_ties():
    assert _auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert _auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert abs(_auc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) - 0.5) < 1e-9
    assert np.isnan(_auc([0.5, 0.5], [1, 1]))


# ------------------------------------------------------------------ one epoch
def test_stack_uses_iou_for_objects_and_objectness_elsewhere(tmp_path):
    g = _graphs(tmp_path)
    cross = [r for r in g.refs if r.tag == 'c'][:1]
    b = g.stack(cross, 'cpu')
    obj = b['kind'] == 2
    assert abs(float(b['target'][obj][0]) - 0.4) < 1e-6 and b['nvalid'][obj].all()
    blk = b['kind'] == 1
    assert b['target'][blk].tolist() == [1.0, 0.0] and b['nvalid'][blk].all()


def test_standardizer_and_model_shapes(tmp_path):
    g = _graphs(tmp_path)
    std = fit_standardizer(g, limit=4, log=lambda *a: None)
    assert std.gm.shape == (11,) and std.em.shape == (29,)
    model = build_unified_model(CFG, g.feat_dim, 'cpu', log=lambda *a: None)
    assert model.num_params() < 100_000 and model.proj.in_features == FEAT


def test_train_one_epoch_and_checkpoint_round_trip(tmp_path):
    g = _graphs(tmp_path)
    model, std, info = train(g, CFG, 'cpu', seed=0, log=lambda *a: None)
    assert len(info['history']) == 1 and np.isfinite(info['history'][0]['loss'])
    path = Path(tmp_path) / 'model.pt'
    save_unified_checkpoint(path, model, std, CFG, extra=dict(seed=0))
    back, std2, ck = load_unified_checkpoint(path, 'cpu')
    assert ck['seed'] == 0 and np.allclose(std2.gm, std.gm)
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), back.named_parameters()):
        assert n1 == n2 and torch.allclose(p1, p2)
    b = g.stack([r for r in g.refs if r.tag == 'i'][:2], 'cpu')
    geom, ef = std2.apply(b['geom'], b['pf'])
    logit, q = back(geom, b['feat'], b['edges'], ef, b['kind'], b['n_frames'], b['age'], b['etype'])
    assert logit.shape == (4,) and q.shape == (8,)
