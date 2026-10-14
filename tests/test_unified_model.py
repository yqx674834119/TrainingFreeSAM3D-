"""UnifiedMergeNet: shapes, parameter budget, the node/edge-type inputs, and the GroupMergeNet init.

CPU and offline (no GPU, no trained weights): the checkpoint copy is exercised with a randomly
initialised ``GroupMergeNet``'s own state dict.
"""
import numpy as np
import torch

from freesam3d.merging.graph import EDGE_DIM, NODE_GEOM
from freesam3d.merging.model import GroupMergeNet, UnifiedMergeNet, init_from_group_merge_net
from freesam3d.streaming.pair_features import pair_feature_names

PAIR_DIM = len(pair_feature_names())
FEAT_DIM = 640


def _graph(num_nodes=7, num_edges=9, seed=0):
    rng = np.random.default_rng(seed)
    geom = torch.as_tensor(rng.normal(size=(num_nodes, NODE_GEOM)), dtype=torch.float32)
    feat = torch.as_tensor(rng.normal(size=(num_nodes, FEAT_DIM)), dtype=torch.float32)
    a = torch.as_tensor(rng.choice(num_nodes, size=(num_edges, 2), replace=True), dtype=torch.int64)
    a = a[a[:, 0] != a[:, 1]]
    edge_feat = torch.as_tensor(rng.normal(size=(len(a), PAIR_DIM)), dtype=torch.float32)
    node_type = torch.as_tensor(rng.integers(0, 3, num_nodes), dtype=torch.int64)
    n_frames = torch.as_tensor(rng.integers(1, 20, num_nodes), dtype=torch.float32)
    age = torch.as_tensor(rng.integers(0, 10, num_nodes), dtype=torch.float32)
    edge_type = torch.as_tensor(rng.integers(0, 3, len(a)), dtype=torch.int64)
    return geom, feat, a, edge_feat, node_type, n_frames, age, edge_type


def test_param_budget_and_output_shapes():
    model = UnifiedMergeNet(NODE_GEOM, FEAT_DIM, PAIR_DIM)
    assert model.num_params() < 100_000, model.num_params()
    geom, feat, edges, ef, nt, nf, age, et = _graph()
    logit, quality = model(geom, feat, edges, ef, nt, nf, age, et)
    assert logit.shape == (len(edges),) and quality.shape == (len(geom),)
    q = torch.sigmoid(quality)
    assert float(q.min()) > 0.0 and float(q.max()) < 1.0


def test_empty_edges_is_a_node_only_call():
    model = UnifiedMergeNet(NODE_GEOM, FEAT_DIM, PAIR_DIM)
    geom, feat, edges, ef, nt, nf, age, et = _graph(num_edges=0)
    logit, quality = model(geom, feat, edges, ef, nt, nf, age, et)
    assert logit.shape == (0,) and quality.shape == (len(geom),)


def test_node_and_edge_types_change_the_output():
    """The one-hots must actually reach the network: same geometry, different types -> different logits."""
    torch.manual_seed(0)
    model = UnifiedMergeNet(NODE_GEOM, FEAT_DIM, PAIR_DIM).eval()
    geom, feat, edges, ef, nt, nf, age, et = _graph()
    with torch.no_grad():
        base, _ = model(geom, feat, edges, ef, nt, nf, age, et)
        nt2 = (nt + 1) % 3
        other_node, _ = model(geom, feat, edges, ef, nt2, nf, age, et)
        et2 = (et + 1) % 3
        other_edge, _ = model(geom, feat, edges, ef, nt, nf, age, et2)
        nf2 = nf + 5.0
        other_age, _ = model(geom, feat, edges, ef, nt, nf2, age, et)
    assert not torch.allclose(base, other_node, atol=1e-6)
    assert not torch.allclose(base, other_edge, atol=1e-6)
    assert not torch.allclose(base, other_age, atol=1e-6)


def test_init_copies_the_shared_parts_of_group_merge_net():
    torch.manual_seed(0)
    src = GroupMergeNet(NODE_GEOM, FEAT_DIM, EDGE_DIM, hidden=64, feature_proj=16, message_passing=1)
    dst = UnifiedMergeNet(NODE_GEOM, FEAT_DIM, PAIR_DIM, hidden=64, feature_proj=16)
    copied = init_from_group_merge_net(dst, src.state_dict(), log=None)
    assert len(copied) == 18
    s, d = src.state_dict(), dst.state_dict()
    assert torch.equal(d['proj.weight'], s['proj.weight'])                    # exact
    assert torch.equal(d['msg.0.weight'], s['msg.0.0.weight'])                # renamed ModuleList entry
    assert torch.equal(d['head.weight'], s['head.weight'])
    assert torch.equal(d['node.0.weight'][:, :NODE_GEOM + 16], s['node.0.weight'])   # grown input
    assert torch.equal(d['edge.0.weight'][:, :2 * 64 + EDGE_DIM], s['edge.0.weight'])
    assert torch.equal(d['edge_norm.weight'][:EDGE_DIM], s['edge_norm.weight'])
    assert torch.equal(d['edge_norm.bias'][:EDGE_DIM], s['edge_norm.bias'])
    # the new columns keep a live init, and the untouched LayerNorm tail stays at weight 1 / bias 0
    assert float(d['edge_norm.weight'][EDGE_DIM:].sub(1).abs().max()) < 1e-6
    assert float(d['edge_norm.bias'][EDGE_DIM:].abs().max()) < 1e-6
    assert float(d['node.0.weight'][:, NODE_GEOM + 16:].abs().max()) > 0
    # the model still runs after the partial copy
    geom, feat, edges, ef, nt, nf, age, et = _graph()
    logit, quality = dst(geom, feat, edges, ef, nt, nf, age, et)
    assert logit.shape == (len(edges),) and quality.shape == (len(geom),)


def test_init_tolerates_a_missing_or_smaller_source():
    dst = UnifiedMergeNet(NODE_GEOM, FEAT_DIM, PAIR_DIM)
    copied = init_from_group_merge_net(dst, {}, log=None)
    assert copied == []
    src = GroupMergeNet(NODE_GEOM, FEAT_DIM, EDGE_DIM, hidden=32, feature_proj=8, message_passing=1)
    copied = init_from_group_merge_net(dst, src.state_dict(), log=None)      # widths differ: only proj-like exacts
    assert 'head.weight' not in copied and 'msg.0.weight' not in copied
