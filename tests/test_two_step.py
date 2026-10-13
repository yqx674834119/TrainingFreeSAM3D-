"""Stage-1 two-step grouping: intra-frame blocks, then a directed block -> object association.

CPU and offline (no GPU, no weights). The end-to-end case is the failure mode stage 0c diagnosed:
a frame's fragments must be glued to the object they overlap even though the *cross-frame* score
never enters the clustering, and the block is compared as a whole.
"""
import numpy as np
import torch

from freesam3d.streaming.two_step import (assign_blocks, block_features, block_object_scores,
                                          block_voxel_sets, intra_blocks, object_candidates)
from freesam3d.streaming.voxel import voxelize
from freesam3d.streaming.window_memory import FragmentFrame, WindowConfig, WindowMemory

VOXEL = 0.05


def _cells(pts, voxel=VOXEL):
    return torch.unique(voxelize(torch.as_tensor(np.asarray(pts, np.float32)), voxel))


def _frame(frag_pts, edges, feat, gt=None, conf=0.9, fid=0):
    """One FragmentFrame from a list of per-fragment point sets."""
    xyz = np.concatenate([np.asarray(p, np.float32) for p in frag_pts])
    groups = np.concatenate([np.full(len(p), i, np.int64) for i, p in enumerate(frag_pts)])
    num = len(frag_pts)
    gt = np.zeros(len(xyz), np.int64) if gt is None else np.asarray(gt, np.int64)
    return FragmentFrame('scene', fid, xyz, np.full_like(xyz, 128), groups,
                         np.asarray(feat, np.float32), np.asarray(edges, np.int64).reshape(-1, 2),
                         np.zeros((len(np.asarray(edges).reshape(-1, 2)), 14), np.float32),
                         np.full(num, conf, np.float32), gt)


def _intra_only(graph):
    """Score every cached intra-frame edge 1.0 and everything else 0."""
    return torch.as_tensor(np.asarray(graph.has_intra, np.float32))


# ------------------------------------------------------------------ pure pieces
def test_intra_blocks_uses_only_the_given_edges():
    # fragments 0-1 glued, 2-3 glued, and the 1-2 edge is *not* in the intra subgraph
    block = intra_blocks(4, [[0, 1], [2, 3]], [0.9, 0.9], 0.5)
    assert block.tolist() == [0, 0, 1, 1]


def test_block_voxel_sets_is_the_union_of_the_members():
    a = np.array([[0.01, 0, 0], [0.02, 0, 0], [0.11, 0, 0]], np.float32)
    b = np.array([[0.12, 0, 0]], np.float32)
    own, vox = block_voxel_sets([0, 0], [_cells(a), _cells(b)], 'cpu')
    assert own.tolist() == [0] * len(vox)
    assert vox.tolist() == sorted(set(_cells(a).tolist()) | set(_cells(b).tolist()))


def test_block_features_are_point_weighted_and_normalised():
    feat = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
    out = block_features([0, 0], 1, feat, [300, 100], 'cpu')
    raw = np.array([0.75, 0.25])                              # 300:100 point-weighted mean
    assert np.allclose(out[0].numpy(), raw / np.linalg.norm(raw), atol=1e-6)
    assert abs(float(out.norm()) - 1.0) < 1e-6


def test_object_candidates_merge_the_nodes_of_one_object():
    vox = [_cells(np.array([[0.01, 0, 0]], np.float32)), _cells(np.array([[0.31, 0, 0]], np.float32))]
    n_own = torch.tensor([0, 1], dtype=torch.int64)
    n_vox = torch.cat(vox)
    ids, c_own, c_vox, c_feat = object_candidates([7, 7], n_own, n_vox,
                                                  np.array([[1.0, 0.0], [0.0, 1.0]], np.float32),
                                                  [100, 300], 'cpu')
    assert ids.tolist() == [7] and c_own.tolist() == [0, 0]
    assert sorted(c_vox.tolist()) == sorted(torch.cat(vox).tolist())
    raw = np.array([0.25, 0.75])                              # 300 of 400 points carry the second feature
    assert np.allclose(c_feat[0].numpy(), raw / np.linalg.norm(raw), atol=1e-6)
    # current fragments (-1) contribute no candidate at all
    ids2, *_ = object_candidates([-1, -1], n_own, n_vox, np.zeros((2, 2), np.float32), [1, 1], 'cpu')
    assert len(ids2) == 0


def test_block_object_scores_containment_and_p2b_gate():
    block = _cells(np.random.default_rng(0).uniform(0, 0.4, (400, 3)))
    inside = _cells(np.random.default_rng(1).uniform(0, 0.4, (400, 3)))
    b_own = torch.zeros(len(block), dtype=torch.int64)
    c_own = torch.zeros(len(inside), dtype=torch.int64)
    b_feat = torch.tensor([[1.0, 0.0]])
    for rule, cos, want in (('p2a', 1.0, 'contain'), ('p2b', 1.0, 'contain'), ('p2b', -1.0, 0.0)):
        c_feat = torch.tensor([[cos, float(np.sqrt(max(0.0, 1 - cos * cos)))]])
        s = float(block_object_scores(b_own, block, b_feat, c_own, inside, c_feat, rule)[0, 0])
        assert s > 0.5 if want == 'contain' else s == 0.0, (rule, cos, s)
    try:
        block_object_scores(b_own, block, b_feat, c_own, inside, b_feat, 'p2c')
    except ValueError:
        pass
    else:
        raise AssertionError('an unknown rule must be rejected')


def test_assign_blocks_threshold_new_object_and_point_floor():
    score = torch.tensor([[0.9, 0.1], [0.2, 0.3], [0.1, 0.2]])
    ids = np.array([5, 9], np.int64)
    made = []

    def new_object():
        made.append(100 + len(made))
        return made[-1]

    # row 0 accepts object 5; row 1 misses the threshold and has enough points -> new; row 2 too small -> -1
    out = assign_blocks(score, ids, np.array([500, 500, 10]), 0.5, 100, new_object)
    assert out.tolist() == [5, 100, -1] and made == [100]
    # no candidate at all: everything is new or unassigned
    out = assign_blocks(torch.zeros((2, 0)), np.zeros(0, np.int64), np.array([500, 500]), 0.5, 100, new_object)
    assert out.tolist() == [101, 102]


# ------------------------------------------------------------------ end to end
def _sequence():
    """Frame 0: object A cut into two touching fragments + a far fragment. Frame 1: A again + a new place."""
    rng = np.random.default_rng(0)
    a1 = np.stack([rng.uniform(0.0, 0.10, 400), rng.uniform(0.0, 0.20, 400), rng.uniform(0.0, 0.20, 400)], 1)
    a2 = np.stack([rng.uniform(0.10, 0.20, 400), rng.uniform(0.0, 0.20, 400), rng.uniform(0.0, 0.20, 400)], 1)
    far = rng.uniform(5.0, 5.2, (200, 3))
    f0 = _frame([a1, a2, far], [[0, 1]], np.array([[1, 0], [1, 0], [0, 1]], np.float32), fid=0)
    whole = rng.uniform(0.0, 0.20, (400, 3))                 # A observed whole, one fragment
    other = rng.uniform(7.0, 7.2, (200, 3))
    f1 = _frame([whole, other], [], np.array([[1, 0], [0, 1]], np.float32), fid=1)
    return f0, f1


def test_two_step_joins_blocks_to_the_object_they_cover():
    f0, f1 = _sequence()
    cfg = WindowConfig(window=1, grouping='two_step', threshold=0.5, cross_rule='p2a',
                       cross_threshold=0.5, admit_min_points=100)
    mem = WindowMemory(cfg, score_fn=_intra_only, device='cpu')
    lab0 = mem.update(f0)
    assert len(mem.objects) == 2                              # block {a1,a2} -> one object, far -> another
    assert lab0[0] == lab0[400] and lab0[0] != lab0[800]
    lab1 = mem.update(f1)
    assert len(mem.objects) == 3                              # the whole-of-A block joined A, `other` opened one
    assert lab1[0] == lab0[0] and lab1[400] != lab0[0]


def test_two_step_threshold_gates_the_join():
    f0, f1 = _sequence()
    cfg = WindowConfig(window=1, grouping='two_step', threshold=0.5, cross_rule='p2a',
                       cross_threshold=1.01, admit_min_points=100)   # nothing can pass
    mem = WindowMemory(cfg, score_fn=_intra_only, device='cpu')
    lab0 = mem.update(f0)
    lab1 = mem.update(f1)
    assert len(mem.objects) == 4 and lab1[0] != lab0[0]


def test_two_step_allows_several_blocks_into_one_object():
    """Two fragments of the same frame that both sit inside A join A instead of opening two objects."""
    rng = np.random.default_rng(3)
    f0, _ = _sequence()
    half1 = rng.uniform(0.0, 0.20, (200, 3))
    half2 = rng.uniform(0.0, 0.20, (200, 3))
    f1 = _frame([half1, half2], [], np.array([[1, 0], [1, 0]], np.float32), fid=1)
    cfg = WindowConfig(window=1, grouping='two_step', threshold=0.5, cross_rule='p2a',
                       cross_threshold=0.5, admit_min_points=100)
    mem = WindowMemory(cfg, score_fn=_intra_only, device='cpu')
    lab0 = mem.update(f0)
    lab1 = mem.update(f1)
    assert lab1[0] == lab1[200] == lab0[0]
    assert len(mem.objects) == 2
