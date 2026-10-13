"""Scoring functions for the window memory that use no GT (stage 0c).

Stage 0 drove ``WindowMemory`` with the GT oracle, so its merge quality was a ceiling. Here the
edges get probabilities from signals that exist at test time:

* edges between two fragments of the *current* frame come from the trained ``GroupMergeNet``
  (``work_dirs/merge_net/full``) — the same model the per-frame pipeline uses, applied to the
  current-fragment subgraph exactly as during training (all cached intra-frame edges, one
  message-passing round);
* every other edge (current-vs-frozen, frozen-vs-frozen) uses a rule read straight off
  ``pair_features``: ``P2a`` = max containment in either direction, ``P2b`` = that times
  ``clamp(feat_cos, 0, 1)``.

The projection head is deliberately *not* used: the merge cache stores features with the frame
mean already subtracted, which is not the distribution the projection head was trained on.

``window_score_fn`` returns a callable ``graph -> (E,) float tensor`` for ``WindowMemory``.
"""
import numpy as np
import torch

from ..merging.graph import edge_feature_names
from .pair_features import cross_feature_names

# ``pair_feat`` columns, derived from the layout itself so the indices cannot drift:
# the 14 cached intra-frame features come first, then the 15 cross-frame ones.
CROSS_OFFSET = len(edge_feature_names())
_CROSS = {name: CROSS_OFFSET + i for i, name in enumerate(cross_feature_names())}
CONTAIN_A_IN_B = _CROSS['contain_a_in_b']
CONTAIN_B_IN_A = _CROSS['contain_b_in_a']
FEAT_COS = _CROSS['feat_cos']

RULES = ('p2a', 'p2b')


def num_current_fragments(graph):
    """Number of leading nodes that are current-frame fragments (they are always the prefix)."""
    return int(np.count_nonzero(np.asarray(graph.node_type) == 0))


def _bool_np(x):
    """``has_intra`` lives on the memory device; numpy needs a host copy."""
    x = x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return x.astype(bool, copy=False)


def load_group_merge_net(path, device='cpu'):
    """The trained per-frame merge model plus its feature standardizer (``GroupMergeNet full``)."""
    from ..merging.train import load_checkpoint
    model, std, ck = load_checkpoint(path, device)
    return model, std, ck


def rule_scores(graph, rule='p2a'):
    """(E,) cross-frame rule score for every edge, from the cached pair features alone.

    ``p2a`` = max(contain_a_in_b, contain_b_in_a); ``p2b`` = p2a * clamp(feat_cos, 0, 1).
    """
    if rule not in RULES:
        raise ValueError(f'unknown rule {rule!r}')
    pf = graph.pair_feat
    contain = torch.maximum(pf[:, CONTAIN_A_IN_B], pf[:, CONTAIN_B_IN_A])
    if rule == 'p2a':
        return contain
    return contain * pf[:, FEAT_COS].clamp(0.0, 1.0)


def intra_scores(graph, model, std, device='cpu'):
    """(E,) GroupMergeNet probability, filled only on the cached intra-frame edges (0 elsewhere).

    The model sees the current-fragment subgraph (nodes 0..n_cur-1, the cached edges among them),
    which reproduces its training-time input; nodes from the window / frozen objects are excluded
    because their statistics are aggregates the model never saw.
    """
    prob = torch.zeros(len(graph.edges), device=graph.pair_feat.device)
    sel = _bool_np(graph.has_intra)
    if not sel.any():
        return prob
    n_cur = num_current_fragments(graph)
    edges = torch.as_tensor(np.asarray(graph.edges)[sel], dtype=torch.int64, device=device)
    if len(edges) == 0:
        return prob
    assert int(edges.max()) < n_cur, 'a cached intra-frame edge points outside the current fragments'
    geom = torch.as_tensor(np.asarray(graph.node_geom[:n_cur]), dtype=torch.float32, device=device)
    feat = torch.as_tensor(np.asarray(graph.node_feat[:n_cur]), dtype=torch.float32, device=device)
    ef = graph.pair_feat[torch.as_tensor(sel, device=graph.pair_feat.device)][:, :CROSS_OFFSET].to(device)
    geom_s, ef_s = std.apply(geom, ef)
    with torch.no_grad():
        logit = model(geom_s, feat, edges, ef_s)
    prob[torch.as_tensor(sel, device=prob.device)] = torch.sigmoid(logit).to(prob.device)
    return prob


def window_score_fn(model=None, std=None, rule='p2a', device='cpu'):
    """``WindowMemory`` scorer: GroupMergeNet on intra-frame edges, ``rule`` on every other edge.

    Pass ``model=None`` to score everything with the rule (the P1-style geometry baseline).
    """
    if rule not in RULES:
        raise ValueError(f'unknown rule {rule!r}')

    def score_fn(graph):
        if model is None:
            return rule_scores(graph, rule)
        prob = intra_scores(graph, model, std, device)
        sel = _bool_np(graph.has_intra)
        cross = torch.as_tensor(~sel, device=prob.device)
        if bool(cross.any()):
            prob[cross] = rule_scores(graph, rule).to(prob.device)[cross]
        return prob

    return score_fn
