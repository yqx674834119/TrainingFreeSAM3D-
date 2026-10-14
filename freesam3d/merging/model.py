"""GroupMergeNet: a small edge classifier (+ node objectness head) on the group adjacency graph.

Per node:  h = MLP([geom(11), P·feat])     P: C -> feature_proj (learned)
Per edge:  e = MLP([h_a + h_b, |h_a - h_b|, edge_feat(14)])
Optional message passing: h <- h + mean_{edges} MLP(e), then recompute e.
Outputs:   edge logit (symmetric in a, b by construction)
           node logit = "this group belongs to a thing object" (objectness),
           used to score / filter merged masks without GT at test time.

``EdgeOnlyNet`` ignores node features entirely (the 14 edge scalars only);
it is the ablation that tells whether learned node embeddings help.
"""
import torch
from torch import nn


def _mlp(i, h, o, dropout=0.0):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, o), nn.ReLU())


class GroupMergeNet(nn.Module):
    has_objectness = True

    def __init__(self, geom_dim, feat_dim, edge_dim, hidden=64, feature_proj=16, message_passing=1, dropout=0.1,
                 use_feat=True, use_geom=True):
        super().__init__()
        self.use_feat, self.use_geom = use_feat and feat_dim > 0, use_geom
        self.proj = nn.Linear(feat_dim, feature_proj) if self.use_feat else None
        nd = (geom_dim if use_geom else 0) + (feature_proj if self.use_feat else 0)
        self.node = _mlp(max(nd, 1), hidden, hidden, dropout)
        self.edge = _mlp(2 * hidden + edge_dim, hidden, hidden, dropout)
        self.rounds = int(message_passing)
        self.msg = nn.ModuleList([_mlp(hidden, hidden, hidden, dropout) for _ in range(self.rounds)])
        self.head = nn.Linear(hidden, 1)
        self.obj_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.nd = nd

    def node_input(self, geom, feat):
        parts = []
        if self.use_geom:
            parts.append(geom)
        if self.use_feat:
            parts.append(self.proj(feat))
        return torch.cat(parts, 1) if parts else geom.new_zeros((len(geom), 1))

    def forward(self, geom, feat, edges, edge_feat, return_objectness=False):
        h = self.node(self.node_input(geom, feat))
        if len(edges) == 0:
            e_logit = h.new_empty((0,))
            return (e_logit, self.obj_head(h).squeeze(1)) if return_objectness else e_logit
        a, b = edges[:, 0].long(), edges[:, 1].long()
        ef = self.edge_norm(edge_feat)
        e = self.edge(torch.cat([h[a] + h[b], (h[a] - h[b]).abs(), ef], 1))
        for m in self.msg:
            agg = h.new_zeros(h.shape).index_add_(0, a, m(e)).index_add_(0, b, m(e))
            deg = torch.bincount(torch.cat([a, b]), minlength=len(h)).clamp_min(1).unsqueeze(1)
            h = h + agg / deg
            e = self.edge(torch.cat([h[a] + h[b], (h[a] - h[b]).abs(), ef], 1))
        e_logit = self.head(e).squeeze(1)
        return (e_logit, self.obj_head(h).squeeze(1)) if return_objectness else e_logit

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class EdgeOnlyNet(nn.Module):
    """Ablation: a 2-layer MLP on the edge scalars only (no node embedding, no objectness)."""
    has_objectness = False

    def __init__(self, edge_dim, hidden=32, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(edge_dim)
        self.net = nn.Sequential(nn.Linear(edge_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, geom, feat, edges, edge_feat, return_objectness=False):
        e = self.net(self.norm(edge_feat)).squeeze(1)
        return (e, geom.new_zeros(len(geom))) if return_objectness else e

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def build_model(kind, geom_dim, feat_dim, edge_dim, cfg):
    hidden = cfg.get('hidden', 32)
    if kind == 'edge_only':
        return EdgeOnlyNet(edge_dim, hidden, cfg.get('dropout', 0.0))
    node_hidden = max(hidden, 64)          # node models need width for the geometry embedding
    if kind == 'geom_only':
        return GroupMergeNet(geom_dim, 0, edge_dim, node_hidden, cfg['feature_proj'], cfg['message_passing'],
                             cfg['dropout'], use_feat=False)
    if kind == 'full':
        return GroupMergeNet(geom_dim, feat_dim, edge_dim, node_hidden, cfg['feature_proj'],
                             cfg['message_passing'], cfg['dropout'])
    raise KeyError(f'unknown model kind {kind!r}')


class ObjectAttention(nn.Module):
    """One self-attention round over the *object* nodes of each frame (stage 3, ranking only).

    Rows flagged in ``obj`` attend to each other within the same ``frame_id`` segment (``None`` =
    every node is in one frame, which is what inference does); all other rows pass through
    unchanged. Single head on a narrow inner width, so the whole layer stays in the
    few-thousand-parameter budget the stage-3 brief allows.
    """

    def __init__(self, dim, inner=32, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj_in = nn.Linear(dim, inner)
        self.q = nn.Linear(inner, inner, bias=False)
        self.k = nn.Linear(inner, inner, bias=False)
        self.v = nn.Linear(inner, inner, bias=False)
        self.proj_out = nn.Linear(inner, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, obj, frame_id=None):
        if not bool(obj.any()):
            return h
        z = self.proj_in(self.norm(h[obj]))
        q, k, v = self.q(z), self.k(z), self.v(z)
        logits = (q @ k.t()) * (q.shape[-1] ** -0.5)
        fid = (torch.zeros(len(h), dtype=torch.int64, device=h.device) if frame_id is None
               else frame_id.long())[obj]
        logits = logits.masked_fill(fid[:, None] != fid[None, :], float('-inf'))
        out = h[obj] + self.drop(self.proj_out(torch.softmax(logits, dim=1) @ v))
        return h.index_copy(0, torch.nonzero(obj).squeeze(1), out)


class UnifiedMergeNet(nn.Module):
    """One small model for the whole merge task (stage 2): intra-frame edges, block <-> object edges,
    object <-> object edges, plus a per-node *quality* head (predicted IoU with the GT instance,
    used as the object ranking score at output time).

    Node input : [node_geom(11), P . feat(C -> feature_proj), node-type one-hot(3),
                  log1p(n_frames), log1p(age)]; the stage-3 relational columns (``rel_dim``)
                  are added to the *quality* head's input only, never to the shared trunk
    Edge input : [h_a + h_b, |h_a - h_b|, LayerNorm(edge_feat), edge-type one-hot(3)]
                 ``edge_feat`` is the 29-column pair layout: the 14 cached intra-frame columns
                 (zero off intra-frame edges) followed by the 15 cross-frame columns, which are
                 computed for *every* edge.
    One message-passing round; both heads read the same (updated) node embedding.

    Edge direction is canonicalised by node index (a < b), so the asymmetric cross columns
    (contain_a_in_b vs contain_b_in_a) are well-defined. ``GroupMergeNet`` stays as the
    per-frame baseline; ``init_from_group_merge_net`` copies what the two share.
    """
    NODE_TYPES = 3       # 0 current-frame fragment, 1 current-frame block, 2 history object
    EDGE_TYPES = 3       # 0 intra-frame, 1 block <-> object, 2 object <-> object

    def __init__(self, geom_dim, feat_dim, edge_dim, hidden=64, feature_proj=16, dropout=0.1,
                 rel_dim=0, attn_layers=0, attn_inner=32):
        super().__init__()
        self.proj = nn.Linear(feat_dim, feature_proj)
        self.rel_dim = int(rel_dim)
        self.attn_layers = int(attn_layers)
        self.node = _mlp(geom_dim + feature_proj + self.NODE_TYPES + 2, hidden, hidden, dropout)
        self.rel_emb = nn.Linear(self.rel_dim, hidden) if self.rel_dim else None
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.edge = _mlp(2 * hidden + edge_dim + self.EDGE_TYPES, hidden, hidden, dropout)
        self.msg = _mlp(hidden, hidden, hidden, dropout)
        self.head = nn.Linear(hidden, 1)
        self.quality = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.attn = nn.ModuleList(ObjectAttention(hidden, attn_inner, dropout)
                                  for _ in range(self.attn_layers))
        self.node_dim = geom_dim + feature_proj + self.NODE_TYPES + 2
        self.edge_dim = edge_dim

    def node_input(self, geom, feat, node_type, n_frames, age):
        oh = torch.nn.functional.one_hot(node_type.long(), self.NODE_TYPES).to(geom.dtype)
        return torch.cat([geom, self.proj(feat), oh,
                          torch.log1p(n_frames)[:, None], torch.log1p(age)[:, None]], 1)

    def quality_input(self, h, rel=None):
        """The quality head's node input: the plain embedding plus the stage-3 relational columns.

        The cues are added *here* - never to the shared trunk - so they can move the ranking
        without touching the edge decisions (the merge behaviour stays exactly the stage-2 one).
        """
        return h if (rel is None or self.rel_emb is None) else h + self.rel_emb(rel)

    def forward(self, geom, feat, edges, edge_feat, node_type, n_frames, age, edge_type,
                rel=None, frame_id=None):
        """-> (edge_logit (E,), quality_logit (G,)). An empty ``edges`` gives a node-only call.

        ``rel`` (G, rel_dim) carries the stage-3 relational node columns; ``frame_id`` (G,) marks
        which frame each node belongs to (a training batch holds several frames; ``None`` = one
        frame). Both reach the *quality* head alone - ``rel`` through an additive embedding, the
        attention through its mask - so the edge decisions keep reading the plain embedding.
        """
        h = self.node(self.node_input(geom, feat, node_type, n_frames, age))
        if len(edges) == 0:
            return h.new_empty((0,)), self.quality(self._attend(self.quality_input(h, rel), node_type, frame_id)).squeeze(1)
        a, b = edges[:, 0].long(), edges[:, 1].long()
        et = torch.nn.functional.one_hot(edge_type.long(), self.EDGE_TYPES).to(h.dtype)
        ef = self.edge_norm(edge_feat)
        e = self.edge(torch.cat([h[a] + h[b], (h[a] - h[b]).abs(), ef, et], 1))
        agg = h.new_zeros(h.shape).index_add_(0, a, self.msg(e)).index_add_(0, b, self.msg(e))
        deg = torch.bincount(torch.cat([a, b]), minlength=len(h)).clamp_min(1).unsqueeze(1)
        h = h + agg / deg
        e = self.edge(torch.cat([h[a] + h[b], (h[a] - h[b]).abs(), ef, et], 1))
        return self.head(e).squeeze(1), self.quality(self._attend(self.quality_input(h, rel), node_type, frame_id)).squeeze(1)

    def _attend(self, h, node_type, frame_id):
        """Object-level self-attention rounds (no-op with ``attn_layers == 0``)."""
        if not self.attn_layers:
            return h
        obj = node_type.long() == self.NODE_TYPES - 1
        for layer in self.attn:
            h = layer(h, obj, frame_id)
        return h

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ``GroupMergeNet`` -> ``UnifiedMergeNet`` parameter map: the appearance projection, the
# message-passing MLP and both heads are shared exactly; the first layers grew their input
# (node 27 -> 32, edge 142 -> 160) and keep the overlapping leading columns.
_GROUP_TO_UNIFIED = {
    'proj.weight': 'proj.weight', 'proj.bias': 'proj.bias',
    'node.0.weight': 'node.0.weight', 'node.0.bias': 'node.0.bias',
    'node.3.weight': 'node.3.weight', 'node.3.bias': 'node.3.bias',
    'edge.0.weight': 'edge.0.weight', 'edge.0.bias': 'edge.0.bias',
    'edge.3.weight': 'edge.3.weight', 'edge.3.bias': 'edge.3.bias',
    'msg.0.0.weight': 'msg.0.weight', 'msg.0.0.bias': 'msg.0.bias',
    'msg.0.3.weight': 'msg.3.weight', 'msg.0.3.bias': 'msg.3.bias',
    'head.weight': 'head.weight', 'head.bias': 'head.bias',
    'edge_norm.weight': 'edge_norm.weight', 'edge_norm.bias': 'edge_norm.bias',
}
# The parameters whose *input* legitimately grew with the unified layout (node: geom+proj then
# type/n_frames/age; edge: [h_a+h_b, |h_a-h_b|, edge_feat] with 14 of the 29 columns shared;
# edge_norm: the first 14 columns are the intra-frame features). Only these may be truncated to
# their overlapping block - a width mismatch anywhere else (a source with a different ``hidden``)
# is skipped instead of silently copying a meaningless sub-block.
_TRUNCATABLE = {'node.0.weight', 'edge.0.weight', 'edge_norm.weight', 'edge_norm.bias'}


def init_from_group_merge_net(model, state_dict, log=print):
    """Initialise ``UnifiedMergeNet`` from a trained ``GroupMergeNet`` state dict.

    Copies every mapped parameter whose shape matches exactly, and truncates the parameters listed
    in ``_TRUNCATABLE`` to their overlapping block (the new columns keep their random init /
    LayerNorm default). Anything else - e.g. a source trained with a different ``hidden`` width -
    is skipped and reported. Returns the names of the copied target parameters.
    """
    own = model.state_dict()
    copied, skipped = [], []
    with torch.no_grad():
        for src, dst in _GROUP_TO_UNIFIED.items():
            if src not in state_dict or dst not in own:
                skipped.append(f'{src} -> {dst} (missing)')
                continue
            s, t = state_dict[src], own[dst]
            if s.shape == t.shape:
                t.copy_(s)
            elif (dst in _TRUNCATABLE and s.dim() == t.dim() and s.shape[:-1] == t.shape[:-1]
                  and all(a <= b for a, b in zip(s.shape, t.shape))):
                t[tuple(slice(0, int(a)) for a in s.shape)].copy_(s)
            else:
                skipped.append(f'{src} -> {dst} ({tuple(s.shape)} -> {tuple(t.shape)})')
                continue
            copied.append(dst)
    if log:
        log(f'init_from_group_merge_net: copied {len(copied)}/{len(_GROUP_TO_UNIFIED)} tensors'
            + (f', skipped: {skipped}' if skipped else ''))
    return copied
