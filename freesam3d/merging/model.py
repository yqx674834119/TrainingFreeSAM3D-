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
