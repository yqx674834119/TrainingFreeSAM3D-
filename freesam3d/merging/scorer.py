"""Mask-quality scorer: GT-free proposal features -> predicted IoU to the best GT instance.

Small MLP on [20 scalars, P·feat (640 -> 32)]. Trained with a regression
loss on IoU (which doubles as the ranking score) plus a BCE term on IoU >= 0.5
so that the top of the ranking is calibrated for the AP matching threshold.
"""
import torch
from torch import nn


class MaskScorer(nn.Module):
    def __init__(self, scalar_dim, feat_dim, hidden=64, feature_proj=32, dropout=0.1):
        super().__init__()
        self.use_feat = feat_dim > 0
        self.proj = nn.Linear(feat_dim, feature_proj) if self.use_feat else None
        d = scalar_dim + (feature_proj if self.use_feat else 0)
        self.norm = nn.LayerNorm(scalar_dim)
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 2))          # [iou regression logit, iou>=0.5 logit]

    def forward(self, scalars, feat):
        x = self.norm(scalars)
        if self.use_feat:
            x = torch.cat([x, self.proj(feat)], 1)
        return self.net(x)

    @torch.no_grad()
    def score(self, scalars, feat):
        out = self.forward(scalars, feat)
        return torch.sigmoid(out[:, 0])          # predicted IoU in [0, 1]

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
