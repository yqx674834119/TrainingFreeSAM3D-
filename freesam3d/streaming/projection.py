"""Contrastive feature projection for cross-frame matching (counterpart of ESAM's MergeHead).

ESAM trains ``MergeHead`` (256 -> 512 -> 256, LayerNorm, L2) with ``ScanNetMergeCriterion_Fast``: an
InfoNCE between instance centres of consecutive frames (tau 0.07). Here the input is the frozen
FastSAM backbone feature pooled per proposal (640-d, un-centred), the positives are proposals of
the same GT instance in nearby frames of the same scene (stream cache, scene-consistent ids), and
the negatives are every other proposal in the paired frame — including same-category neighbours,
which is exactly the confusion the voxel memory cannot resolve geometrically.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=640, hidden=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def multi_positive_infonce(za, zb, same, tau=0.07):
    """InfoNCE from frame a to frame b with possibly several positives per anchor (same instance seen as
    several proposals). Anchors without a positive are skipped. Symmetrised by the caller."""
    logits = za @ zb.T / tau
    has = same.any(1)
    if not has.any():
        return logits.new_zeros(())
    lp = F.log_softmax(logits[has], 1)
    return -(torch.logsumexp(lp.masked_fill(~same[has], -1e9), 1)).mean()


def load_projection(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = ProjectionHead(ck['in_dim'], ck['hidden'], ck['out_dim']).to(device).eval()
    m.load_state_dict(ck['state_dict'])
    return m


def load_scene_frames(scene_dir, min_iou=0.5, feat_key='feat_raw'):
    """[(feat (P,640), gt_id (P,), frame_id)] for proposals that are objects (true IoU >= min_iou)."""
    out = []
    for f in sorted(scene_dir.glob('*.npz'), key=lambda q: int(q.stem)):
        with np.load(f) as z:
            ok = z['iou'] >= min_iou
            out.append((z[feat_key][ok].astype(np.float32), z['best_gt'][ok].astype(np.int64), int(z['frame_id'])))
    return out
