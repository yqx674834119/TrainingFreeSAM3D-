"""ESAM's OnlineMerge as a reference memory (no training): the rule baseline of the v2 evaluation.

Ported from ESAM ``oneformer3d/instance_merge.py:180`` (ESAM-E stream config: ``use_bbox=True``,
``inscat_topk_insts=200``). Per frame:

    S = clamp(cos(feat_mem, feat_new), min=0) * bbox_iou_3d(mem, new)

Hungarian assignment on ``-S``; every matched pair with ``S > 0`` merges (voxel union,
count-weighted running means of feature and score); unmatched instances are appended. No
admission gate, no maintenance merge, no forgetting; the output is truncated to ``topk``
instances by score.

Deviation from ESAM (README「与 ESAM 的差异」): masks are stored as voxel tables and rendered by
voxel lookup instead of point concatenation + KNN, so the memory grows with the scene volume
instead of with the number of frames.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .voxel import devoxelize, mask_voxel_sets, render_voxel_table, voxelize


def bbox_iou_3d(lo_a, hi_a, lo_b, hi_b):
    """(A,B) 3-D bbox IoU from per-instance bbox corners."""
    inter = (torch.minimum(hi_a[:, None], hi_b[None]) - torch.maximum(lo_a[:, None], lo_b[None])).clamp_min(0).prod(-1)
    vol_a = (hi_a - lo_a).clamp_min(0).prod(-1)
    vol_b = (hi_b - lo_b).clamp_min(0).prod(-1)
    return inter / (vol_a[:, None] + vol_b[None] - inter).clamp_min(1e-9)


def _bbox_from_voxels(vox, owner, num, voxel):
    c = devoxelize(vox, voxel)
    lo = torch.full((num, 3), float('inf'), device=c.device).scatter_reduce(0, owner[:, None].expand(-1, 3), c, 'amin')
    hi = torch.full((num, 3), float('-inf'), device=c.device).scatter_reduce(0, owner[:, None].expand(-1, 3), c, 'amax')
    empty = ~torch.isfinite(lo[:, 0])
    lo[empty], hi[empty] = 0.0, 0.0
    return lo, hi


class EsamOnlineMerge:
    def __init__(self, topk=200, voxel=0.05, device='cuda'):
        self.topk = topk
        self.voxel = voxel
        self.device = device
        self.reset()

    def reset(self):
        self.vox = []              # per-instance voxel id tensors
        self.lo = self.hi = None
        self.feat = None           # (K,C) running mean (ESAM's mix_feats)
        self.score = None          # (K,) running mean
        self.counts = None         # (K,) merge counts
        self.times = {}

    @torch.no_grad()
    def update(self, fr):
        """``fr``: FrameInput of one frame's instances. Returns (P,) bool: matched to an existing instance."""
        xyz = torch.as_tensor(np.asarray(fr.xyz, np.float32), device=self.device)
        masks = torch.as_tensor(np.asarray(fr.masks, bool), device=self.device)
        owner, vox = mask_voxel_sets(masks, voxelize(xyz, self.voxel))
        num = len(fr.masks)
        chunks = torch.split(vox, torch.bincount(owner, minlength=num).tolist())
        lo, hi = _bbox_from_voxels(vox, owner, num, self.voxel)
        feat = torch.as_tensor(np.asarray(fr.feat, np.float32), device=self.device)
        feat = feat / feat.norm(dim=1, keepdim=True).clamp_min(1e-6)
        score = torch.as_tensor(np.asarray(fr.scores, np.float32), device=self.device)
        matched = np.zeros(num, bool)
        if self.feat is None or len(self.feat) == 0:
            self.vox = list(chunks)
            self.lo, self.hi = lo, hi
            self.feat, self.score = feat, score.clone()
            self.counts = torch.zeros(num, device=self.device)
            return matched
        s = (self.feat @ feat.t()).clamp_min(0) * bbox_iou_3d(self.lo, self.hi, lo, hi)
        s_np = s.detach().cpu().numpy()
        rows, cols = linear_sum_assignment(-s_np)
        accept = s_np[rows, cols] > 0          # numpy-side mask: a torch bool index would act as an integer
        rows, cols = rows[accept], cols[accept]
        for ri, ci in zip(rows.tolist(), cols.tolist()):
            count = self.counts[ri] + 1                       # count-weighted running mean, as ESAM
            self.feat[ri] = (self.feat[ri] * self.counts[ri] + feat[ci]) / count
            self.score[ri] = (self.score[ri] * self.counts[ri] + score[ci]) / count
            self.counts[ri] = count
            self.vox[ri] = torch.unique(torch.cat([self.vox[ri], chunks[ci]]))
            self.lo[ri] = torch.minimum(self.lo[ri], lo[ci])
            self.hi[ri] = torch.maximum(self.hi[ri], hi[ci])
        matched[cols] = True
        new = np.setdiff1d(np.arange(num), cols)
        if len(new):
            idx = torch.as_tensor(new, device=self.device)
            self.vox += [chunks[i] for i in new]
            self.lo = torch.cat([self.lo, lo[idx]])
            self.hi = torch.cat([self.hi, hi[idx]])
            self.feat = torch.cat([self.feat, feat[idx]])
            self.score = torch.cat([self.score, score[idx]])
            self.counts = torch.cat([self.counts, torch.zeros(len(new), device=self.device)])
        return matched

    @torch.no_grad()
    def render(self, xyz):
        """(K,V) masks over arbitrary points; top-``topk`` instances by score, as ESAM's output truncation."""
        xyz_t = torch.as_tensor(np.asarray(xyz, np.float32), device=self.device)
        if self.feat is None or len(self.feat) == 0:
            return np.zeros((0, len(xyz_t)), bool), np.zeros(0, np.float32)
        keep = torch.argsort(-self.score)[:self.topk]
        rec_vox = [self.vox[i] for i in keep.tolist()]
        rec_own = [torch.full((len(v),), i, dtype=torch.int64, device=self.device) for i, v in enumerate(rec_vox)]
        rv, ro = torch.cat(rec_vox), torch.cat(rec_own)
        order = torch.argsort(rv)
        masks = render_voxel_table(rv[order], ro[order], len(rec_vox), voxelize(xyz_t, self.voxel))
        return masks.cpu().numpy(), self.score[keep].cpu().numpy()
