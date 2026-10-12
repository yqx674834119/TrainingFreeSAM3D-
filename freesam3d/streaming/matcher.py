"""Cross-frame match scorer: (proposal, query) pair features -> probability of being the same object.

Trained on the dense stream cache (consecutive 40-frame-stride frames of a scene, GT instance ids are
scene-consistent) by simulating the memory with GT-labelled queries and collecting every candidate pair.
Architecture mirrors ``merging/scorer.py::MaskScorer`` (LayerNorm + 2-layer MLP, ~1.5k params).
"""
import numpy as np
import torch
from torch import nn

from .memory import MatchFeatures


class MatchScorer(nn.Module):
    def __init__(self, dim=12, hidden=32):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(self.norm(x)).squeeze(-1)

    @torch.no_grad()
    def score_pairs(self, feats: MatchFeatures):      # QueryMemory expects callable(MatchFeatures) -> (P,K)
        x = (feats.stack() - self.mu) / self.sd
        return torch.sigmoid(self.forward(x))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def load_matcher(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = MatchScorer(ck['dim'], ck['hidden']).to(device).eval()
    m.load_state_dict(ck['state_dict'])
    m.mu = torch.as_tensor(ck['mu'], device=device, dtype=torch.float32)
    m.sd = torch.as_tensor(ck['sd'], device=device, dtype=torch.float32)
    return m.score_pairs


def collect_pairs(frames, device, voxel=0.05, match_dilation=1, top_k=20, label_purity=0.5):
    """Run the memory over one scene's frames with GT-labelled queries; return (X, y) for all candidate pairs.

    ``frames``: list of dicts from the stream cache (masks packed, scalars, feat_raw, iou, gt_instance, xyz, frame_id)
    with a 'scores' key (proposal scores). Query labels are maintained exactly by tracking which proposal
    created / was matched to each query and the memory's merge map is not needed because we rebuild the
    label from the query's voxels against the GT voxels seen so far.
    """
    from .memory import FrameInput, MemoryConfig, QueryMemory
    from .voxel import mask_voxel_sets, voxelize
    mem = QueryMemory(MemoryConfig(voxel=voxel, match_dilation=match_dilation, match_score='max_contain',
                                   match_threshold=0.3, merge_rule='iou', overlap_merge_iou=0.5, admit_score=0.0), device=device)
    X, Y = [], []
    gt_vox = {}                                        # GT id -> set of voxel ids seen so far (for query labelling)
    q_lab = np.zeros(0, int)
    for d in frames:
        masks = np.unpackbits(d['masks'], axis=1, count=int(d['num_points'])).astype(bool)
        s = d['scores']
        keep = np.argsort(-s)[:top_k]
        masks, s, feat = masks[keep], s[keep], d['feat_raw'].astype(np.float32)[keep]
        gt = d['gt_instance']
        lab = np.array([np.bincount(gt[m][gt[m] > 0]).argmax() if (gt[m] > 0).sum() >= label_purity * m.sum() and (gt[m] > 0).any() else 0 for m in masks])
        xyz = torch.as_tensor(d['xyz'].astype(np.float32), device=device)
        vox = voxelize(xyz, voxel)
        if mem.num_queries and len(masks):
            po, pv = mask_voxel_sets(torch.as_tensor(masks, device=device), vox)
            ps = torch.bincount(po, minlength=len(masks)).float()
            f = mem.match_features(po, pv, ps, torch.as_tensor(s, device=device, dtype=torch.float32), torch.as_tensor(feat, device=device))
            cand = (f.voxel_iou > 0) | (f.centroid_dist < 0.5)
            pl = torch.as_tensor(lab, device=device); ql = torch.as_tensor(q_lab, device=device)
            valid = cand & (pl[:, None] > 0) & (ql[None, :] > 0)
            if valid.any():
                X.append(f.stack()[valid].cpu().numpy()); Y.append((pl[:, None] == ql[None, :])[valid].cpu().numpy())
        mem.update(FrameInput(d['xyz'].astype(np.float32), masks, s.astype(np.float32), feat, int(d['frame_id'])))
        v_np = vox.cpu().numpy()
        for k in np.unique(gt[gt > 0]):
            gt_vox.setdefault(int(k), set()).update(v_np[gt == k].tolist())
        # relabel queries exactly from accumulated GT voxels
        q_lab = np.zeros(mem.num_queries, int)
        for q in range(mem.num_queries):
            qv = set(mem.voxel[mem.owner == q].cpu().numpy().tolist())
            best, bk = 0, 0
            for k, kv in gt_vox.items():
                o = len(qv & kv)
                if o > best:
                    best, bk = o, k
            q_lab[q] = bk if best >= 0.3 * len(qv) else 0
    if not X:
        return np.zeros((0, 12), np.float32), np.zeros(0, bool)
    return np.concatenate(X), np.concatenate(Y)
