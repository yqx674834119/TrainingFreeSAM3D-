"""Node statistics and edge features for the unified (intra + cross-frame) merge graph.

``NodeStats`` holds the additive statistics of a node — a current FastSAM fragment, a past-frame
block of fragments, or a frozen object. They are additive on purpose: a block is the sum of its
fragments and an object the sum of everything frozen into it, so a node is featurised the same
way whatever its history. ``geom`` reproduces ``merging.graph_gpu``'s 11-d node geometry exactly.

``pair_features`` concatenates the 14 cached intra-frame edge features (zeros for pairs that are
not intra-frame edges) with 15 cross-frame features computed from the statistics and the voxel
overlap counts of the pair.
"""
from dataclasses import dataclass, fields

import torch

from ..merging.graph import edge_feature_names

CROSS_FEATURE_NAMES = ['inter0_frac_min', 'inter1_frac_min', 'iou0', 'contain_a_in_b', 'contain_b_in_a',
                       'centroid_dist', 'height_diff', 'log_size_ratio', 'extent_ratio', 'bbox_iou',
                       'feat_cos', 'frame_gap', 'same_frame', 'has_intra', 'frozen_count']


def cross_feature_names():
    return list(CROSS_FEATURE_NAMES)


def pair_feature_names():
    """Full edge feature layout: the 14 intra-frame features, then the 15 cross-frame ones."""
    return edge_feature_names() + cross_feature_names()


@dataclass
class NodeStats:
    """Additive per-node statistics; all fields are (G, ...) tensors on one device."""
    n: torch.Tensor          # point count
    sum_xyz: torch.Tensor    # (G,3)
    sum_xx: torch.Tensor     # (G,9) row-major sum of xyz (x) xyz
    sum_rgb: torch.Tensor    # (G,3) on the 0-1 scale
    lo: torch.Tensor         # (G,3) bbox min (+inf when the node is empty)
    hi: torch.Tensor         # (G,3) bbox max (-inf when the node is empty)
    feat_sum: torch.Tensor   # (G,C) sum of the member fragments' node_feat rows
    n_feat: torch.Tensor     # number of members that carry a feature
    first: torch.Tensor      # first frame position the node was seen at
    last: torch.Tensor       # last frame position (age = pos - last)
    n_frames: torch.Tensor   # number of observations
    conf_sum: torch.Tensor   # sum of member confidences (FastSAM score)

    @staticmethod
    def zeros(num, feat_dim, device):
        z = lambda *shape: torch.zeros(*shape, device=device)
        return NodeStats(
            n=z(num), sum_xyz=z(num, 3), sum_xx=z(num, 9), sum_rgb=z(num, 3),
            lo=torch.full((num, 3), float('inf'), device=device),
            hi=torch.full((num, 3), float('-inf'), device=device),
            feat_sum=z(num, feat_dim), n_feat=z(num),
            first=torch.full((num,), float('inf'), device=device),
            last=torch.full((num,), float('-inf'), device=device),
            n_frames=z(num), conf_sum=z(num))

    @staticmethod
    def from_points(xyz, rgb, feat=None, conf=None, pos=0, device='cpu'):
        """Single node from a point set; ``rgb`` on the 0-255 scale (divided by 255 like the graph)."""
        xyz = torch.as_tensor(xyz, dtype=torch.float32, device=device)
        rgb = torch.as_tensor(rgb, dtype=torch.float32, device=device) / 255
        out = NodeStats.zeros(1, 0 if feat is None else len(feat), device)
        out.n[0] = len(xyz)
        out.sum_xyz[0] = xyz.sum(0)
        out.sum_xx[0] = (xyz[:, :, None] * xyz[:, None, :]).reshape(-1, 9).sum(0)
        out.sum_rgb[0] = rgb.sum(0)
        out.lo[0] = xyz.amin(0)
        out.hi[0] = xyz.amax(0)
        if feat is not None:
            out.feat_sum[0] = torch.as_tensor(feat, dtype=torch.float32, device=device)
            out.n_feat[0] = 1
        out.conf_sum[0] = 0.0 if conf is None else float(conf)
        out.first[0] = out.last[0] = float(pos)
        out.n_frames[0] = 1
        return out

    @staticmethod
    def cat(parts):
        if len(parts) == 1:
            return parts[0]
        return NodeStats(*[torch.cat([getattr(p, f.name) for p in parts], 0) for f in fields(NodeStats)])

    def select(self, idx):
        return NodeStats(*[getattr(self, f.name)[idx] for f in fields(NodeStats)])

    def add(self, other):
        """Elementwise accumulation (bbox = union, first/last = extremes)."""
        return NodeStats(n=self.n + other.n, sum_xyz=self.sum_xyz + other.sum_xyz,
                         sum_xx=self.sum_xx + other.sum_xx, sum_rgb=self.sum_rgb + other.sum_rgb,
                         lo=torch.minimum(self.lo, other.lo), hi=torch.maximum(self.hi, other.hi),
                         feat_sum=self.feat_sum + other.feat_sum, n_feat=self.n_feat + other.n_feat,
                         first=torch.minimum(self.first, other.first), last=torch.maximum(self.last, other.last),
                         n_frames=self.n_frames + other.n_frames, conf_sum=self.conf_sum + other.conf_sum)

    def centroid(self):
        return self.sum_xyz / self.n.clamp_min(1)[:, None]

    def extent(self):
        ext = self.hi - self.lo
        return torch.where(self.n[:, None] > 0, ext, torch.zeros_like(ext)).clamp_min(0)

    def mean_rgb(self):
        return self.sum_rgb / self.n.clamp_min(1)[:, None]

    def feat(self):
        """Member-mean feature, L2-normalised (zeros stay zeros)."""
        f = self.feat_sum / self.n_feat.clamp_min(1)[:, None]
        return f / f.norm(dim=1, keepdim=True).clamp_min(1e-6)

    def geom(self, z_ref):
        """(G,11) node geometry, identical to ``graph_gpu.build_graph_gpu``:
        [log1p(size), height, extent(3), sqrt eigenvalues desc(3), mean rgb(3)]."""
        cent = self.centroid()
        cov = self.sum_xx.reshape(-1, 3, 3) / self.n.clamp_min(1)[:, None, None] - cent[:, :, None] * cent[:, None, :]
        eig = torch.linalg.eigvalsh(cov).clamp_min(0).sqrt().flip(1)
        return torch.cat([torch.log1p(self.n)[:, None], (cent[:, 2] - z_ref)[:, None],
                          self.extent(), eig, self.mean_rgb()], 1)


def pair_features(stats, a, b, inter0, inter1, va, vb, frame_gap, same_frame, has_intra, frozen_count,
                  edge_feat=None):
    """(E, 29) edge features for the node pairs ``(a, b)``.

    ``inter0``/``inter1`` are the voxel intersection counts without / with one-sided dilation,
    ``va``/``vb`` the per-node voxel counts (from ``overlap_matrix``). ``edge_feat`` (E,14) are
    the cached intra-frame features; pass ``None`` to zero them (non-intra pairs).
    """
    dev = stats.n.device
    cent, ext = stats.centroid(), stats.extent()
    n_a, n_b = stats.n[a].clamp_min(1), stats.n[b].clamp_min(1)
    va_, vb_ = va[a].clamp_min(1), vb[b].clamp_min(1)
    vmin = torch.minimum(va_, vb_)
    iou0 = inter0 / (va_ + vb_ - inter0).clamp_min(1)
    b_inter = (torch.minimum(stats.hi[a], stats.hi[b]) - torch.maximum(stats.lo[a], stats.lo[b])).clamp_min(0).prod(1)
    b_union = (torch.maximum(stats.hi[a], stats.hi[b]) - torch.minimum(stats.lo[a], stats.lo[b])).clamp_min(0).prod(1)
    feat = stats.feat()
    ext_max = ext.max(1).values
    cross = torch.stack([
        inter0 / vmin, inter1 / vmin, iou0,
        inter0 / va_, inter0 / vb_,
        (cent[a] - cent[b]).norm(dim=1), (cent[a, 2] - cent[b, 2]).abs(),
        (n_a / n_b).log().abs(),
        ((ext_max[a] + 1e-3) / (ext_max[b] + 1e-3)).log().abs(),
        b_inter / b_union.clamp_min(1e-9),
        (feat[a] * feat[b]).sum(1),
        frame_gap.float(), same_frame.float(), has_intra.float(), frozen_count.float(),
    ], 1)
    if edge_feat is None:
        edge_feat = torch.zeros((len(cross), len(edge_feature_names())), device=dev)
    return torch.cat([edge_feat, cross], 1)
