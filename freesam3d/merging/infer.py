"""Single-call inference: RGB-D frame -> merged instance masks, everything on the GPU.

This is the deployable counterpart of ESAM's per-frame decoder + post-processing:

    FastSAM (masks, feature maps on device)
      -> exclusive group map            (tensor min over masks, 0.2 ms)
      -> back-projection + 20k sample   (tensor ops)
      -> KMeans fill of uncovered pts   (GPU Lloyd, ~8 ms)
      -> point / group features          (grid_sample + index_add on device)
      -> adjacency graph + 14 edge scalars (cdist + scatter, ~7 ms)
      -> GroupMergeNet                  (~2 ms)
      -> average-linkage clustering     (CPU, ~5 ms; G ≈ 70 nodes)

No KD-tree, no sklearn, no per-edge Python loop. The only CPU step left is the
final agglomeration over ~70 nodes / ~400 edges.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..lifting.geometry import backproject_torch
from ..lifting.grouping import exclusive_group_map_torch, kmeans_torch
from ..lifting.pooling import pool
from ..models.fastsam import FastSAMRunner, sample_point_features
from .cluster import merge_groups
from .graph_gpu import build_graph_gpu
from .proposal_data import ScalarStandardizer
from .proposals_gpu import build_proposals_gpu, nested_nms_gpu
from .scorer import MaskScorer
from .train import load_checkpoint


@dataclass
class MergeResult:
    xyz: np.ndarray            # (N,3) sampled world points
    pixels: np.ndarray         # (N,2) (u,v)
    groups: np.ndarray         # (N,) FastSAM exclusive group id (after KMeans fill)
    instances: np.ndarray      # (N,) merged instance id per point (exclusive)
    instance_scores: np.ndarray  # (K,) objectness score per merged instance (index = instance id)
    edge_prob: np.ndarray      # (E,) merge probability per adjacency edge
    edges: np.ndarray          # (E,2)
    timing_ms: dict
    masks: np.ndarray = None   # (P,N) bool: final proposal masks after scoring + nested NMS (may overlap)
    mask_scores: np.ndarray = None  # (P,) predicted IoU per final mask


class GroupMerger:
    """Frozen FastSAM + trained GroupMergeNet, end to end on one device."""

    def __init__(self, lifting_cfg, merge_cfg, checkpoint, device='cuda', scorer_checkpoint=None,
                 nms_iou=0.5, nms_contain=0.8, top_k=None):
        self.device = torch.device(device)
        fs = lifting_cfg['fastsam']
        feats = sorted({merge_cfg['feature'], merge_cfg['boundary_feature']})
        self.runner = FastSAMRunner(fs['checkpoint'], str(self.device), fs['imgsz'], fs['conf'], fs['iou'],
                                    fs['fallback_iou'], feats, keep_on_device=True)
        self.lc, self.gc, self.mc = lifting_cfg['lifting'], merge_cfg['graph'], merge_cfg
        self.pooling = lifting_cfg['pooling']['method']
        self.feature, self.boundary_feature = merge_cfg['feature'], merge_cfg['boundary_feature']
        self.model, self.std, ck = load_checkpoint(checkpoint, str(self.device))
        self.threshold = merge_cfg['merge']['threshold'] or ck['threshold'] or 0.5
        self.linkage = merge_cfg['merge']['linkage']
        self.min_points = merge_cfg['merge'].get('min_points', 0)
        self.scorer = None
        if scorer_checkpoint is not None:
            ck = torch.load(scorer_checkpoint, map_location=str(self.device), weights_only=False)
            self.scorer = MaskScorer(ck['scalar_dim'], ck['feat_dim'], ck['hidden']).to(self.device).eval()
            self.scorer.load_state_dict(ck['state_dict'])
            self.scorer_std = ScalarStandardizer.from_state(ck['standardizer'])
            self.scorer_feat = ck['feat_dim'] > 0
            self.nms_iou, self.nms_contain, self.top_k = nms_iou, nms_contain, top_k

    @torch.inference_mode()
    def __call__(self, rgb, depth, intrinsic, pose_c2w, seed=0):
        t = {}
        s = _tick()
        out = self.runner(rgb)                                           # masks (M,H,W) np.bool, maps on device
        masks = torch.as_tensor(out.masks, device=self.device)
        t['fastsam'] = _tock(s)

        s = _tick()
        gmap = exclusive_group_map_torch(masks)                          # (H,W) long, -1 uncovered
        xyz, px = backproject_torch(depth, intrinsic, pose_c2w, self.lc['depth_scale'], self.lc['min_depth_raw'],
                                    self.device)
        n = len(xyz)
        g = torch.Generator(device=self.device).manual_seed(seed)
        keep = torch.randperm(n, generator=g, device=self.device)[:self.lc['num_points']] if n >= self.lc['num_points'] \
            else torch.randint(n, (self.lc['num_points'],), generator=g, device=self.device)
        xyz, px = xyz[keep], px[keep]
        rgb_t = torch.as_tensor(rgb, device=self.device)[px[:, 1], px[:, 0]]
        raw = gmap[px[:, 1], px[:, 0]]
        missing = raw < 0
        groups = raw.clone()
        if missing.any():
            x = torch.cat([xyz[missing], rgb_t[missing].float()], 1) if self.lc['fill_features'] == 'xyz_rgb' else xyz[missing]
            k = min(self.lc['fill_clusters'], int(missing.sum()))
            lab = torch.zeros(int(missing.sum()), dtype=torch.long, device=self.device) if int(missing.sum()) < self.lc['fill_min_points'] \
                else kmeans_torch(x, k, seed=seed)
            groups[missing] = lab + len(out.masks)
        uniq, groups = torch.unique(groups, return_inverse=True)         # compact 0..G-1
        mask_index = torch.where(uniq < len(out.masks), uniq, torch.full_like(uniq, -1))
        t['lift'] = _tock(s)

        s = _tick()
        px_np = px.cpu().numpy()
        pf = {name: sample_point_features(m, px_np, out) for name, m in out.feature_maps.items()}
        gf = pool(self.pooling, pf[self.feature], groups, len(uniq))
        t['features'] = _tock(s)

        s = _tick()
        graph = build_graph_gpu(xyz, rgb_t, groups, gf, pf[self.boundary_feature], out.scores,
                                mask_index.cpu().numpy(), out.masks, self.gc['adjacency_eps'], self.gc['voxel'],
                                self.gc['boundary_k'], self.gc.get('feature_centering', True), device=str(self.device))
        t['graph'] = _tock(s)

        s = _tick()
        geom = torch.as_tensor(graph.node_geom, device=self.device)
        feat = torch.as_tensor(graph.node_feat, device=self.device)
        edges = torch.as_tensor(graph.edges, device=self.device)
        ef = torch.as_tensor(graph.edge_feat, device=self.device)
        geom, ef = self.std.apply(geom, ef)
        logit, obj = self.model(geom, feat, edges, ef, return_objectness=True)
        prob, obj = torch.sigmoid(logit).cpu().numpy(), torch.sigmoid(obj).cpu().numpy()
        t['model'] = _tock(s)

        s = _tick()
        groups_np = groups.cpu().numpy()
        inst, node_inst = merge_groups(groups_np, graph.edges, prob, self.threshold, self.linkage, return_mapping=True)
        # objectness-weighted score per instance; drop tiny instances (ESAM npoint_thr)
        sizes = graph.sizes.astype(np.float64)
        K = int(node_inst.max()) + 1
        score = np.bincount(node_inst, weights=obj * sizes, minlength=K) / np.bincount(node_inst, weights=sizes, minlength=K)
        counts = np.bincount(inst, minlength=K)
        drop = counts < self.min_points
        score[drop] = -1.0
        t['cluster'] = _tock(s)
        res = MergeResult(xyz.cpu().numpy(), px_np, groups_np, inst, score, prob, graph.edges, t)
        if self.scorer is not None:
            s = _tick()
            merged_t = torch.as_tensor(node_inst, device=self.device)[groups]          # (N,) instance per point
            merged_masks = F.one_hot(merged_t, int(node_inst.max()) + 1).T.bool()         # (K,N) on device
            raw_pts = masks[:, px[:, 1], px[:, 0]]
            P = build_proposals_gpu(xyz, groups, graph.node_feat, pf[self.boundary_feature], raw_pts, out.scores,
                                    mask_index, obj, merged_masks, self.min_points, device=str(self.device))
            if len(P.masks):
                X = self.scorer_std.apply(torch.as_tensor(P.scalars, device=self.device))
                Fe = torch.as_tensor(P.feat, device=self.device)
                sc = self.scorer.score(X, Fe if self.scorer_feat else Fe[:, :0]).cpu().numpy()
                keep = nested_nms_gpu(P.masks, sc, self.nms_iou, self.nms_contain, str(self.device))
                if self.top_k:
                    keep = keep[:self.top_k]
                res.masks, res.mask_scores = P.masks[keep], sc[keep]
            else:
                res.masks, res.mask_scores = np.zeros((0, len(res.xyz)), bool), np.zeros(0)
            t['proposals+scorer+nms'] = _tock(s)
        return res


def _tick():
    torch.cuda.synchronize()
    import time
    return time.perf_counter()


def _tock(s):
    torch.cuda.synchronize()
    import time
    return (time.perf_counter() - s) * 1000
