"""Training-free Query Lifting: RGB-D + FastSAM -> 3D groups -> group queries.

ESAM Query Lifting                  | here
------------------------------------|------------------------------------------
SAM masks -> exclusive groups        | same (FastSAM, smaller mask wins)
depth + pose -> world pts, 20k samp. | same
KMeans groups for uncovered points   | same
Stage 1: 3D U-Net + Memory Adapter   | FastSAM feature map sampled at each point
Stage 2: Geometric-aware Pooling     | mean pooling + L2 (baseline)
"""
from dataclasses import dataclass

import numpy as np

from ..io.frames import RGBDFrame
from ..models.fastsam import FastSAMOutput, FastSAMRunner, sample_point_features
from .geometry import backproject, sample_indices
from .grouping import compact_ids, exclusive_group_map, fill_uncovered
from .pooling import pool


@dataclass
class LiftedFrame:
    frame: RGBDFrame
    fastsam: FastSAMOutput
    group_map: np.ndarray          # HxW, FastSAM mask index after exclusivity, -1 = uncovered
    xyz: np.ndarray                # (N,3) world coords (ESAM ``points`` convention)
    rgb: np.ndarray                # (N,3) uint8
    pixels: np.ndarray             # (N,2) source (x, y) pixel
    group_ids: np.ndarray          # (N,) compact group id 0..G-1
    group_mask_index: np.ndarray   # (G,) FastSAM mask index, -1 = KMeans-filled group
    point_feats: dict              # feature name -> (N,C) tensor
    group_feats: dict              # feature name -> (G,C) L2-normalised tensor

    @property
    def num_groups(self):
        return len(self.group_mask_index)


class QueryLifting:
    def __init__(self, cfg):
        self.cfg = cfg
        fs = cfg['fastsam']
        self.runner = FastSAMRunner(fs['checkpoint'], fs['device'], fs['imgsz'], fs['conf'],
                                    fs['iou'], fs['fallback_iou'], cfg['features'])

    def __call__(self, frame: RGBDFrame) -> LiftedFrame:
        lc = self.cfg['lifting']
        out = self.runner(frame.rgb)

        # 1. overlapping masks -> exclusive pixel groups; lift and sample points.
        group_map = exclusive_group_map(out.masks)
        xyz, pixels = backproject(frame.depth, frame.intrinsic, frame.pose_c2w,
                                  lc['depth_scale'], lc['min_depth_raw'])
        keep = sample_indices(len(xyz), lc['num_points'], lc['seed'])
        xyz, pixels = xyz[keep], pixels[keep]
        rgb = frame.rgb[pixels[:, 1], pixels[:, 0]]

        # Uncovered points -> KMeans groups (ESAM clusters on [xyz, rgb]).
        raw_ids = group_map[pixels[:, 1], pixels[:, 0]]
        fill_x = np.concatenate([xyz, rgb.astype(np.float32)], 1) if lc['fill_features'] == 'xyz_rgb' else xyz
        filled = fill_uncovered(raw_ids, fill_x, lc['fill_clusters'], lc['fill_min_points'],
                                lc['seed'], base=len(out.masks))
        group_ids, original = compact_ids(filled)
        group_mask_index = np.where(original < len(out.masks), original, -1)

        # 2. per-point features straight from FastSAM; 3. group pooling.
        point_feats = {n: sample_point_features(m, pixels, out) for n, m in out.feature_maps.items()}
        group_feats = {n: pool(self.cfg['pooling']['method'], f, group_ids, len(original))
                       for n, f in point_feats.items()}
        return LiftedFrame(frame, out, group_map, xyz, rgb, pixels, group_ids,
                           group_mask_index, point_feats, group_feats)
