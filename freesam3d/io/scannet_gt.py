"""ScanNet 3D instance GT, transferred to lifted points the same way as ESAM.

ESAM (``data/scannet200-mv/load_scannet_mv_data_fast.py``) labels each sampled
point with the instance of its nearest mesh vertex, then zeroes points that
fall outside that instance's axis-aligned bbox (+5 cm). We reproduce that, but
use a CPU KD-tree instead of ``pointops.knn_query``.
"""
import csv
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree


@dataclass
class PointGT:
    instance: np.ndarray       # (N,) int, 0 = unlabeled
    knn_distance: np.ndarray   # (N,) metres to the nearest mesh vertex
    categories: dict           # instance id -> category name


@lru_cache(maxsize=1)
def _raw_to_category(label_tsv):
    with open(label_tsv) as f:
        return {row['raw_category']: row['category'] for row in csv.DictReader(f, delimiter='\t')}


def _axis_align(meta_txt):
    for line in Path(meta_txt).read_text().splitlines():
        if line.startswith('axisAlignment'):
            return np.array([float(v) for v in line.split('=')[1].split()]).reshape(4, 4)
    return np.eye(4)


class ScanNetSceneGT:
    def __init__(self, scene, scans_roots, label_tsv, bbox_margin=0.05):
        root = next((Path(r) / scene for r in scans_roots if (Path(r) / scene / f'{scene}_vh_clean_2.ply').exists()), None)
        if root is None:
            raise FileNotFoundError(f'no ScanNet mesh for {scene} under {scans_roots}')
        self.scene = scene
        self.bbox_margin = bbox_margin
        ply = PlyData.read(str(root / f'{scene}_vh_clean_2.ply'))['vertex']
        self.vertices = np.stack([ply['x'], ply['y'], ply['z']], axis=1).astype(np.float64)
        seg_indices = np.asarray(json.loads((root / f'{scene}_vh_clean_2.0.010000.segs.json').read_text())['segIndices'])
        groups = json.loads((root / f'{scene}.aggregation.json').read_text())['segGroups']
        self.axis_align = _axis_align(root / f'{scene}.txt')

        raw2cat = _raw_to_category(str(label_tsv))
        self.vertex_instance = np.zeros(len(self.vertices), np.int64)
        self.categories = {}
        for g in groups:
            inst = int(g['objectId']) + 1       # ESAM/ScanNet convention: objectId + 1
            self.vertex_instance[np.isin(seg_indices, g['segments'])] = inst
            self.categories[inst] = raw2cat.get(g['label'], g['label'])

        aligned = self._align(self.vertices)
        self.aligned_bbox = {}
        for inst in self.categories:
            pts = aligned[self.vertex_instance == inst]
            if len(pts):
                self.aligned_bbox[inst] = (pts.min(0) - bbox_margin, pts.max(0) + bbox_margin)
        self.tree = cKDTree(self.vertices)

    def _align(self, xyz):
        return xyz @ self.axis_align[:3, :3].T + self.axis_align[:3, 3]

    def label_points(self, xyz_world):
        """Instance id per point (unaligned world coords), with ESAM bbox denoise."""
        dist, idx = self.tree.query(np.asarray(xyz_world, np.float64), k=1, workers=-1)
        inst = self.vertex_instance[idx].copy()
        aligned = self._align(np.asarray(xyz_world, np.float64))
        for i, (lo, hi) in self.aligned_bbox.items():
            sel = inst == i
            if sel.any():
                inside = np.all((aligned[sel] >= lo) & (aligned[sel] <= hi), axis=1)
                inst[np.flatnonzero(sel)[~inside]] = 0
        return PointGT(inst, dist.astype(np.float32), self.categories)
