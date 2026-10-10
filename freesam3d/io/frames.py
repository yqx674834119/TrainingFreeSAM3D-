"""RGB-D frame container and loaders."""
from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np


@dataclass
class RGBDFrame:
    """One posed RGB-D frame on the depth pixel grid.

    ``rgb`` is resized to the depth resolution on load, so masks, depth and
    features all share one ``(H, W)`` grid (ESAM uses 640x480 for both).
    """
    name: str
    scene: str
    frame_id: int
    rgb: np.ndarray        # HxWx3 uint8, RGB order
    depth: np.ndarray      # HxW raw depth (ScanNet: uint16 millimetres)
    pose_c2w: np.ndarray   # 4x4 camera-to-world (ScanNet unaligned world)
    intrinsic: np.ndarray  # 3x3 depth intrinsic


_NAME_RE = re.compile(r'(scene\d{4}_\d{2})_(\d+)$')


def load_npz_frame(path, rgb_key='rgb640'):
    """Load a frame saved by ESAM ``local/feature_benchmark/prepare_inputs.py``."""
    path = Path(path)
    m = _NAME_RE.search(path.stem)
    if m is None:
        raise ValueError(f'cannot parse scene/frame from {path.name}')
    with np.load(path) as d:
        depth = d['depth']
        rgb = d[rgb_key]
        pose = d['pose'].astype(np.float64)
        intrinsic = d['intrinsic'].astype(np.float64)[:3, :3]
    if rgb.shape[:2] != depth.shape:
        rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
    return RGBDFrame(path.stem, m.group(1), int(m.group(2)), rgb, depth, pose, intrinsic)


def list_frames(inputs):
    """Accept a directory of ``.npz`` frames, a single file, or a list of files."""
    if isinstance(inputs, (list, tuple)):
        return [Path(p) for p in inputs]
    p = Path(inputs)
    return sorted(p.glob('*.npz')) if p.is_dir() else [p]
