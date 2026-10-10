"""PLY export for inspecting lifted points in MeshLab / CloudCompare."""
import numpy as np
from plyfile import PlyData, PlyElement

from .style import id_colors


def write_ply(path, xyz, colors):
    """``colors`` in [0,1] float or uint8."""
    colors = np.asarray(colors)
    if colors.dtype != np.uint8:
        colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    v = np.empty(len(xyz), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    v['x'], v['y'], v['z'] = xyz.T
    v['red'], v['green'], v['blue'] = colors.T
    PlyData([PlyElement.describe(v, 'vertex')]).write(str(path))


def export_lifted(lifted, gt_instance, out_dir):
    write_ply(out_dir / 'points_rgb.ply', lifted.xyz, lifted.rgb)
    write_ply(out_dir / 'points_groups.ply', lifted.xyz, id_colors(lifted.group_ids))
    gt_col = np.where(gt_instance[:, None] > 0, id_colors(gt_instance), 0.6)
    write_ply(out_dir / 'points_gt.ply', lifted.xyz, gt_col)
