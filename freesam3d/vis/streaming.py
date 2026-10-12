"""Scene-level streaming results as PLY / PNG: RGB | GT | QueryMemory | ESAM on the same mesh vertices.

Every predicted mask is painted in the colour of the GT instance it overlaps best (IoU >= 0.25), so a
correct instance has the same colour in every panel; masks that match no GT get their own distinct
colours; vertices covered by nothing are grey. Overlapping masks are painted in ascending score order
so the highest-scoring mask wins (as ESAM's ``mask_to_color`` does).
"""
import numpy as np

from ..evaluation.instance import GTSet, iou_matrix
from .pointcloud import write_ply
from .style import id_colors

BACKGROUND = 0.78


def paint_masks(masks, scores, colors, n):
    """(K,V) bool masks -> (V,3) colours; later (higher-score) masks overwrite earlier ones."""
    out = np.full((n, 3), BACKGROUND, np.float32)
    for k in np.argsort(np.asarray(scores)):
        out[masks[k]] = colors[k]
    return out


def gt_panel(gts: GTSet):
    """One stable colour per GT instance id; returns (K,3) colours and the painted (V,3) panel."""
    col = np.asarray(id_colors(gts.ids), np.float32).reshape(len(gts.ids), 3)
    out = np.full((gts.onehot.shape[0], 3), BACKGROUND, np.float32)
    for k in range(len(gts.ids)):
        out[gts.onehot[:, k]] = col[k]
    return col, out


def pred_colors(masks, gts: GTSet, gt_col, iou_thr=0.25):
    """Colour per predicted mask: its best GT's colour when IoU >= iou_thr, else a distinct fallback colour.
    Returns (P,3) colours and the (P,) matched flag."""
    if len(masks) == 0:
        return np.zeros((0, 3), np.float32), np.zeros(0, bool)
    iou = iou_matrix(masks, gts)
    best, ok = iou.argmax(1), iou.max(1) >= iou_thr
    col = np.asarray(id_colors(10_000 + np.arange(len(masks))), np.float32).reshape(len(masks), 3)
    col[ok] = gt_col[best[ok]]
    return col, ok


def side_by_side(xyz, panels, gap=1.0):
    """Lay the same points out along +x once per panel; returns (n_panels*V, 3) xyz and colours."""
    w = float(xyz[:, 0].max() - xyz[:, 0].min()) + gap
    xs, cs = [], []
    for i, (_, c) in enumerate(panels):
        p = xyz.copy()
        p[:, 0] += i * w
        xs.append(p)
        cs.append(c)
    return np.concatenate(xs), np.concatenate(cs)


def top_down_png(path, xyz, panels, max_points=250_000, ceiling=2.0, seed=0):
    """Top-down scatter of each panel (ScanNet z is up): points above floor + ``ceiling`` metres are dropped so the
    ceiling does not hide the room, and higher points are drawn last."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z0 = np.percentile(xyz[:, 2], 5)
    keep = np.nonzero(xyz[:, 2] < z0 + ceiling)[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(keep, min(max_points, len(keep)), replace=False)
    idx = idx[np.argsort(xyz[idx, 2])]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5.5))
    for ax, (name, c) in zip(np.atleast_1d(axes), panels):
        ax.scatter(xyz[idx, 0], xyz[idx, 1], c=np.clip(c[idx], 0, 1), s=0.25, linewidths=0)
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_scene_comparison(prefix, xyz, rgb, gts: GTSet, pred_masks, pred_scores, esam_masks=None, esam_scores=None,
                           min_points=100, label='QueryMemory'):
    """Write <prefix>_compare.ply (panels side by side), one PLY per panel and <prefix>_compare.png.
    Masks smaller than ``min_points`` vertices are dropped, as in the metrics. Returns the written paths."""
    prefix = str(prefix)
    rgb = np.asarray(rgb, np.float32)
    rgb = rgb / 255.0 if rgb.max() > 1.0 else rgb
    gt_col, gt_img = gt_panel(gts)
    panels = [('RGB', rgb), ('GT (thing instances)', gt_img)]
    stats = {}
    for name, m, s in ((label, pred_masks, pred_scores), ('ESAM', esam_masks, esam_scores)):
        if m is None:
            continue
        m = np.asarray(m, bool)
        s = np.asarray(s, np.float32)
        keep = m.sum(1) >= min_points
        m, s = m[keep], s[keep]
        col, ok = pred_colors(m, gts, gt_col)
        panels.append((f'{name}: {int(ok.sum())}/{len(m)} masks match a GT', paint_masks(m, s, col, len(xyz))))
        stats[name] = dict(masks=int(len(m)), matched=int(ok.sum()))
    paths = []
    x_all, c_all = side_by_side(xyz, panels)
    write_ply(prefix + '_compare.ply', x_all, c_all)
    paths.append(prefix + '_compare.ply')
    for (name, c), suffix in zip(panels, ('rgb', 'gt', 'memory', 'esam')):
        write_ply(f'{prefix}_{suffix}.ply', xyz, c)
        paths.append(f'{prefix}_{suffix}.ply')
    top_down_png(prefix + '_compare.png', xyz, panels)
    paths.append(prefix + '_compare.png')
    return paths, stats
