"""Per-frame image diagnostics: lifting overview and feature PCA."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .style import id_colors, savefig


def _color_ids(id_map, background=(0, 0, 0)):
    out = np.zeros(id_map.shape + (3,), np.float32) + np.asarray(background, np.float32)
    valid = id_map >= 0
    out[valid] = id_colors(id_map[valid])
    return out


def _overlay_masks(rgb, masks):
    img = rgb.astype(np.float32) / 255
    if len(masks) == 0:
        return img
    colors = id_colors(np.arange(len(masks)))
    order = np.argsort(-masks.reshape(len(masks), -1).sum(1))   # large first, small on top
    for i in order:
        img[masks[i]] = 0.55 * img[masks[i]] + 0.45 * colors[i]
    return img


def overview(lifted, gt_dense, categories, path):
    """RGB | depth | raw FastSAM masks | exclusive groups | GT | sampled 3D groups."""
    fr, out = lifted.frame, lifted.fastsam
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    axes = axes.ravel()
    axes[0].imshow(fr.rgb)
    axes[0].set_title(f'{fr.name}  RGB')
    axes[1].imshow(np.where(fr.depth > 0, fr.depth / 1000.0, np.nan), cmap='Blues')
    axes[1].set_title('depth (m), white = invalid')
    axes[2].imshow(_overlay_masks(fr.rgb, out.masks))
    axes[2].set_title(f'FastSAM raw masks: {len(out.masks)} (overlapping)')
    axes[3].imshow(_color_ids(lifted.group_map))
    n_ex = len(np.unique(lifted.group_map[lifted.group_map >= 0]))
    axes[3].set_title(f'exclusive groups: {n_ex} (small wins), black = uncovered')
    axes[4].imshow(_color_ids(np.where(gt_dense > 0, gt_dense, -1), background=(0.85, 0.85, 0.85)))
    for inst in np.unique(gt_dense[gt_dense > 0]):
        ys, xs = np.nonzero(gt_dense == inst)
        if len(ys) > 1500:
            axes[4].text(np.median(xs), np.median(ys), f'{categories.get(int(inst), "?")} #{inst}',
                         fontsize=7, ha='center', va='center',
                         bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75))
    axes[4].set_title('ScanNet GT instances (3D mesh -> pixels), gray = none')
    is_fill = lifted.group_mask_index[lifted.group_ids] < 0
    cols = id_colors(lifted.group_ids)
    px = lifted.pixels
    axes[5].set_facecolor('black')
    axes[5].scatter(px[~is_fill, 0], px[~is_fill, 1], c=cols[~is_fill], s=0.6, linewidths=0)
    axes[5].scatter(px[is_fill, 0], px[is_fill, 1], c=cols[is_fill], s=2.5, marker='x', linewidths=0.4)
    axes[5].set_xlim(0, fr.rgb.shape[1])
    axes[5].set_ylim(fr.rgb.shape[0], 0)
    axes[5].set_aspect('equal')
    axes[5].set_title(f'{len(px)} sampled pts, {lifted.num_groups} groups (x = KMeans-filled)')
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    savefig(fig, path)


def _pca_rgb(fmap):
    c, h, w = fmap.shape[1:]
    x = fmap[0].reshape(c, -1).T.numpy().astype(np.float64)
    x -= x.mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    y = x @ vt[:3].T
    lo, hi = np.percentile(y, 1, axis=0), np.percentile(y, 99, axis=0)
    return np.clip((y - lo) / np.maximum(hi - lo, 1e-6), 0, 1).reshape(h, w, 3)


def feature_pca(lifted, path):
    """First 3 PCA components of each dense feature map, cropped to the image."""
    out, fr = lifted.fastsam, lifted.frame
    names = list(out.feature_maps)
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(4.2 * (len(names) + 1), 3.6))
    axes[0].imshow(fr.rgb)
    axes[0].set_title('RGB')
    h, w = fr.rgb.shape[:2]
    for ax, name in zip(axes[1:], names):
        fmap = out.feature_maps[name]
        s = out.input_hw[0] / fmap.shape[2]            # stride of this map
        x0, y0 = out.pad
        crop = fmap[..., int(y0 / s):int(np.ceil((y0 + h * out.gain) / s)),
                    int(x0 / s):int(np.ceil((x0 + w * out.gain) / s))]
        ax.imshow(_pca_rgb(crop), interpolation='nearest')
        ax.set_title(f'{name}: C={fmap.shape[1]}, {fmap.shape[2]}x{fmap.shape[3]}')
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    savefig(fig, path)
