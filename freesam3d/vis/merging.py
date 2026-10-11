"""Qualitative panels for GroupMergeNet: RGB | FastSAM groups | merged | oracle | GT, on the image plane."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .style import id_colors, savefig


def merge_panel(fr, merged, oracle, gt, rgb_path, path, title_extra='', prob=None, threshold=None):
    """``fr`` is a cached frame dict (with pixels); ids arrays are per-point."""
    rgb = np.asarray(Image.open(rgb_path))
    h, w = rgb.shape[:2]
    px = fr['pixels'].astype(int)
    panels = [('FastSAM groups', fr['groups'], len(np.unique(fr['groups']))),
              ('GroupMergeNet', merged, len(np.unique(merged))),
              ('Oracle-adj', oracle, len(np.unique(oracle))),
              ('GT (3D)', np.where(gt > 0, gt, -1), len(np.unique(gt[gt > 0])))]
    ncol = 5 + (prob is not None)
    fig, axes = plt.subplots(1, ncol, figsize=(4.4 * ncol, 3.6))
    axes[0].imshow(rgb)
    axes[0].set_title(f"{fr['name']}{title_extra}", fontsize=10)
    for ax, (title, ids, n) in zip(axes[1:5], panels):
        ids = np.asarray(ids)
        ax.set_facecolor('black')
        ok = ids >= 0
        ax.scatter(px[~ok, 0], px[~ok, 1], c='#555555', s=1.2, linewidths=0)
        ax.scatter(px[ok, 0], px[ok, 1], c=id_colors(ids[ok] * 7 + 3), s=1.2, linewidths=0)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect('equal')
        ax.set_title(f'{title}: {n}', fontsize=10)
    for ax in axes[:5]:
        ax.set_xticks([])
        ax.set_yticks([])
    if prob is not None:
        axes[5].hist(prob, bins=25, color='#2a78d6')
        if threshold is not None:
            axes[5].axvline(threshold, color='#eb6834', ls='--')
        axes[5].set_title(f'edge p(merge), {len(prob)} edges', fontsize=10)
        axes[5].set_xlabel('probability')
    savefig(fig, path, dpi=90)


def comparison_bars(means, path, esam_ref=None, keys=('mean_best_iou', 'recall@0.5', 'AP50')):
    titles = {'mean_best_iou': 'mean best IoU', 'recall@0.5': 'recall @ IoU 0.5', 'AP50': 'class-agnostic AP50'}
    methods = list(means)
    if esam_ref:
        methods += [f'{m} (ESAM)' for m in esam_ref]
    colors = {'FastSAM': '#9ec5f4', 'cosine': '#6da7ec', 'Oracle-adj': '#1baf7a', 'Oracle': '#008300'}
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 3.6))
    for ax, k in zip(axes, keys):
        vals = [means[m][k] for m in means] + ([esam_ref[m][k] for m in esam_ref] if esam_ref else [])
        cols = [colors.get(m, '#eb6834' if 'ESAM' in m else '#2a78d6') for m in methods]
        ax.bar(range(len(methods)), vals, color=cols, width=0.65)
        for i, v in enumerate(vals):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
        ax.set_title(titles[k], fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#e5e4df', lw=0.6)
        ax.set_axisbelow(True)
    savefig(fig, path)
