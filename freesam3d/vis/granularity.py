"""Figures for the ESAM granularity-alignment study."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .style import TEXT_SECONDARY, id_colors, savefig

METHOD_COLORS = {'FastSAM': '#2a78d6', 'ESAM': '#eb6834', 'ESAM-snap': '#eda100', 'Oracle-adj': '#e87ba4',
                 'Oracle': '#1baf7a'}
# ESAM preprocessing back-projects with this unified intrinsic (640x480).
ESAM_K = np.array([[577.870605, 0, 319.5], [0, 577.870605, 239.5], [0, 0, 1]])


def _clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e5e4df', lw=0.6)
    ax.set_axisbelow(True)


def summary_bars(summary, path, subset='thing'):
    panels = [('mean_best_iou', 'mean best IoU per GT'), ('recall@0.5', 'GT recall @ IoU 0.5'),
              ('AP50', 'class-agnostic AP50'), ('mean_fragments', 'fragments per GT (purity ≥ 0.8)')]
    methods = [m for m in METHOD_COLORS if f'{subset}/{m}' in summary]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.3))
    for ax, (key, title) in zip(axes, panels):
        vals = [summary[f'{subset}/{m}'][key] for m in methods]
        ax.bar(range(len(methods)), vals, width=0.6, color=[METHOD_COLORS[m] for m in methods])
        for i, v in enumerate(vals):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=8)
        ax.set_title(title, fontsize=10)
        _clean(ax)
    fig.suptitle(f'Per-frame comparison on ESAM points ({subset} GT instances)', fontsize=11)
    savefig(fig, path)


def composition(acc, path):
    merged = np.array([r['groups_merged'] for r in acc.esam_mask_rows])
    inc = np.concatenate(acc.inclusion_values) if acc.inclusion_values else np.zeros(0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    bins = np.arange(0, min(merged.max(), 25) + 2) - 0.5
    axes[0].hist(np.clip(merged, 0, 25), bins=bins, color=METHOD_COLORS['ESAM'])
    axes[0].set_xlabel('FastSAM groups with ≥50% of their points inside the ESAM mask')
    axes[0].set_ylabel('ESAM masks')
    axes[0].set_title(f'groups merged per ESAM mask (mean {merged.mean():.2f})', fontsize=10)
    axes[1].hist(inc, bins=np.linspace(0.05, 1, 39), color=METHOD_COLORS['ESAM'])
    axes[1].set_xlabel('fraction of a FastSAM group inside an ESAM mask (>0.05)')
    axes[1].set_ylabel('(mask, group) pairs')
    mid = ((inc > 0.2) & (inc < 0.8)).mean() if len(inc) else float('nan')
    axes[1].set_title(f'whole groups (≈1) vs split groups (0.2–0.8: {mid:.1%})', fontsize=10)
    for ax in axes:
        _clean(ax)
    savefig(fig, path)


def per_gt_scatter(acc, path):
    g = acc.gt_rows
    x = np.array([r['FastSAM_iou'] for r in g])
    y = np.array([r['ESAM_iou'] for r in g])
    frags = np.array([r['FastSAM_frags'] for r in g])
    fig, ax = plt.subplots(figsize=(5.4, 5))
    for lo, hi, c, lab in [(0, 1, '#9ec5f4', '≤1 fragment'), (2, 3, '#3987e5', '2–3 fragments'),
                           (4, 10 ** 6, '#0d366b', '≥4 fragments')]:
        s = (frags >= lo) & (frags <= hi)
        ax.scatter(x[s], y[s], s=10, color=c, label=f'{lab} ({s.sum()})', linewidths=0)
    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, lw=0.8, ls='--')
    ax.set_xlabel('best single FastSAM group IoU')
    ax.set_ylabel('best ESAM mask IoU')
    ax.set_title('per GT (thing) instance: above the diagonal = ESAM merged', fontsize=10)
    ax.legend(fontsize=8, loc='lower right', framealpha=0.9, edgecolor='none')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _clean(ax)
    savefig(fig, path)


def _project(xyz, pose_c2w, hw=(480, 640)):
    w2c = np.linalg.inv(pose_c2w)
    cam = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    uv = cam @ ESAM_K.T
    uv = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6)
    ok = (cam[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < hw[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < hw[0])
    return uv, ok


def frame_panel(fd, oracle_groups, frames_2d, path):
    """RGB | FastSAM groups | ESAM masks | Oracle merge | GT, as points on the image plane."""
    rgb = np.asarray(Image.open(frames_2d / 'color' / f'{fd.frame}.jpg'))
    pose = np.loadtxt(frames_2d / 'pose' / f'{fd.frame}.txt')
    uv, ok = _project(fd.xyz, pose)
    esam_id = np.full(len(fd.xyz), -1)
    for p in np.argsort(fd.esam_scores):            # highest score painted last
        esam_id[fd.esam_masks[p]] = p
    panels = [('FastSAM groups', fd.groups, len(np.unique(fd.groups))),
              ('ESAM per-frame masks', esam_id, len(fd.esam_masks)),
              ('Oracle merge (θ=0.8)', oracle_groups, len(np.unique(oracle_groups))),
              ('ScanNet GT', np.where(fd.gt > 0, fd.gt, -1), len(np.unique(fd.gt[fd.gt > 0])))]
    fig, axes = plt.subplots(1, 5, figsize=(22, 3.6))
    axes[0].imshow(rgb)
    axes[0].set_title(f'{fd.scene} frame {fd.frame}', fontsize=10)
    for ax, (title, ids, n) in zip(axes[1:], panels):
        ax.imshow(rgb, alpha=0.25)
        col = np.where(ids[:, None] >= 0, id_colors(np.maximum(ids, 0) * 7 + 3), 0.55)
        ax.scatter(uv[ok, 0], uv[ok, 1], c=col[ok], s=1.2, linewidths=0)
        ax.set_title(f'{title}: {n}', fontsize=10)
    for ax in axes:
        ax.set_xlim(0, 640)
        ax.set_ylim(480, 0)
        ax.set_xticks([])
        ax.set_yticks([])
    savefig(fig, path, dpi=90)
