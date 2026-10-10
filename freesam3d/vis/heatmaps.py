"""Similarity heatmaps (category-labelled) and same/diff pair distributions."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .style import DIFF_COLOR, SAME_COLOR, SIM_CMAP, TEXT_SECONDARY, savefig


def _boundaries(keys):
    return [i for i in range(1, len(keys)) if keys[i] != keys[i - 1]]


def similarity_heatmap(view, title, path, minor_keys=None, major_keys=None):
    """Cosine heatmap; thin lines split ``minor_keys`` blocks, thick ``major_keys``."""
    k = len(view.sim)
    if k == 0:
        return
    side = float(np.clip(0.17 * k + 3.5, 6, 24))
    fs = float(np.clip(9 - k / 22, 4.5, 9))
    fig, ax = plt.subplots(figsize=(side + 1.2, side))
    off = view.sim[~np.eye(k, dtype=bool)]
    vmin = float(np.floor(np.percentile(off, 2) * 20) / 20) if off.size else 0.0
    im = ax.imshow(view.sim, cmap=SIM_CMAP, vmin=min(vmin, 0.95), vmax=1.0, interpolation='nearest')
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(view.labels, rotation=90, fontsize=fs)
    ax.set_yticklabels(view.labels, fontsize=fs)
    for keys, lw in ((minor_keys, 0.5), (major_keys, 1.4)):
        for b in _boundaries(keys) if keys is not None else []:
            ax.axhline(b - 0.5, color=TEXT_SECONDARY, lw=lw)
            ax.axvline(b - 0.5, color=TEXT_SECONDARY, lw=lw)
    if k <= 25:
        for i in range(k):
            for j in range(k):
                v = view.sim[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=fs - 1,
                        color='white' if v > vmin + 0.6 * (1 - vmin) else '#0b0b0b')
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label='cosine similarity')
    savefig(fig, path)


def pair_distributions(pairs, metrics, path, title):
    """Small multiples: same-GT vs different-GT mask-pair similarity, per feature."""
    names = list(pairs)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.4), squeeze=False)
    for ax, name in zip(axes[0], names):
        vals, same = pairs[name]
        bins = np.linspace(min(vals.min(), 0.0) if len(vals) else 0, 1, 41)
        ax.hist(vals[~same], bins=bins, density=True, color=DIFF_COLOR, alpha=0.75,
                label=f'different GT ({(~same).sum()})')
        ax.hist(vals[same], bins=bins, density=True, color=SAME_COLOR, alpha=0.75,
                label=f'same GT ({same.sum()})')
        m = metrics[name]['mask']
        ax.set_title(f'{name}\nAUC={m["auc"]:.3f}  gap={m["gap"]:.3f}  NN-hit={m["nn_hit_rate"]:.2f}', fontsize=9)
        ax.set_xlabel('cosine similarity of mask pair')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0][0].set_ylabel('density')
    axes[0][0].legend(fontsize=8, frameon=False, loc='upper left')
    fig.suptitle(title, fontsize=11)
    savefig(fig, path)


def summary_bars(table, path):
    """Run-level comparison of features: one panel per metric, mean ± std over frames."""
    metrics = [('mask_auc', 'mask-pair AUC, all masks'),
               ('mask_pure_auc', 'mask-pair AUC, purity-filtered'),
               ('mask_pure_nn_hit_rate', 'NN-hit rate, purity-filtered'),
               ('mask_gap', 'mean sim gap (same - diff)')]
    names = list(table)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 3.4))
    for ax, (key, label) in zip(axes, metrics):
        mean = [table[n][key]['mean'] for n in names]
        std = [table[n][key]['std'] for n in names]
        ax.bar(range(len(names)), mean, yerr=std, width=0.55, color=SAME_COLOR,
               error_kw=dict(ecolor=TEXT_SECONDARY, lw=1, capsize=3))
        for i, v in enumerate(mean):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8, color='#0b0b0b')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#e5e4df', lw=0.6)
        ax.set_axisbelow(True)
        if key.endswith('auc'):
            ax.axhline(0.5, color=TEXT_SECONDARY, lw=0.8, ls='--')
            ax.set_ylim(0.4, 1.0)
    savefig(fig, path)
