"""阶段 4 可视化：实例特征头的训练曲线（只画图，不做算法）。"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def training_curves(history, out_png, title=''):
    """history: [{epoch, loss, pos_cos, neg_cos, pool, tau, sec}] -> 2x2 曲线图。"""
    ep = [h['epoch'] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.5))
    ax = axes[0, 0]
    ax.plot(ep, [h['loss'] for h in history], marker='o')
    ax.set_title('InfoNCE loss')
    ax = axes[0, 1]
    ax.plot(ep, [h['pos_cos'] for h in history], marker='o', label='same instance')
    ax.plot(ep, [h['neg_cos'] for h in history], marker='s', label='different instance')
    ax.legend()
    ax.set_title('mean cosine')
    ax = axes[1, 0]
    ax.plot(ep, [h['tau'] for h in history], marker='o')
    ax.set_title('temperature')
    ax = axes[1, 1]
    ax.plot(ep, [h['pool'] for h in history], marker='o')
    ax.set_title('points per step (anchors + mined)')
    for ax in axes.ravel():
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
