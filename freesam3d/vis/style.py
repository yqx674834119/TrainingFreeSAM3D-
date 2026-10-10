"""Shared colors and figure saving."""
import colorsys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Single-hue sequential ramp (light -> dark blue) for similarity magnitude.
SIM_CMAP = LinearSegmentedColormap.from_list(
    'sim_blue', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'])
SAME_COLOR = '#2a78d6'   # same GT instance pairs
DIFF_COLOR = '#eb6834'   # different GT instance pairs
TEXT_SECONDARY = '#52514e'


def id_colors(ids):
    """Deterministic, well-spread colors for arbitrary integer ids (segment maps)."""
    ids = np.asarray(ids, np.int64)
    hue = (ids * 0.618033988749895) % 1.0
    sat = 0.55 + 0.35 * ((ids * 7) % 3) / 2
    val = 0.95 - 0.25 * ((ids * 5) % 2)
    return np.array([colorsys.hsv_to_rgb(h, s, v) for h, s, v in zip(hue, sat, val)], np.float32).reshape(ids.shape + (3,))


def savefig(fig, path, dpi=110):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
