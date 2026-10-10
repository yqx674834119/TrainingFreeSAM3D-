"""SAM masks -> mutually exclusive groups, and grouping of uncovered points."""
import numpy as np
from sklearn.cluster import KMeans


def exclusive_group_map(masks):
    """Paint masks in descending-area order so smaller masks win overlaps (ESAM).

    Returns an HxW map whose value is the *original* mask index, or -1 where
    no mask covers the pixel. Masks fully hidden by smaller ones disappear.
    """
    masks = np.asarray(masks, bool)
    group = np.full(masks.shape[1:], -1, np.int64)
    if len(masks) == 0:
        return group
    areas = masks.reshape(len(masks), -1).sum(1)
    for i in np.argsort(-areas, kind='stable'):
        group[masks[i]] = i
    return group


def fill_uncovered(group_ids, fill_features, n_clusters=20, min_points=20, seed=0, base=None):
    """Assign ``-1`` points to new groups with KMeans, as ESAM does.

    ``group_ids`` holds mask indices (>=0) or -1. New groups get ids
    ``base, base+1, ...``; ``base`` defaults to ``max+1`` like ESAM.
    """
    group_ids = np.asarray(group_ids, np.int64).copy()
    missing = group_ids < 0
    if not missing.any():
        return group_ids
    if base is None:
        base = group_ids.max() + 1 if (~missing).any() else 0
    x = np.asarray(fill_features, np.float64)[missing]
    if len(x) < min_points:
        labels = np.zeros(len(x), np.int64)
    else:
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(x).labels_
    group_ids[missing] = labels + base
    return group_ids


def compact_ids(group_ids):
    """Relabel to ``0..G-1``; return the new ids and the original id per group."""
    unique, compact = np.unique(group_ids, return_inverse=True)
    return compact.astype(np.int64), unique
