"""Does pooled FastSAM feature similarity separate GT instances?

Two views per feature:
* GT view   — pool point features inside each GT instance; instance x instance
  cosine. Tells whether *different* objects are distinguishable at all.
* mask view — FastSAM groups (KMeans-filled groups excluded), each tagged with
  its majority GT instance. Pairs of masks on the same GT instance should be
  more similar than pairs on different instances; that is exactly what a
  later similarity-based merging step needs.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from ..lifting.pooling import pool

UNLABELED = 'unlabeled'


@dataclass
class SimilarityView:
    sim: np.ndarray          # (K,K) cosine similarity
    labels: list             # tick label per row
    gt_ids: np.ndarray       # (K,) GT instance id per row, 0 = unlabeled
    categories: list         # category per row
    purity: np.ndarray = None  # (K,) majority-GT purity (mask view only)


def majority_gt(group_ids, gt_instance, num_groups):
    """Majority GT instance, its point fraction (purity) and point count per group."""
    pairs = np.stack([group_ids, gt_instance], 1)
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    major = np.zeros(num_groups, np.int64)
    best = np.zeros(num_groups, np.int64)
    for (g, inst), c in zip(uniq, counts):
        if c > best[g]:
            best[g], major[g] = c, inst
    sizes = np.bincount(group_ids, minlength=num_groups)
    return major, best / np.maximum(sizes, 1), sizes


def pair_metrics(sim, gt_ids):
    """Same-GT vs different-GT mask pairs (unlabeled rows ignored)."""
    keep = gt_ids > 0
    s, g = sim[np.ix_(keep, keep)], gt_ids[keep]
    iu = np.triu_indices(len(g), 1)
    vals, same = s[iu], (g[:, None] == g[None, :])[iu]
    res = dict(n_masks=int(keep.sum()), n_pairs=int(len(vals)), n_same_pairs=int(same.sum()),
               mean_same=float(vals[same].mean()) if same.any() else float('nan'),
               mean_diff=float(vals[~same].mean()) if (~same).any() else float('nan'),
               auc=float(roc_auc_score(same, vals)) if 0 < same.sum() < len(same) else float('nan'))
    res['gap'] = res['mean_same'] - res['mean_diff']
    # For masks whose GT instance is split into >=2 masks: is the most similar
    # other mask on the same instance? (what greedy merging would pick)
    s_nn = s.copy()
    np.fill_diagonal(s_nn, -np.inf)
    split = np.array([(g == x).sum() >= 2 for x in g], bool)
    res['nn_hit_rate'] = float((g[s_nn[split].argmax(1)] == g[split]).mean()) if split.any() else float('nan')
    res['n_split_masks'] = int(split.sum())
    return res, vals, same


def gt_metrics(sim, categories):
    off = ~np.eye(len(sim), dtype=bool)
    cats = np.asarray(categories)
    same_cat = (cats[:, None] == cats[None, :]) & off
    return dict(n_instances=len(sim),
                mean_offdiag=float(sim[off].mean()) if off.any() else float('nan'),
                mean_same_category=float(sim[same_cat].mean()) if same_cat.any() else float('nan'),
                mean_diff_category=float(sim[off & ~same_cat].mean()) if (off & ~same_cat).any() else float('nan'),
                mean_max_offdiag=float(np.where(off, sim, -np.inf).max(1).mean()) if len(sim) > 1 else float('nan'))


def _order(gt_ids, categories, sizes):
    key = [(c == UNLABELED, c, int(g), -int(n)) for g, c, n in zip(gt_ids, categories, sizes)]
    return sorted(range(len(key)), key=key.__getitem__)


def gt_view(point_feats, gt_instance, category_of, min_points, pooling='mean_l2'):
    ids, counts = np.unique(gt_instance[gt_instance > 0], return_counts=True)
    ids = ids[counts >= min_points]
    # Pool every point (others go to a dummy last group) so that centering, if
    # used, subtracts the same whole-frame mean as in lifting.
    remap = np.full(len(gt_instance), len(ids), np.int64)
    sel = np.isin(gt_instance, ids)
    remap[sel] = np.searchsorted(ids, gt_instance[sel])
    feats = pool(pooling, point_feats, remap, len(ids) + 1)[:len(ids)].numpy()

    cats = [category_of.get(int(i), '?') for i in ids]
    order = _order(ids, cats, np.zeros(len(ids)))
    ids, cats, feats = ids[order], [cats[i] for i in order], feats[order]
    return SimilarityView(feats @ feats.T, [f'{c} #{i}' for c, i in zip(cats, ids)], ids, cats)


def mask_view(group_feats, group_mask_index, major, purity, sizes, category_of, min_points):
    rows = np.flatnonzero((group_mask_index >= 0) & (sizes >= min_points))
    cats = [category_of.get(int(major[r]), '?') if major[r] > 0 else UNLABELED for r in rows]
    order = _order(major[rows], cats, sizes[rows])
    rows, cats = rows[order], [cats[i] for i in order]
    labels = [(f'{c} #{major[r]}' if major[r] > 0 else c) + f' | m{group_mask_index[r]} ({purity[r]:.0%})'
              for r, c in zip(rows, cats)]
    feats = group_feats[rows]
    return SimilarityView(feats @ feats.T, labels, major[rows], cats, purity[rows])


def evaluate_frame(lifted, point_gt, min_points=10, min_purity=0.8, pooling='mean_l2'):
    """Return (metrics, views, pair_samples) for one lifted frame.

    ``mask`` metrics use every FastSAM mask; ``mask_pure`` only masks whose
    majority-GT purity >= ``min_purity`` (removes GT noise / under-segmentation).
    """
    major, purity, sizes = majority_gt(lifted.group_ids, point_gt.instance, lifted.num_groups)
    is_mask = lifted.group_mask_index >= 0
    labeled = point_gt.instance > 0
    gt_ids, gt_counts = np.unique(point_gt.instance[labeled], return_counts=True)
    big_gt = gt_ids[gt_counts >= min_points]
    masks_per_gt = np.array([((major == i) & is_mask & (sizes >= min_points)).sum() for i in big_gt])
    seg = dict(
        num_fastsam_masks=int(len(lifted.fastsam.masks)),
        num_masks_after_exclusive=int(len(np.unique(lifted.group_map[lifted.group_map >= 0]))),
        num_mask_groups_in_points=int(is_mask.sum()),
        num_fill_groups=int((~is_mask).sum()),
        uncovered_point_ratio=float((~is_mask[lifted.group_ids]).mean()),
        point_weighted_purity=float((purity * sizes)[is_mask].sum() / max(sizes[is_mask].sum(), 1)),
        gt_labeled_point_ratio=float(labeled.mean()),
        gt_knn_distance_median=float(np.median(point_gt.knn_distance)),
        num_gt_instances=int(len(big_gt)),
        num_gt_split_into_multiple_masks=int((masks_per_gt >= 2).sum()),
        mean_masks_per_gt=float(masks_per_gt.mean()) if len(masks_per_gt) else float('nan'),
    )
    metrics, views, pairs = dict(segmentation=seg, features={}), {}, {}
    for name in lifted.group_feats:
        gv = gt_view(lifted.point_feats[name], point_gt.instance, point_gt.categories, min_points, pooling)
        mv = mask_view(lifted.group_feats[name].numpy(), lifted.group_mask_index, major, purity, sizes,
                       point_gt.categories, min_points)
        pm, vals, same = pair_metrics(mv.sim, mv.gt_ids)
        pure = np.asarray(mv.purity) >= min_purity
        pm_pure, _, _ = pair_metrics(mv.sim[np.ix_(pure, pure)], mv.gt_ids[pure])
        metrics['features'][name] = dict(gt=gt_metrics(gv.sim, gv.categories), mask=pm, mask_pure=pm_pure)
        views[name] = dict(gt=gv, mask=mv)
        pairs[name] = (vals, same)
    return metrics, views, pairs
