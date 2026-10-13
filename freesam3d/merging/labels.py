"""GT edge labels: do two adjacent groups belong to the same instance?

Definition shared with ``evaluation.instance.oracle_merge``: a group is
*pure* if, among its labeled points, the majority instance holds >= purity
and labeled points are >= min_labeled of the group. Only pure-pure edges get
a label; the rest are masked out of the loss (they need splitting, which a
merge-only model cannot express).

The cache stores, per group, the majority instance and its purity, so the
purity threshold can be chosen at training time (``relabel``).
"""
import numpy as np


def group_majority(groups, gt, num_groups, purity=0.8, min_labeled=0.3):
    """Return (major_id, is_pure) per group; major_id = 0 when unlabeled."""
    major, purity_of, labeled_frac = group_purity(groups, gt, num_groups)
    pure = (labeled_frac >= min_labeled) & (purity_of >= purity) & (major > 0)
    return major, pure


def group_purity(groups, gt, num_groups):
    """Per group: majority GT id (0 = none), purity among labeled points, labeled fraction."""
    major = np.zeros(num_groups, np.int64)
    best = np.zeros(num_groups, np.int64)
    labeled = gt > 0
    sizes = np.bincount(groups, minlength=num_groups)
    lab_count = np.bincount(groups[labeled], minlength=num_groups)
    pairs = np.stack([groups[labeled], gt[labeled]], 1)
    if len(pairs):
        uniq, counts = np.unique(pairs, axis=0, return_counts=True)
        for (g, k), c in zip(uniq, counts):
            if c > best[g]:
                best[g], major[g] = c, k
    purity = best / np.maximum(lab_count, 1)
    return major, purity.astype(np.float32), (lab_count / np.maximum(sizes, 1)).astype(np.float32)


def edge_labels(groups, gt, edges, purity=0.8, min_labeled=0.3):
    """(y, valid): y[e] = 1 if both ends are pure and share the majority GT."""
    major, pure = group_majority(groups, gt, int(groups.max()) + 1, purity, min_labeled)
    return labels_from_majority(major, pure, edges)


def labels_from_majority(major, pure, edges):
    a, b = edges[:, 0], edges[:, 1]
    valid = pure[a] & pure[b]
    y = (major[a] == major[b]).astype(np.float32)
    return y, valid


def relabel(major, purity_of, labeled_frac, edges, purity, min_labeled):
    """Edge labels from cached per-group statistics at a chosen purity threshold."""
    pure = (labeled_frac >= min_labeled) & (purity_of >= purity) & (major > 0)
    return labels_from_majority(major, pure, edges)


def objectness_labels(major, purity_of, labeled_frac, sem_major, stuff, purity=0.8, min_labeled=0.3):
    """Per-group target for the objectness head: pure (>= purity), labeled, majority instance is a *thing*.

    ``sem_major`` is the majority semantic id per group; ``stuff`` the stuff id set in the same id space.
    Groups that are unlabeled or mixed are masked out (valid=False) rather than used as negatives:
    "unlabeled" in ScanNet frequently means "object not annotated", not "not an object".
    """
    is_thing = (major > 0) & ~np.isin(sem_major, list(stuff)) & (sem_major > 0)
    is_stuff = (major > 0) & np.isin(sem_major, list(stuff))
    labeled = (labeled_frac >= min_labeled) & (purity_of >= purity)
    y = is_thing.astype(np.float32)
    valid = labeled & (is_thing | is_stuff)
    return y, valid


def group_gt_votes(groups, gt, num_groups):
    """Sparse per-group GT votes: (node, gt_id, count) over the labeled points of each group.

    Used by the window memory (oracle / training mode) where a node's GT votes are an
    additive statistic accumulated over the fragments that were glued into it.
    """
    lab = gt > 0
    pairs = np.stack([groups[lab], gt[lab]], 1)
    if len(pairs) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    return uniq[:, 0].astype(np.int64), uniq[:, 1].astype(np.int64), counts.astype(np.int64)
