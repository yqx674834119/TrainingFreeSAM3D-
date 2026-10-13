"""Stage-1 probe: which pair features carry signal for the *cross-frame* merge decision?

Stage 2's plan replaces the hand-written cross-frame rule with a learned edge scorer, so the
question here is whether the graph's edge features separate "same object" from "different object"
across frames, and which relations are hard. Everything is a *diagnostic*: labels come from GT
(a node's majority instance and purity over its accumulated GT votes), nothing feeds a merge, and
every number is an analysis figure / upper bound, never a method result.

The unit is one edge of ``WindowMemory``'s per-frame graph with ``window = 1``: current-frame
fragments (node type 0) against the frozen objects they touch (type 2), plus frozen-object pairs.
``pair_feature_names()`` has 29 columns; the 14 cached intra-frame features are identically zero on
these edges (they are not intra-frame edges), so only the 15 cross-frame columns can carry signal.
"""
import numpy as np
from scipy.stats import rankdata

from .pair_features import cross_feature_names, pair_feature_names
from .two_step import as_bool_np

POS_PURITY = 0.6
PART_WHOLE_CONTAIN = 0.8
RELATIONS = ('reobserve', 'part_whole', 'adjacent')
RELATION_LABELS = {'reobserve': '同一物体再观测', 'part_whole': '部件–整体（同 GT 且包含度 ≥0.8）',
                   'adjacent': '相邻不同物体'}


def relation_of(same_gt, contain_max, part_whole=PART_WHOLE_CONTAIN):
    """Name one labeled edge: same object re-observed, part-whole of one object, or adjacent objects."""
    if not same_gt:
        return 'adjacent'
    return 'part_whole' if contain_max >= part_whole else 'reobserve'


def collect_cross_edges(frames, variant, model, std, cfg, device, purity=POS_PURITY):
    """Stream the frames through ``variant``'s memory and keep every labeled *cross-frame* edge.

    Returns flat aligned arrays over all frames (``pair_feat`` (E, 29), ``y`` 1 = same instance,
    ``valid`` = both ends pure enough to label, ``relation``, ``scene``, ``frame_index``,
    ``types`` (E, 2)) plus ``nodes`` — one dict per frame with that frame's ``major``/``purity``/
    ``prev_obj`` per graph node, which the top-1 ranking needs.
    """
    from .holdout import make_memory
    mem = make_memory(variant, model, std, cfg, device)
    names = pair_feature_names()
    ca, cb = names.index('contain_a_in_b'), names.index('contain_b_in_a')
    acc = {k: [] for k in ('pair_feat', 'y', 'valid', 'relation', 'scene', 'frame_index', 'types', 'edges')}
    nodes = []
    for fi, fr in enumerate(frames):
        mem.update(fr)
        g = mem.last_graph
        nodes.append(dict(major=np.asarray(g.node_major, np.int64),
                          purity=np.asarray(g.node_purity, np.float32),
                          prev_obj=np.asarray(g.node_prev_obj, np.int64),
                          node_type=np.asarray(g.node_type, np.int64),
                          num_cur=int(np.count_nonzero(np.asarray(g.node_type) == 0)),
                          scene=fr.scene, frame_id=fr.frame_id))
        cross = ~as_bool_np(g.has_intra)
        if not cross.any():
            continue
        sel = np.flatnonzero(cross)
        pf = g.pair_feat.cpu().numpy()[sel]
        pure = (nodes[-1]['purity'] >= purity) & (nodes[-1]['major'] > 0)
        a, b = g.edges[sel, 0], g.edges[sel, 1]
        same = nodes[-1]['major'][a] == nodes[-1]['major'][b]
        contain = np.maximum(pf[:, ca], pf[:, cb])
        acc['pair_feat'].append(pf)
        acc['y'].append(same.astype(np.int8))
        acc['valid'].append(pure[a] & pure[b])
        acc['relation'].append(np.asarray([relation_of(bool(s), float(c)) for s, c in zip(same, contain)]))
        acc['scene'].append(np.full(len(sel), fr.scene, dtype=object))
        acc['frame_index'].append(np.full(len(sel), fi, np.int64))
        acc['types'].append(np.stack([nodes[-1]['node_type'][a], nodes[-1]['node_type'][b]], 1))
        raw = g.edges
        raw = raw.detach().cpu().numpy() if hasattr(raw, 'detach') else np.asarray(raw)
        acc['edges'].append(raw.astype(np.int64)[sel])
    out = dict(nodes=nodes, names=names)
    empty = {'pair_feat': lambda: np.zeros((0, len(names))), 'edges': lambda: np.zeros((0, 2), np.int64)}
    for k, v in acc.items():
        out[k] = np.concatenate(v) if v else empty.get(k, lambda: np.zeros(0))()
    return out


def column_auc(x, y):
    """AUC of one feature column used directly as a ranking score.

    A constant column scores 0.5 — it cannot discriminate; ``single_column_auc`` reports the
    non-zero share next to it so an uninformative column is not mistaken for a useless-but-live one.
    """
    x, y = np.asarray(x, np.float64), np.asarray(y, bool)
    if len(x) == 0 or y.all() or not y.any():
        return float('nan')
    r = rankdata(x)                                   # average ranks, so ties score 0.5
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def single_column_auc(pair_feat, y, names=None):
    """Per column: AUC, the share of non-zero values, and the mean on each side of the label."""
    names = names or pair_feature_names()
    y = np.asarray(y, bool)
    rows = {}
    for i, n in enumerate(names):
        col = np.asarray(pair_feat[:, i], np.float64)
        rows[n] = dict(auc=column_auc(col, y), nonzero=float(np.mean(col != 0)),
                       mean_pos=float(col[y].mean()) if y.any() else float('nan'),
                       mean_neg=float(col[~y].mean()) if (~y).any() else float('nan'))
    return rows


def logistic_auc(pair_feat, y, groups, folds=5, C=1.0):
    """Out-of-fold logistic-regression AUC, folds taken over *scenes* so no scene leaks across them."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    y, groups = np.asarray(y, int), np.asarray(groups)
    uniq = np.unique(groups)
    if len(uniq) < 2 or y.sum() == 0 or y.sum() == len(y):
        return float('nan')
    fold_of = {s: i % min(folds, len(uniq)) for i, s in enumerate(uniq)}
    fold = np.asarray([fold_of[s] for s in groups])
    scores = np.full(len(y), np.nan)
    for f in range(min(folds, len(uniq))):
        te, tr = fold == f, fold != f
        if not te.any() or y[tr].sum() == 0 or y[tr].sum() == tr.sum():
            continue
        sc = StandardScaler().fit(pair_feat[tr])
        lr = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(pair_feat[tr]), y[tr])
        scores[te] = lr.predict_proba(sc.transform(pair_feat[te]))[:, 1]
    ok = ~np.isnan(scores)
    if not ok.any() or y[ok].sum() == 0 or y[ok].sum() == ok.sum():
        return float('nan')
    return float(roc_auc_score(y[ok], scores[ok]))


def cross_columns(pair_feat, names=None):
    """The 15 cross-frame columns of a (E, 29) block (the 14 intra ones are zero on these edges)."""
    names = names or pair_feature_names()
    off = len(names) - len(cross_feature_names())
    return np.asarray(pair_feat)[:, off:]


def rule_scores_of(pair_feat, rule='p2a', names=None):
    """The cross-frame rule's own score per edge, read off the cached columns."""
    names = names or pair_feature_names()
    ca, cb, fc = names.index('contain_a_in_b'), names.index('contain_b_in_a'), names.index('feat_cos')
    contain = np.maximum(pair_feat[:, ca], pair_feat[:, cb])
    if rule == 'p2b':
        return contain * np.clip(pair_feat[:, fc], 0, 1)
    if rule != 'p2a':
        raise ValueError(f'unknown rule {rule!r}')
    return contain


RELATION_PAIRS = (('reobserve', 'adjacent'), ('part_whole', 'adjacent'), ('reobserve', 'part_whole'))
RELATION_PAIR_LABELS = {
    ('reobserve', 'adjacent'): '同一物体再观测 vs 相邻不同物体',
    ('part_whole', 'adjacent'): '部件-整体 vs 相邻不同物体',
    ('reobserve', 'part_whole'): '同一物体再观测 vs 部件-整体',
}


def _cross_off(names):
    """First column of the cross-frame block of ``pair_feature_names()``."""
    return len(names) - len(cross_feature_names())


def cross_column_rows(pair_feat, y, names=None):
    """Per *cross-frame* column, in layout order, as rows instead of a name-keyed dict.

    ``pair_feature_names()`` repeats three names (``centroid_dist``, ``height_diff`` and
    ``extent_ratio`` sit in both the intra and the cross block), so a dict keyed by name keeps only
    one of the two rows and the reader cannot tell which. The 14 intra columns are identically zero
    on cross-frame edges, so only the 15 cross columns are reported.
    """
    names = names or pair_feature_names()
    off = _cross_off(names)
    y = np.asarray(y, bool)
    rows = []
    for i, n in enumerate(cross_feature_names()):
        col = np.asarray(pair_feat[:, off + i], np.float64)
        rows.append(dict(name=n, index=off + i, auc=column_auc(col, y),
                         nonzero=float(np.mean(col != 0)),
                         mean_pos=float(col[y].mean()) if y.any() else float('nan'),
                         mean_neg=float(col[~y].mean()) if (~y).any() else float('nan')))
    return rows


def relation_pair_auc(pair_feat, y, relation, scene, a, b, names=None):
    """AUC *between* two relations - the only well-posed reading of a relation stratum.

    ``relation_of`` is a function of the label (different GT -> adjacent, same GT -> reobserve or
    part_whole), so every edge inside one relation carries the same label and an AUC within it is
    undefined. Comparing two relations is fine: the label becomes "is this relation ``a``".
    """
    pair_feat, relation = np.asarray(pair_feat), np.asarray(relation)
    sel = (relation == a) | (relation == b)
    if not sel.any():
        return dict(n=0, positive=0, logistic=float('nan'), best_column='-', best_auc=float('nan'))
    yy = relation[sel] == a
    rows = cross_column_rows(pair_feat[sel], yy, names)
    best = max(rows, key=lambda r: (r['auc'] if r['auc'] == r['auc'] else -1.0))
    return dict(n=int(sel.sum()), positive=int(yy.sum()),
                logistic=logistic_auc(cross_columns(pair_feat[sel], names), yy.astype(int),
                                      np.asarray(scene)[sel]),
                best_column=best['name'], best_auc=best['auc'])


def touching_negative_auc(pair_feat, y, scene, names=None, touch_column='inter0_frac_min'):
    """Positive edges against the *touching* negatives only - the discrimination that matters.

    A negative that does not touch the fragment is easy. The interesting failure mode of the
    cross-frame rule is a *different* object that does touch it, so this keeps only negatives with a
    non-zero voxel intersection.
    """
    names = names or pair_feature_names()
    off = _cross_off(names)
    pair_feat, y = np.asarray(pair_feat), np.asarray(y, bool)
    touch = np.asarray(pair_feat[:, off + cross_feature_names().index(touch_column)], np.float64) > 0
    sel = y | (~y & touch)
    n_neg = int((~y & touch).sum())
    if not sel.any() or y[sel].all() or not y[sel].any():
        return dict(n=int(sel.sum()), n_negative=n_neg, logistic=float('nan'), rule_auc=float('nan'))
    ca = off + cross_feature_names().index('contain_a_in_b')
    cb = off + cross_feature_names().index('contain_b_in_a')
    rule = np.maximum(pair_feat[sel, ca], pair_feat[sel, cb])
    return dict(n=int(sel.sum()), n_negative=n_neg,
                logistic=logistic_auc(cross_columns(pair_feat[sel], names), y[sel].astype(int),
                                      np.asarray(scene)[sel]),
                rule_auc=column_auc(rule, y[sel]))


def top1_stats(edges, types, frame_index, nodes, scores, purity=POS_PURITY):
    """Top-1 ranking quality of each current fragment's historical-object candidates.

    For every current-frame fragment with at least one scored edge to a frozen object whose GT is
    known, rank those objects by ``scores`` and look at where the *correct* one (same majority
    instance) lands. Edges between two historical objects take no part: there is no "current" end.

    Returns a dict: ``n_fragments`` (fragments that had a labeled candidate), ``n_with_correct``,
    ``accuracy`` (share whose best candidate is correct), ``mean_best_rank`` and
    ``mean_distractors_above`` (how many wrong objects outrank the correct one — the quantity
    stage 0c measured by hand in F0c.7), and ``n_no_correct_candidate``.
    """
    first = ranks = distract = 0
    n_frag = n_with = n_none = 0
    for fi in np.unique(frame_index):
        sel = np.asarray(frame_index) == fi
        e, t, s = np.asarray(edges)[sel], np.asarray(types)[sel], np.asarray(scores)[sel]
        major, pur = nodes[int(fi)]['major'], nodes[int(fi)]['purity']
        cand = {}
        for k in range(len(e)):
            a, b = int(e[k, 0]), int(e[k, 1])
            if t[k, 0] != 0 and t[k, 1] != 0:
                continue
            cur, hist = (a, b) if t[k, 0] == 0 else (b, a)
            if pur[cur] < purity or major[cur] <= 0 or pur[hist] < purity or major[hist] <= 0:
                continue
            cand.setdefault(cur, []).append((float(s[k]), int(major[hist] == major[cur])))
        for lst in cand.values():
            lst.sort(key=lambda z: -z[0])
            n_frag += 1
            best = next((i + 1 for i, (_, hit) in enumerate(lst) if hit), None)
            if best is None:
                n_none += 1
                continue
            n_with += 1
            first += int(best == 1)
            ranks += best
            distract += best - 1
    return dict(n_fragments=n_frag, n_with_correct=n_with, n_no_correct_candidate=n_none,
                accuracy=(first / n_with if n_with else float('nan')),
                mean_best_rank=(ranks / n_with if n_with else float('nan')),
                mean_distractors_above=(distract / n_with if n_with else float('nan')))
