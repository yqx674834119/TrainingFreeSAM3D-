"""Is the FastSAM feature worth learning from? Answered before training.

Edge-level question: can we tell "same instance" from "different instance"
for adjacent group pairs? We compare, with scene-held-out cross-validation
(no frame of a validation scene is seen in training):

  geometry      — 10 geometric / colour / confidence scalars
  + group_cos   — plus the pooled-feature cosine
  + boundary    — plus the boundary (proto) contrast
  + nested      — plus raw-mask nesting
  all           — all 14 edge scalars

A logistic-regression AUC on these columns is the training-free lower bound;
the gain of "+ feature" columns over "geometry" is the signal the feature
carries beyond geometry. Also reported: each single column's AUC.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from .graph import edge_feature_names

GROUPS = {
    'geometry': ['min_dist', 'contact_norm', 'centroid_dist', 'height_diff', 'extent_ratio', 'size_ratio',
                 'color_dist', 'conf_diff', 'normal_angle', 'contact_depth_gap'],
    '+group_cos': ['group_cos'],
    '+boundary': ['boundary_cos', 'boundary_cos_min'],
    '+nested': ['nested'],
}


def _cols(names):
    all_names = edge_feature_names()
    return [all_names.index(n) for n in names]


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float('nan')


def _cv_auc(x, y, scene, folds=5, seed=0):
    """Scene-grouped k-fold logistic regression; returns mean AUC and AP over folds."""
    scenes = np.unique(scene)
    rng = np.random.default_rng(seed)
    rng.shuffle(scenes)
    fold_of = {s: i % folds for i, s in enumerate(scenes)}
    f = np.array([fold_of[s] for s in scene])
    aucs, aps = [], []
    for k in range(folds):
        tr, te = f != k, f == k
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
        clf = LogisticRegression(max_iter=500, class_weight='balanced', C=1.0).fit((x[tr] - mu) / sd, y[tr])
        p = clf.predict_proba((x[te] - mu) / sd)[:, 1]
        aucs.append(_auc(y[te], p))
        aps.append(float(average_precision_score(y[te], p)))
    return float(np.mean(aucs)), float(np.mean(aps))


def probe(x, y, scene, folds=5, seed=0):
    """Return a report dict: cumulative column groups, every single column, and class balance."""
    report = dict(edges=int(len(y)), positive_rate=float(y.mean()), scenes=int(len(np.unique(scene))), cumulative={},
                  single={})
    cols = []
    for name, members in GROUPS.items():
        cols += _cols(members)
        auc, ap = _cv_auc(x[:, cols], y, scene, folds, seed)
        report['cumulative'][name] = dict(columns=len(cols), auc=auc, ap=ap)
    for name in edge_feature_names():
        c = _cols([name])[0]
        s = x[:, c]
        a = _auc(y, s)
        report['single'][name] = dict(auc=max(a, 1 - a), sign='+' if a >= 0.5 else '-')
    # feature groups alone, without geometry, to see their standalone value
    for name, members in GROUPS.items():
        if name != 'geometry':
            auc, ap = _cv_auc(x[:, _cols(members)], y, scene, folds, seed)
            report['cumulative'][name + ' alone'] = dict(columns=len(members), auc=auc, ap=ap)
    return report


def to_markdown(report, title):
    lines = [f'# {title}', '', f"edges {report['edges']}, positive rate {report['positive_rate']:.3f}, "
             f"scenes {report['scenes']}, scene-grouped 5-fold CV", '',
             '## Cumulative column groups (logistic regression)', '', '| columns | #cols | AUC | AP |', '|---|---|---|---|']
    for name, r in report['cumulative'].items():
        lines.append(f"| {name} | {r['columns']} | {r['auc']:.3f} | {r['ap']:.3f} |")
    lines += ['', '## Single columns (raw value as score)', '', '| column | AUC | direction |', '|---|---|---|']
    for name, r in sorted(report['single'].items(), key=lambda kv: -kv[1]['auc']):
        lines.append(f"| {name} | {r['auc']:.3f} | {r['sign']} |")
    return '\n'.join(lines) + '\n'
