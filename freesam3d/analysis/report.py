"""Aggregate per-frame metrics into a feature-comparison table."""
import numpy as np

ROW_KEYS = [('mask', 'auc'), ('mask_pure', 'auc'), ('mask', 'nn_hit_rate'), ('mask_pure', 'nn_hit_rate'),
            ('mask', 'gap'), ('mask', 'mean_same'),
            ('mask', 'mean_diff'), ('gt', 'mean_offdiag'), ('gt', 'mean_same_category'),
            ('gt', 'mean_diff_category'), ('gt', 'mean_max_offdiag')]
SEG_KEYS = ['num_fastsam_masks', 'num_masks_after_exclusive', 'num_mask_groups_in_points', 'num_fill_groups',
            'uncovered_point_ratio', 'point_weighted_purity', 'gt_labeled_point_ratio',
            'gt_knn_distance_median', 'num_gt_instances', 'num_gt_split_into_multiple_masks']


def _stat(values):
    v = np.asarray([x for x in values if x == x], np.float64)   # drop NaN
    return dict(mean=float(v.mean()) if len(v) else float('nan'),
                std=float(v.std()) if len(v) else float('nan'), n=int(len(v)))


def aggregate(per_frame):
    """``per_frame``: {frame name: metrics from ``evaluate_frame``}."""
    frames = list(per_frame.values())
    features = list(frames[0]['features'])
    table = {f: {f'{lvl}_{k}': _stat([m['features'][f][lvl][k] for m in frames]) for lvl, k in ROW_KEYS}
             for f in features}
    seg = {k: _stat([m['segmentation'][k] for m in frames]) for k in SEG_KEYS}
    return dict(features=table, segmentation=seg, num_frames=len(frames))


def to_markdown(summary, per_frame):
    lines = [f'# Query Lifting summary ({summary["num_frames"]} frames)', '',
             '## Feature comparison (mean ± std over frames)', '',
             '| feature | ' + ' | '.join(f'{lvl}.{k}' for lvl, k in ROW_KEYS) + ' |',
             '|' + '---|' * (len(ROW_KEYS) + 1)]
    for f, row in summary['features'].items():
        lines.append(f'| {f} | ' + ' | '.join(
            f'{row[f"{lvl}_{k}"]["mean"]:.3f} ± {row[f"{lvl}_{k}"]["std"]:.3f}' for lvl, k in ROW_KEYS) + ' |')
    lines += ['', '## Segmentation / lifting statistics', '', '| stat | mean | std |', '|---|---|---|']
    for k, s in summary['segmentation'].items():
        lines.append(f'| {k} | {s["mean"]:.3f} | {s["std"]:.3f} |')
    lines += ['', '## Mask-pair AUC per frame (all masks / purity-filtered)', '',
              '| frame | ' + ' | '.join(summary['features']) + ' |', '|' + '---|' * (len(summary['features']) + 1)]
    for name, m in per_frame.items():
        lines.append(f'| {name} | ' + ' | '.join(f'{m["features"][f]["mask"]["auc"]:.3f} / {m["features"][f]["mask_pure"]["auc"]:.3f}'
                            for f in summary['features']) + ' |')
    return '\n'.join(lines) + '\n'
