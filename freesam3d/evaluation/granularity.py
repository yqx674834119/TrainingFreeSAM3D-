"""Does ESAM's decoder re-align FastSAM groups to ScanNet's instance granularity?

Per frame, on ESAM's own 20k points, compare three mask sets against the
per-frame GT ESAM itself trains on (``instance_mask/<scene>/<frame>.bin``):

* FastSAM  — the exclusive FastSAM groups (= ESAM's input queries, incl. the
             KMeans groups for uncovered points), ``super_points/*.bin``.
* ESAM     — the decoder's per-frame masks after ESAM's own post-processing,
             before cross-frame merging.
* ESAM-snap — each ESAM mask rounded to the union of the FastSAM groups that
             are >= 50% inside it. If it scores like ESAM, ESAM's gain is merging,
             not its point-level splitting.
* Oracle   — merge-only GT oracle on the FastSAM groups (upper bound of any
             method that only merges whole groups).
* Oracle-adj — same, but only through 3D-adjacent groups (< ``adj_eps``):
             the upper bound of a merger that looks at adjacent pairs only.

Plus how each ESAM mask is composed of FastSAM groups (merge vs split).
"""
import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .instance import (GTSet, class_agnostic_ap, coverage_stats, fragments_per_gt, group_inclusion,
                       groups_to_masks, oracle_merge, oracle_merge_adjacent)

STUFF = {'wall', 'floor', 'ceiling'}
METHODS = ('FastSAM', 'ESAM', 'ESAM-snap', 'Oracle-adj', 'Oracle')


def stuff_ids(label_tsv):
    with open(label_tsv) as f:
        return {int(r['id']) for r in csv.DictReader(f, delimiter='\t') if r['category'] in STUFF}


@dataclass
class FrameData:
    scene: str
    frame: int
    xyz: np.ndarray
    groups: np.ndarray       # FastSAM group id per point
    gt: np.ndarray           # GT instance id per point (0 = unlabeled)
    sem: np.ndarray          # raw ScanNet label id per point
    esam_masks: np.ndarray   # (P, N) bool
    esam_scores: np.ndarray


def load_scene_frames(mv_root, pred_npz, scene):
    mv_root = Path(mv_root)
    files = sorted((mv_root / 'points' / scene).glob('*.bin'), key=lambda p: int(p.stem))
    pred = np.load(pred_npz)
    if int(pred['num_frames']) != len(files):
        raise ValueError(f'{scene}: {len(files)} frames on disk, {int(pred["num_frames"])} predicted')
    for i, f in enumerate(files):
        def read(kind, dtype):
            return np.fromfile(mv_root / kind / scene / f.name, dtype)
        yield FrameData(scene, int(f.stem), read('points', np.float32).reshape(-1, 6)[:, :3],
                        read('super_points', np.int64), read('instance_mask', np.int64),
                        read('semantic_mask', np.int64), pred[f'frame_masks_{i}'], pred[f'frame_scores_{i}'])


@dataclass
class Accumulator:
    """Collects per-GT and per-mask records across frames."""
    gt_rows: list = field(default_factory=list)      # one dict per (frame, GT instance)
    frame_rows: list = field(default_factory=list)   # one dict per (frame, method, subset)
    esam_mask_rows: list = field(default_factory=list)
    inclusion_values: list = field(default_factory=list)


def evaluate_frame(fd: FrameData, stuff, acc: Accumulator, purity=0.8, min_gt_points=100,
                   thresholds=(0.25, 0.5), adj_eps=0.05):
    ids, sizes = np.unique(fd.gt[fd.gt > 0], return_counts=True)
    ids = ids[sizes >= min_gt_points]
    sem_of = {k: np.bincount(fd.sem[fd.gt == k]).argmax() for k in ids}
    subsets = {'thing': [k for k in ids if sem_of[k] not in stuff], 'all': list(ids)}

    fs_masks, _ = groups_to_masks(fd.groups)
    oracle_masks, _ = groups_to_masks(oracle_merge(fd.groups, fd.gt, purity))
    adj_masks, _ = groups_to_masks(oracle_merge_adjacent(fd.groups, fd.gt, fd.xyz, purity, adj_eps))
    if len(fd.esam_masks):
        snapped = (group_inclusion(fd.esam_masks, fd.groups) >= 0.5).astype(np.float32) @ fs_masks.astype(np.float32) > 0
    else:
        snapped = fd.esam_masks
    preds = {'FastSAM': (fs_masks, fs_masks.sum(1).astype(float)),     # score = size (no confidence stored)
             'ESAM': (fd.esam_masks, fd.esam_scores),
             'ESAM-snap': (snapped, fd.esam_scores),
             'Oracle-adj': (adj_masks, adj_masks.sum(1).astype(float)),
             'Oracle': (oracle_masks, oracle_masks.sum(1).astype(float))}

    for subset, keep in subsets.items():
        if not keep:
            continue
        gts = GTSet.from_labels(fd.gt, keep)
        per_method = {}
        for m, (masks, scores) in preds.items():
            cov = coverage_stats(masks, gts, thresholds)
            frag = fragments_per_gt(masks, gts, purity)
            row = dict(scene=fd.scene, frame=fd.frame, method=m, subset=subset, num_gt=len(keep),
                       num_pred=len(masks), mean_best_iou=cov['mean_best_iou'],
                       mean_fragments=float(frag.mean()))
            for t in thresholds:
                row[f'recall@{t}'] = cov[f'recall@{t}']
                row[f'precision@{t}'] = cov[f'precision@{t}']
                row[f'AP{int(t * 100)}'] = class_agnostic_ap(masks, scores, gts, t)
            acc.frame_rows.append(row)
            per_method[m] = (cov['best_iou'], frag)
        if subset == 'thing':
            for j, k in enumerate(gts.ids):
                acc.gt_rows.append(dict(scene=fd.scene, frame=fd.frame, gt=int(k), size=int(gts.sizes[j]),
                                        **{f'{m}_iou': float(per_method[m][0][j]) for m in METHODS},
                                        **{f'{m}_frags': int(per_method[m][1][j]) for m in METHODS}))

    # Composition of ESAM masks in terms of FastSAM groups.
    if len(fd.esam_masks):
        inc = group_inclusion(fd.esam_masks, fd.groups)                  # (P, G)
        gsize = np.bincount(fd.groups[fd.groups >= 0])
        gsize = gsize[gsize > 0]
        msize = fd.esam_masks.sum(1)
        for p in range(len(inc)):
            whole = inc[p] >= 0.8
            partial = (inc[p] > 0.2) & (inc[p] < 0.8)
            acc.esam_mask_rows.append(dict(
                scene=fd.scene, frame=fd.frame, size=int(msize[p]),
                groups_merged=int((inc[p] >= 0.5).sum()),
                whole_group_point_frac=float((inc[p] * gsize)[whole].sum() / max(msize[p], 1)),
                groups_split=int(partial.sum())))
        acc.inclusion_values.append(inc[inc > 0.05].ravel())


def summarize(acc: Accumulator):
    """Mean over frames (frame_rows) and over GT instances / ESAM masks."""
    out = {}
    for subset in ('thing', 'all'):
        for m in METHODS:
            rows = [r for r in acc.frame_rows if r['subset'] == subset and r['method'] == m]
            if not rows:
                continue
            keys = [k for k in rows[0] if k not in ('scene', 'frame', 'method', 'subset')]
            out[f'{subset}/{m}'] = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    g = acc.gt_rows
    if g:
        frag = np.array([r['FastSAM_frags'] for r in g])
        split = frag >= 2
        out['per_gt'] = dict(
            num_gt=len(g),
            fastsam_fragmented_ratio=float(split.mean()),
            **{f'{m}_mean_iou': float(np.mean([r[f'{m}_iou'] for r in g])) for m in METHODS},
            **{f'{m}_mean_iou_on_fragmented': float(np.mean([r[f'{m}_iou'] for r, s in zip(g, split) if s]))
               for m in METHODS if split.any()},
            **{f'{m}_recall50_on_fragmented': float(np.mean([r[f'{m}_iou'] >= 0.5 for r, s in zip(g, split) if s]))
               for m in METHODS if split.any()},
            esam_beats_best_group=float(np.mean([r['ESAM_iou'] > r['FastSAM_iou'] + 0.05 for r in g])),
            esam_worse_than_best_group=float(np.mean([r['ESAM_iou'] < r['FastSAM_iou'] - 0.05 for r in g])),
            esam_gap_to_oracle=float(np.mean([r['Oracle_iou'] - r['ESAM_iou'] for r in g])))
    e = acc.esam_mask_rows
    if e:
        merged = np.array([r['groups_merged'] for r in e])
        out['esam_masks'] = dict(
            num_masks=len(e),
            mean_groups_merged=float(merged.mean()),
            frac_masks_merging_2plus_groups=float((merged >= 2).mean()),
            mean_whole_group_point_frac=float(np.mean([r['whole_group_point_frac'] for r in e])),
            mean_groups_split=float(np.mean([r['groups_split'] for r in e])))
        inc = np.concatenate(acc.inclusion_values) if acc.inclusion_values else np.zeros(0)
        out['esam_masks']['inclusion_mid_frac'] = float(((inc > 0.2) & (inc < 0.8)).mean()) if len(inc) else float('nan')
    return out
