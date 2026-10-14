"""Stage-3 holdout evaluation: one streaming pass, many ranking-side variants.

The stage-2 sweep re-streams every scene once per threshold triple and scores a single ranking
(the model's node quality). Stage 3 asks ranking-side questions - containment suppression at
several thresholds, the oracle ceiling - that do not change the streamed objects at all, so this
module streams once per (triple, memory override) and then scores every ranking variant on the
same objects and the same voxel GT.

GT construction and metrics are exactly ``evaluate.voxel_scene_eval``'s: 5 cm cells, cross-frame
majority instance, thing and >= ``min_gt_voxels`` cells, greedy score-ordered matching
(class-agnostic AP at 0.5 / 0.25, mean best IoU, recall@0.5, plus ``association_stats``).
``tests/test_phase3.py`` asserts the two agree on a synthetic scene.
"""
import numpy as np
import torch

from ..evaluation.instance import GTSet, association_stats, iou_matrix
from .evaluate import scene_metrics, voxel_majority_gt
from .fragment_fate import object_voxel_sets
from .holdout import load_scene_frames
from .unified_dedup import dedup_objects
from .unified_flow import make_unified_memory

RANK_KEYS = ('ap50', 'ap25', 'iou', 'r50', 'masks', 'dup_rate', 'overmerge')


def voxel_gt(frames, stuff, voxel=0.05, min_gt_voxels=20):
    """(cells, GTSet | None) of one streamed scene - the GT ``voxel_scene_eval`` builds internally."""
    cells, gt_vox, sem_vox = voxel_majority_gt(frames, voxel)
    ids, counts = np.unique(gt_vox[gt_vox > 0], return_counts=True)
    stuff = set(stuff)
    keep = []
    for k, n in zip(ids, counts):
        if n < min_gt_voxels:
            continue
        s = sem_vox[gt_vox == k]
        s = s[s > 0]
        if len(s) and int(np.bincount(s).argmax()) in stuff:
            continue
        keep.append(int(k))
    return cells, (GTSet.from_labels(gt_vox, keep) if keep else None)


def pred_masks(obj_vox, cells, device='cpu'):
    """(K, V) bool masks of the object voxel sets over the scene's cell array."""
    masks = np.zeros((len(obj_vox), len(cells)), bool)
    for i, v in enumerate(obj_vox):
        v = v.cpu().numpy() if torch.is_tensor(v) else np.asarray(v, np.int64)
        if len(v) == 0:
            continue
        pos = np.clip(np.searchsorted(cells, v), 0, len(cells) - 1)
        masks[i, pos[cells[pos] == v]] = True
    return masks


def eval_objects(masks, scores, gts, min_pred_voxels=20, device='cpu'):
    """One metrics dict for a (masks, scores) variant; no GT gives NaN AP, like ``voxel_scene_eval``."""
    if gts is None:
        return dict(ap50=float('nan'), ap25=float('nan'), iou=0.0, r50=0.0, masks=0,
                    dup_rate=float('nan'), overmerge=float('nan'))
    met = scene_metrics(masks, np.asarray(scores, np.float32), gts, min_points=min_pred_voxels,
                        device=device)
    st = association_stats(masks, gts, device=device)
    met.update(dup_rate=st['dup_rate'], overmerge=st['overmerge_rate'])
    return met


def oracle_scores(masks, gts, device='cpu'):
    """Best voxel IoU per prediction - the perfect-ranking ceiling (GT used; a bound, never a result)."""
    if gts is None or len(masks) == 0:
        return np.zeros(len(masks), np.float32)
    return iou_matrix(np.asarray(masks), gts, device).max(1).astype(np.float32)


def stream_scene(scene, cfg, model, std, device, cache_dir, mem_overrides=None):
    """Stream one holdout scene -> (frames, mem, obj_vox, attrs, merges)."""
    frames = load_scene_frames(cache_dir, scene, gt='2d')
    mem = make_unified_memory(cfg, model, std, device, **(mem_overrides or {}))
    merges = []
    for fr in frames:
        mem.update(fr)
        merges.extend(mem.last_unified['merged'])
    return frames, mem, object_voxel_sets(mem), mem.object_attrs(), merges


def rank_variants(obj_vox, quality, cells, gts, dedup_thresholds=(), device='cpu'):
    """{name: metrics} for quality / quality + dedup@t / oracle, all on the same streamed objects."""
    masks = pred_masks(obj_vox, cells, device)
    quality = np.asarray(quality, np.float32)
    out = {'quality': eval_objects(masks, quality, gts, device=device)}
    for thr in dedup_thresholds:
        keep = dedup_objects(obj_vox, quality, thr, device=device)
        out[f'dedup_{float(thr):g}'] = eval_objects(masks[keep], quality[keep], gts, device=device)
    out['oracle'] = eval_objects(masks, oracle_scores(masks, gts, device), gts, device=device)
    return out


def aggregate(rows, keys=RANK_KEYS):
    """Scene-mean of the per-scene metrics (nan-safe), the stage-3 metric set."""
    out = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    out['scenes'] = len(rows)
    return out
