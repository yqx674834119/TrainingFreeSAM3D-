"""Stage-1 holdout harness: one merge configuration over one frame sequence, scored on the voxel proxy.

The 60 holdout scenes have no reconstructed mesh (only the 16 val scenes do), so the configuration
that gets carried to the final val-16 run is picked here on :func:`evaluate.voxel_scene_eval` —
5 cm cells, GT = the cross-frame majority vote of ScanNet's per-frame instance PNG. Three families
of cross-frame decision are compared, all with ``window = 1`` so the past is frozen immediately:

``linkage``   one average-linkage pass over the whole graph, the cross-frame rule threshold swept
              (the stage-0c P2 pipeline with a properly swept threshold);
``calib``     the same one pass, but each edge is binarised at *its own* threshold first
              (``window_scores.calibrated_score_fn``) — fix 1 of the stage-1 brief;
``two_step``  intra-frame blocks first, then a directed block -> object containment
              (``two_step.py``) — fix 2 of the stage-1 brief.

Nothing here consults GT while merging; GT enters only through the reported metrics.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..merging.graph import edge_feature_names
from .evaluate import frames_from_merge_cache, voxel_scene_eval
from .fragment_fate import object_voxel_sets
from .pair_features import cross_feature_names
from .window_memory import WindowConfig, WindowMemory
from .window_scores import calibrated_score_fn, window_score_fn

MODES = ('linkage', 'calib', 'two_step')
RULES = ('p2a', 'p2b')
# pair_feat layout: the 14 cached intra-frame features, then the 15 cross-frame ones
CROSS_OFFSET = len(edge_feature_names())
_CROSS = {n: CROSS_OFFSET + i for i, n in enumerate(cross_feature_names())}


@dataclass(frozen=True)
class Variant:
    """One merge configuration: a mode, the cross-frame rule, and the thresholds it needs."""
    mode: str
    rule: str
    intra_threshold: float = 0.65
    cross_threshold: float = 0.5

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f'unknown mode {self.mode!r}')
        if self.rule not in RULES:
            raise ValueError(f'unknown rule {self.rule!r}')

    @property
    def tag(self):
        return f'{self.mode}_{self.rule}_{self.cross_threshold:g}'

    def as_dict(self):
        return dict(mode=self.mode, rule=self.rule, intra_threshold=self.intra_threshold,
                    cross_threshold=self.cross_threshold)


def default_variants(thresholds, modes=MODES, rules=RULES, intra_threshold=0.65):
    """The grid the holdout sweeps: every mode x rule x threshold, plus each mode at the checkpoint value."""
    out = []
    for mode in modes:
        for rule in rules:
            for thr in thresholds:
                out.append(Variant(mode, rule, intra_threshold, float(thr)))
    return out


def make_memory(variant, model, std, cfg, device):
    """The WindowMemory of one variant. ``window = 1``: the past is frozen, only this frame is revisable."""
    wc = cfg['stream_v2']
    common = dict(window=1, past_unit=wc['past_unit'], voxel=wc['voxel'],
                  touch_dilation=wc['touch_dilation'], linkage=wc['linkage'],
                  admit_min_points=wc['admit_min_points'], object_merge=bool(wc['object_merge']),
                  purity=wc['purity'])
    if variant.mode == 'calib':
        # each edge is already a yes/no, so the linkage threshold is a plain majority
        score_fn = calibrated_score_fn(model, std, variant.rule, device,
                                       variant.intra_threshold, variant.cross_threshold)
        wcfg = WindowConfig(threshold=0.5, grouping='linkage', **common)
    elif variant.mode == 'two_step':
        score_fn = window_score_fn(model, std, variant.rule, device)
        wcfg = WindowConfig(threshold=variant.intra_threshold, grouping='two_step',
                            cross_rule=variant.rule, cross_threshold=variant.cross_threshold, **common)
    else:
        score_fn = window_score_fn(model, std, variant.rule, device)
        wcfg = WindowConfig(threshold=variant.cross_threshold, grouping='linkage', **common)
    return WindowMemory(wcfg, score_fn=score_fn, device=device)


def load_scene_frames(cache_dir, scene, gt='2d'):
    """The scene's frames in frame order (``gt='2d'``: the per-frame instance PNG, scene-consistent)."""
    files = sorted(Path(cache_dir).glob(f'{scene}_*.npz'), key=lambda q: int(q.stem.rsplit('_', 1)[1]))
    return frames_from_merge_cache(files, gt=gt)


def run_scene(frames, variant, model, std, cfg, device):
    """Stream the frames through the variant's memory -> (per-object voxel sets, attrs, ms/frame)."""
    mem = make_memory(variant, model, std, cfg, device)
    times = []
    for fr in frames:
        mem.update(fr)
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
    return object_voxel_sets(mem), mem.object_attrs(), mem.times


def run_scene_masks(frames, variant, model, std, cfg, device, mesh):
    """Stream the frames through the variant's memory and render it onto ``mesh`` (the val protocol).

    Returns ``(masks (K, V) bool, scores (K,), attrs)`` with ``attrs`` in the same row order as
    ``masks``, so the stage-0b/0c post-processing can be replayed unchanged.
    """
    mem = make_memory(variant, model, std, cfg, device)
    for fr in frames:
        mem.update(fr)
    masks, scores = mem.render(mesh)
    return masks, scores, mem.object_attrs()


def evaluate_scene(obj_voxels, attrs, frames, stuff, voxel=0.05, min_gt_voxels=20, min_pred_voxels=20,
                   device='cpu'):
    """Voxel-proxy scene metrics for one run (scores = the objects' mean member confidence)."""
    met = voxel_scene_eval(obj_voxels, attrs['conf'], frames, stuff, voxel=voxel,
                           min_gt_voxels=min_gt_voxels, min_pred_voxels=min_pred_voxels, device=device)
    met['objects'] = int(len(obj_voxels))
    met['obs_frames_mean'] = float(np.mean(attrs['n_obs'])) if len(attrs['n_obs']) else 0.0
    met['single_frame_objects'] = int((attrs['n_obs'] <= 1).sum())
    return met
