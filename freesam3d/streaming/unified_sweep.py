"""Holdout threshold selection for the stage-2 unified flow (60 holdout scenes, voxel proxy).

Every triple ``(intra, bo, oo)`` streams the holdout scenes once through the unified flow, ranks the
objects by the model's node quality (NMS only, no mask is dropped) and scores them with
``voxel_scene_eval`` against the same voxel proxy the phase-1 selection used. The first pass uses one
shared threshold for all three edge types; the second gives each type its own, around the shared
optimum. Only the holdout scenes are used here - the 16 val scenes are never touched by a decision.
"""
import json
from pathlib import Path

import numpy as np

from .evaluate import voxel_scene_eval
from .fragment_fate import object_voxel_sets
from .holdout import load_scene_frames
from .unified_flow import make_unified_memory

METRIC_KEYS = ('ap50', 'ap25', 'iou', 'r50')


def triples(intra, bo, oo):
    """The full product of the three threshold lists, as ``(intra, bo, oo)`` tuples."""
    return [(float(i), float(b), float(o)) for i in intra for b in bo for o in oo]


def run_triple(triple, scenes, cfg, model, std, device, cache_dir, stuff, log=print):
    """Stream every scene once with these thresholds -> one metrics dict per scene."""
    intra, bo, oo = triple
    wc = cfg['stream_v2']
    rows = []
    for scene in scenes:
        frames = load_scene_frames(cache_dir, scene, gt='2d')
        mem = make_unified_memory(cfg, model, std, device, threshold=intra, cross_threshold=bo,
                                  bo_threshold=bo, oo_threshold=oo)
        for fr in frames:
            mem.update(fr)
        obj_vox = object_voxel_sets(mem)
        quality = np.asarray(mem.object_attrs()['quality'], np.float32)
        m = voxel_scene_eval(obj_vox, quality, frames, stuff, voxel=float(wc['voxel']),
                             min_gt_voxels=20, min_pred_voxels=20, device=device)
        rows.append(dict(scene=scene, **{k: float(v) for k, v in m.items()}))
    log(f'  intra {intra:.2f} bo {bo:.2f} oo {oo:.2f}: ' +
        ' '.join(f'{k} {aggregate(rows)[k]:.4f}' for k in METRIC_KEYS) +
        f' masks {aggregate(rows)["masks"]:.1f}')
    return rows


def aggregate(rows):
    """Scene-mean of the per-scene metrics (nan-safe) plus the mean mask count."""
    out = {k: float(np.nanmean([r[k] for r in rows])) for k in METRIC_KEYS}
    out['masks'] = float(np.mean([r['masks'] for r in rows]))
    out['scenes'] = len(rows)
    return out


def pick_best(entries):
    """The entry with the highest mean AP50; ties go to the fewest masks."""
    ok = [e for e in entries if np.isfinite(e['metrics']['ap50'])]
    if not ok:
        raise SystemExit('no finite AP50 in the sweep')
    return max(ok, key=lambda e: (e['metrics']['ap50'], -e['metrics']['masks']))


def save_entry(path, tag, triple, rows, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = dict(tag=tag, triple=list(triple), metrics=aggregate(rows), scenes=rows, **(extra or {}))
    path.write_text(json.dumps(entry, indent=2))
    return entry


def load_entries(sweep_dir):
    return [json.loads(p.read_text()) for p in sorted(Path(sweep_dir).glob('*.json'))
            if p.name != 'selected.json']


def select(sweep_dir, checkpoint=None, extra=None):
    """The best triple over every sweep file in the directory (writes ``selected.json``)."""
    entries = load_entries(sweep_dir)
    if not entries:
        raise SystemExit(f'no sweep files in {sweep_dir}')
    best = pick_best(entries)
    out = dict(tag=best['tag'], triple=best['triple'], metrics=best['metrics'],
               intra=best['triple'][0], bo=best['triple'][1], oo=best['triple'][2],
               checkpoint=checkpoint, candidates=[dict(tag=e['tag'], triple=e['triple'],
                                                       ap50=e['metrics']['ap50'],
                                                       masks=e['metrics']['masks']) for e in entries],
               **(extra or {}))
    (Path(sweep_dir) / 'selected.json').write_text(json.dumps(out, indent=2))
    return out
