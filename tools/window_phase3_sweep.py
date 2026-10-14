"""Stage-3 holdout sweep: thresholds x memory variants, scored under quality / dedup / oracle rankings.

    source tools/env.sh
    CK=work_dirs/stream_v2/phase3/model/rel/seed0/model.pt
    python tools/window_phase3_sweep.py --checkpoint $CK --tag rel_s0 \
        --intra 0.75 0.80 0.85 --bo 0.4 0.6 0.8 --oo 1.01
    python tools/window_phase3_sweep.py --checkpoint $CK --tag rel_s0_d2 --touch-dilation 2 ...
    python tools/window_phase3_sweep.py --checkpoint $CK --tag rel_s0_cand --cand-dist 0.3 --cand-cos 0.5 ...
    python tools/window_phase3_sweep.py --report

Every (intra, bo, oo) triple streams the 60 holdout scenes once; the model's quality ranking,
quality + containment suppression at each ``--dedup`` threshold, and the oracle ranking (GT best
IoU; a bound) are scored on the same objects and the same voxel GT. Writes one JSON per triple
under <out_root>/phase3/sweep; --report picks the best (entry, variant) by AP50.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.streaming.phase3_eval import aggregate, rank_variants, stream_scene, voxel_gt  # noqa: E402
from freesam3d.streaming.unified_sweep import triples  # noqa: E402
from freesam3d.streaming.unified_train import load_unified_checkpoint  # noqa: E402


def variant_names(dedup):
    return ['quality'] + [f'dedup_{float(t):g}' for t in dedup] + ['oracle']


def report(out):
    rows = []
    for f in sorted(Path(out).glob('*.json')):
        if f.name == 'selected.json':
            continue
        e = json.loads(f.read_text())
        for name, m in e['variants'].items():
            if name == 'oracle':
                continue
            rows.append(dict(file=f.name, tag=e['tag'], triple=e['triple'], variant=name,
                             ap50=m['ap50'], ap25=m['ap25'], iou=m['iou'], r50=m['r50'],
                             masks=m['masks'], dup_rate=m.get('dup_rate'),
                             overmerge=m.get('overmerge'), oracle_ap50=e['variants']['oracle']['ap50']))
    rows.sort(key=lambda r: (-(r['ap50'] if np.isfinite(r['ap50']) else -1.0), r['masks']))
    print(f'{"file":36s} {"variant":10s} {"AP50":>7s} {"AP25":>7s} {"IoU":>7s} {"r50":>7s} '
          f'{"masks":>6s} {"dup":>6s} {"oracle":>7s}')
    for r in rows[:30]:
        print(f'{r["file"]:36s} {r["variant"]:10s} {r["ap50"]:7.4f} {r["ap25"]:7.4f} {r["iou"]:7.4f} '
              f'{r["r50"]:7.4f} {r["masks"]:6.1f} {r["dup_rate"]:6.3f} {r["oracle_ap50"]:7.4f}')
    if rows:
        (Path(out) / 'selected.json').write_text(json.dumps(rows[0], indent=2))
        print('best:', json.dumps(rows[0]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--tag', default='p3')
    p.add_argument('--intra', nargs='+', type=float, default=[0.80])
    p.add_argument('--bo', nargs='+', type=float, default=[0.60])
    p.add_argument('--oo', nargs='+', type=float, default=[1.01])
    p.add_argument('--dedup', nargs='*', type=float, default=[0.6, 0.7, 0.8, 0.9],
                   help='containment suppression thresholds scored on every streamed scene')
    p.add_argument('--touch-dilation', type=int, default=None)
    p.add_argument('--cand-dist', type=float, default=None)
    p.add_argument('--cand-cos', type=float, default=None)
    p.add_argument('--shard', default='0/1')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--report', action='store_true')
    p.add_argument('--out', default=None)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    out = Path(a.out) if a.out else resolve_path(cfg['phase3']['out_root']) / 'sweep'
    if a.report:
        report(out)
        return
    if not a.checkpoint:
        raise SystemExit('--checkpoint is required (or use --report)')
    dev = a.device if torch.cuda.is_available() else 'cpu'
    if str(dev).startswith('cuda'):
        print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    model, std, ck = load_unified_checkpoint(resolve_path(a.checkpoint), dev)
    print(f'checkpoint {a.checkpoint} (seed {ck.get("seed")}, rel_dim {ck.get("rel_dim", 0)}, '
          f'attn {ck.get("attn_layers", 0)})', flush=True)
    cache = resolve_path(cfg['cache_root']) / 'seq' / 'train'
    stuff = set(cfg['eval']['stuff_nyu40'])
    voxel = float(cfg['stream_v2']['voxel'])
    scenes = list(cfg['stream_v2']['holdout_scenes'])
    i, n = map(int, a.shard.split('/'))
    scenes = [s for k, s in enumerate(scenes) if k % n == i]
    if a.limit:
        scenes = scenes[:a.limit]
    mem_over = {}
    if a.touch_dilation is not None:
        mem_over['touch_dilation'] = int(a.touch_dilation)
    if a.cand_dist is not None:
        mem_over['cand_dist'] = float(a.cand_dist)
    if a.cand_cos is not None:
        mem_over['cand_cos'] = float(a.cand_cos)
    grid = triples(a.intra, a.bo, a.oo)
    names = variant_names(a.dedup)
    print(f'{len(grid)} triples x {len(scenes)} holdout scenes, overrides {mem_over} -> {out}', flush=True)
    for triple in grid:
        intra, bo, oo = triple
        dst = out / f'{a.tag}_{intra:.2f}_{bo:.2f}_{oo:.2f}.json'
        if dst.exists():
            print(f'  {dst.name}: already done', flush=True)
            continue
        over = dict(mem_over, threshold=intra, cross_threshold=bo, bo_threshold=bo, oo_threshold=oo)
        rows = {name: [] for name in names}
        for scene in scenes:
            frames, mem, obj_vox, attrs, _ = stream_scene(scene, cfg, model, std, dev, cache, over)
            cells, gts = voxel_gt(frames, stuff, voxel=voxel, min_gt_voxels=20)
            for name, met in rank_variants(obj_vox, attrs['quality'], cells, gts, a.dedup,
                                           device=dev).items():
                rows[name].append(dict(scene=scene, **{k: float(v) for k, v in met.items()}))
        entry = dict(tag=a.tag, triple=[float(t) for t in triple], mem_overrides=over,
                     dedup=[float(t) for t in a.dedup], checkpoint=str(a.checkpoint),
                     variants={name: aggregate(r) for name, r in rows.items()}, scenes=rows)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(entry, indent=1, default=float))
        v = entry['variants']
        print(f'  intra {intra:.2f} bo {bo:.2f} oo {oo:.2f}: ' +
              ' | '.join(f'{name} {v[name]["ap50"]:.4f}/{v[name]["masks"]:.1f}/{v[name]["dup_rate"]:.3f}'
                         for name in names), flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
