"""Build the proposal cache (raw FastSAM masks ∪ GroupMergeNet merged masks) with IoU targets.

    source tools/env.sh
    python tools/proposal_build_cache.py --split val
    for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$((i+1)) python tools/proposal_build_cache.py --split train --shard $i/3 & done

Reads the existing merge cache for the frame list and GT (2D PNG for train, ESAM 3D GT for val),
re-runs lifting on the GPU path, applies the trained GroupMergeNet, and writes
<cache_root>/proposals/<split>/<scene>_<frame>.npz with masks (packed bits), scalars, feat,
is_raw, iou target, best GT id and GT class (thing / stuff / none).
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.merging.cache import CacheBuilder, load, load_frame  # noqa: E402
from freesam3d.merging.cluster import merge_groups  # noqa: E402
from freesam3d.merging.data import list_cache  # noqa: E402
from freesam3d.merging.graph_gpu import build_graph_gpu  # noqa: E402
from freesam3d.merging.proposals import build_proposals, proposal_targets  # noqa: E402
from freesam3d.merging.train import load_checkpoint, predict_frame  # noqa: E402
from freesam3d.evaluation.instance import groups_to_masks  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--split', choices=['train', 'val'], required=True)
    p.add_argument('--checkpoint', default='full')
    p.add_argument('--shard', default='0/1')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    ds, ev = cfg['dataset'], cfg['eval']
    lc = load_config(resolve_path(cfg['lifting_config']))
    lc['fastsam']['device'] = a.device
    lc['keep_on_device'] = True
    lc['lifting']['fill_backend'] = 'torch'
    builder = CacheBuilder(lc, dict(cfg['graph'], backend='gpu'), cfg['labels'], cfg['feature'], cfg['boundary_feature'])
    model, std, ck = load_checkpoint(resolve_path('work_dirs/merge_net') / a.checkpoint / 'model.pt', a.device)
    files = list_cache(resolve_path(cfg['cache_root']) / a.split)
    i, n = map(int, a.shard.split('/'))
    files = [f for k, f in enumerate(files) if k % n == i]
    out = resolve_path(cfg['cache_root']) / 'proposals' / a.split
    out.mkdir(parents=True, exist_ok=True)
    gt_key = 'gt_instance_3d' if a.split == 'val' else 'gt_instance'
    sem_key = 'gt_sem_3d' if a.split == 'val' else 'gt_nyu40'
    stuff = set(ev['stuff_raw'] if a.split == 'val' else ev['stuff_nyu40'])
    print(f'split={a.split} frames={len(files)} shard={a.shard} gt={gt_key}', flush=True)
    done = skipped = 0
    t0 = time.time()
    for k, f in enumerate(files):
        dst = out / f.name
        if dst.exists():
            skipped += 1
            continue
        d = load(f)
        scene, fid = f.stem.rsplit('_', 1)
        frame, _ = load_frame(Path(ds['root']), scene, int(fid), ds['intrinsic'])
        L = builder.lifting(frame)
        xyz = L.xyz
        # GT for the fresh point sample: nearest cached point of the same frame
        _, idx = cKDTree(d['xyz']).query(xyz, k=1)
        gt, sem = d[gt_key][idx], d[sem_key][idx]
        g = build_graph_gpu(xyz, L.rgb, L.group_ids, L.group_feats[cfg['feature']], L.point_feats[cfg['boundary_feature']],
                            L.fastsam.scores, L.group_mask_index, L.fastsam.masks, cfg['graph']['adjacency_eps'],
                            cfg['graph']['voxel'], cfg['graph']['boundary_k'], True, device=a.device)
        fr = dict(node_geom=g.node_geom, node_feat=g.node_feat, edges=g.edges, edge_feat=g.edge_feat)
        prob, obj = predict_frame(model, std, fr, a.device, with_objectness=True)
        merged, _ = groups_to_masks(merge_groups(L.group_ids, g.edges, prob, ck['threshold'], cfg['merge']['linkage']))
        px = L.pixels
        raw_pts = L.fastsam.masks[:, px[:, 1], px[:, 0]]
        P = build_proposals(xyz, L.group_ids, g.node_feat, L.point_feats[cfg['boundary_feature']].cpu().numpy(), raw_pts,
                            L.fastsam.scores, L.group_mask_index, obj, merged, cfg['merge'].get('min_points', 100))
        # targets against thing GT instances with >= min_gt_points (same protocol as evaluation)
        ids, counts = np.unique(gt[gt > 0], return_counts=True)
        ids = ids[counts >= ev['min_gt_points']]
        sem_of = {int(t): int(np.bincount(sem[gt == t]).argmax()) for t in ids}
        thing_ids = [t for t in ids if sem_of[t] not in stuff]
        iou, best = proposal_targets(P.masks, gt, keep_ids=np.array(thing_ids, np.int64)) if thing_ids else \
            (np.zeros(len(P.masks), np.float32), np.zeros(len(P.masks), np.int64))
        # stuff / unlabeled fraction of each proposal (for analysis and as optional targets)
        lab = gt > 0
        stuff_pt = np.isin(sem, list(stuff)) & lab
        pm = P.masks.astype(np.float32)
        np.savez_compressed(dst, masks=np.packbits(P.masks, axis=1), num_points=len(xyz), scalars=P.scalars,
                            feat=P.feat.astype(np.float16), is_raw=P.is_raw, source_index=P.source_index,
                            iou=iou, best_gt=best, stuff_frac=(pm @ stuff_pt.astype(np.float32)) / np.maximum(pm.sum(1), 1),
                            labeled_frac=(pm @ lab.astype(np.float32)) / np.maximum(pm.sum(1), 1),
                            num_thing_gt=len(thing_ids), xyz=xyz.astype(np.float16), gt_instance=gt.astype(np.int32),
                            gt_sem=sem.astype(np.int32), groups=L.group_ids.astype(np.int32))
        done += 1
        if (k + 1) % 200 == 0 or k + 1 == len(files):
            el = time.time() - t0
            print(f'[{a.shard}] {k + 1}/{len(files)}  built {done} skipped {skipped}  {el / max(done, 1):.2f}s/frame  {el / 60:.1f} min', flush=True)
    (out / f'manifest_{a.shard.replace("/", "of")}.json').write_text(json.dumps(dict(
        split=a.split, frames=len(files), built=done, checkpoint=a.checkpoint, gt=gt_key), indent=2))


if __name__ == '__main__':
    main()
