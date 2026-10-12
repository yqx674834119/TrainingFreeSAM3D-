"""Dense per-scene proposal cache for training / evaluating the cross-frame matcher.

    source tools/env.sh
    for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$((i+1)) python tools/stream_build_cache.py --split train --shard $i/3 & done

train: ESAM 2D export, train scenes, 40-frame stride (2 export steps), the first ``frames_per_scene``
consecutive frames (GT instance ids are scene-consistent, so cross-frame matches are labelled for free).
Each frame -> <cache_root>/stream/<split>/<scene>/<frame>.npz in the proposal-cache format
(masks packed, scalars, feat, is_raw, iou target, gt_instance, xyz float16) plus the pose.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.merging.cache import CacheBuilder, frame_ids, load_frame, read_scene_list  # noqa: E402
from freesam3d.merging.cluster import merge_groups  # noqa: E402
from freesam3d.merging.graph_gpu import build_graph_gpu  # noqa: E402
from freesam3d.merging.proposals_gpu import build_proposals_gpu  # noqa: E402
from freesam3d.merging.proposals import proposal_targets  # noqa: E402
from freesam3d.merging.train import load_checkpoint, predict_frame  # noqa: E402
from freesam3d.evaluation.instance import groups_to_masks  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--split', choices=['train'], default='train')
    p.add_argument('--frames-per-scene', type=int, default=12)
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
    val = set(read_scene_list(ds['val_list']))
    scenes = [s for s in read_scene_list(ds['train_list']) if s not in val]
    i, n = map(int, a.shard.split('/'))
    scenes = [s for k, s in enumerate(scenes) if k % n == i]
    root = Path(ds['root'])
    out = resolve_path(cfg['cache_root']) / 'stream' / a.split
    stuff = set(ev['stuff_nyu40'])
    print(f'split={a.split} scenes={len(scenes)} frames/scene={a.frames_per_scene} shard={a.shard}', flush=True)
    done = skipped = 0
    t0 = time.time()
    for si, scene in enumerate(scenes):
        fids = frame_ids(root / scene, ds['frame_stride'], 0)[:a.frames_per_scene]
        sdir = out / scene
        sdir.mkdir(parents=True, exist_ok=True)
        for fid in fids:
            dst = sdir / f'{fid}.npz'
            if dst.exists():
                skipped += 1
                continue
            frame, gt_imgs = load_frame(root, scene, fid, ds['intrinsic'])
            if frame is None:
                continue
            inst_img, label_img = gt_imgs
            L = builder.lifting(frame)
            xyz, px = L.xyz, L.pixels
            gt, sem = inst_img[px[:, 1], px[:, 0]], label_img[px[:, 1], px[:, 0]]
            g = build_graph_gpu(xyz, L.rgb, L.group_ids, L.group_feats[cfg['feature']], L.point_feats[cfg['boundary_feature']],
                                L.fastsam.scores, L.group_mask_index, L.fastsam.masks, cfg['graph']['adjacency_eps'],
                                cfg['graph']['voxel'], cfg['graph']['boundary_k'], True, device=a.device)
            fr = dict(node_geom=g.node_geom, node_feat=g.node_feat, edges=g.edges, edge_feat=g.edge_feat)
            prob, obj = predict_frame(model, std, fr, a.device, with_objectness=True)
            merged, _ = groups_to_masks(merge_groups(L.group_ids, g.edges, prob, ck['threshold'], cfg['merge']['linkage']))
            raw_pts = L.fastsam.masks[:, px[:, 1], px[:, 0]]
            P = build_proposals_gpu(xyz, L.group_ids, g.node_feat, L.point_feats[cfg['boundary_feature']], raw_pts,
                                    L.fastsam.scores, L.group_mask_index, obj, merged, cfg['merge'].get('min_points', 100),
                                    device=a.device)
            ids, counts = np.unique(gt[gt > 0], return_counts=True)
            ids = ids[counts >= ev['min_gt_points']]
            thing = [t for t in ids if np.bincount(sem[gt == t]).argmax() not in stuff]
            iou, best = proposal_targets(P.masks, gt, keep_ids=np.array(thing, np.int64)) if thing else \
                (np.zeros(len(P.masks), np.float32), np.zeros(len(P.masks), np.int64))
            # un-centred pooled backbone feature per proposal, for cross-frame comparison experiments
            raw_gf = L.group_feats[cfg['feature']]
            raw_gf = (raw_gf / raw_gf.norm(dim=1, keepdim=True).clamp_min(1e-8)).cpu().numpy()
            onehot = np.zeros((len(xyz), int(L.group_ids.max()) + 1), np.float32)
            onehot[np.arange(len(xyz)), L.group_ids] = 1
            w = (P.masks.astype(np.float32) @ onehot)
            w /= np.maximum(w.sum(1, keepdims=True), 1)
            feat_raw = w @ raw_gf
            feat_raw /= np.maximum(np.linalg.norm(feat_raw, axis=1, keepdims=True), 1e-8)
            np.savez_compressed(dst, masks=np.packbits(P.masks, axis=1), num_points=len(xyz), scalars=P.scalars,
                                feat=P.feat.astype(np.float16), feat_raw=feat_raw.astype(np.float16), is_raw=P.is_raw,
                                source_index=P.source_index, iou=iou, best_gt=best, xyz=xyz.astype(np.float16),
                                gt_instance=gt.astype(np.int32), gt_sem=sem.astype(np.int32), groups=L.group_ids.astype(np.int32),
                                pose=frame.pose_c2w.astype(np.float32), frame_id=fid)
            done += 1
        if (si + 1) % 25 == 0 or si + 1 == len(scenes):
            el = time.time() - t0
            print(f'[{a.shard}] {si + 1}/{len(scenes)} scenes  built {done} skipped {skipped}  {el / max(done, 1):.2f}s/frame  {el / 60:.1f} min', flush=True)
    (out / f'manifest_{a.shard.replace("/", "of")}.json').write_text(json.dumps(dict(
        split=a.split, scenes=len(scenes), built=done, frames_per_scene=a.frames_per_scene, checkpoint=a.checkpoint), indent=2))


if __name__ == '__main__':
    main()
