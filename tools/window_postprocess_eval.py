"""Stage 0b: output-side processing on top of the perfect (GT-oracle) merge.

One cache per scene (a single ``window=1`` oracle WindowMemory run) holds the object masks
rendered on ESAM's mesh vertices before/after segment smoothing plus per-object attributes;
every ranking x cleaning x top-K combination is then post-processed purely on that cache,
plus the wrong-answer destination decomposition and three reference rows.

    source tools/env.sh
    python tools/window_postprocess_eval.py --stage cache    # one GPU pass over the 16 val scenes
    python tools/window_postprocess_eval.py --stage eval     # numpy only, reads the cache

Writes work_dirs/stream_v2/phase0b/{cache/<scene>.npz,summary.md,summary.json,per_scene.json}.
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
from freesam3d.evaluation.granularity import stuff_ids  # noqa: E402
from freesam3d.evaluation.instance import association_stats  # noqa: E402
from freesam3d.merging.labels import group_gt_votes  # noqa: E402
from freesam3d.streaming.evaluate import (frames_from_merge_cache, mesh_gtset, scene_metrics,  # noqa: E402
                                          segment_smooth)
from freesam3d.streaming.postprocess import (apply_topk, best_iou, classify_masks, filter_impure,  # noqa: E402
                                             filter_single, greedy_nms, rank_scores)
from freesam3d.streaming.voxel import voxelize  # noqa: E402
from freesam3d.streaming.window_memory import WindowConfig, WindowMemory, oracle_scores  # noqa: E402

SCORES = ('conf', 'size', 'nobs', 'oracle')
CLEANS = ('none', 'nms', 'nms_impure', 'nms_single')
TOPK = (0, 40)
CLASSES = ('correct', 'duplicate', 'partial', 'junk')
FLAGS = ('impure', 'stuff', 'small', 'single')
SMALL_VERTICES = 500      # "very small" in the destination decomposition
NMS_IOU = 0.5
IMPURE_THR = 0.5          # "mostly impure members" (decomposition + nms+impure)
REF_NAMES = ('esam', 'esam_oracle', 'gt_voxel', 'gt_voxel_raw')
REF_LABELS = {'esam': 'ESAM-E 原始输出', 'esam_oracle': 'ESAM-E + S_oracle 排序',
              'gt_voxel': 'GT 实例 5 cm 体素渲染 + S_oracle（段平滑后）',
              'gt_voxel_raw': 'GT 实例 5 cm 体素渲染 + S_oracle（未平滑）'}
METRIC_KEYS = ('masks', 'iou', 'r50', 'ap50', 'ap25', 'dup_rate')


def load_mesh(ws, scene):
    mesh6 = np.fromfile(ws / 'data/scannet200/points' / f'{scene}.bin', np.float32).reshape(-1, 6)
    mesh_gt = np.fromfile(ws / 'data/scannet200/instance_mask' / f'{scene}.bin', np.int64)
    mesh_sem = np.fromfile(ws / 'data/scannet200/semantic_mask' / f'{scene}.bin', np.int64)
    sp = np.fromfile(ws / 'data/scannet200/super_points' / f'{scene}.bin', np.int64)
    return mesh6[:, :3], mesh_gt, mesh_sem, sp


def build_cache(scene, files, mesh, mesh_sem, sp, cfg, dev, stuff, out_path):
    """One window=1 oracle run -> masks (raw + smoothed) and per-object attributes on disk."""
    frames = frames_from_merge_cache(files, gt=cfg['eval']['gt'])
    wc = cfg['stream_v2']
    wcfg = WindowConfig(window=1, past_unit=wc['past_unit'], voxel=wc['voxel'],
                        touch_dilation=wc['touch_dilation'], threshold=wc['threshold'],
                        linkage=wc['linkage'], admit_min_points=wc['admit_min_points'],
                        object_merge=True, purity=wc['purity'])
    mem = WindowMemory(wcfg, score_fn=lambda g: oracle_scores(g, wcfg.purity), device=dev)
    for fr in frames:
        mem.update(fr)
    masks_raw, scores = mem.render(mesh)
    attrs = mem.object_attrs()
    assert np.allclose(attrs['conf'], scores, atol=1e-4), 'object_attrs / render score mismatch'
    masks_smooth = segment_smooth(masks_raw, sp, device=dev)
    n_vertices = masks_raw.sum(1)
    is_stuff = np.isin(mesh_sem, stuff).astype(np.float32)
    cov = masks_raw.astype(np.float32) @ is_stuff if len(masks_raw) else np.zeros(0, np.float32)
    stuff_flag = (cov / np.maximum(n_vertices, 1) > 0.5) if len(masks_raw) else np.zeros(0, bool)
    np.savez_compressed(
        out_path, masks_raw=masks_raw, masks_smooth=masks_smooth, scores=scores,
        n_obs=attrs['n_obs'], n_pts=attrs['n_pts'],
        impure_frac=(attrs['imp_pts'] / np.maximum(attrs['n_pts'], 1)).astype(np.float32),
        vc_raw=n_vertices.astype(np.int32), vc_smooth=masks_smooth.sum(1).astype(np.int32),
        stuff=stuff_flag)
    return dict(frames=len(frames), objects=int(len(scores)),
                masks_ge100_smooth=int((masks_smooth.sum(1) >= cfg['eval']['min_gt_points']).sum()))


def metrics_of(masks, scores, gts):
    met = scene_metrics(masks, scores, gts)
    met['dup_rate'] = association_stats(masks, gts)['dup_rate']
    return met


def gt_voxel_masks(mesh, mesh_gt, sp, gts, dev, voxel=0.05):
    """GT instances pushed through the 5 cm voxel grid and rendered back on the mesh vertices
    (majority GT per voxel); scores come from the oracle best-IoU ranking."""
    _, inv = np.unique(voxelize(mesh, voxel), return_inverse=True)
    major = np.zeros(int(inv.max()) + 1, np.int64)
    vi, gi, ci = group_gt_votes(inv, mesh_gt, len(major))
    if len(vi):
        order = np.lexsort((gi, -ci, vi))
        vs, first = vi[order], np.r_[True, vi[order][1:] != vi[order][:-1]]
        major[vs[first]] = gi[order][first]
    label = major[inv]
    masks = label[None, :] == np.asarray(gts.ids)[:, None]
    return masks, segment_smooth(masks, sp, device=dev)


def evaluate_scene(d, gts, mesh, mesh_gt, sp, esam_path, min_vertices, dev):
    out = {'combo': {}, 'decomp': None, 'ref': {}}
    for set_name, mk, vk in (('smooth', 'masks_smooth', 'vc_smooth'), ('raw', 'masks_raw', 'vc_raw')):
        base = np.where(d[vk] >= min_vertices)[0]
        masks = d[mk][base]
        conf, n_obs, impure = d['scores'][base], d['n_obs'][base], d['impure_frac'][base]
        iou_gt = best_iou(masks, gts)
        for sm in SCORES:
            s = rank_scores(sm, masks=masks, conf=conf, n_obs=n_obs, iou_gt=iou_gt)
            order = np.argsort(-s, kind='stable')
            for cl in CLEANS:
                keep = greedy_nms(masks, s, NMS_IOU) if cl != 'none' else order
                if cl == 'nms_impure':
                    keep = filter_impure(keep, impure, IMPURE_THR)
                elif cl == 'nms_single':
                    keep = filter_single(keep, n_obs)
                for k in TOPK:
                    kk = apply_topk(keep, s, k)
                    out['combo'][f'{sm}|{cl}|k{k}|{set_name}'] = metrics_of(masks[kk], s[kk], gts)
        if set_name == 'smooth':
            out['decomp'] = dict(
                labels=classify_masks(masks, rank_scores('conf', conf=conf), gts),
                impure=impure > IMPURE_THR, stuff=np.asarray(d['stuff'])[base],
                small=d[vk][base] < SMALL_VERTICES, single=n_obs == 1)
    if esam_path.exists():
        z = np.load(esam_path)
        em, es = z['scene_masks'], z['scene_scores']
        out['ref']['esam'] = metrics_of(em, es, gts)
        out['ref']['esam_oracle'] = metrics_of(em, best_iou(em, gts), gts)
    gm, gsm = gt_voxel_masks(mesh, mesh_gt, sp, gts, dev)
    out['ref']['gt_voxel_raw'] = metrics_of(gm, best_iou(gm, gts), gts)
    out['ref']['gt_voxel'] = metrics_of(gsm, best_iou(gsm, gts), gts)
    return out


def combo_row(name, r):
    return (f"| {name} | {r['masks']:.1f} | {r['iou']:.3f} | {r['r50']:.3f} | "
            f"{r['ap50']:.3f} | {r['ap25']:.3f} | {r['dup_rate']:.3f} |")


def write_report(out_dir, per_scene, args, cfg):
    if not per_scene:
        raise SystemExit('no scenes evaluated; build the cache first (--stage cache)')
    n = len(per_scene)
    keys = [f'{sm}|{cl}|k{k}|{set_name}'
            for set_name in ('smooth', 'raw') for sm in SCORES for cl in CLEANS for k in TOPK]
    combo = {key: {m: float(np.mean([r['combo'][key][m] for r in per_scene])) for m in METRIC_KEYS}
             for key in keys}
    counts = {c: 0 for c in CLASSES}
    hits = {f: {c: 0 for c in CLASSES} for f in FLAGS}
    for r in per_scene:
        lab, dc = r['decomp']['labels'], r['decomp']
        for ci, c in enumerate(CLASSES):
            sel = lab == ci
            counts[c] += int(sel.sum())
            for f in FLAGS:
                hits[f][c] += int(np.logical_and(sel, dc[f]).sum())
    refs = {name: {m: float(np.mean([r['ref'][name][m] for r in per_scene])) for m in METRIC_KEYS}
            for name in REF_NAMES if all(name in r['ref'] for r in per_scene)}

    lines = ['# 第 0b 阶段：输出端处理后的 AP 上限（win_w1 oracle 合并）', '',
             f'{n} 个验证场景，ESAM 网格协议（thing GT ≥ {cfg["eval"]["min_gt_points"]} 顶点，段平滑 thr 0.5，'
             '场景平均）。合并与记忆只跑一次（window=1，GT oracle），以下全部是同一份缓存上的后处理。'
             'S_oracle 与 nms+impure 用到 GT，只作上限；不用 GT 的组合阈值固定，未在验证集上调参。', '',
             '## 1. 错答案去向分解（段平滑后，S_conf 排序，未清理）', '',
             '按 AP50 的贪心匹配顺序归类每个 mask；占比列为该类内部的比例（16 场景合计）。', '',
             '| 类别 | masks/场景 | 混合碎片(>50% 不纯点) | stuff | <500 顶点 | 只看到 1 帧 |',
             '|---|---|---|---|---|---|']
    for c in CLASSES:
        pct = [100.0 * hits[f][c] / max(counts[c], 1) for f in FLAGS]
        lines.append(f'| {c} | {counts[c] / n:.1f} | ' + ' | '.join(f'{p:.0f}%' for p in pct) + ' |')
    tot = max(sum(counts.values()), 1)
    lines.append(f'| **合计** | {sum(counts.values()) / n:.1f} | '
                 + ' | '.join(f'{100.0 * sum(hits[f].values()) / tot:.0f}%' for f in FLAGS) + ' |')

    for set_name, title in (('smooth', '## 2. 后处理组合（段平滑后）'), ('raw', '## 3. 后处理组合（未平滑）')):
        lines += ['', title, '',
                  '| 排序 | 清理 | topK | masks/场景 | 平均最佳 IoU | 召回@0.5 | AP50 | AP25 | dup_rate |',
                  '|---|---|---|---|---|---|---|---|---|']
        for sm in SCORES:
            for cl in CLEANS:
                for k in TOPK:
                    r = combo[f'{sm}|{cl}|k{k}|{set_name}']
                    lines.append(combo_row(f'S_{sm} | {cl} | {k if k else "all"}', r))
    lines += ['', '## 4. 参考行（同一评估代码）', '',
              '| 参考 | masks/场景 | 平均最佳 IoU | 召回@0.5 | AP50 | AP25 | dup_rate |',
              '|---|---|---|---|---|---|---|']
    for name in REF_NAMES:
        if name in refs:
            lines.append(combo_row(REF_LABELS[name], refs[name]))
    md = '\n'.join(lines) + '\n'
    (out_dir / 'summary.md').write_text(md)
    (out_dir / 'summary.json').write_text(json.dumps(
        dict(combo=combo, decomp=dict(counts=counts, hits=hits, scenes=n), refs=refs, args=vars(args)),
        indent=2))
    (out_dir / 'per_scene.json').write_text(json.dumps(
        [dict(scene=r['scene'], ref=r['ref'],
              combo={k: v for k, v in r['combo'].items() if k.endswith('|smooth')},
              decomp_counts=np.bincount(r['decomp']['labels'], minlength=4).tolist()) for r in per_scene], indent=1))
    print(md)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--stage', default='all', choices=['cache', 'eval', 'all'])
    p.add_argument('--scenes', nargs='*')
    p.add_argument('--device', default='cuda')
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    dev = a.device if torch.cuda.is_available() else 'cpu'
    ws = resolve_path('work_dirs/esam_eval')
    stuff = stuff_ids('/home/yqx/SegmentAnything3D/ESAM/data/scannet200/meta_data/scannetv2-labels.combined.tsv')
    out_dir = resolve_path(cfg['stream_v2']['out_root']) / 'phase0b'
    cache_dir = out_dir / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = sorted((resolve_path(cfg['cache_root']) / 'val').glob('*.npz'))
    by_scene = {}
    for f in files:
        by_scene.setdefault(f.stem.rsplit('_', 1)[0], []).append(f)
    scenes = a.scenes or sorted(by_scene)

    if a.stage in ('cache', 'all'):
        for scene in scenes:
            path = cache_dir / f'{scene}.npz'
            if path.exists() and not a.overwrite:
                print(f'{scene}: cache exists, skip', flush=True)
                continue
            fs = sorted(by_scene[scene], key=lambda q: int(q.stem.rsplit('_', 1)[1]))
            mesh, mesh_gt, mesh_sem, sp = load_mesh(ws, scene)
            t0 = time.perf_counter()
            info = build_cache(scene, fs, mesh, mesh_sem, sp, cfg, dev, stuff, path)
            print(f"{scene}: {info['frames']} frames, {info['objects']} objects, "
                  f"{info['masks_ge100_smooth']} masks>=100 (smooth), {time.perf_counter() - t0:.1f} s", flush=True)
    if a.stage == 'cache':
        return

    min_vertices = cfg['eval']['min_gt_points']
    per_scene = []
    for scene in scenes:
        cache = cache_dir / f'{scene}.npz'
        if not cache.exists():
            print(f'{scene}: no cache, skip', flush=True)
            continue
        mesh, mesh_gt, mesh_sem, sp = load_mesh(ws, scene)
        gts = mesh_gtset(mesh_gt, mesh_sem, stuff, min_vertices)
        if gts is None:
            continue
        t0 = time.perf_counter()
        d = np.load(cache)
        res = evaluate_scene(d, gts, mesh, mesh_gt, sp, ws / 'predictions' / f'{scene}.npz', min_vertices, dev)
        res['scene'] = scene
        per_scene.append(res)
        k = 'conf|none|k0|smooth'
        esam = res['ref'].get('esam', {}).get('ap50', float('nan'))
        print(f"{scene}: conf|none|k0|smooth AP50 {res['combo'][k]['ap50']:.3f} "
              f"masks {res['combo'][k]['masks']:.0f} | esam {esam:.3f} | {time.perf_counter() - t0:.1f} s", flush=True)
    write_report(out_dir, per_scene, a, cfg)


if __name__ == '__main__':
    main()
