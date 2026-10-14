"""Stage-3 final report on the 16 val scenes: mesh protocol, dedup variants, fates, timing.

    source tools/env.sh
    CK=work_dirs/stream_v2/phase3/model/rel/seed0/model.pt
    for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python tools/window_phase3_final.py --stage run \
        --checkpoint $CK --thresholds 0.8 0.6 1.01 --dedup 0.8 --shard $i/4 & done
    python tools/window_phase3_final.py --stage report

One memory pass per scene with the final configuration. Rows: nobs / quality / oracle rankings,
each with and without the containment suppression; the oracle row uses GT (best IoU, a bound,
never a result). Plus the voxel-proxy cross-check, the block-fate table and the per-frame stage
timing. Writes <out_root>/phase3/final/<scene>.json and, on --stage report, {summary.md,summary.json}.
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
from freesam3d.streaming.evaluate import (frames_from_merge_cache, mesh_gtset, mesh_rows,  # noqa: E402
                                          segment_smooth, voxel_scene_eval)
from freesam3d.streaming.fragment_fate import object_voxel_sets  # noqa: E402
from freesam3d.streaming.holdout import load_scene_frames  # noqa: E402
from freesam3d.streaming.postprocess import best_iou  # noqa: E402
from freesam3d.streaming.unified_flow import make_unified_memory  # noqa: E402
from freesam3d.streaming.unified_probe import FATES, add_fates, unassigned_block_fates  # noqa: E402
from freesam3d.streaming.unified_train import load_unified_checkpoint  # noqa: E402

# the semantic-label table ESAM ships (read-only: the mesh stuff ids come from it)
LABELS = '/home/yqx/SegmentAnything3D/ESAM/data/scannet200/meta_data/scannetv2-labels.combined.tsv'
RANKINGS = ('nobs', 'quality', 'oracle')
CLEANS = ('nodedup', 'dedup')
KEYS = ('ap50', 'ap25', 'iou', 'r50', 'masks', 'dup_rate', 'overmerge', 'frags_gt')
STAGES = ('freeze_ms', 'build_ms', 'score_ms', 'cluster_ms', 'assign_ms')


def load_mesh(ws, scene):
    """ESAM's mesh for one scene (same files tools/window_phase2_final.py reads)."""
    mesh6 = np.fromfile(ws / 'data/scannet200/points' / f'{scene}.bin', np.float32).reshape(-1, 6)
    mesh_gt = np.fromfile(ws / 'data/scannet200/instance_mask' / f'{scene}.bin', np.int64)
    mesh_sem = np.fromfile(ws / 'data/scannet200/semantic_mask' / f'{scene}.bin', np.int64)
    sp = np.fromfile(ws / 'data/scannet200/super_points' / f'{scene}.bin', np.int64)
    return mesh6[:, :3], mesh_gt, mesh_sem, sp


def run_scene(scene, by_scene, cfg, model, std, dev, thr, dedup, stuff_mesh, stuff_proxy, ws,
              mem_overrides=None):
    mesh, mesh_gt, mesh_sem, sp = load_mesh(ws, scene)
    frames_gt3d = frames_from_merge_cache(
        sorted(by_scene[scene], key=lambda q: int(q.stem.rsplit('_', 1)[1])), gt='3d')
    frames = load_scene_frames(resolve_path(cfg['cache_root']) / 'val', scene, gt='3d')
    gts = mesh_gtset(mesh_gt, mesh_sem, stuff_mesh, int(cfg['eval']['min_gt_points']))
    t0 = time.perf_counter()
    mem = make_unified_memory(cfg, model, std, dev, threshold=float(thr[0]),
                              cross_threshold=float(thr[1]), bo_threshold=float(thr[1]),
                              oo_threshold=float(thr[2]), **(mem_overrides or {}))
    fates = {}
    times = {k: 0.0 for k in STAGES}
    for fr in frames:
        mem.update(fr)
        for k in STAGES:
            times[k] += float(mem.times.get(k, 0.0))
        fates = add_fates(fates, unassigned_block_fates(
            mem.last_unified, fr, admit_min_points=int(cfg['stream_v2']['admit_min_points'])))
    masks_raw, scores = mem.render(mesh)
    masks_smooth = segment_smooth(masks_raw, sp, device=dev)
    attrs = mem.object_attrs()
    obj_vox = object_voxel_sets(mem)
    iou_gt = best_iou(masks_smooth, gts, dev) if gts is not None else None
    rows = {}
    for name, dc in (('nodedup', None), ('dedup', dedup)):
        rows[name] = mesh_rows(masks_raw, masks_smooth, scores, attrs['n_obs'], gts, dev,
                               min_points=int(cfg['eval']['min_gt_points']), quality=attrs['quality'],
                               iou_gt=iou_gt, dedup_contain=dc)
    proxy = voxel_scene_eval(obj_vox, attrs['quality'], frames_gt3d, stuff_proxy,
                             voxel=float(cfg['stream_v2']['voxel']), min_gt_voxels=20,
                             min_pred_voxels=20, device=dev)
    n = max(len(frames), 1)
    times = {k: v / n for k, v in times.items()}
    times['total'] = sum(times[k] for k in STAGES)
    return dict(rows=rows, proxy=proxy, fates=fates, objects=int(len(obj_vox)),
                frames=len(frames), times=times, seconds=time.perf_counter() - t0)


def write_report(out, cfg, thresholds, dedup, checkpoint):
    per = [json.loads(f.read_text()) for f in sorted(out.glob('*.json')) if f.name != 'summary.json']
    if not per:
        raise SystemExit(f'no per-scene results under {out}')
    L = ['# 第 3 阶段 · 最终报告（16 个验证场景，ESAM 网格协议，segment_smooth 后）', '',
         f'模型 {checkpoint}；阈值 intra {thresholds[0]} / block-object {thresholds[1]} / '
         f'object-object {thresholds[2]}；包含度抑制 {dedup}（阈值与配置只在 60 个留出场景上选）。'
         f'{len(per)} 个场景，每场景一次记忆扫描。',
         '每行都是平滑 + NMS 后的结果，**不丢弃单帧物体**；``oracle`` 行用 GT IoU 排序，是完美排序上界，'
         '只是界，不是结果。``nodedup`` = 现有 NMS；``dedup`` = NMS 后再做包含度抑制。', '',
         '| 清理 | 排序 | AP50 | AP25 | mean best IoU | recall@0.5 | masks | dup_rate | overmerge | frags/GT |',
         '|---|---|---|---|---|---|---|---|---|---|']
    summary = {}
    for clean in CLEANS:
        for rk in RANKINGS:
            vals = {k: [r['rows'][clean].get(rk, {}).get(k) for r in per] for k in KEYS}
            m = {k: float(np.nanmean([v for v in vals[k] if v is not None and v == v])) for k in KEYS}
            summary[f'{clean}|{rk}'] = m
            star = '（上界）' if rk == 'oracle' else ''
            L.append(f'| {clean} | {rk}{star} | {m["ap50"]:.3f} | {m["ap25"]:.3f} | {m["iou"]:.3f} | '
                     f'{m["r50"]:.3f} | {m["masks"]:.1f} | {m["dup_rate"]:.3f} | {m["overmerge"]:.3f} | '
                     f'{m["frags_gt"]:.2f} |')
    L += ['', '参考：第 2 阶段最终 AP50 0.422 / ESAM-E 0.589（同一网格协议）。']
    L += ['', '## 体素代理（留出场景上选阈值用的口径，验证集只做交叉核对）', '',
          '| 清理 | 代理 AP50 | 代理 mean best IoU |', '|---|---|---|']
    for clean in CLEANS:
        a = float(np.nanmean([r['proxy']['ap50'] for r in per]))
        i = float(np.nanmean([r['proxy']['iou'] for r in per]))
        summary[f'proxy|{clean}'] = dict(ap50=a, iou=i)
        L.append(f'| {clean} | {a:.4f} | {i:.4f} |')
    L += ['', '## 没有拿到物体的 block：为什么（GT 只用于分类）', '',
          '| 类别 | block 数 | 占比 | 点数占比 |', '|---|---|---|---|']
    tot = {k: sum(r['fates'].get(k, 0) for r in per) for k in FATES}
    ptot = {k: sum(r['fates'].get('points', {}).get(k, 0.0) for r in per) for k in FATES}
    n = sum(tot.values()) or 1
    pn = sum(ptot.values()) or 1.0
    names = {'no_object_yet': '物体尚未出现（首次出现）', 'not_a_candidate': '物体存在但不在候选中',
             'scored_below': '是候选但分数低于阈值', 'unlabeled': 'block 自身 GT 不纯/无标注'}
    for k in FATES:
        L.append(f'| {names[k]} | {tot[k]} | {tot[k] / n:.3f} | {ptot[k] / pn:.3f} |')
    small = sum(r['fates'].get('small', 0) for r in per)
    L.append(f'| （其中低于点数下限的 {small} 个，仍计入上面三类） | | | |')
    tms = {k: float(np.mean([r['times'][k] for r in per])) for k in STAGES}
    tsd = {k: float(np.std([r['times'][k] for r in per])) for k in STAGES}
    summary['times_ms'] = dict(tms, total=float(np.mean([r['times']['total'] for r in per])))
    summary['times_ms_sd'] = tsd
    L += ['', '## 逐帧耗时（阶段 3 模块，均值 ± 标准差，按场景）', '',
          'freeze %.1f±%.1f ＋ build %.1f±%.1f ＋ score %.1f±%.1f ＋ cluster %.1f±%.1f ＋ assign %.1f±%.1f = **%.1f ms/帧**'
          % (tms['freeze_ms'], tsd['freeze_ms'], tms['build_ms'], tsd['build_ms'],
             tms['score_ms'], tsd['score_ms'], tms['cluster_ms'], tsd['cluster_ms'],
             tms['assign_ms'], tsd['assign_ms'], summary['times_ms']['total']),
          '（另有 FastSAM 约 72 ms/帧；`tools/merge_benchmark.py --unified` 复核。）']
    (out / 'summary.md').write_text('\n'.join(L) + '\n')
    (out / 'summary.json').write_text(json.dumps(dict(summary=summary, fates=tot, fate_points=ptot,
                                                      scenes=len(per), thresholds=list(thresholds),
                                                      dedup=dedup, checkpoint=str(checkpoint)),
                                                 indent=2))
    print('\n'.join(L), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--stage', default='all', choices=['run', 'report', 'all'])
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--thresholds', nargs=3, type=float, default=None, metavar=('INTRA', 'BO', 'OO'))
    p.add_argument('--sweep', default=None, help='JSON from tools/window_phase3_sweep.py --report')
    p.add_argument('--dedup', type=float, default=None, help='containment suppression threshold')
    p.add_argument('--touch-dilation', type=int, default=None)
    p.add_argument('--cand-dist', type=float, default=None)
    p.add_argument('--cand-cos', type=float, default=None)
    p.add_argument('--shard', default='0/1')
    p.add_argument('--scenes', nargs='*', default=None)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--out', default=None)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    out = Path(a.out) if a.out else resolve_path(cfg['phase3']['out_root']) / 'final'
    thr, dedup = a.thresholds, a.dedup
    if a.sweep:
        sw = json.loads(resolve_path(a.sweep).read_text())
        thr = thr or [float(t) for t in sw['triple']]
        if dedup is None and str(sw.get('variant', '')).startswith('dedup_'):
            dedup = float(str(sw['variant']).split('_', 1)[1])
    if a.stage == 'report':
        write_report(out, cfg, thr or [float('nan')] * 3, dedup, a.checkpoint or '(see per-scene json)')
        return
    if a.checkpoint is None or thr is None:
        raise SystemExit('--checkpoint and --thresholds (or --sweep) are required to run')
    dev = a.device if torch.cuda.is_available() else 'cpu'
    if str(dev).startswith('cuda'):
        print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    model, std, ck = load_unified_checkpoint(resolve_path(a.checkpoint), dev)
    ws = resolve_path('work_dirs/esam_eval')
    stuff_mesh = stuff_ids(str(resolve_path(LABELS)))
    stuff_proxy = set(cfg['eval']['stuff_raw'])          # the 3D GT semantic ids are ScanNet raw ids
    frag_files = sorted((resolve_path(cfg['cache_root']) / 'val').glob('*.npz'))
    by_scene = {}
    for f in frag_files:
        by_scene.setdefault(f.stem.rsplit('_', 1)[0], []).append(f)
    scenes = a.scenes or sorted(by_scene)
    i, n = map(int, a.shard.split('/'))
    mine = [s for k, s in enumerate(scenes) if k % n == i]
    if a.limit:
        mine = mine[:a.limit]
    mem_over = {}
    if a.touch_dilation is not None:
        mem_over['touch_dilation'] = int(a.touch_dilation)
    if a.cand_dist is not None:
        mem_over['cand_dist'] = float(a.cand_dist)
    if a.cand_cos is not None:
        mem_over['cand_cos'] = float(a.cand_cos)
    print(f'{len(mine)}/{len(scenes)} val scenes, thresholds {thr}, dedup {dedup}, overrides {mem_over}, '
          f'checkpoint seed {ck.get("seed")}', flush=True)
    t0 = time.time()
    for si, scene in enumerate(mine):
        dst = out / f'{scene}.json'
        if dst.exists() and not a.overwrite:
            continue
        res = run_scene(scene, by_scene, cfg, model, std, dev, thr, dedup, stuff_mesh, stuff_proxy, ws,
                        mem_overrides=mem_over)
        res.update(scene=scene, thresholds=list(thr), dedup=dedup, checkpoint=str(a.checkpoint))
        out.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(res, indent=1, default=float))
        f = res['rows']['dedup' if dedup is not None else 'nodedup']
        print(f'[{a.shard}] {si + 1}/{len(mine)} {scene}: quality {f["quality"]["ap50"]:.3f} '
              f'oracle {f["oracle"]["ap50"]:.3f} masks {f["quality"]["masks"]} '
              f'({(time.time() - t0) / 60:.1f} min)', flush=True)
    if a.stage == 'run':
        return
    write_report(out, cfg, thr, dedup, a.checkpoint)


if __name__ == '__main__':
    main()
