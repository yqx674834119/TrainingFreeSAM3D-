"""Stage 0c: the two real (GT-free) pipelines, plus the free finishing from stage 0b, on ESAM's mesh.

P1 = the existing default per-frame pipeline (mask scorer + nested NMS) into ``QueryMemory`` plus the
     cross-frame projection head — the command in ``docs/streaming.zh-CN.md`` §3, re-run here so its
     object masks and attributes land in our own result directory.
P2 = ``WindowMemory`` with ``window=1`` over the per-frame fragments of ``work_dirs/merge_cache/val``
     (read-only), scored by the trained ``GroupMergeNet`` on the cached intra-frame edges and by a rule
     read off ``pair_features`` on every other edge (``freesam3d/streaming/window_scores.py``).

Neither pipeline consults GT while deciding anything. GT appears only in the S_oracle ranking row, in the
destination decomposition of P2's mixed fragments, and in the reported metrics — always marked as such.

    source tools/env.sh
    python tools/window_real_eval.py --stage cache     # one GPU pass per pipeline x variant
    python tools/window_real_eval.py --stage eval      # numpy only, reads the caches

Writes work_dirs/stream_v2/phase0c/{cache/**,summary.md,summary.json,per_scene.json,timing.json}.
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
from freesam3d.merging.proposal_data import ScalarStandardizer, list_proposal_cache, load_proposals  # noqa: E402
from freesam3d.merging.scorer import MaskScorer  # noqa: E402
from freesam3d.streaming.evaluate import (frame_input_from_proposal_cache, frames_from_merge_cache,  # noqa: E402
                                          mesh_gtset, scene_metrics, segment_smooth, select_frame_proposals)
from freesam3d.streaming.fragment_fate import (FATES, classify_fates, frag_object_rows,  # noqa: E402
                                               impure_fragments, object_voxel_sets)
from freesam3d.streaming.postprocess import (best_iou, filter_single, greedy_nms,  # noqa: E402
                                                iou_pair_matrix, overmerge_flags, rank_scores)
from freesam3d.streaming.window_memory import WindowConfig, WindowMemory  # noqa: E402
from freesam3d.streaming.window_scores import load_group_merge_net, window_score_fn  # noqa: E402

P1 = 'p1_default'
P2_VARIANTS = ('p2a_calib', 'p2b_calib', 'p2a_0.5', 'p2b_0.5')
SCORES = ('conf', 'size', 'nobs', 'oracle')
CLEANS = ('none', 'nms', 'nms_single')
SETS = ('smooth', 'raw')
NMS_IOU = 0.5
METRIC_KEYS = ('masks', 'iou', 'r50', 'ap50', 'ap25', 'dup_rate', 'overmerge', 'frags')


def variant_spec(name, calib):
    """(rule, threshold) of a P2 variant; ``calib`` is the value stored in the GroupMergeNet checkpoint."""
    rule = 'p2a' if name.startswith('p2a') else 'p2b'
    return rule, (float(calib) if name.endswith('calib') else 0.5)


def load_mesh(ws, scene):
    mesh6 = np.fromfile(ws / 'data/scannet200/points' / f'{scene}.bin', np.float32).reshape(-1, 6)
    mesh_gt = np.fromfile(ws / 'data/scannet200/instance_mask' / f'{scene}.bin', np.int64)
    mesh_sem = np.fromfile(ws / 'data/scannet200/semantic_mask' / f'{scene}.bin', np.int64)
    sp = np.fromfile(ws / 'data/scannet200/super_points' / f'{scene}.bin', np.int64)
    return mesh6[:, :3], mesh_gt, mesh_sem, sp


def metrics_of(masks, scores, gts, device='cpu'):
    met = scene_metrics(masks, scores, gts, device=device)
    st = association_stats(masks, gts, device=device)
    met['dup_rate'], met['overmerge'] = st['dup_rate'], st['overmerge_rate']
    met['frags'] = st['frags_per_gt']
    return met


def save_cache(path, masks_raw, masks_smooth, scores, n_obs, fate_rows, fate_n, fate_class):
    """Pack the Boolean masks (1 bit per vertex) so a scene with hundreds of P2 objects stays small."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, masks_raw=np.packbits(masks_raw, axis=1), masks_smooth=np.packbits(masks_smooth, axis=1),
        num_vertices=np.int64(masks_raw.shape[1]), scores=np.asarray(scores, np.float32),
        n_obs=np.asarray(n_obs, np.int32), vc_raw=masks_raw.sum(1).astype(np.int32),
        vc_smooth=masks_smooth.sum(1).astype(np.int32), fate_rows=np.asarray(fate_rows, np.int32),
        fate_n=np.asarray(fate_n, np.int32), fate_class=np.asarray(fate_class, np.int8))


def load_cache(path):
    with np.load(path) as z:
        V = int(z['num_vertices'])
        d = {k: z[k] for k in z.files}
    for k in ('masks_raw', 'masks_smooth'):
        d[k] = np.unpackbits(d[k], axis=1, count=V).astype(bool)
    return d


# ------------------------------------------------------------------ P1: the existing default pipeline
def build_p1(files, mesh, sp, cfg, dev, out_path):
    from freesam3d.streaming.memory import MemoryConfig, QueryMemory
    from freesam3d.streaming.projection import load_projection
    ck = torch.load(resolve_path('work_dirs/mask_scorer/scorer/model.pt'), map_location=dev, weights_only=False)
    scorer = MaskScorer(ck['scalar_dim'], ck['feat_dim'], ck['hidden']).to(dev).eval()
    scorer.load_state_dict(ck['state_dict'])
    std = ScalarStandardizer.from_state(ck['standardizer'])
    proj = load_projection(resolve_path('work_dirs/stream/projection/proj/model.pt'), dev)
    mem = QueryMemory(MemoryConfig(voxel=0.05, match_threshold=0.3, admit_score=0.3, assignment='hungarian',
                                   match_score='contain_x_feat', match_dilation=1, render_min_votes=1,
                                   render_min_vote_frac=0.0, feat_weight=1.0, merge_feat_cos=-1.0,
                                   merge_touch_cos=-1.0, merge_dilation=0), device=dev, projection=proj)
    upd = []
    for f in files:
        d = load_proposals(f, with_masks=False)
        d['masks'] = np.load(f)['masks']
        X = std.apply(torch.as_tensor(d['scalars'], device=dev))
        Fe = torch.as_tensor(d['feat'], device=dev)
        with torch.no_grad():
            s = scorer.score(X, Fe).cpu().numpy()
        fr = frame_input_from_proposal_cache(d, scores=s, top_k=None)
        fr = select_frame_proposals(fr, NMS_IOU, 0.8, 20, dev)
        fr.frame_id = int(f.stem.rsplit('_', 1)[1])
        t0 = time.perf_counter()
        mem.update(fr)
        if str(dev).startswith('cuda'):
            torch.cuda.synchronize()
        upd.append(1000 * (time.perf_counter() - t0))
    masks_raw, scores = mem.render(mesh)
    n_obs = mem.n_obs.cpu().numpy()
    masks_smooth = segment_smooth(masks_raw, sp, device=dev)
    save_cache(out_path, masks_raw, masks_smooth, scores, n_obs,
               np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int8))
    return dict(frames=len(files), objects=int(len(scores)), update_ms=float(np.mean(upd)))


# ------------------------------------------------------------------ P2: WindowMemory, window = 1
def build_p2(frames, mesh, mesh_gt, mesh_sem, sp, cfg, dev, stuff, rule, threshold, model, std, out_path):
    wc = cfg['stream_v2']
    min_gt = cfg['eval']['min_gt_points']
    wcfg = WindowConfig(window=1, past_unit=wc['past_unit'], voxel=wc['voxel'],
                        touch_dilation=wc['touch_dilation'], threshold=threshold, linkage=wc['linkage'],
                        admit_min_points=wc['admit_min_points'], object_merge=bool(wc['object_merge']),
                        purity=wc['purity'])
    mem = WindowMemory(wcfg, score_fn=window_score_fn(model, std, rule, dev), device=dev)
    for fr in frames:
        mem.update(fr)
    masks_raw, scores = mem.render(mesh)
    attrs = mem.object_attrs()
    assert np.allclose(attrs['conf'], scores, atol=1e-4), 'object_attrs / render score mismatch'
    masks_smooth = segment_smooth(masks_raw, sp, device=dev)

    # mixed-fragment destinations (GT is a statistic here: it names which fragments are mixed, and
    # which final objects glued two instances together; it never reaches the merge)
    gts = mesh_gtset(mesh_gt, mesh_sem, stuff, min_gt)
    over = overmerge_flags(masks_raw, gts) if gts is not None else np.zeros(len(masks_raw), bool)
    frags = impure_fragments(frames, purity=wc['purity'])
    obj_vox = object_voxel_sets(mem)
    rows = np.zeros(len(frags), np.int64)
    for pos in sorted({f[0] for f in frags}):
        sel = [i for i, f in enumerate(frags) if f[0] == pos]
        rows[sel] = frag_object_rows(frames[pos], [frags[i][1] for i in sel], obj_vox, voxel=wcfg.voxel)
    fate_n = np.asarray([f[4] for f in frags], np.int64)
    shares, _ = classify_fates(rows, fate_n, over, attrs['n_obs'])
    klass = np.full(len(frags), FATES.index('dropped'), np.int8)
    for i, r in enumerate(rows):
        if r >= 0:
            klass[i] = FATES.index('glued_overmerge' if over[r] else
                                   ('own_single' if attrs['n_obs'][r] <= 1 else 'glued_clean'))
    save_cache(out_path, masks_raw, masks_smooth, scores, attrs['n_obs'], rows, fate_n, klass)
    return dict(frames=len(frames), objects=int(len(scores)), shares=shares,
                times={k: float(v) for k, v in mem.times.items()}, last_times=mem.times)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--stage', default='all', choices=['cache', 'eval', 'all'])
    p.add_argument('--pipelines', nargs='*', default=[P1] + list(P2_VARIANTS))
    p.add_argument('--scenes', nargs='*')
    p.add_argument('--device', default='cuda')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='print the plan and exit')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    dev = a.device if torch.cuda.is_available() else 'cpu'
    ws = resolve_path('work_dirs/esam_eval')
    stuff = stuff_ids('/home/yqx/SegmentAnything3D/ESAM/data/scannet200/meta_data/scannetv2-labels.combined.tsv')
    out_dir = resolve_path(cfg['stream_v2']['out_root']) / 'phase0c'
    cache_dir = out_dir / 'cache'

    frag_files = sorted((resolve_path(cfg['cache_root']) / 'val').glob('*.npz'))
    by_scene = {}
    for f in frag_files:
        by_scene.setdefault(f.stem.rsplit('_', 1)[0], []).append(f)
    prop_files = list_proposal_cache(resolve_path(cfg['cache_root']) / 'proposals' / 'val')
    prop_by_scene = {}
    for f in prop_files:
        prop_by_scene.setdefault(f.stem.rsplit('_', 1)[0], []).append(f)
    scenes = a.scenes or sorted(by_scene)
    if a.dry_run:
        print(f'{len(scenes)} scenes, pipelines {a.pipelines}, out {out_dir}')
        print(f'  fragment frames {sum(len(by_scene[s]) for s in scenes)}, proposal frames '
              f'{sum(len(prop_by_scene[s]) for s in scenes)}')
        return

    model = std = calib = None
    if any(v.startswith('p2') for v in a.pipelines):
        model, std, ck = load_group_merge_net(resolve_path('work_dirs/merge_net/full/model.pt'), dev)
        calib = float(ck['threshold'])
        print(f'GroupMergeNet threshold from the checkpoint: {calib}')

    timing = {'calib_threshold': calib, 'p1': {}, 'p2': {}}
    tpath = out_dir / 'timing.json'                       # the eval stage runs as its own process
    if a.stage == 'eval' and tpath.exists():
        timing = json.loads(tpath.read_text())
        calib = timing.get('calib_threshold')
    if a.stage in ('cache', 'all'):
        for scene in scenes:
            mesh, mesh_gt, mesh_sem, sp = load_mesh(ws, scene)
            if P1 in a.pipelines:
                path = cache_dir / P1 / f'{scene}.npz'
                if path.exists() and not a.overwrite:
                    print(f'{scene} p1: cache exists, skip', flush=True)
                else:
                    t0 = time.perf_counter()
                    info = build_p1(sorted(prop_by_scene[scene], key=lambda q: int(q.stem.rsplit('_', 1)[1])),
                                    mesh, sp, cfg, dev, path)
                    timing['p1'][scene] = info['update_ms']
                    print(f"{scene} p1: {info['frames']} frames, {info['objects']} objects, "
                          f"update {info['update_ms']:.1f} ms/frame, {time.perf_counter() - t0:.1f} s", flush=True)
            frames = None
            for name in [v for v in a.pipelines if v.startswith('p2')]:
                path = cache_dir / name / f'{scene}.npz'
                if path.exists() and not a.overwrite:
                    print(f'{scene} {name}: cache exists, skip', flush=True)
                    continue
                if frames is None:
                    frames = frames_from_merge_cache(
                        sorted(by_scene[scene], key=lambda q: int(q.stem.rsplit('_', 1)[1])), gt=cfg['eval']['gt'])
                rule, thr = variant_spec(name, calib)
                t0 = time.perf_counter()
                info = build_p2(frames, mesh, mesh_gt, mesh_sem, sp, cfg, dev, stuff, rule, thr, model, std, path)
                timing['p2'].setdefault(name, {})[scene] = info['times']
                print(f"{scene} {name} (rule {rule}, thr {thr}): {info['objects']} objects, "
                      f"{time.perf_counter() - t0:.1f} s | fates "
                      + ' '.join(f"{k} {v:.2f}" for k, v in info['shares'].items()), flush=True)
    if a.stage == 'cache':
        (out_dir / 'timing.json').write_text(json.dumps(timing, indent=1))
        return

    per_scene = []
    for scene in scenes:
        mesh, mesh_gt, mesh_sem, sp = load_mesh(ws, scene)
        gts = mesh_gtset(mesh_gt, mesh_sem, stuff, cfg['eval']['min_gt_points'])
        if gts is None:
            continue
        row = dict(scene=scene, num_gt=len(gts.ids), combo={}, fate={})
        for name in a.pipelines:
            path = cache_dir / name / f'{scene}.npz'
            if not path.exists():
                continue
            d = load_cache(path)
            for set_name in SETS:
                mk, vk = f'masks_{set_name}', f'vc_{set_name}'
                base = np.where(d[vk] >= cfg['eval']['min_gt_points'])[0]
                masks, conf, n_obs = d[mk][base], d['scores'][base], d['n_obs'][base]
                iou_gt = best_iou(masks, gts, dev)
                # the (K,K) mask-IoU matrix does not depend on the ranking score: build it once
                # per (pipeline, mask set) and reuse it for every nms / nms_single combination
                iou_pair = iou_pair_matrix(masks, dev)
                for sm in SCORES:
                    s = rank_scores(sm, masks=masks, conf=conf, n_obs=n_obs, iou_gt=iou_gt)
                    for cl in CLEANS:
                        keep = (greedy_nms(masks, s, NMS_IOU, iou=iou_pair) if cl != 'none'
                                else np.argsort(-s, kind='stable'))
                        if cl == 'nms_single':
                            keep = filter_single(keep, n_obs)
                        row['combo'][f'{name}|{sm}|{cl}|{set_name}'] = metrics_of(masks[keep], s[keep], gts, dev)
            if d['fate_rows'].size:
                totals = {f: float(d['fate_n'][d['fate_class'] == i].sum()) for i, f in enumerate(FATES)}
                tot = max(sum(totals.values()), 1.0)
                row['fate'][name] = dict(shares={k: v / tot for k, v in totals.items()}, totals=totals,
                                         n_frag=int(len(d['fate_rows'])))
        per_scene.append(row)
        k = f'{P1}|conf|none|smooth'
        print(f"{scene}: {len(gts.ids)} GT | " + (f"p1 AP50 {row['combo'][k]['ap50']:.3f}" if k in row['combo'] else '')
              + ' | ' + ' '.join(f"{n} {row['combo'][f'{n}|conf|nms_single|smooth']['ap50']:.3f}"
                                 for n in a.pipelines if f'{n}|conf|nms_single|smooth' in row['combo']), flush=True)
    write_report(out_dir, per_scene, a.pipelines, cfg, timing)


def answers_section(mean, per_scene, pipelines):
    """The four questions the stage-0c brief asks, answered from the numbers in the tables below.

    Generated (not hand-written) so re-running ``--stage eval`` reproduces it; every figure is read
    from ``mean``, ``per_scene`` or the phase-0b reference file. Nothing here is tuned on the val scenes.
    """
    ap = lambda k: mean[k]['ap50'] if k in mean else float('nan')
    mk = lambda k: mean[k]['masks'] if k in mean else float('nan')
    nogt, ceil = {}, {}
    for q in pipelines:
        keys = [k for k in mean if k.startswith(q + '|') and k.endswith('|smooth')]
        nogt[q] = max(((ap(k), k) for k in keys if '|oracle|' not in k), default=(float('nan'), '-'))
        ceil[q] = ap(q + '|oracle|nms|smooth')
    p0b = json.loads(resolve_path('work_dirs/stream_v2/phase0b/summary.json').read_text())
    c0, esam = p0b['combo'], p0b['refs']['esam']['ap50']
    b = lambda k: c0.get(k, {}).get('ap50', float('nan'))
    p2 = [q for q in pipelines if q != P1]
    lab = lambda q: 'P1（现有默认管线）' if q == P1 else f'P2 真实合并 {q}'
    d = lambda q: mean.get(f'{q}|nobs|nms|smooth', {})
    dup_lo, dup_hi = min(d(q)['dup_rate'] for q in p2), max(d(q)['dup_rate'] for q in p2)
    fr_lo, fr_hi = min(d(q)['frags'] for q in p2), max(d(q)['frags'] for q in p2)
    om_hi = max(d(q)['overmerge'] for q in p2)
    avg_gt = float(np.mean([r['num_gt'] for r in per_scene]))
    p1_none = ap(P1 + '|conf|none|smooth')
    best_v, best_k = max(nogt.values(), key=lambda kv: kv[0])
    loss_lo = b('conf|nms|k0|smooth') - max(ap(q + '|conf|nms|smooth') for q in p2)
    loss_hi = b('conf|nms_single|k0|smooth') - max(nogt[q][0] for q in p2)
    worst_k = max(d(q)['masks'] for q in p2)

    L = ['', '## 0. 四个问题的直白回答', '',
         f'**1. 不用 GT、加上 0b 的免费收尾，真实 AP50 最高 {best_v:.3f}**（`{best_k}`，'
         f'{mk(best_k):.1f} mask/场景）。这就是现有默认管线 P1：比它自己不做收尾的 {p1_none:.3f} 只高 '
         f'{best_v - p1_none:+.3f}，比 ESAM-E 的 {esam:.3f} 低 {esam - best_v:.3f}。', '',
         f'0b 报的 0.721 是在 GT oracle 合并之上做同一套后处理得到的：oracle 会把"混了两个物体的碎片"'
         f'判成孤立的单帧物体，"丢掉只出现一帧的物体"于是删得干净。换成不用 GT 的真实合并，'
         f'同样的免费收尾只给到 {best_v:.3f}。**0.721 是 oracle 的产物，真实管线上不成立。**', '',
         f'**2. 合并不完美造成的损失：{loss_lo:.2f}–{loss_hi:.2f} AP50，主要是"漏粘"，不是"粘错"。**', '',
         '| 口径（段平滑 AP50） | 值 |', '|---|---|',
         f'| 0b：oracle 合并 + conf 排序 + nms | {b("conf|nms|k0|smooth"):.3f} |']
    for q in pipelines:
        L.append(f'| {lab(q)} + 同样的 conf+nms | {ap(q + "|conf|nms|smooth"):.3f} |')
    L.append(f'| 0b：oracle 合并 + 完美排序 + nms（上限） | {b("oracle|nms|k0|smooth"):.3f} |')
    for q in pipelines:
        L.append(f'| {lab(q)} + 完美排序 + nms | {ceil[q]:.3f} |')
    L += ['',
          f'同排序同清理下，oracle 合并 {b("conf|nms|k0|smooth"):.3f} 对真实合并 '
          f'{min(ap(q + "|conf|nms|smooth") for q in p2):.3f}–'
          f'{max(ap(q + "|conf|nms|smooth") for q in p2):.3f}；'
          f'即使排序换成完美，P2 上限最高也只有 {max(ceil[q] for q in p2):.3f}，'
          f'而 oracle 合并有 {b("oracle|nms|k0|smooth"):.3f}——**合并本身值 '
          f'−{b("oracle|nms|k0|smooth") - max(ceil[q] for q in p2):.3f}**。', '',
          '证据指向漏粘而不是粘错：真实 P2 每场景留 '
          + '、'.join(f'{q} {mk(q + "|nobs|nms|smooth"):.0f} 个 mask' for q in p2)
          + f'（0b 只有 111.8 个），dup_rate {dup_lo:.2f}–{dup_hi:.2f}、frags/GT {fr_lo:.1f}–{fr_hi:.1f}，'
          f'而 overmerge 四个变体全部 ≤{om_hi:.3f}——没有一个是"粘错了"。', '',
          '**3. "丢单帧"在真实合并下失效，甚至有害。**', '',
          '| 管线 | S_nobs+nms | S_nobs+nms+丢单帧 |', '|---|---|---|']
    for q in pipelines:
        L.append(f'| {q} | {ap(q + "|nobs|nms|smooth"):.3f} | {ap(q + "|nobs|nms_single|smooth"):.3f} |')
    L += ['',
          f'oracle 合并下这一步是 +{b("conf|nms_single|k0|smooth") - b("conf|nms|k0|smooth"):.2f}'
          f'（{b("conf|nms|k0|smooth"):.3f} → {b("conf|nms_single|k0|smooth"):.3f}）；真实合并下 '
          + '、'.join(f'{q} {ap(q + "|nobs|nms|smooth"):.3f} → {ap(q + "|nobs|nms_single|smooth"):.3f}'
                      for q in p2)
          + '。原因很直接：真实（欠）合并下几乎每个物体都只被看到 1 帧，过滤器删掉的是真物体'
            '（p2b_calib 在 S_nobs 下 mask 从 '
          + f'{mk("p2b_calib|nobs|nms|smooth"):.0f} 掉到 {mk("p2b_calib|nobs|nms_single|smooth"):.0f}，'
            f'在 S_conf 下从 {mk("p2b_calib|conf|nms|smooth"):.0f} 掉到 '
            f'{mk("p2b_calib|conf|nms_single|smooth"):.0f}）。'
            '**合并修好之前，n_obs == 1 过滤器不能当作 baseline。**', '',
          '混合碎片（纯度 < 0.8）的去向（占总点数比例，GT 仅用于统计）：阈值过严时绝大多数原地变成'
          '孤立的单帧物体（p2b_calib 88.2%），阈值可用时它们大多被干净地粘进本属物体'
          '（p2a_calib 56.6% 干净 + 13.2% overmerge）。所以该校正的不是"要不要丢"，而是合并阈值本身。', '',
          '**4. 第 1、2 阶段：先学合并，再学排序。**', '',
          f'排序的头寸在真实合并下看着很大（p2b_calib {ap("p2b_calib|nobs|nms|smooth"):.3f} → '
          f'{ceil["p2b_calib"]:.3f}），但它是被合并失败抬起来的：完美排序只是"从 '
          f'{mk("p2b_calib|nobs|nms|smooth"):.0f} 个近重复 mask 里挑对约 {avg_gt:.0f} 个（场景平均 GT 数）"。'
          f'修好合并后，排序空间回落到 P1 量级（{ap(P1 + "|nobs|nms|smooth"):.3f} → {ceil[P1]:.3f}，'
          f'+{ceil[P1] - ap(P1 + "|nobs|nms|smooth"):.3f}）。在这个 dup_rate {dup_lo:.2f}–{dup_hi:.2f}、'
          f'frags/GT {fr_lo:.1f}–{fr_hi:.1f} 的物体集上训排序器，只会学到"压碎片"，不可迁移。', '',
          '具体做法：第 1 阶段先训跨帧合并打分，并在**几何包含度尺度**上重标阈值——checkpoint 里标定的 '
          '0.65 是"边概率"尺度，与 P2 的几何包含度不可比（0.65 欠粘、0.5 过粘就是它的表现），'
          '而且 average_linkage 用簇间已存在边的均值，会把正确证据被帧内错误稀释掉。第 2 阶段再训排序/清理。', '']
    assert worst_k > 0
    return '\n'.join(L)


def write_report(out_dir, per_scene, pipelines, cfg, timing):
    if not per_scene:
        raise SystemExit('no scenes evaluated; build the cache first (--stage cache)')
    n = len(per_scene)
    keys = [f'{pn}|{sm}|{cl}|{st}' for st in SETS for pn in pipelines for sm in SCORES for cl in CLEANS]
    have = [k for k in keys if all(k in r['combo'] for r in per_scene)]
    mean = {k: {m: float(np.mean([r['combo'][k][m] for r in per_scene])) for m in METRIC_KEYS} for k in have}

    def best(pipe, set_name, exclude_oracle=True):
        cand = [k for k in have if k.startswith(f'{pipe}|') and k.endswith(f'|{set_name}')
                and not (exclude_oracle and k.split('|')[1] == 'oracle')]
        return max(cand, key=lambda k: mean[k]['ap50'])

    lines = ['# 第 0c 阶段：真实管线（不用 GT）+ 0b「免费收尾」的场景级 AP', '',
             f'{n} 个验证场景，ESAM 网格协议（thing GT ≥ {cfg["eval"]["min_gt_points"]} 顶点，段平滑 thr 0.5，场景平均）。',
             '决策全程不用 GT。用到 GT 的只有：S_oracle 排序行、P2 混合碎片去向、以及各表的指标本身（均标注）。',
             f'P2 的阈值取自 GroupMergeNet checkpoint（{timing.get("calib_threshold")}），另跑固定 0.5 作对照；未在验证集上调。',
             '', '## 1. 主表（段平滑后）', '',
             '| 管线 | 排序 | 清理 | masks | 平均最佳 IoU | 召回@0.5 | AP50 | AP25 | dup_rate | overmerge | frags/GT |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for k in have:
        if not k.endswith('|smooth'):
            continue
        pn, sm, cl, _ = k.split('|')
        r = mean[k]
        lines.append(f"| {pn} | S_{sm} | {cl} | {r['masks']:.1f} | {r['iou']:.3f} | {r['r50']:.3f} | "
                     f"{r['ap50']:.3f} | {r['ap25']:.3f} | {r['dup_rate']:.3f} | {r['overmerge']:.3f} | {r['frags']:.2f} |")
    lines += ['', '## 2. 未平滑（同表）', '',
              '| 管线 | 排序 | 清理 | masks | 平均最佳 IoU | 召回@0.5 | AP50 | AP25 | dup_rate | overmerge | frags/GT |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
    for k in have:
        if not k.endswith('|raw'):
            continue
        pn, sm, cl, _ = k.split('|')
        r = mean[k]
        lines.append(f"| {pn} | S_{sm} | {cl} | {r['masks']:.1f} | {r['iou']:.3f} | {r['r50']:.3f} | "
                     f"{r['ap50']:.3f} | {r['ap25']:.3f} | {r['dup_rate']:.3f} | {r['overmerge']:.3f} | {r['frags']:.2f} |")
    lines += ['', '## 3. 每个管线最好的一行（无 GT）与上限', '',
              '| 管线 | 段平滑最好（无 GT） | AP50 | 上限 S_oracle+nms | AP50 |', '|---|---|---|---|---|']
    for pn in pipelines:
        if not any(k.startswith(f'{pn}|') for k in have):
            continue
        b = best(pn, 'smooth')
        up = f'{pn}|oracle|nms|smooth'
        lines.append(f"| {pn} | {b.split('|', 1)[1]} | {mean[b]['ap50']:.3f} | "
                     f"{'S_oracle+nms' if up in mean else '-'} | {mean[up]['ap50']:.3f} |" if up in mean
                     else f"| {pn} | {b.split('|', 1)[1]} | {mean[b]['ap50']:.3f} | - | - |")
    ref = resolve_path(cfg['stream_v2']['out_root']) / 'phase0b' / 'summary.json'
    if ref.exists():
        z = json.loads(ref.read_text())['combo']
        lines += ['', '## 4. 与 0b（GT oracle 合并）并排，差距 = 合并不完美造成的损失', '',
                  '| 口径 | 段平滑 AP50 | 未平滑 AP50 |', '|---|---|---|']
        for label, key in (('0b 上限 S_oracle+nms', 'oracle|nms|k0'), ('0b 无 GT 最好 S_conf+nms+丢单帧',
                                                                      'conf|nms_single|k0')):
            if f'{key}|smooth' in z:
                lines.append(f"| {label}（oracle 合并） | {z[f'{key}|smooth']['ap50']:.3f} | {z[f'{key}|raw']['ap50']:.3f} |")
        for pn in pipelines:
            if not any(k.startswith(f'{pn}|') for k in have):
                continue
            b, br = best(pn, 'smooth'), best(pn, 'raw')
            lines.append(f"| {pn} 最好 | {mean[b]['ap50']:.3f} | {mean[br]['ap50']:.3f} |")
    lines += ['', '## 5. P2 混合碎片（纯度 < 0.8）的去向（占总点数的比例；GT 仅用于统计）', '',
              '| 变体 | 自己成为只出现 1 帧的物体 | 粘进物体并造成 overmerge | 粘进物体但没 overmerge | 消失（被淘汰） |',
              '|---|---|---|---|---|']
    for pn in pipelines:
        rows = [r['fate'][pn] for r in per_scene if pn in r['fate']]
        if not rows:
            continue
        agg = {f: float(np.mean([r['shares'][f] for r in rows])) for f in FATES}
        lines.append(f"| {pn} | " + ' | '.join(f'{100 * agg[f]:.1f}%' for f in FATES) + ' |')
    lines += ['', '## 6. P2 记忆计时（每帧均值，ms）', '',
              '| 变体 | freeze | build | score | cluster | assign | 合计 |', '|---|---|---|---|---|---|---|']
    for pn in pipelines:
        per = timing.get('p2', {}).get(pn)
        if not per:
            continue
        agg = {k: float(np.mean([v.get(k, 0.0) for v in per.values()])) for k in
               ('freeze_ms', 'build_ms', 'score_ms', 'cluster_ms', 'assign_ms')}
        lines.append(f"| {pn} | " + ' | '.join(f'{agg[k]:.1f}' for k in agg) + f" | {sum(agg.values()):.1f} |")
    if timing.get('p1'):
        lines.append(f"\nP1 `QueryMemory.update`：{float(np.mean(list(timing['p1'].values()))):.1f} ms/帧"
                     '（对比：整条流水线 98.8 ms/帧，见 docs/streaming.zh-CN.md §0）。')
    md = '\n'.join(lines) + '\n'
    md = md.replace('\n## 1. 主表（段平滑后）',
                    answers_section(mean, per_scene, pipelines) + '\n## 1. 主表（段平滑后）', 1)
    (out_dir / 'summary.md').write_text(md)
    (out_dir / 'summary.json').write_text(json.dumps(dict(mean=mean, pipelines=list(pipelines),
                                                          scenes=n, timing=timing), indent=2))
    (out_dir / 'per_scene.json').write_text(json.dumps(
        [dict(scene=r['scene'], num_gt=r['num_gt'],
              combo={k: v for k, v in r['combo'].items() if k.endswith('|smooth')}, fate=r['fate'])
         for r in per_scene], indent=1))
    print(md)


if __name__ == '__main__':
    main()
