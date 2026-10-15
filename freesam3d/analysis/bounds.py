"""阶段 5 · P3：掩码集合上界（不训练；任务书 §4 P3）。

三个子问题：
1. 原始 mask 池上界：把每帧全部 FastSAM 原始 mask（在缓存 20k 点上的成员 → 5 cm 体素键）跨帧
   直接并池（不合并、不去重），完美排序算 AP50；另给「原始 ∪ 现有候选」与「原始池 + 包含度
   去重 0.7」两个版本；GT 实例渲染池作口径上界确认（≈ 1.0）。
2. 拆分上界：把每个候选按 GT 切成「与某个 GT 一致的核心」（share ≥ thr），完美排序算 AP50——
   这是任何无监督拆分机制的上界（用 GT 切，只是界，不是结果）。
3. 冗余：包含度去重 0.7 对 oracle 上界与 mask 数的影响。

口径与评估完全一致：体素代理 GT（5 cm、跨帧多数实例、thing ≥20 体素）、候选 ≥20 体素、
贪心匹配（``streaming/phase3_eval.py::eval_objects``）。CPU 单测见 ``tests/test_phase5_bounds.py``。
"""
import json
from pathlib import Path

import numpy as np
import torch

from ..config import resolve_path

SPLIT_THRS = (0.1, 0.2)


# ------------------------------------------------------------------ 纯函数
def share_matrix(masks, gt_vox, keep):
    """(P,K)：每个候选落在每个 GT 实例内的体素占比。keep 为空 → (P,0)。"""
    if len(keep) == 0 or len(masks) == 0:
        return np.zeros((len(masks), 0), np.float32)
    onehot = np.stack([gt_vox == k for k in keep], 1)
    inter = masks.astype(np.float32) @ onehot.astype(np.float32)
    return inter / np.maximum(masks.sum(1, keepdims=True), 1)


def pieces_pool(masks, gt_vox, keep, thr):
    """把每个候选切成「与某个 GT 一致的核心」（share ≥ thr）；GT 只用于切，不参与打分。"""
    if len(keep) == 0 or len(masks) == 0:
        return np.zeros((0, masks.shape[1]), bool)
    onehot = np.stack([gt_vox == k for k in keep], 1)
    sh = share_matrix(masks, gt_vox, keep)
    rows = [masks[p] & onehot[:, k] for p in range(len(masks)) for k in np.where(sh[p] >= thr)[0]]
    rows = [r for r in rows if bool(r.any())]
    return np.stack(rows) if rows else np.zeros((0, masks.shape[1]), bool)


def iou_matrix_np(masks, gt_vox, keep):
    """(P,K) 体素 IoU（评估口径的 numpy 版，只用于诊断 recall）。"""
    if len(keep) == 0 or len(masks) == 0:
        return np.zeros((len(masks), 0), np.float32)
    onehot = np.stack([gt_vox == k for k in keep], 1)
    inter = masks.astype(np.float32) @ onehot.astype(np.float32)
    union = masks.sum(1, keepdims=True) + onehot.sum(0)[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def recall_at_50(iou):
    """(matched, n_gt)：按最佳 IoU 降序贪心匹配、IoU ≥ 0.5 且 GT 未被占用。"""
    if iou.shape[1] == 0 or iou.shape[0] == 0:
        return 0, int(iou.shape[1])
    taken = np.zeros(iou.shape[1], bool)
    m = 0
    for p in np.argsort(-iou.max(1)):
        free = np.where(~taken & (iou[p] >= 0.5))[0]
        if len(free):
            taken[free[int(np.argmax(iou[p, free]))]] = True
            m += 1
    return m, int(iou.shape[1])


def pool_ap50(masks, gt_vox, keep, device='cpu', dedup=None):
    """完美排序（oracle 分数）下该池的 (AP50, 池大小)；dedup = 先做包含度抑制。"""
    from ..evaluation.instance import GTSet
    from ..streaming.phase3_eval import eval_objects, oracle_scores
    from ..streaming.unified_dedup import dedup_masks
    if len(keep) == 0 or len(masks) == 0:
        return float('nan'), 0
    gts = GTSet.from_labels(gt_vox, keep)
    s = np.asarray(oracle_scores(masks, gts, device), np.float32)
    m = masks
    if dedup is not None:
        k = dedup_masks(m, s, float(dedup), device=device)
        m, s = m[k], s[k]
    return float(eval_objects(m, s, gts, device=device)['ap50']), int(len(m))


def render_keys(key_rows, cells):
    """体素键列表 → (P,V) bool 掩码（与 ``phase3_eval.pred_masks`` 同一渲染逻辑）。"""
    masks = np.zeros((len(key_rows), len(cells)), bool)
    for i, v in enumerate(key_rows):
        v = np.asarray(v, np.int64)
        if len(v) == 0:
            continue
        pos = np.clip(np.searchsorted(cells, v), 0, len(cells) - 1)
        masks[i, pos[cells[pos] == v]] = True
    return masks


# ------------------------------------------------------------------ 原始 mask 池（GPU 采集）
@torch.no_grad()
def raw_pool_scene(scene, cfg, runner, cache_dir, image_root, voxel=0.05):
    """每帧 FastSAM 原始 mask 在缓存点上的成员 → 场景 5 cm 单元掩码池（跨帧直接并池）。

    用缓存的 ``pixels`` 采样原始 mask，保证与缓存同一批点、同一体素单元；不合并、不去重。
    """
    from ..models.instance_head import load_rgb
    from ..streaming.evaluate import voxel_majority_gt
    from ..streaming.holdout import load_scene_frames
    from ..streaming.voxel import voxelize
    frames = load_scene_frames(cache_dir, scene, gt='2d')
    cells, _, _ = voxel_majority_gt(frames, voxel)
    rows, per_frame = [], []
    for fr in frames:
        with np.load(Path(cache_dir) / f'{scene}_{fr.frame_id}.npz') as z:
            pixels = np.asarray(z['pixels'])
        out = runner(load_rgb(image_root, scene, fr.frame_id))
        m = np.asarray(out.masks)
        per_frame.append(int(m.shape[0]))
        if m.shape[0] == 0:
            continue
        pts = m[:, pixels[:, 1], pixels[:, 0]]        # (M,N) 原始 mask 在 20k 点上的成员
        key = voxelize(fr.xyz, voxel)
        key = np.asarray(key[0] if isinstance(key, tuple) else key).reshape(-1)
        for i in range(pts.shape[0]):
            v = np.unique(key[pts[i]])
            if len(v):
                rows.append(v)
    return dict(cells=cells, pool=render_keys(rows, cells), per_frame=per_frame)


def cos_keep(pf, mean_dir, tau):
    """(N,) bool：与均值方向余弦 ≥ tau（tau < 0 = 全保留）。纯函数，CPU 单测用。"""
    if tau < 0:
        return np.ones(len(pf), bool)
    return (np.asarray(pf, np.float32) @ np.asarray(mean_dir, np.float32)) >= tau


# ------------------------------------------------------------------ 外观一致性裁剪（GPU 采集）
@torch.no_grad()
def feature_trim_scene(scene, cfg, model, std, enc, device, cache_dir, image_root, gt_vox, keep,
                       taus=(0.3, 0.5, 0.7), mem_overrides=None, min_gt_voxels=20):
    """外观一致性裁剪的上界：每个候选只保留与自身 proto 均值方向余弦 ≥ tau 的成员点，重新体素化。

    单遍采集（逐点 proto 单位特征，减帧均值 + L2，与 F1.1/F4b.5 同口径；逐帧缓存在 CPU），
    折叠到最终物体后取均值方向，再按 tau 裁剪成员点、按 5 cm 体素重建掩码。``tau=-1`` 为不裁剪
    对照（应复现该场景的 oracle）。GT 只用于打分，不进入裁剪。
    """
    from ..models.instance_head import load_rgb, sample_features
    from ..streaming.holdout import load_scene_frames
    from ..streaming.phase3_eval import voxel_gt
    from ..streaming.unified_flow import make_unified_memory
    from ..streaming.voxel import voxelize
    from ..streaming.window_data import final_object_labels
    wc = cfg['stream_v2']
    voxel = float(wc['voxel'])
    stuff = set(cfg['eval']['stuff_nyu40'])
    frames = load_scene_frames(cache_dir, scene, gt='2d')
    mem = make_unified_memory(cfg, model, std, device, **(mem_overrides or {}))
    pt_raw, pf_frames, sums, merges = [], [], {}, []
    for fr in frames:
        labels = np.asarray(mem.update(fr), np.int64)
        merges.extend(mem.last_unified['merged'])
        with np.load(Path(cache_dir) / f'{scene}_{fr.frame_id}.npz') as z:
            pixels = np.asarray(z['pixels'])
        rgb = load_rgb(image_root, scene, fr.frame_id)
        maps, geom = enc.encode(rgb)
        pf = sample_features(maps['proto'], pixels, geom)
        pf = pf - pf.mean(0, keepdim=True)
        pf = pf / pf.norm(dim=1, keepdim=True).clamp_min(1e-6)
        pf_frames.append(pf.cpu().numpy().astype(np.float32))
        pt_raw.append(labels.astype(np.int32))
        sel = labels >= 0
        if sel.any():
            sel_t = torch.as_tensor(sel, device=device)
            lab_t = torch.as_tensor(labels[sel], device=device)
            oids, inv = torch.unique(lab_t, return_inverse=True)
            S = torch.zeros(len(oids), pf.shape[1], device=device).index_add_(0, inv, pf[sel_t])
            n = torch.bincount(inv, minlength=len(oids)).to(torch.float64).cpu().numpy()
            for k, o in enumerate(oids.cpu().tolist()):
                s0, n0 = sums.get(int(o), (np.zeros(pf.shape[1], np.float64), 0.0))
                sums[int(o)] = (s0 + S[k].cpu().numpy().astype(np.float64), n0 + float(n[k]))
    final_of, _ = final_object_labels(mem, frames, voxel, stuff, min_gt_voxels, merges, device)
    cells, _ = voxel_gt(frames, stuff, voxel=voxel, min_gt_voxels=min_gt_voxels)
    assert len(cells) == len(gt_vox), (scene, len(cells), len(gt_vox))
    fsum = {}
    for oid, (s0, n0) in sums.items():
        f = int(final_of.get(int(oid), -1))
        if f < 0:
            continue
        a = fsum.setdefault(f, [np.zeros_like(s0), 0.0])
        a[0] += s0; a[1] += n0
    mean_dir = {f: s / max(float(np.linalg.norm(s)), 1e-9) for f, (s, _) in fsum.items()}
    lut = np.full((max(final_of.keys(), default=-1) + 2,), -1, np.int64)
    for oid, f in final_of.items():
        lut[int(oid)] = int(f)
    keys = {tau: {} for tau in taus}
    for t, fr in enumerate(frames):
        lab = pt_raw[t]
        fin = np.where(lab >= 0, lut[np.clip(lab, 0, len(lut) - 1)], -1)
        pf = pf_frames[t]
        key = np.asarray(voxelize(np.asarray(fr.xyz, np.float32), voxel)).reshape(-1)
        for f, m in mean_dir.items():
            idx = np.flatnonzero(fin == f)
            if not len(idx):
                continue
            for tau in taus:
                kept = idx[cos_keep(pf[idx], m, tau)]
                if len(kept):
                    keys[tau].setdefault(f, []).append(key[kept])
    out = dict(scene=scene)
    for tau in taus:
        rows = [np.unique(np.concatenate(keys[tau][f])) for f in sorted(mean_dir) if f in keys[tau]]
        masks = render_keys(rows, cells)
        ap, n = pool_ap50(masks, gt_vox, keep, device)
        out[f'trim_{tau}_ap50'] = ap
        out[f'trim_{tau}_n'] = n
    return out


# ------------------------------------------------------------------ 汇总（CPU）
def scene_bounds(d, cfg_dedup=0.7, device='cpu'):
    """一个留出场景的三类上界（输入 = P1 缓存的 npz 字典）。"""
    from ..streaming.unified_dedup import dedup_masks
    from ..streaming.phase3_eval import oracle_scores
    from ..evaluation.instance import GTSet
    gt_vox, keep, masks = d['gt_vox'], d['keep'], d['masks']
    out = dict(scene=d['scene'], n=int(len(masks)))
    if len(keep) == 0:
        return out
    gts = GTSet.from_labels(gt_vox, keep)
    iou = iou_matrix_np(masks, gt_vox, keep)
    m_cur, n_gt = recall_at_50(iou)
    out['recall_cur'] = [m_cur, n_gt]
    out['oracle_ap50'], _ = pool_ap50(masks, gt_vox, keep, device)
    out['oracle_ap50_dedup'], out['n_dedup'] = pool_ap50(masks, gt_vox, keep, device, dedup=cfg_dedup)
    for thr in SPLIT_THRS:
        p = pieces_pool(masks, gt_vox, keep, thr)
        out[f'split_{thr}_ap50'], out[f'split_{thr}_n'] = pool_ap50(p, gt_vox, keep, device)
    g = np.stack([gt_vox == k for k in keep])
    out['gt_render_ap50'], _ = pool_ap50(g, gt_vox, keep, device)
    sh = share_matrix(masks, gt_vox, keep)
    best = np.asarray(oracle_scores(masks, gts, device), np.float32)
    is_mixed = (sh >= 0.2).sum(1) >= 2
    out['mixed'] = int(is_mixed.sum())
    out['missed'] = int((best < 0.5).sum())
    out['missed_mixed'] = int((is_mixed & (best < 0.5)).sum())
    return out


def write_report(out_dir, cfg, log=print):
    """汇总 bounds 采集结果 + P1 缓存 → bounds.md/json。"""
    out_dir = Path(out_dir)
    raws = [json.loads(f.read_text()) for f in sorted((out_dir / 'raw').glob('*.json'))]
    p1 = resolve_path(cfg['phase5']['out_root']) / 'rank_probe' / 'cache'
    from .rank_probe import load_scene
    data = [load_scene(f) for f in sorted(p1.glob('*.npz'))]
    sb = [scene_bounds(d) for d in data]
    n = len(sb)

    def m(key):
        return float(np.nanmean([s.get(key, float('nan')) for s in sb]))
    cur_oracle = m('oracle_ap50')
    lines = ['# 阶段 5 · P3：掩码集合上界（60 个留出场景，体素代理口径，不训练）', '',
             f'参考：现有候选池 oracle（完美排序）**{cur_oracle:.4f}**；GT 渲染池 '
             f'{m("gt_render_ap50"):.4f}（口径上界确认）。', '',
             '## 1. 原始 FastSAM mask 池（每帧全部原始 mask，跨帧直接并池，不合并/不去重）', '']
    if raws:
        def mr(key):
            return float(np.nanmean([r.get(key, float('nan')) for r in raws]))
        tot = sum(r['recall_raw'][1] for r in raws); hit = sum(r['recall_raw'][0] for r in raws)
        cur_tot = sum(r['recall_cur'][1] for r in raws); cur_hit = sum(r['recall_cur'][0] for r in raws)
        lines += [f'- 池大小 {mr("pool_n"):.1f}/场景（原始 mask 数/帧 {mr("masks_per_frame"):.1f}）；'
                  f'oracle AP50 **{mr("raw_ap50"):.4f}**，+去重 0.7 后 {mr("raw_ap50_dedup"):.4f}'
                  f'（{mr("raw_n_dedup"):.1f} 个）。',
                  f'- 原始 ∪ 现有候选：oracle AP50 **{mr("union_ap50"):.4f}**，+去重 0.7 后 '
                  f'{mr("union_ap50_dedup"):.4f}。',
                  f'- oracle recall@0.5：原始池 {hit}/{tot} = {hit / max(tot, 1):.3f}；'
                  f'现有池 {cur_hit}/{cur_tot} = {cur_hit / max(cur_tot, 1):.3f}。', '']
    else:
        lines += ['- （未采集：先跑 `--stage collect`）', '']
    lines += ['## 2. 拆分上界（把每个候选按 GT 切成 share ≥ thr 的核心；任何无监督拆分机制的上界）', '',
              '| 阈值 | oracle AP50 | 切后池大小/场景 |', '|---|---|---|']
    for thr in SPLIT_THRS:
        lines.append(f'| {thr} | **{m(f"split_{thr}_ap50"):.4f}** | {m(f"split_{thr}_n"):.1f} |')
    lines += ['', '## 3. 冗余（包含度去重 0.7 对 oracle 上界的影响）', '',
              f'- 去重后 oracle **{m("oracle_ap50_dedup"):.4f}**（去重前 {cur_oracle:.4f}）；'
              f'mask 数 {m("n"):.1f} → {m("n_dedup"):.1f}/场景。',
              f'- 混合候选（≥2 个 GT 各占 ≥20%）：{m("mixed"):.1f}/场景；'
              f'未被 ≥0.5 匹配的候选 {m("missed"):.1f}/场景，其中混合的只有 {m("missed_mixed"):.1f}。', '']
    trims = ([json.loads(f.read_text()) for f in sorted((out_dir / 'trim').glob('*.json'))]
             if (out_dir / 'trim').exists() else [])
    if trims:
        lines += ['', '## 4. 外观一致性裁剪上界（只保留与候选 proto 均值方向余弦 ≥ tau 的成员点，'
                  '重新体素化；机制上界，不训练）', '']
        for tau in ('-1', '0.3', '0.5', '0.7'):
            vals = [t.get(f'trim_{tau}_ap50') for t in trims if t.get(f'trim_{tau}_ap50') is not None]
            if vals:
                lines.append(f'- tau {tau}: oracle **{float(np.nanmean(vals)):.4f}**'
                             + ('（不裁剪对照）' if tau == '-1' else ''))
    (out_dir / 'bounds.md').write_text('\n'.join(lines) + '\n')
    (out_dir / 'bounds.json').write_text(json.dumps(dict(
        per_scene=sb, raw_per_scene=raws,
        raw_ap50=float(np.nanmean([r.get('raw_ap50', np.nan) for r in raws])) if raws else None,
        union_ap50=float(np.nanmean([r.get('union_ap50', np.nan) for r in raws])) if raws else None,
        scenes=n), indent=1, ensure_ascii=False, default=float))
    log('\n'.join(lines))
    return dict(per_scene=sb, raw=raws)
