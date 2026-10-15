"""阶段 5 · P1：掩码质量打分的排序信息上限（不训练；任务书 §2）。

问题：第 2/3 阶段的排序缺口（60 个留出场景体素代理上，现有节点质量头 AP50 0.3069 → 完美排序
0.4519；16 个验证场景网格协议上 0.422 → 0.635）里，多少是「信息不够」、多少是「训练不够」？
本模块把每个候选的特征按族做「IoU ≥ 0.5」逻辑回归（按场景留一交叉验证），量出每族能到达的
AP50 天花板。GT 只用于打标签与上界，不进入任何特征。

流程（与任务书 §2 逐条对应）：
1. 用第 3 阶段最终配置在 60 个留出场景跑一遍管线，导出每个候选掩码（= 记忆的最终物体）：
   成员点（逐帧「点 → 最终物体」映射 ``pt_final``，−1 = 未分配）、体素（场景 5 cm 单元上的
   bool 掩码）、来源（``n_frag``/``n_obs`` 判定：单个 FastSAM 碎片 / 合并 mask）、与最佳 GT 的
   真实体素 IoU（``window_data.final_object_labels``，与现有评估完全同一口径）。
2. 口径校验：真实 IoU 完美排序复现已发布的 oracle（留出 0.4519；16 场景网格 0.635 读已记录
   结果复核，不重跑验证集）。
3. 特征族：自身属性 / 外观一致性（冻结 backbone、proto；阶段 4 的头作消融）/ 关系 / 时间 /
   现有头输出（对照）。每族：与真实 IoU 的 Spearman（pooled + 场景均值）+ 留一逻辑回归 AP50。

口径：体素代理 GT（5 cm 单元、跨帧多数实例、thing ≥20 体素）、候选 ≥20 体素、按分数贪心匹配，
与 ``streaming/phase3_eval.py::eval_objects`` 完全一致。CPU 单测见 ``tests/test_rank_probe.py``。
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from ..config import resolve_path

SELF_NAMES = ('n_pts', 'n_vox', 'n_frag', 'log_bbox_vol', 'compactness',
              'conf_mean', 'boundary_cos_mean', 'boundary_cos_min')
APPEARANCE_SUFFIX = ('pair_cos', 'var', 'consistency')
REL_NAMES = ('rel_contain_in_max', 'rel_contain_out_max', 'rel_iou_max', 'rel_n_overlap',
             'rel_is_subset')
TIME_NAMES = ('n_obs', 'age', 'obs_regularity', 'vox_growth')
BOUNDARY_COS_COL = 11        # merging.graph.edge_feature_names() 的 'boundary_cos'
BOUNDARY_COS_MIN_COL = 12    # 'boundary_cos_min'

# 任务书 §2 判定：全特征集拟合 AP50 = X，现有头 0.422，完美 0.635（16 场景网格口径）。
# 留出（体素代理）与 16 场景（网格）不是同一尺度，判定用与尺度无关的「缺口恢复比例」
# r = (X − head) / (perfect − head)；任务书阈值换算：0.58 → r = 0.742；0.48 → r = 0.272。
R_MOSTLY_TRAINING = (0.58 - 0.422) / (0.635 - 0.422)
R_PARTIAL = (0.48 - 0.422) / (0.635 - 0.422)


# ------------------------------------------------------------------ 特征族布局
def appearance_names(space):
    return tuple(f'{space}_{s}' for s in APPEARANCE_SUFFIX)


def family_names(spaces=('backbone', 'proto'), head_space='head'):
    """特征族 → 列名；'appearance' 只含冻结 backbone/proto（任务书口径），头单独一族作消融。"""
    fams = {'self': list(SELF_NAMES),
            'appearance': [n for s in spaces for n in appearance_names(s)],
            'rel': list(REL_NAMES),
            'time': list(TIME_NAMES),
            'control': ['quality']}
    if head_space:
        fams['appearance_head'] = list(appearance_names(head_space))
    return fams


def feature_layout(spaces=('backbone', 'proto'), head_space='head'):
    """(全列名, 各族切片)；'control'（现有头输出）在最后、不进「全特征集」。"""
    fams = family_names(spaces, head_space)
    order = (['self', 'appearance'] + (['appearance_head'] if head_space else [])
             + ['rel', 'time', 'control'])
    names, slices = [], {}
    for f in order:
        slices[f] = slice(len(names), len(names) + len(fams[f]))
        names.extend(fams[f])
    return names, slices


def full_columns(slices, with_head=False, with_control=False):
    """「全特征集」列下标：self + appearance + rel + time（+ 头 / + 现有头对照）。"""
    parts = ['self', 'appearance', 'rel', 'time']
    if with_head:
        parts.append('appearance_head')
    if with_control:
        parts.append('control')
    return np.concatenate([np.arange(slices[p].start, slices[p].stop) for p in parts])


# ------------------------------------------------------------------ 特征计算（纯函数）
def appearance_from_sums(sum_f, sum_f2, n):
    """(pair_cos, var, consistency)，输入是逐点**单位**特征的可加统计量。

    ``sum_f`` (K,C) = Σf_i，``sum_f2`` (K,) = Σ‖f_i‖²，``n`` (K,) = 点数：
    平均成对余弦 = (‖Σf‖² − Σ‖f‖²) / (n(n−1))；逐维平均方差（协方差迹 / C）=
    (Σ‖f‖² − ‖Σf‖²/n) / (n·C)；一致性 = ‖Σf‖/n（单位特征均值的模长，∈ [0,1]）。
    n < 2 时平均成对余弦为 NaN。
    """
    s = np.asarray(sum_f, np.float64)
    s2 = np.asarray(sum_f2, np.float64)
    n = np.asarray(n, np.float64)
    C = s.shape[1] if s.ndim == 2 else 0
    ss = np.einsum('ij,ij->i', s, s) if C else np.zeros(len(n))
    denom = n * np.maximum(n - 1.0, 0.0)
    pair = np.where(denom > 0, (ss - s2) / np.maximum(denom, 1e-12), np.nan)
    var = (s2 - ss / np.maximum(n, 1.0)) / np.maximum(n * C, 1.0)
    cons = np.sqrt(ss) / np.maximum(n, 1.0)
    return pair, var, cons


def relation_features(size, contain, subset_thr=0.8):
    """候选间关系特征（纯几何，不用 GT）。``contain[i, j]`` = i 的体素落在 j 内的比例。"""
    k = len(size)
    if k == 0:
        return {n: np.zeros(0, np.float64) for n in REL_NAMES}
    size = np.asarray(size, np.float64)
    contain = np.asarray(contain, np.float64)
    inter = contain * size[:, None]
    union = size[:, None] + size[None, :] - inter
    iou = inter / np.maximum(union, 1.0)
    eye = np.eye(k, dtype=bool)
    big = size[None, :] > size[:, None]
    fin = lambda v: np.where(np.isfinite(v), v, 0.0)   # 只有 1 个候选时 max 为空
    return {
        'rel_contain_in_max': fin(np.where(eye, -np.inf, contain).max(1)),
        'rel_contain_out_max': fin(np.where(eye, -np.inf, contain).T.max(1)),
        'rel_iou_max': fin(np.where(eye, -np.inf, iou).max(1)),
        'rel_n_overlap': ((inter > 0) & ~eye).sum(1).astype(np.float64),
        'rel_is_subset': ((contain >= subset_thr) & big).any(1).astype(np.float64),
    }


def point_features(maps, head, pixels, geom, spaces=('backbone', 'proto'), head_space='head'):
    """逐点单位特征：双线性采样 → 减帧均值 → L2（F1.1 / F4b.5 的统一口径）。"""
    from ..models.instance_head import sample_features
    out = {}
    for s in spaces:
        pf = sample_features(maps[s], pixels, geom)
        pf = pf - pf.mean(0, keepdim=True)
        out[s] = pf / pf.norm(dim=1, keepdim=True).clamp_min(1e-6)
    if head_space:
        dense = head(maps)
        pf = sample_features(dense, pixels, geom)
        pf = pf - pf.mean(0, keepdim=True)
        out[head_space] = pf / pf.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return out


def _entry(spaces):
    return dict(sum={s: None for s in spaces}, f2={s: 0.0 for s in spaces}, n=0.0,
                n_frag=0, b_sum=0.0, b_w=0.0, b_min=float('inf'), vox={})


def accumulate_frame(acc, labels, frag_obj, pf, fr, device='cpu'):
    """把一帧的可加统计量累加进 per-oid 累加器（分析路径，per-oid 循环只做字典合并）。

    ``labels`` (N,) 逐点历史物体 id（−1 = 未分配）；``frag_obj`` (G,) 每个碎片的物体 id；
    ``pf`` {space: (N,C) 逐点单位特征}。边界对比度用缓存帧内边的第 11/12 列
    （boundary_cos / boundary_cos_min，proto 层局部对比）：按碎片取均值/最小值，再按点数加权
    聚合到物体；没有帧内边的碎片不参与。
    """
    spaces = list(pf)
    lab = torch.as_tensor(np.asarray(labels, np.int64), device=device)
    sel = lab >= 0
    if bool(sel.any()):
        oids, inv = torch.unique(lab[sel], return_inverse=True)
        n_cnt = torch.bincount(inv, minlength=len(oids)).to(torch.float64).cpu().numpy()
        oid_list = [int(o) for o in oids.cpu().tolist()]
        for s in spaces:
            F = pf[s][sel]
            S = torch.zeros(len(oids), F.shape[1], device=device).index_add_(0, inv, F)
            S2 = torch.zeros(len(oids), device=device).index_add_(0, inv, (F * F).sum(1))
            S_np, S2_np = S.cpu().numpy().astype(np.float64), S2.cpu().numpy().astype(np.float64)
            for k, o in enumerate(oid_list):
                e = acc.setdefault(o, _entry(spaces))
                e['sum'][s] = S_np[k] if e['sum'][s] is None else e['sum'][s] + S_np[k]
                e['f2'][s] += float(S2_np[k])
        for k, o in enumerate(oid_list):
            acc.setdefault(o, _entry(spaces))['n'] += float(n_cnt[k])
    fo = np.asarray(frag_obj, np.int64)
    live = fo >= 0
    for o, c in zip(*np.unique(fo[live], return_counts=True)):
        acc.setdefault(int(o), _entry(spaces))['n_frag'] += int(c)
    edges = np.asarray(fr.edges, np.int64).reshape(-1, 2)
    G = int(len(fr.node_feat))
    n_g = np.bincount(np.asarray(fr.groups, np.int64), minlength=G).astype(np.float64)
    if len(edges) and live.any():
        ef = np.asarray(fr.edge_feat, np.float32)
        e0 = torch.as_tensor(edges[:, 0], device=device)
        e1 = torch.as_tensor(edges[:, 1], device=device)
        bc = torch.as_tensor(ef[:, BOUNDARY_COS_COL], device=device)
        bcm = torch.as_tensor(ef[:, BOUNDARY_COS_MIN_COL], device=device)
        deg = torch.bincount(torch.cat([e0, e1]), minlength=G).float()
        sb = torch.zeros(G, device=device).index_add_(0, e0, bc).index_add_(0, e1, bc)
        mn = torch.full((G,), float('inf'), device=device)
        mn = mn.scatter_reduce_(0, e0, bcm, 'amin').scatter_reduce_(0, e1, bcm, 'amin')
        has = (deg > 0).cpu().numpy() & live
        if has.any():
            obj_of = torch.as_tensor(np.clip(fo, 0, None), device=device)
            w = torch.as_tensor(n_g, dtype=torch.float32, device=device)
            K = int(fo.max()) + 1
            has_t = torch.as_tensor(has, device=device)
            osum = torch.zeros(K, device=device).index_add_(
                0, obj_of[has_t], (sb / deg.clamp_min(1))[has_t] * w[has_t])
            ow = torch.zeros(K, device=device).index_add_(0, obj_of[has_t], w[has_t])
            omin = torch.full((K,), float('inf'), device=device).scatter_reduce_(
                0, obj_of[has_t], mn[has_t], 'amin')
            osum, ow, omin = osum.cpu().numpy(), ow.cpu().numpy(), omin.cpu().numpy()
            for o in np.unique(fo[has]):
                e = acc.setdefault(int(o), _entry(spaces))
                e['b_sum'] += float(osum[o]); e['b_w'] += float(ow[o])
                e['b_min'] = min(e['b_min'], float(omin[o]))


def record_voxels(acc, mem, pos, spaces):
    """每个物体的当前体素数（定稿 ∪ 窗口成员）——用于「面积增长」。"""
    from ..streaming.fragment_fate import object_voxel_sets
    for oid, v in zip(sorted(mem.objects), object_voxel_sets(mem)):
        acc.setdefault(int(oid), _entry(spaces))['vox'][pos] = int(len(v))


# ------------------------------------------------------------------ 场景采集
@torch.no_grad()
def collect_scene(scene, cfg, model, std, enc, head, device, cache_dir, image_root,
                  spaces=('backbone', 'proto'), head_space='head', mem_overrides=None,
                  min_gt_voxels=20):
    """跑一遍最终配置并导出该场景的候选特征/掩码/GT/成员点（任务书 §2.1）。"""
    from ..models.instance_head import load_rgb
    from ..streaming.evaluate import voxel_majority_gt
    from ..streaming.fragment_fate import object_voxel_sets
    from ..streaming.holdout import load_scene_frames
    from ..streaming.phase3_eval import eval_objects, oracle_scores, pred_masks, voxel_gt
    from ..streaming.postprocess import rank_scores
    from ..streaming.unified_dedup import voxel_containment
    from ..streaming.unified_flow import make_unified_memory, object_state
    from ..streaming.window_data import final_object_labels

    wc = cfg['stream_v2']
    voxel = float(wc['voxel'])
    stuff = set(cfg['eval']['stuff_nyu40'])
    frames = load_scene_frames(cache_dir, scene, gt='2d')
    mem = make_unified_memory(cfg, model, std, device, **(mem_overrides or {}))
    all_spaces = tuple(spaces) + ((head_space,) if head_space else ())
    acc, pt_raw, merges = {}, [], []
    t0 = time.perf_counter()
    for pos, fr in enumerate(frames):
        labels = np.asarray(mem.update(fr), np.int64)
        merges.extend(mem.last_unified['merged'])
        rgb = load_rgb(image_root, scene, fr.frame_id)
        maps, geom = enc.encode(rgb)
        with np.load(Path(cache_dir) / f'{scene}_{fr.frame_id}.npz') as z:
            pixels = np.asarray(z['pixels'])          # 与缓存完全同一批采样点像素
        pf = point_features(maps, head, pixels, geom, spaces, head_space)
        _, first_idx = np.unique(np.asarray(fr.groups, np.int64), return_index=True)
        frag_obj = labels[first_idx]
        accumulate_frame(acc, labels, frag_obj, pf, fr, device=device)
        record_voxels(acc, mem, pos, all_spaces)
        pt_raw.append(labels.astype(np.int16))
    obj_vox = object_voxel_sets(mem)
    attrs = mem.object_attrs()
    ids = [int(o) for o in attrs['ids']]
    final_of, iou_of = final_object_labels(mem, frames, voxel, stuff, min_gt_voxels, merges, device)
    cells, gts = voxel_gt(frames, stuff, voxel=voxel, min_gt_voxels=min_gt_voxels)
    _, inst_vox, _ = voxel_majority_gt(frames, voxel)
    masks = pred_masks(obj_vox, cells, device)
    _, o_stats, _ = object_state(mem)
    lo, hi = o_stats.lo.cpu().numpy(), o_stats.hi.cpu().numpy()
    first, last = o_stats.first.cpu().numpy(), o_stats.last.cpu().numpy()

    rev = {}
    for oid, f in final_of.items():
        rev.setdefault(int(f), []).append(int(oid))
    names, slices = feature_layout(spaces, head_space)
    K = len(ids)
    feat = np.full((K, len(names)), np.nan, np.float64)
    n_pts = np.zeros(K); n_frag = np.zeros(K); src = np.zeros(K, np.int8)
    n_vox_arr = np.asarray([len(v) for v in obj_vox], np.int64)
    for k, o in enumerate(ids):
        e = _entry(all_spaces)
        for m in rev.get(o, [o]):
            a = acc.get(m)
            if a is None:
                continue
            for s in all_spaces:
                if a['sum'][s] is not None:
                    e['sum'][s] = a['sum'][s] if e['sum'][s] is None else e['sum'][s] + a['sum'][s]
                e['f2'][s] += a['f2'][s]
            e['n'] += a['n']; e['n_frag'] += a['n_frag']
            e['b_sum'] += a['b_sum']; e['b_w'] += a['b_w']
            e['b_min'] = min(e['b_min'], a['b_min'])
            for p, c in a['vox'].items():
                e['vox'][p] = e['vox'].get(p, 0) + c
        n = e['n']
        for s in all_spaces:
            if e['sum'][s] is None:
                continue
            pair, var, cons = appearance_from_sums(e['sum'][s][None], np.asarray([e['f2'][s]]),
                                                   np.asarray([n]))
            feat[k, names.index(f'{s}_pair_cos')] = pair[0]
            feat[k, names.index(f'{s}_var')] = var[0]
            feat[k, names.index(f'{s}_consistency')] = cons[0]
        ext = np.clip(hi[k] - lo[k], 0, None)
        bbox_vol = float(np.prod(np.maximum(ext, 1e-6)))
        bbox_vox = float(np.prod(np.floor(ext / voxel) + 1.0))
        n_obs = int(attrs['n_obs'][k])
        age = float(max(last[k] - first[k], 0.0))
        pos_counts = sorted((p, c) for p, c in e['vox'].items() if c > 0)
        growth = (pos_counts[-1][1] / pos_counts[0][1]) if len(pos_counts) >= 2 else 1.0
        feat[k, names.index('n_pts')] = n
        feat[k, names.index('n_vox')] = n_vox_arr[k]
        feat[k, names.index('n_frag')] = e['n_frag']
        feat[k, names.index('log_bbox_vol')] = np.log1p(bbox_vol)
        feat[k, names.index('compactness')] = n_vox_arr[k] / max(bbox_vox, 1.0)
        feat[k, names.index('conf_mean')] = float(attrs['conf'][k])
        feat[k, names.index('boundary_cos_mean')] = e['b_sum'] / e['b_w'] if e['b_w'] > 0 else np.nan
        feat[k, names.index('boundary_cos_min')] = e['b_min'] if np.isfinite(e['b_min']) else np.nan
        feat[k, names.index('n_obs')] = n_obs
        feat[k, names.index('age')] = age
        feat[k, names.index('obs_regularity')] = n_obs / (age + 1.0)
        feat[k, names.index('vox_growth')] = growth
        feat[k, names.index('quality')] = float(attrs['quality'][k])
        n_pts[k] = n; n_frag[k] = e['n_frag']
        src[k] = 0 if (e['n_frag'] == 1 and n_obs == 1) else 1
    contain = voxel_containment(obj_vox, device) if len(obj_vox) else np.zeros((0, 0), np.float32)
    for nm, v in relation_features(n_vox_arr, contain).items():
        feat[:, names.index(nm)] = v

    iou = np.asarray([float(iou_of.get(o, 0.0)) for o in ids], np.float32)
    quality = np.asarray(attrs['quality'], np.float32)
    nobs_score = rank_scores('nobs', masks=masks, conf=np.asarray(attrs['conf'], np.float32),
                             n_obs=np.asarray(attrs['n_obs'], np.float32))
    oracle = oracle_scores(masks, gts, device) if gts is not None else np.zeros(len(masks), np.float32)
    met = dict(quality=float(eval_objects(masks, quality, gts, device=device)['ap50']),
               oracle=float(eval_objects(masks, oracle, gts, device=device)['ap50']),
               nobs=float(eval_objects(masks, nobs_score, gts, device=device)['ap50']),
               masks=int(len(masks)))
    lut = np.full((max(final_of.keys(), default=-1) + 2,), -1, np.int16)
    for oid, f in final_of.items():
        lut[int(oid)] = int(f)
    pt_final = (np.stack([np.where(l >= 0, lut[np.clip(l, 0, len(lut) - 1)], -1)
                          for l in pt_raw]) if pt_raw else np.zeros((0, 0), np.int16))
    return dict(scene=scene, names=names, feat=feat.astype(np.float32), iou=iou, quality=quality,
                n_obs=np.asarray(attrs['n_obs'], np.int32), source=src,
                n_pts=n_pts.astype(np.int32), n_vox=n_vox_arr.astype(np.int32),
                n_frag=n_frag.astype(np.int32), masks_packed=np.packbits(masks, axis=1),
                ncells=int(len(cells)), gt_vox=np.asarray(inst_vox, np.int32),
                keep=np.asarray(sorted(gts.ids), np.int32) if gts is not None else np.zeros(0, np.int32),
                pt_final=pt_final, frame_ids=np.asarray([f.frame_id for f in frames], np.int32),
                voxel=voxel, metrics=met, seconds=time.perf_counter() - t0)


def save_scene(path, d):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, scene=np.asarray(d['scene']), names_json=np.asarray(json.dumps(d['names'])),
        feat=d['feat'], iou=d['iou'], quality=d['quality'], n_obs=d['n_obs'], source=d['source'],
        n_pts=d['n_pts'], n_vox=d['n_vox'], n_frag=d['n_frag'], masks_packed=d['masks_packed'],
        ncells=np.int64(d['ncells']), gt_vox=d['gt_vox'], keep=d['keep'], pt_final=d['pt_final'],
        frame_ids=d['frame_ids'], voxel=np.float32(d['voxel']),
        metrics_json=np.asarray(json.dumps(d['metrics'])), seconds=np.float64(d['seconds']))


def load_scene(path):
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    d['scene'] = str(d['scene'])
    d['names'] = json.loads(str(d['names_json']))
    d['metrics'] = json.loads(str(d['metrics_json']))
    d['masks'] = np.unpackbits(d['masks_packed'], axis=1, count=int(d['ncells'])).astype(bool)
    return d


# ------------------------------------------------------------------ 统计
def _spearman(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or x[ok].std() == 0 or y[ok].std() == 0:
        return float('nan')
    return float(np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1])


def spearman_table(X, iou, scene, names):
    rows = {}
    scene = np.asarray(scene)
    iou = np.asarray(iou, np.float64)
    for i, n in enumerate(names):
        x = np.asarray(X[:, i], np.float64)
        per = [_spearman(x[scene == s], iou[scene == s]) for s in np.unique(scene)]
        per = [v for v in per if np.isfinite(v)]
        rows[n] = dict(pooled=_spearman(x, iou),
                       scene_mean=float(np.mean(per)) if per else float('nan'),
                       nonzero=float(np.mean(np.isfinite(x) & (x != 0))))
    return rows


def loocv_logistic_scores(X, y, scene, C=1.0):
    """按场景留一的逻辑回归分数（拟合 y =「IoU ≥ 0.5」）；NaN 特征按训练折列均值填补。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = np.asarray(X, np.float64)
    y = np.asarray(y, int)
    scene = np.asarray(scene)
    scores = np.full(len(y), np.nan)
    for s in np.unique(scene):
        te, tr = scene == s, scene != s
        if not te.any() or y[tr].sum() == 0 or y[tr].sum() == tr.sum():
            continue
        mu = np.nanmean(X[tr], 0)
        Xtr = np.where(np.isfinite(X[tr]), X[tr], mu)
        Xte = np.where(np.isfinite(X[te]), X[te], mu)
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr), y[tr])
        scores[te] = lr.predict_proba(sc.transform(Xte))[:, 1]
    return scores


def gbm_loocv_scores(X, y, scene, seed=0, max_iter=150, learning_rate=0.08, max_depth=3,
                      min_samples_leaf=20):
    """非线性稳健性检查：与 ``loocv_logistic_scores`` 同一留一协议，换成直方图梯度提升。

    用途：在按线性估计器下「信息不够」的结论之前，排除「线性估计器太弱」这一替代解释。
    NaN 由 HistGBM 原生处理，不做填补。
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = np.asarray(X, np.float64)
    y = np.asarray(y, int)
    scene = np.asarray(scene)
    scores = np.full(len(y), np.nan)
    for s in np.unique(scene):
        te, tr = scene == s, scene != s
        if not te.any() or y[tr].sum() == 0 or y[tr].sum() == tr.sum():
            continue
        m = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=learning_rate,
                                           max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                           random_state=seed)
        m.fit(X[tr], y[tr])
        scores[te] = m.predict_proba(X[te])[:, 1]
    return scores


def aggregate_ap50(scenes_data, scores, dedup=None, device='cpu'):
    """逐场景 AP50 的场景均值（nan-safe）；``dedup`` = 先按该分数做包含度抑制再评。"""
    from ..evaluation.instance import GTSet
    from ..streaming.phase3_eval import eval_objects
    from ..streaming.unified_dedup import dedup_masks
    per, off = [], 0
    for d in scenes_data:
        k = len(d['iou'])
        s = np.nan_to_num(np.asarray(scores[off:off + k], np.float64), nan=0.0).astype(np.float32)
        off += k
        masks = d['masks']
        if dedup is not None:
            keep = dedup_masks(masks, s, float(dedup), device=device)
            masks, s = masks[keep], s[keep]
        if len(d['keep']) == 0:
            per.append(float('nan'))
            continue
        gts = GTSet.from_labels(d['gt_vox'], d['keep'])
        per.append(float(eval_objects(masks, s, gts, device=device)['ap50']))
    return float(np.nanmean(per)), per


def decide(X, head, perfect):
    """任务书 §2 的判定（用与尺度无关的缺口恢复比例 r）。"""
    r = float((X - head) / max(perfect - head, 1e-9))
    if r >= R_MOSTLY_TRAINING:
        v = 'A 档：缺口主要是训练 → 走 P2 全阶梯'
    elif r >= R_PARTIAL:
        v = 'B 档：部分可补 → 只做 P2 的第 1、2 条（头 vs 自己的输入、序列长度外推）'
    else:
        v = 'C 档：缺口是信息 → 不再训排序头，走 P3 与 P4'
    return r, v


# ------------------------------------------------------------------ 报告
def eval_labels(data, device='cpu'):
    """每个候选与最佳 GT 的真实 IoU，口径 = 评估的 oracle（``phase3_eval.oracle_scores``）。

    任务书要求「匹配口径与现有评估完全一致」，因此标签用本函数在报告端从保存的掩码 + GT 重算；
    训练标签口径 ``window_data.final_object_labels`` 在少数候选上与评估口径不同（本轮 45/2793
    个在 0.5 处翻转，见 summary 的 label_flips）。
    """
    from ..evaluation.instance import GTSet
    from ..streaming.phase3_eval import oracle_scores
    out = []
    for d in data:
        if len(d['keep']) == 0:
            out.append(np.full(len(d['iou']), np.nan, np.float32))
            continue
        out.append(np.asarray(oracle_scores(d['masks'], GTSet.from_labels(d['gt_vox'], d['keep']),
                                            device), np.float32))
    return np.concatenate(out)


def write_report(out, cfg, device='cpu', spaces=('backbone', 'proto'), head_space='head', log=print):
    from ..streaming.unified_train import _auc
    out = Path(out)
    files = sorted((out / 'cache').glob('*.npz'))
    if not files:
        raise SystemExit(f'no cached scenes under {out / "cache"}; run --stage collect first')
    data = [load_scene(f) for f in files]
    names, slices = feature_layout(spaces, head_space)
    X = np.concatenate([d['feat'] for d in data])
    iou = eval_labels(data, device)          # 评估口径的真实 IoU（任务书要求与现有评估一致）
    saved_iou = np.concatenate([d['iou'] for d in data])
    label_flips = int(((iou >= 0.5) != (saved_iou >= 0.5)).sum())
    scene = np.concatenate([np.full(len(d['iou']), d['scene']) for d in data])
    y = (iou >= 0.5).astype(int)
    dedup_thr = float(cfg['phase5']['rank_probe'].get('dedup', 0.7))

    # ---- 0. 口径校验：与已发布的留出扫描逐场景对齐（同一口径的独立重跑）----
    p3 = resolve_path(cfg['phase3']['out_root'])
    ref = json.loads((p3 / 'sweep_A_s0' / 'A_s0_0.75_0.60_1.01.json').read_text())
    rec_q = {r['scene']: r['ap50'] for r in ref['scenes']['quality']}
    rec_o = {r['scene']: r['ap50'] for r in ref['scenes']['oracle']}
    dq = [abs(d['metrics']['quality'] - rec_q[d['scene']]) for d in data if d['scene'] in rec_q]
    do = [abs(d['metrics']['oracle'] - rec_o[d['scene']]) for d in data if d['scene'] in rec_o]
    my_q = float(np.nanmean([d['metrics']['quality'] for d in data]))
    my_o = float(np.nanmean([d['metrics']['oracle'] for d in data]))
    my_nobs = float(np.nanmean([d['metrics']['nobs'] for d in data]))
    my_masks = float(np.mean([d['metrics']['masks'] for d in data]))
    fin = [json.loads(f.read_text()) for f in sorted((p3 / 'final').glob('scene*.json'))]
    val16 = {k: float(np.nanmean([r['rows']['nodedup'][k]['ap50'] for r in fin]))
             for k in ('quality', 'oracle')}
    caliber = dict(scenes=len(data), recorded_scenes=len(rec_q),
                   quality=my_q, quality_recorded=float(ref['variants']['quality']['ap50']),
                   oracle=my_o, oracle_recorded=float(ref['variants']['oracle']['ap50']),
                   masks=my_masks, masks_recorded=float(ref['variants']['quality']['masks']),
                   per_scene_max_abs_diff=float(np.nanmax(dq + do)) if (dq or do) else None,
                   val16_recorded_quality=val16['quality'], val16_recorded_oracle=val16['oracle'])
    caliber['ok'] = bool(len(data) == len(rec_q) and abs(my_q - caliber['quality_recorded']) <= 0.005
                         and abs(my_o - caliber['oracle_recorded']) <= 0.005
                         and not ((caliber['per_scene_max_abs_diff'] or 0) > 0.01))

    # ---- 1. 每族留一逻辑回归 AP50 ----
    results = {}
    scores_quality = np.concatenate([d['quality'] for d in data])
    ap_q, per_q = aggregate_ap50(data, scores_quality)
    ap_qd, _ = aggregate_ap50(data, scores_quality, dedup=dedup_thr)
    results['quality（现有头）'] = dict(ap50=ap_q, ap50_dedup=ap_qd, auc=_auc(scores_quality, y),
                                        per_scene=per_q)
    for f in ('self', 'appearance', 'appearance_head', 'rel', 'time'):
        cols = np.arange(slices[f].start, slices[f].stop)
        scores = loocv_logistic_scores(X[:, cols], y, scene)
        ap, per = aggregate_ap50(data, scores)
        ap_d, _ = aggregate_ap50(data, scores, dedup=dedup_thr)
        results[f] = dict(ap50=ap, ap50_dedup=ap_d, auc=_auc(scores, y), per_scene=per)
    for tag, kw in (('full（a+b+c+d）', dict()), ('full+head', dict(with_head=True)),
                    ('full+head+quality', dict(with_head=True, with_control=True))):
        cols = full_columns(slices, **kw)
        scores = loocv_logistic_scores(X[:, cols], y, scene)
        ap, per = aggregate_ap50(data, scores)
        ap_d, _ = aggregate_ap50(data, scores, dedup=dedup_thr)
        results[tag] = dict(ap50=ap, ap50_dedup=ap_d, auc=_auc(scores, y), per_scene=per)
        g = gbm_loocv_scores(X[:, cols], y, scene)     # 稳健性：非线性估计器，同一留一协议
        g_ap, _ = aggregate_ap50(data, g)
        g_apd, _ = aggregate_ap50(data, g, dedup=dedup_thr)
        results[tag + '·GBM'] = dict(ap50=g_ap, ap50_dedup=g_apd, auc=_auc(g, y), per_scene=None)
    ap_o, per_o = aggregate_ap50(data, iou)

    # ---- 2. Spearman ----
    sp = spearman_table(X, iou, scene, names)

    # ---- 3. 判定 ----
    X_full = results['full（a+b+c+d）']['ap50']
    r, verdict = decide(X_full, ap_q, ap_o)
    src = np.concatenate([d['source'] for d in data])
    secs = [float(d['seconds']) for d in data]

    # ---- 4. 写文件 ----
    per_scene = [dict(scene=d['scene'], candidates=int(len(d['iou'])),
                      quality=d['metrics']['quality'], oracle=d['metrics']['oracle'],
                      nobs=d['metrics']['nobs'],
                      full_ap50=results['full（a+b+c+d）']['per_scene'][i],
                      seconds=float(d['seconds']))
                 for i, d in enumerate(data)]
    summary = dict(caliber=caliber, label_flips=label_flips, families=results,
                   oracle_ap50=ap_o, nobs_ap50=my_nobs,
                   full_X=X_full, gap_recovery=r, verdict=verdict,
                   r_thresholds=dict(mostly_training=R_MOSTLY_TRAINING, partial=R_PARTIAL),
                   dedup_thr=dedup_thr, scenes=len(data), candidates=int(len(iou)),
                   pos_rate=float(y.mean()), merged_share=float(src.mean()),
                   spearman=sp, seconds=dict(total=float(np.sum(secs)),
                                             per_scene=float(np.mean(secs))),
                   names=names)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float))
    (out / 'per_scene.json').write_text(json.dumps(per_scene, indent=1, ensure_ascii=False))

    L = ['# 阶段 5 · P1：掩码质量打分的排序信息上限（不训练）', '',
         f'60 个留出场景、第 3 阶段最终配置（A 轮 seed0 @ 0.75/0.60/1.01）；'
         f'{len(data)} 个场景、{len(iou)} 个候选（合并 mask 占比 {src.mean():.3f}），'
         f'正类率（真实 IoU ≥ 0.5）{y.mean():.3f}。口径：体素代理 GT（5 cm、跨帧多数实例、'
         'thing ≥20 体素）、候选 ≥20 体素、贪心匹配（与第 3 阶段留出扫描完全一致）。', '',
         '## 0. 口径校验（独立重跑 vs 已发布的留出扫描）', '',
         '| 排序 | 重跑 AP50 | 已记录 AP50 | 逐场景最大绝对差 |', '|---|---|---|---|',
         f'| quality（现有头） | {my_q:.4f} | {caliber["quality_recorded"]:.4f} | '
         f'{np.nanmax(dq) if dq else float("nan"):.4f} |',
         f'| oracle（完美排序，上界） | {my_o:.4f} | {caliber["oracle_recorded"]:.4f} | '
         f'{np.nanmax(do) if do else float("nan"):.4f} |',
         f'| nobs（朴素观测帧数排序） | {my_nobs:.4f} | —（留出口径未发布） | — |', '',
         f'每场景 mask 数 {my_masks:.1f}（已记录 {caliber["masks_recorded"]:.1f}）。'
         f'**口径校验：{"通过" if caliber["ok"] else "不通过——先修口径，结论无效"}**。', '',
         f'标签口径：候选「真实 IoU」= 评估的 oracle（oracle_scores）在报告端重算；与训练标签'
         f'口径（final_object_labels）相比，{label_flips} 个候选（{label_flips / len(iou):.2%}）'
         f'在 0.5 处翻转。本报告所有标签与上界均用前者。', '',
         f'16 个验证场景网格协议（读已记录结果，未重跑）：quality {val16["quality"]:.3f}、'
         f'oracle {val16["oracle"]:.3f}（任务书的 0.422 / 0.635 即此口径）。', '',
         '## 1. 每族留一逻辑回归（「IoU ≥ 0.5」）的 AP50 天花板', '',
         '| 特征族 | 列数 | AP50（留一拟合分数排序） | +去重 0.7 | 分数 AUC |', '|---|---|---|---|---|']
    fam_n = {f: len(family_names(spaces, head_space)[f]) for f in
             ('self', 'appearance', 'appearance_head', 'rel', 'time')}
    fam_n['full（a+b+c+d）'] = len(full_columns(slices))
    fam_n['full+head'] = len(full_columns(slices, with_head=True))
    fam_n['full+head+quality'] = len(full_columns(slices, with_head=True, with_control=True))
    fam_n['quality（现有头）'] = 1
    for t in ('full（a+b+c+d）', 'full+head', 'full+head+quality'):
        fam_n[t + '·GBM'] = fam_n[t]
    for tag in ('quality（现有头）', 'self', 'appearance', 'appearance_head', 'rel', 'time',
                'full（a+b+c+d）', 'full+head', 'full+head+quality',
                'full（a+b+c+d）·GBM', 'full+head·GBM', 'full+head+quality·GBM'):
        m = results[tag]
        L.append(f'| {tag} | {fam_n[tag]} | **{m["ap50"]:.4f}** | {m["ap50_dedup"]:.4f} | '
                 f'{m["auc"]:.4f} |')
    L += [f'| oracle（完美排序，上界） | 1 | **{ap_o:.4f}** | — | 1.0 |', '',
          '## 2. 逐列 Spearman（与真实 IoU）', '',
          '| 特征 | pooled | 场景均值 | 非零率 |', '|---|---|---|---|']
    for n in names:
        L.append(f'| {n} | {sp[n]["pooled"]:.3f} | {sp[n]["scene_mean"]:.3f} | {sp[n]["nonzero"]:.2f} |')
    L += ['', '## 3. 判定（任务书 §2）', '',
          f'- 现有头（留出口径）**{ap_q:.4f}**；完美排序上界 **{ap_o:.4f}**；'
          f'全特征集拟合 AP50 **X = {X_full:.4f}**。',
          f'- 缺口恢复比例 r = (X − head) / (perfect − head) = **{r:.3f}**'
          f'（任务书阈值换算：≥{R_MOSTLY_TRAINING:.3f} 走 P2；≥{R_PARTIAL:.3f} 只做 P2 第 1、2 条；'
          f'低于 {R_PARTIAL:.3f} 走 P3/P4）。',
          f'- 稳健性（非线性估计器 GBM，同一留一协议）：full '
          f'{results["full（a+b+c+d）·GBM"]["ap50"]:.4f}、full+head '
          f'{results["full+head·GBM"]["ap50"]:.4f}、full+head+quality '
          f'{results["full+head+quality·GBM"]["ap50"]:.4f}；即使把现有头输出也当特征、并用 GBM，'
          f'缺口恢复比例 r = '
          f'{(results["full+head+quality·GBM"]["ap50"] - ap_q) / max(ap_o - ap_q, 1e-9):.3f}。',
          f'- **{verdict}**', '',
          f'计时：采集 {np.sum(secs) / 60:.1f} 分钟（{np.mean(secs):.1f} s/场景，含 FastSAM 前向、'
          f'逐点特征与评估）。']
    (out / 'summary.md').write_text('\n'.join(L) + '\n')
    log('\n'.join(L))
    return summary
