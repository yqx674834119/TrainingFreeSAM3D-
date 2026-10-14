"""Phase 2, task 0: validate the voxel-proxy AP50 as a stand-in for the mesh AP50.

The 60 holdout scenes have no reconstructed mesh, so stage-1 (and stage-2) threshold selection
runs on the *voxel proxy* (``streaming/evaluate.voxel_scene_eval``: 5 cm cells, GT = the
cross-frame majority vote of the per-frame GT, thing and >= 20 cells). This tool checks the
assumption behind that choice: on the 16 val scenes - the only scenes that have both a mesh and
the voxel GT - do the proxy and the mesh protocol *rank* a set of already-fixed plans the same way?
Nothing here selects or tunes anything; every plan was fixed and reported by stage 0c / stage 1.

    source tools/env.sh
    for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python tools/window_phase2_proxy.py --stage run --shard $i/4 & done
    python tools/window_phase2_proxy.py --stage report

Per (plan, scene) the memory runs once; the proxy metrics come from its per-object voxel sets, the
mesh metrics from rendering the same objects onto ESAM's mesh. Two mesh conventions are reported:
``conf|none|smooth`` (the closest mesh analogue of the proxy - the memory's own confidence ranking,
no post-processing) and ``nobs|nms|smooth`` (the stage-1 headline row). GT enters only the metrics.

Writes work_dirs/stream_v2/phase2/proxy/<tag>/<scene>.json and, on ``--stage report``,
work_dirs/stream_v2/phase2/proxy/{summary.md,summary.json}.
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
from freesam3d.streaming.evaluate import (frame_input_from_proposal_cache, frames_from_merge_cache,  # noqa: E402
                                          mesh_gtset, scene_metrics, segment_smooth, select_frame_proposals,
                                          voxel_scene_eval)
from freesam3d.streaming.fragment_fate import object_voxel_sets  # noqa: E402
from freesam3d.streaming.holdout import Variant, load_scene_frames, make_memory  # noqa: E402
from freesam3d.streaming.postprocess import greedy_nms, rank_scores  # noqa: E402
from freesam3d.streaming.window_scores import load_group_merge_net  # noqa: E402

P1 = 'p1_default'
# The plans the brief names (P1, 0c's p2a/p2b_calib, two_step_p2a_0.2) plus further configurations
# that the stage-1 holdout sweep already fixed and reported - extra points make the rank correlation
# meaningful without selecting anything here. 'p2a_calib' is stage 0c's name for "linkage + rule at
# the checkpoint threshold 0.65"; 'calib_p2a_*' is stage 1's per-scale binarisation mode.
FIXED_VARIANTS = [
    ('p2a_calib', Variant('linkage', 'p2a', 0.65, 0.65)),
    ('p2b_calib', Variant('linkage', 'p2b', 0.65, 0.65)),
    ('two_step_p2a_0.2', Variant('two_step', 'p2a', 0.65, 0.2)),
    ('two_step_p2a_0.3', Variant('two_step', 'p2a', 0.65, 0.3)),
    ('two_step_p2a_0.4', Variant('two_step', 'p2a', 0.65, 0.4)),
    ('two_step_p2a_0.5', Variant('two_step', 'p2a', 0.65, 0.5)),
    ('two_step_p2b_0.2', Variant('two_step', 'p2b', 0.65, 0.2)),
    ('calib_p2a_0.3', Variant('calib', 'p2a', 0.65, 0.3)),
    ('calib_p2a_0.4', Variant('calib', 'p2a', 0.65, 0.4)),
    ('linkage_p2a_0.2', Variant('linkage', 'p2a', 0.65, 0.2)),
    ('linkage_p2a_0.4', Variant('linkage', 'p2a', 0.65, 0.4)),
]
MESH_KEYS = ('ap50', 'ap25', 'iou', 'r50', 'masks', 'dup_rate', 'overmerge', 'frags')


def load_mesh(ws, scene):
    mesh6 = np.fromfile(ws / 'data/scannet200/points' / f'{scene}.bin', np.float32).reshape(-1, 6)
    mesh_gt = np.fromfile(ws / 'data/scannet200/instance_mask' / f'{scene}.bin', np.int64)
    mesh_sem = np.fromfile(ws / 'data/scannet200/semantic_mask' / f'{scene}.bin', np.int64)
    sp = np.fromfile(ws / 'data/scannet200/super_points' / f'{scene}.bin', np.int64)
    return mesh6[:, :3], mesh_gt, mesh_sem, sp


def mesh_combos(masks_raw, masks_smooth, scores, n_obs, gts, dev, min_points=100):
    """Mesh-protocol AP for the two conventions this check reports, both segment-smoothed."""
    out = {}
    for set_name, mk, vc in (('smooth', masks_smooth, masks_smooth.sum(1)), ('raw', masks_raw, masks_raw.sum(1))):
        base = np.where(vc >= min_points)[0]
        masks, conf, nobs = mk[base], scores[base], n_obs[base]
        if len(masks) == 0:
            continue
        for sm, s in (('conf', conf), ('nobs', rank_scores('nobs', masks=masks, conf=conf, n_obs=nobs))):
            for cl in ('none', 'nms'):
                keep = np.argsort(-s, kind='stable') if cl == 'none' else greedy_nms(masks, s, 0.5)
                met = scene_metrics(masks[keep], s[keep], gts, device=dev)
                st = association_stats(masks[keep], gts, device=dev)
                met.update(dup_rate=st['dup_rate'], overmerge=st['overmerge_rate'], frags=st['frags_per_gt'])
                out[f'{sm}|{cl}|{set_name}'] = met
    return out


def p1_memory(cfg, dev):
    """The default per-frame pipeline (stage 0c's P1), assembled exactly as tools/window_real_eval.py."""
    from freesam3d.merging.proposal_data import ScalarStandardizer
    from freesam3d.merging.scorer import MaskScorer
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
    return mem, scorer, std


def run_p1_scene(prop_files, mesh, sp, cfg, dev, stuff, mesh_gt, mesh_sem, frames_gt3d):
    from freesam3d.merging.proposal_data import load_proposals
    mem, scorer, std = p1_memory(cfg, dev)
    for f in prop_files:
        d = load_proposals(f, with_masks=False)
        d['masks'] = np.load(f)['masks']
        X = std.apply(torch.as_tensor(d['scalars'], device=dev))
        Fe = torch.as_tensor(d['feat'], device=dev)
        with torch.no_grad():
            s = scorer.score(X, Fe).cpu().numpy()
        fr = frame_input_from_proposal_cache(d, scores=s, top_k=None)
        fr = select_frame_proposals(fr, 0.5, 0.8, 20, dev)
        fr.frame_id = int(f.stem.rsplit('_', 1)[1])
        mem.update(fr)
    masks_raw, scores = mem.render(mesh)
    obj_vox = [mem.voxel[mem.owner == k] for k in range(mem.num_queries)]
    n_obs = mem.n_obs.cpu().numpy()
    return masks_raw, scores, n_obs, obj_vox


def run_window_scene(frames, variant, model, std, cfg, dev, mesh):
    mem = make_memory(variant, model, std, cfg, dev)
    for fr in frames:
        mem.update(fr)
    obj_vox = object_voxel_sets(mem)
    attrs = mem.object_attrs()
    masks_raw, scores = mem.render(mesh)
    return masks_raw, scores, attrs['n_obs'], obj_vox


def evaluate_scene_pair(masks_raw, scores, n_obs, obj_vox, sp, frames_gt3d, mesh_gt, mesh_sem,
                        stuff_mesh, stuff_proxy, cfg, dev):
    masks_smooth = segment_smooth(masks_raw, sp, device=dev)
    proxy = voxel_scene_eval(obj_vox, scores, frames_gt3d, stuff_proxy,
                             voxel=float(cfg['stream_v2']['voxel']), min_gt_voxels=20, min_pred_voxels=20,
                             device=dev)
    gts = mesh_gtset(mesh_gt, mesh_sem, stuff_mesh, cfg['eval']['min_gt_points'])
    mesh = mesh_combos(masks_raw, masks_smooth, scores, n_obs, gts, dev, cfg['eval']['min_gt_points'])
    return dict(proxy={k: v for k, v in proxy.items()}, mesh=mesh)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--stage', default='all', choices=['run', 'report', 'all'])
    p.add_argument('--shard', default='0/1')
    p.add_argument('--scenes', nargs='*', default=None)
    p.add_argument('--variants', nargs='*', default=None, help=f'default: {P1} + {len(FIXED_VARIANTS)} fixed variants')
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', default=None, help='default: <stream_v2.out_root>/phase2/proxy')
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    dev = a.device if torch.cuda.is_available() else 'cpu'
    if str(dev).startswith('cuda'):
        print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    out = Path(a.out) if a.out else resolve_path(cfg['stream_v2']['out_root']) / 'phase2' / 'proxy'
    ws = resolve_path('work_dirs/esam_eval')
    stuff_mesh = stuff_ids('/home/yqx/SegmentAnything3D/ESAM/data/scannet200/meta_data/scannetv2-labels.combined.tsv')
    stuff_proxy = set(cfg['eval']['stuff_raw'])          # gt='3d' semantic ids are ScanNet raw ids
    variants = dict(FIXED_VARIANTS)
    tags = a.variants if a.variants else [P1] + list(variants)

    frag_files = sorted((resolve_path(cfg['cache_root']) / 'val').glob('*.npz'))
    by_scene = {}
    for f in frag_files:
        by_scene.setdefault(f.stem.rsplit('_', 1)[0], []).append(f)
    prop_files = sorted((resolve_path(cfg['cache_root']) / 'proposals' / 'val').glob('*.npz'))
    prop_by_scene = {}
    for f in prop_files:
        prop_by_scene.setdefault(f.stem.rsplit('_', 1)[0], []).append(f)
    scenes = a.scenes or sorted(by_scene)

    if a.stage in ('run', 'all'):
        i, n = map(int, a.shard.split('/'))
        mine = [s for k, s in enumerate(scenes) if k % n == i]
        model = std = None
        if any(t != P1 for t in tags):
            model, std, _ = load_group_merge_net(resolve_path('work_dirs/merge_net/full/model.pt'), dev)
        t0 = time.time()
        for si, scene in enumerate(mine):
            mesh, mesh_gt, mesh_sem, sp = load_mesh(ws, scene)
            frames_gt3d = frames_from_merge_cache(
                sorted(by_scene[scene], key=lambda q: int(q.stem.rsplit('_', 1)[1])), gt='3d')
            for tag in tags:
                dst = out / tag / f'{scene}.json'
                if dst.exists() and not a.overwrite:
                    continue
                t1 = time.perf_counter()
                if tag == P1:
                    got = run_p1_scene(sorted(prop_by_scene[scene], key=lambda q: int(q.stem.rsplit('_', 1)[1])),
                                       mesh, sp, cfg, dev, stuff_mesh, mesh_gt, mesh_sem, frames_gt3d)
                else:
                    frames = load_scene_frames(resolve_path(cfg['cache_root']) / 'val', scene, gt='3d')
                    got = run_window_scene(frames, variants[tag], model, std, cfg, dev, mesh)
                masks_raw, scores, n_obs, obj_vox = got
                met = evaluate_scene_pair(masks_raw, scores, n_obs, obj_vox, sp, frames_gt3d, mesh_gt,
                                          mesh_sem, stuff_mesh, stuff_proxy, cfg, dev)
                met.update(scene=scene, tag=tag, frames=len(frames_gt3d), objects=int(len(obj_vox)),
                           seconds=time.perf_counter() - t1)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(json.dumps(met, indent=1, default=float))
                print(f'[{a.shard}] {si + 1}/{len(mine)} {scene} {tag}: '
                      f"proxy {met['proxy']['ap50']:.4f} mesh(conf|none) "
                      f"{met['mesh'].get('conf|none|smooth', {}).get('ap50', float('nan')):.4f} "
                      f"({met['seconds']:.0f}s, total {(time.time() - t0) / 60:.1f} min)", flush=True)
    if a.stage == 'run':
        return
    write_report(out, tags, scenes, cfg)


def write_report(out, tags, scenes, cfg):
    from scipy.stats import spearmanr
    rows = {}
    for tag in tags:
        per = [json.loads(f.read_text()) for f in sorted((out / tag).glob('*.json'))] if (out / tag).exists() else []
        if per:
            rows[tag] = per
    if not rows:
        raise SystemExit(f'no results under {out}; run --stage run first')
    n = max(len(v) for v in rows.values())

    def mean(tag, path, key=None):
        vals = []
        for r in rows[tag]:
            v = r
            for k in path:
                v = v.get(k, {}) if isinstance(v, dict) else {}
            if isinstance(v, (int, float)) and v == v:
                vals.append(float(v))
        return float(np.mean(vals)) if vals else float('nan')

    proxy = {t: mean(t, ['proxy', 'ap50']) for t in rows}
    mesh_conf = {t: mean(t, ['mesh', 'conf|none|smooth', 'ap50']) for t in rows}
    mesh_nobs = {t: mean(t, ['mesh', 'nobs|nms|smooth', 'ap50']) for t in rows}
    ranked = sorted(rows, key=lambda t: -proxy[t])

    def spearman(a, b, subset=None):
        ts = [t for t in rows if subset is None or t in subset]
        x, y = [a[t] for t in ts], [b[t] for t in ts]
        ok = [i for i in range(len(ts)) if x[i] == x[i] and y[i] == y[i]]
        if len(ok) < 3:
            return float('nan')
        return float(spearmanr([x[i] for i in ok], [y[i] for i in ok]).statistic)

    named = {P1, 'p2a_calib', 'p2b_calib', 'two_step_p2a_0.2'}
    r_conf = spearman(proxy, mesh_conf)
    r_nobs = spearman(proxy, mesh_nobs)
    r_conf_named = spearman(proxy, mesh_conf, named)
    p0b = resolve_path(cfg['stream_v2']['out_root']) / 'phase0b' / 'summary.json'
    esam = json.loads(p0b.read_text())['refs']['esam'] if p0b.exists() else None

    L = ['# 第 2 阶段 · 任务 0：体素代理指标与网格指标排得一样吗（只查评估方式，不选方案）', '',
         f'16 个验证场景（唯一同时有 ESAM 网格和逐帧 GT 的场景），{len(rows)} 个**已固定**的方案：'
         'P1（默认管线）＋ 0c 的 p2a/p2b_calib ＋ 第 1 阶段留出扫描已经定下并报告过的方案。'
         '本表不做任何选择或调参，只检查两套协议的排序是否一致。', '',
         '每对（方案，场景）只跑一次记忆：**体素代理**用该次运行的逐物体体素集合 '
         '（5 cm 体素，GT = 跨帧点投票的多数实例，thing 且 ≥ 20 体素，无后处理，分数 = 记忆自身置信度）；'
         '**网格**把同一批物体渲染到 ESAM 网格（thing GT ≥ 100 顶点，段平滑 0.5）。'
         '网格给两套口径：`conf|none`（与代理最接近：自身分数、无后处理）与 `nobs|nms`（第 1 阶段的主口径）。'
         'GT 只进指标。', '',
         '## 1. 每个方案的两种指标', '',
         '| 方案 | 体素代理 AP50 | 网格 AP50（conf\|none\|smooth） | 网格 AP50（nobs\|nms\|smooth） | 体素 IoU | 网格 IoU(conf) |',
         '|---|---|---|---|---|---|']
    for t in ranked:
        L.append(f"| {t} | {proxy[t]:.4f} | {mesh_conf[t]:.4f} | {mesh_nobs[t]:.4f} | "
                 f"{mean(t, ['proxy', 'iou']):.4f} | {mean(t, ['mesh', 'conf|none|smooth', 'iou']):.4f} |")
    if esam:
        L.append(f"| ESAM-E 官方导出（参考） | — | {esam['ap50']:.3f} | {esam['ap50']:.3f} | — | {esam['iou']:.3f} |")
    L += ['', '## 2. 排序一致性（Spearman）', '',
          '| 对比 | 方案数 | Spearman |', '|---|---|---|',
          f'| 体素代理 vs 网格 conf\\|none\\|smooth | {len(rows)} | **{r_conf:.3f}** |',
          f'| 体素代理 vs 网格 nobs\\|nms\\|smooth | {len(rows)} | {r_nobs:.3f} |',
          f'| 体素代理 vs 网格 conf\\|none\\|smooth（只看任务点名的 4 个方案） | 4 | {r_conf_named:.3f} |', '',
          '判定：Spearman ≥ 0.7 视为两套协议排序一致，留出场景上的体素代理可以继续用来选方案；'
          '低于 0.7 就停下报告（按任务书）。', '',
          '## 3. 与已发布数字的对照（sanity check）', '']
    p1c = resolve_path(cfg['stream_v2']['out_root']) / 'phase0c' / 'summary.json'
    p1v = resolve_path(cfg['stream_v2']['out_root']) / 'phase1' / 'val16' / 'summary.json'
    if p1c.exists():
        m = json.loads(p1c.read_text())['mean']
        for label, tag, key in (('0c: p1_default conf|none|smooth', P1, 'p1_default|conf|none|smooth'),
                                ('0c: p2a_calib conf|none|smooth', 'p2a_calib', 'p2a_calib|conf|none|smooth'),
                                ('0c: p2b_calib conf|none|smooth', 'p2b_calib', 'p2b_calib|conf|none|smooth')):
            if key in m and tag in mesh_conf:
                L.append(f"- {label} = {m[key]['ap50']:.3f}（本次重跑 {mesh_conf[tag]:.3f}）")
    if p1v.exists():
        m = json.loads(p1v.read_text())['mean']
        if 'nobs|nms|smooth' in m:
            L.append(f"- 第 1 阶段 two_step_p2a_0.2 nobs|nms|smooth = {m['nobs|nms|smooth']['ap50']:.3f}"
                     f"（本次重跑 {mesh_nobs.get('two_step_p2a_0.2', float('nan')):.3f}）")
    verdict = '一致（≥ 0.7）' if r_conf >= 0.7 else '不一致（< 0.7）：按任务书停下报告'
    L += ['', f'**结论：{verdict}。**', '']
    md = '\n'.join(L) + '\n'
    (out / 'summary.md').write_text(md)
    (out / 'summary.json').write_text(json.dumps(dict(
        proxy=proxy, mesh_conf=mesh_conf, mesh_nobs=mesh_nobs, spearman_conf=r_conf,
        spearman_nobs=r_nobs, spearman_conf_named=r_conf_named, scenes=n, tags=list(rows)), indent=2))
    print(md)


if __name__ == '__main__':
    main()
