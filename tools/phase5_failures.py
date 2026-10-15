"""阶段 5 · P3 失败案例诊断：把不好的预测按体素来源拆开着色（定性证据，CPU，不训练）。

    source tools/env.sh
    python tools/phase5_failures.py --stage collect [--limit N]   # 逐预测统计（60 留出场景）
    python tools/phase5_failures.py --stage render                # 挑最差的画图 + PLY + summary.md
    python tools/phase5_failures.py --stage all

数据全部来自 P1 缓存（`phase5/rank_probe/cache`）＋ merge_cache 的逐帧 npz，不再跑模型；
GT 只用于诊断着色，不参与任何选择。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.analysis.rank_probe import load_scene                      # noqa: E402
from freesam3d.config import load_config, resolve_path                    # noqa: E402
from freesam3d.evaluation.instance import GTSet, iou_matrix               # noqa: E402
from freesam3d.streaming.evaluate import voxel_majority_gt                # noqa: E402
from freesam3d.streaming.holdout import load_scene_frames                 # noqa: E402
from freesam3d.streaming.voxel import devoxelize                          # noqa: E402
from freesam3d.vis import failures as fv                                  # noqa: E402
from freesam3d.vis import streaming as vs                                 # noqa: E402


def scene_data(cfg, scene):
    """一个留出场景的诊断输入：体素坐标、GT、语义、预测掩码、分数、IoU 矩阵。"""
    ck = resolve_path(cfg['phase5']['out_root']) / 'rank_probe' / 'cache' / f'{scene}.npz'
    d = load_scene(ck)
    cache_dir = resolve_path(cfg['cache_root']) / 'seq' / 'train'
    frames = load_scene_frames(cache_dir, scene, gt='2d')
    cells, gt_vox, sem = voxel_majority_gt(frames, float(cfg['stream_v2']['voxel']))
    assert len(cells) == int(d['ncells']), (scene, len(cells), int(d['ncells']))
    gts = GTSet.from_labels(gt_vox, d['keep'])
    iou = iou_matrix(d['masks'], gts).max(1) if len(gts.ids) else np.zeros(len(d['masks']), np.float32)
    return dict(scene=scene, xyz=devoxelize(cells, float(cfg['stream_v2']['voxel'])), cells=cells,
                gt_vox=gt_vox, sem=sem, gts=gts, masks=d['masks'], quality=d['quality'].astype(np.float32),
                iou=iou, keep=d['keep'], stuff=set(int(s) for s in cfg['eval']['stuff_nyu40']))


def collect(cfg, out, scenes, limit=0):
    dst = out / 'preds'
    dst.mkdir(parents=True, exist_ok=True)
    for i, scene in enumerate(scenes[:limit] if limit else scenes):
        f = dst / f'{scene}.json'
        if f.exists():
            print(f'  {scene}: cached', flush=True)
            continue
        d = scene_data(cfg, scene)
        rows = []
        for p in range(len(d['masks'])):
            st = fv.object_stats(d['masks'][p], d['gt_vox'], d['sem'], d['keep'], d['stuff'])
            st['ptr'], st['iou'] = p, float(d['iou'][p])
            rows.append(st)
        f.write_text(json.dumps(dict(scene=scene, n_cells=int(len(d['cells'])), preds=rows)))
        print(f'[{i + 1}/{len(scenes)}] {scene}: {len(rows)} preds, '
              f'{(d["iou"] >= 0.5).sum()} matched', flush=True)


def load_rows(out):
    rows = []
    for f in sorted((out / 'preds').glob('*.json')):
        d = json.loads(f.read_text())
        for r in d['preds']:
            rows.append(dict(scene=d['scene'], **r))
    return rows


def aggregate(rows):
    """全留出场景的体素来源汇总：命中 / 未命中两组的体素构成 + 混合与尺寸比。"""
    def block(sel, name):
        sub = [r for r in rows if sel(r)]
        n = sum(r['n_pred'] for r in sub)
        out = dict(name=name, n_pred=len(sub), n_preds_voxels=n)
        for k in ('core', 'other_gt', 'stuff', 'unlabeled'):
            v = sum(r[k] for r in sub)
            out[k] = v; out[k + '_share'] = v / max(n, 1)
        out['mixed_preds'] = sum(1 for r in sub if r['mixed'] >= 2)
        out['into_other_dominant'] = sum(1 for r in sub if r['other_gt_share'] > 0.5)
        out['median_size_ratio'] = float(np.median([r['size_ratio'] for r in sub])) if sub else float('nan')
        out['median_iou'] = float(np.median([r['iou'] for r in sub])) if sub else float('nan')
        return out
    return dict(all=block(lambda r: True, 'all'),
                matched=block(lambda r: r['iou'] >= 0.5, 'matched (IoU >= 0.5)'),
                unmatched=block(lambda r: r['iou'] < 0.5, 'unmatched (IoU < 0.5)'))


def pick_cases(rows):
    """挑要画的案例：膨胀、混合、完全不落在物体上、漏掉的 GT、以及一个中等命中的对照。"""
    un = [r for r in rows if r['iou'] < 0.5]
    by_size = sorted(un, key=lambda r: -r['n_pred'])
    by_other = sorted([r for r in un if r['mixed'] == 0], key=lambda r: -r['other_gt_share'])
    by_stuff = sorted([r for r in un if r['best'] >= 0], key=lambda r: -r['stuff_share'])
    cases = []
    for r in by_size[:3]:
        cases.append(('A_inflated_or_big', r))
    for r in by_other[:2]:
        cases.append(('B_into_other_or_nothing', r))
    for r in by_stuff[:1]:
        cases.append(('C_stuff_heavy', r))
    for r in sorted([r for r in rows if r['best'] < 0], key=lambda r: -r['n_pred'])[:1]:
        cases.append(('D_covers_no_thing', r))
    for r in sorted([r for r in rows if 0.5 <= r['iou'] < 0.65], key=lambda r: -r['n_pred'])[:1]:
        cases.append(('E_matched_but_loose', r))
    return cases


def render(cfg, out, rows, scenes):
    vis = out / 'vis'
    vis.mkdir(parents=True, exist_ok=True)
    cases = pick_cases(rows)
    todo = {}
    for name, r in cases:
        todo.setdefault(r['scene'], None)
    missing = [s for s in scenes if s in todo]
    # 每个涉及的场景取一次数据，画场景总览
    lines = []
    for si, scene in enumerate(missing):
        d = scene_data(cfg, scene)
        gt_col, gt_img = vs.gt_panel(d['gts'])
        pcol, _ = vs.pred_colors(d['masks'], d['gts'], gt_col)
        panels = [('GT (thing instances)', gt_img),
                  ('Ours (painted by best GT)', vs.paint_masks(d['masks'], d['quality'], pcol, len(d['cells'])))]
        tag = f'{si + 1:02d}_{scene}'
        vs.top_down_png(vis / f'{tag}_scene.png', d['xyz'], panels)
        vs.write_ply(vis / f'{tag}_scene.ply', *vs.side_by_side(d['xyz'], panels))
        print(f'  scene {scene} -> {tag}_scene.png', flush=True)

    for ci, (kind, r) in enumerate(cases):
        d = scene_data(cfg, r['scene'])
        pred = d['masks'][r['ptr']]
        cat, best = fv.object_categories(pred, d['gt_vox'], d['sem'], d['keep'], d['stuff'])
        sel = pred | (d['gt_vox'] == best) if best >= 0 else pred
        note = (f'{r["scene"]} pred#{r["ptr"]}  IoU {r["iou"]:.3f}  |pred| {r["n_pred"]}  |GT| {r["n_gt"]}  '
                f'core {r["core_share"]*100:.0f}%  into-other {r["other_gt_share"]*100:.0f}%  '
                f'stuff {r["stuff_share"]*100:.0f}%  unlabeled {r["unlabeled_share"]*100:.0f}%')
        tag = f'{ci + 1:02d}_{kind}_{r["scene"]}_p{r["ptr"]}'
        fv.object_png(vis / f'{tag}.png', d['xyz'], cat, sel, f'{kind}: {r["scene"]} pred#{r["ptr"]}', note)
        fv.write_category_ply(vis / f'{tag}.ply', d['xyz'], cat, sel)
        lines.append(dict(tag=tag, kind=kind, **{k: r[k] for k in
                    ('scene', 'ptr', 'iou', 'best', 'n_pred', 'n_gt', 'size_ratio', 'mixed',
                     'core_share', 'other_gt_share', 'stuff_share', 'unlabeled_share')}))
        print(f'  case {tag}: {note}', flush=True)

    # 一个「漏掉的 GT」：没有任何预测覆盖它
    worst = None
    for scene in scenes[:12]:
        d = scene_data(cfg, scene)
        if len(d['gts'].ids) == 0:
            continue
        best_gt = iou_matrix(d['masks'], d['gts']).max(0) if len(d['masks']) else np.zeros(len(d['gts'].ids))
        for k, g in enumerate(d['gts'].ids):
            if best_gt[k] < 0.2 and int(d['gts'].sizes[k]) > 1500:
                if worst is None or int(d['gts'].sizes[k]) > worst[2]:
                    worst = (scene, int(g), int(d['gts'].sizes[k]), float(best_gt[k]), d)
    if worst is not None:
        scene, g, size, biou, d = worst
        pred = np.zeros(len(d['gt_vox']), bool)
        cat, best = fv.object_categories(pred, d['gt_vox'], d['sem'], d['keep'], d['stuff'], best=g)
        tag = f'F_missing_gt_{scene}_{g}'
        fv.object_png(vis / f'{tag}.png', d['xyz'], cat, d['gt_vox'] == g,
                      f'missing GT: {scene} #{g}', f'|GT| {size} voxels, best predicted IoU {biou:.3f}')
        fv.write_category_ply(vis / f'{tag}.ply', d['xyz'], cat, d['gt_vox'] == g)
        lines.append(dict(tag=tag, kind='F_missing_gt', scene=scene, best=g, n_gt=size, iou=biou))
        print(f'  case {tag}: |GT| {size}, best IoU {biou:.3f}', flush=True)
    return lines


def write_summary(out, agg, cases, n_scenes):
    L = ['# 阶段 5 · P3 失败案例诊断（60 个留出场景，体素代理口径；不训练）', '',
         f'场景数 {n_scenes}，预测总数 {agg["all"]["n_pred"]}（命中 {agg["matched"]["n_pred"]}，'
         f'未命中 {agg["unmatched"]["n_pred"]}）。GT 只用于着色与统计。', '',
         '## 预测体素的来源构成', '',
         '| 组 | 预测数 | 体素占比：核心 | 吃进另一个物体 | 墙/地板/天花板 | 无标注 | 混合(≥2 物体) | 中位尺寸比 | 中位 IoU |',
         '|---|---|---|---|---|---|---|---|---|']
    for k in ('all', 'matched', 'unmatched'):
        b = agg[k]
        L.append(f'| {b["name"]} | {b["n_pred"]} | {b["core_share"]*100:.1f}% | {b["other_gt_share"]*100:.1f}% | '
                 f'{b["stuff_share"]*100:.1f}% | {b["unlabeled_share"]*100:.1f}% | {b["mixed_preds"]} | '
                 f'{b["median_size_ratio"]:.2f} | {b["median_iou"]:.3f} |')
    L += ['', f'- 未命中的预测里，体素有一半以上落在**另一个 thing 物体**里的只有 '
          f'{agg["unmatched"]["into_other_dominant"]} 个（占未命中 {agg["unmatched"]["n_pred"]} 个的 '
          f'{agg["unmatched"]["into_other_dominant"] / max(agg["unmatched"]["n_pred"], 1) * 100:.1f}%）。',
          '- 图例：绿 = 落在最好那个 GT 里；橙 = 落在另一个 thing 物体里；土黄 = 墙/地板/天花板；'
          '灰 = 无标注；蓝灰 = GT 有而预测没覆盖。', '',
          '## 画出来的案例', '',
          '| 文件 | 类别 | 场景 | pred# | IoU | 核心 | 另一个物体 | 墙地 | 无标注 |',
          '|---|---|---|---|---|---|---|---|---|']
    for c in cases:
        if c['kind'] == 'F_missing_gt':
            L.append(f'| `{c["tag"]}` | 漏掉的 GT | {c["scene"]} | GT#{c["best"]} | {c["iou"]:.3f} | — | — | — | — |')
        else:
            L.append(f'| `{c["tag"]}` | {c["kind"]} | {c["scene"]} | {c["ptr"]} | {c["iou"]:.3f} | '
                     f'{c["core_share"]*100:.0f}% | {c["other_gt_share"]*100:.0f}% | {c["stuff_share"]*100:.0f}% | '
                     f'{c["unlabeled_share"]*100:.0f}% |')
    (out / 'summary.md').write_text('\n'.join(L) + '\n')
    (out / 'cases.json').write_text(json.dumps(dict(aggregate=agg, cases=cases, scenes=n_scenes),
                                               indent=1, ensure_ascii=False, default=float))
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--stage', default='all', choices=['collect', 'render', 'all'])
    p.add_argument('--scenes', nargs='*', default=None)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    out = resolve_path(cfg['phase5']['out_root']) / 'failures'
    out.mkdir(parents=True, exist_ok=True)
    scenes = a.scenes or list(cfg['stream_v2']['holdout_scenes'])
    if a.stage in ('collect', 'all'):
        collect(cfg, out, scenes, limit=a.limit)
    if a.stage in ('render', 'all'):
        rows = load_rows(out)
        n_scenes = len({r['scene'] for r in rows})
        agg = aggregate(rows)
        cases = render(cfg, out, rows, scenes)
        write_summary(out, agg, cases, n_scenes)
    print('done', flush=True)


if __name__ == '__main__':
    main()
