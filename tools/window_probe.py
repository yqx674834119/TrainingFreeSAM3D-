"""Stage 1 probe: which pair features carry signal for the cross-frame merge decision?

    source tools/env.sh
    python tools/window_probe.py                     # uses phase1/selected.json
    python tools/window_probe.py --mode two_step --rule p2a --cross-threshold 0.4

Runs ``WindowMemory`` (window 1) with the holdout-selected configuration over the 60 holdout scenes,
collects every *cross-frame* edge (current fragment <-> frozen object, frozen object <-> frozen
object) and labels it with GT: both ends pure (>= 0.6) and the same majority instance -> positive,
both pure and different -> negative, anything else is dropped. GT is used for labels and metrics
only; nothing here feeds a merge.

Reports per-column AUC of ``pair_feature_names()``, a scene-folded logistic-regression AUC, both
stratified by relation (same-object re-observation / part-whole / adjacent different object), and
the top-1 ranking accuracy of each current fragment's historical-object candidates.

Writes work_dirs/stream_v2/phase1/probe/{summary.md,probe.json}.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.streaming.holdout import Variant, load_scene_frames  # noqa: E402
from freesam3d.streaming.pair_features import cross_feature_names, pair_feature_names  # noqa: E402
from freesam3d.streaming.window_probe import (POS_PURITY, RELATION_LABELS, RELATION_PAIRS,  # noqa: E402
                                              RELATION_PAIR_LABELS, RELATIONS, collect_cross_edges,
                                              cross_column_rows, cross_columns, logistic_auc,
                                              relation_pair_auc, rule_scores_of, single_column_auc,
                                              top1_stats, touching_negative_auc)
from freesam3d.streaming.window_scores import load_group_merge_net  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--mode', default=None, choices=['linkage', 'calib', 'two_step'])
    p.add_argument('--rule', default='p2a', choices=['p2a', 'p2b'])
    p.add_argument('--intra-threshold', type=float, default=0.65)
    p.add_argument('--cross-threshold', type=float, default=0.5)
    p.add_argument('--scenes', nargs='*', default=None)
    p.add_argument('--purity', type=float, default=POS_PURITY)
    p.add_argument('--cache-dir', default=None)
    p.add_argument('--out', default=None)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    dev = a.device if torch.cuda.is_available() else 'cpu'
    cache_dir = Path(a.cache_dir) if a.cache_dir else resolve_path(cfg['cache_root']) / 'seq' / 'train'
    out = Path(a.out) if a.out else resolve_path(cfg['stream_v2']['out_root']) / 'phase1' / 'probe'
    out.mkdir(parents=True, exist_ok=True)
    scenes = a.scenes if a.scenes else list(cfg['stream_v2']['holdout_scenes'])
    if a.mode:
        variant = Variant(a.mode, a.rule, a.intra_threshold, a.cross_threshold)
    else:
        sel = resolve_path(cfg['stream_v2']['out_root']) / 'phase1' / 'selected.json'
        if not sel.exists():
            raise SystemExit(f'{sel} missing: run the holdout sweep first, or pass --mode')
        d = json.loads(sel.read_text())['variant']
        variant = Variant(d['mode'], d['rule'], d['intra_threshold'], d['cross_threshold'])
    model, std, _ = load_group_merge_net(resolve_path('work_dirs/merge_net/full/model.pt'), dev)
    print(f'probe variant {variant.as_dict()} over {len(scenes)} holdout scenes, purity {a.purity}', flush=True)

    parts, nodes, offset, used = [], [], 0, []
    for scene in scenes:
        files = sorted(cache_dir.glob(f'{scene}_*.npz'))
        if len(files) < 2:
            continue
        got = collect_cross_edges(load_scene_frames(cache_dir, scene, gt='2d'), variant, model, std, cfg,
                                  dev, purity=a.purity)
        got['frame_index'] = got['frame_index'] + offset
        offset += len(got['nodes'])
        parts.append(got)
        nodes.extend(got['nodes'])
        used.append(scene)
    if not parts:
        raise SystemExit('no holdout scene had a cached sequence; build work_dirs/merge_cache/seq/train first')
    pf = np.concatenate([g['pair_feat'] for g in parts])
    y = np.concatenate([g['y'] for g in parts]).astype(bool)
    valid = np.concatenate([g['valid'] for g in parts])
    relation = np.concatenate([g['relation'] for g in parts])
    scene = np.concatenate([g['scene'] for g in parts])
    frame_index = np.concatenate([g['frame_index'] for g in parts])
    types = np.concatenate([g['types'] for g in parts])
    edges = np.concatenate([g['edges'] for g in parts])
    names = pair_feature_names()
    print(f'{len(pf)} cross-frame edges over {len(used)} scenes, {int(valid.sum())} labeled '
          f'({int(y[valid].sum())} positive)', flush=True)

    report = dict(variant=variant.as_dict(), purity=a.purity, scenes=used,
                  counts=dict(edges=int(len(pf)), labeled=int(valid.sum()), positive=int(y[valid].sum()),
                              by_relation={r: int(((relation == r) & valid).sum()) for r in RELATIONS}))
    Xv, yv, relv, scv = pf[valid], y[valid], relation[valid], scene[valid]
    report['column_auc'] = single_column_auc(Xv, yv, names)
    report['cross_column_rows'] = cross_column_rows(Xv, yv, names)
    report['relation_pairs'] = {f'{a}|{b}': relation_pair_auc(Xv, yv, relv, scv, a, b, names)
                                for a, b in RELATION_PAIRS}
    report['touching_negative'] = touching_negative_auc(Xv, yv, scv, names)
    report['logistic_auc'] = dict(all_29=logistic_auc(Xv, yv, scv),
                                  cross_15=logistic_auc(cross_columns(Xv, names), yv, scv))
    scores = rule_scores_of(pf, variant.rule, names)
    report['top1'] = top1_stats(edges, types, frame_index, nodes, scores, purity=a.purity)
    report['top1_by_rule'] = {r: top1_stats(edges, types, frame_index, nodes, rule_scores_of(pf, r, names),
                                            purity=a.purity) for r in ('p2a', 'p2b')}
    report['single_column_auc_would_be'] = {n: report['column_auc'][n]['auc'] for n in cross_feature_names()}

    write_report(out, report, names)
    (out / 'probe.json').write_text(json.dumps(report, indent=2, default=float))
    print((out / 'summary.md').read_text())


def write_report(out, rep, names):
    c = rep['counts']
    L = ['# 第 1 阶段探针：跨帧配对的哪些特征有信号（留出场景，GT 仅用于打标签）', '',
         f'方案 `{rep["variant"]["mode"]}` / 规则 `{rep["variant"]["rule"]}` / 跨帧阈值 '
         f'{rep["variant"]["cross_threshold"]}，window=1，{len(rep["scenes"])} 个留出场景。', '',
         f'跨帧边共 {c["edges"]} 条，其中两端纯度 ≥ {rep["purity"]} 可打标签的 {c["labeled"]} 条'
         f'（正 {c["positive"]} / 负 {c["labeled"] - c["positive"]}）。'
         '本帧碎片↔历史物体与历史物体↔历史物体都算跨帧边。', '',
         '## 1. 每条跨帧边特征的单列 AUC（可打标签的边）', '',
         '| 特征 | 列号 | AUC | 非零占比 | 正类均值 | 负类均值 |', '|---|---|---|---|---|---|']
    for r in rep['cross_column_rows']:
        L.append(f"| {r['name']} | {r['index']} | {r['auc']:.4f} | {r['nonzero']:.3f} | "
                 f"{r['mean_pos']:.4f} | {r['mean_neg']:.4f} |")
    L += ['', '注：29 列里前 14 列是缓存的帧内边特征，在跨帧边上恒为 0，故不列出。'
              '`pair_feature_names()` 有三个名字（centroid_dist / height_diff / extent_ratio）'
              '同时出现在帧内块和跨帧块，按名字索引会张冠李戴，所以这里按列号列出。', '',
          '## 2. 逻辑回归 AUC（按场景 5 折）', '',
          '| 特征集 | AUC |', '|---|---|',
          f"| 全部 29 列 | {rep['logistic_auc']['all_29']:.4f} |",
          f"| 只用 15 个跨帧列 | {rep['logistic_auc']['cross_15']:.4f} |", '',
          '## 3. 按关系分层', '',
          '| 关系 | 边数 |', '|---|---|']
    for r in RELATIONS:
        L.append(f"| {RELATION_LABELS[r]} | {c['by_relation'][r]} |")
    L += ['', '关系是标签的函数（不同 GT → 相邻；同 GT → 再观测 / 部件-整体），'
              '单个关系内部全是同一类，AUC 没有定义。有意义的问法是**两个关系之间**分不分得开：', '',
          '| 关系对 | 边数 | 逻辑回归 AUC（15 列） | 最好的单列（AUC） |', '|---|---|---|---|']
    for a, b in RELATION_PAIRS:
        q = rep['relation_pairs'][f'{a}|{b}']
        L.append(f"| {RELATION_PAIR_LABELS[(a, b)]} | {q['n']} | {q['logistic']:.4f} | "
                 f"{q['best_column']} ({q['best_auc']:.4f}) |")
    tn = rep['touching_negative']
    L += ['', f"把负类收紧成**接触到的**相邻物体（`inter0_frac_min > 0`，{tn['n_negative']} 条）后："
              f"逻辑回归 AUC {tn['logistic']:.4f}，"
              f"现用规则（max 两向包含度）单列 AUC {tn['rule_auc']:.4f}。", '']
    t = rep['top1']
    L += ['', '## 4. 排名第一的准确率', '',
          f"对每个本帧碎片，取分数最高的历史物体，看它是不是同一个实例（只用两端 GT 已知的候选）。", '',
          '| 口径 | 值 |', '|---|---|',
          f"| 有候选的本帧碎片数 | {t['n_fragments']} |",
          f"| 其中候选里有正确物体的 | {t['n_with_correct']} |",
          f"| **排名第一就是正确的（准确率）** | **{t['accuracy']:.4f}** |",
          f"| 正确物体平均排名 | {t['mean_best_rank']:.4f} |",
          f"| 平均有多少个错误物体排在正确物体前面 | {t['mean_distractors_above']:.4f} |",
          f"| 候选里根本没有正确物体的碎片数 | {t['n_no_correct_candidate']} |", '',
          '| 规则 | 排名第一准确率 | 正确物体平均排名 | 平均错误物体排在前面 |', '|---|---|---|---|']
    for r in ('p2a', 'p2b'):
        q = rep['top1_by_rule'][r]
        L.append(f"| {r} | {q['accuracy']:.4f} | {q['mean_best_rank']:.4f} | {q['mean_distractors_above']:.4f} |")
    (out / 'summary.md').write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
