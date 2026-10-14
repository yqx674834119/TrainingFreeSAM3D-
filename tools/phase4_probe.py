"""Phase-4 gate probe on the 60 holdout scenes, with the F1.7-identical pairs.

    python tools/phase4_probe.py --head work_dirs/stream_v2/phase4/head/seed0/head.pt
    # 逐层消融：特征缓存与 --tag 换成对应的 drop_<layer>
    python tools/phase4_probe.py --head .../head_ablate/drop_proto/seed0/head.pt \
        --feats work_dirs/stream_v2/phase4/feats/holdout_drop_proto --tag drop_proto
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.streaming.holdout import Variant, load_scene_frames  # noqa: E402
from freesam3d.streaming.phase4_probe import collect_phase4_pairs, phase4_report  # noqa: E402
from freesam3d.streaming.window_scores import load_group_merge_net  # noqa: E402


def write_summary(out, rep):
    g = rep['gate']
    L = [f"# 阶段 4 探针（{rep['tag']}）：学到的实例特征 vs 手工特征（60 个留出场景，GT 只用于打标签）", '',
         f"配对与 F1.7 完全相同（window=1，`{rep['variant']['mode']}` / 规则 `{rep['variant']['rule']}` / "
         f"跨帧阈值 {rep['variant']['cross_threshold']}，{len(rep['scenes'])} 个场景，纯度 ≥ {rep['purity']}）。", '',
         '## 门槛', '',
         f"- (c) 最难配对（再观测 vs 相邻不同物体）逻辑回归 AUC = "
         f"**{rep['hardest_reobserve_vs_adjacent']['learned_logistic']:.4f}**（要求 ≥ 0.85；手工特征 0.7923）",
         f"- (a) 帧内相邻碎片 AUC = **{rep['intra']['learned_col']:.4f}** vs GroupMergeNet 概率 "
         f"{rep['intra']['groupmerge_prob_col']:.4f}（要求不低于后者）",
         f"- **结论：{'通过' if g['passed'] else '未通过'}**", '',
         '## (a) 帧内相邻碎片', '', '| 分数 | 列 AUC | 逻辑回归 AUC |', '|---|---|---|',
         f"| 学到的余弦 | {rep['intra']['learned_col']:.4f} | {rep['intra']['learned_logistic']:.4f} |",
         f"| 现有 group_cos | {rep['intra']['group_cos_col']:.4f} | {rep['intra']['group_cos_logistic']:.4f} |",
         f"| GroupMergeNet 概率 | {rep['intra']['groupmerge_prob_col']:.4f} | "
         f"{rep['intra']['groupmerge_prob_logistic']:.4f} |",
         '', f"帧内边 {rep['intra']['n']} 条（正 {rep['intra']['n_pos']}）。", '',
         '## (b)(c)(d) 跨帧配对', '',
         '| 配对集合 | 边数 | 学到余弦（列 / 逻辑回归） | 15 列手工 | 手工几何 10 列 | 学到+接触量 | 手工15+学到+接触 |',
         '|---|---|---|---|---|---|---|']
    for key, label in (('cross', '全部跨帧边'), ('hardest_reobserve_vs_adjacent', '再观测 vs 相邻（最难）'),
                       ('touching_negative', '接触的负样本')):
        q = rep[key]
        L.append(f"| {label} | {q['n']} | {q['learned_col']:.4f} / {q['learned_logistic']:.4f} | "
                 f"{q['hand15_logistic']:.4f} | {q['hand_geom10_logistic']:.4f} | "
                 f"{q['learned_plus_contact']:.4f} | {q['hand15_plus_learned_contact']:.4f} |")
    L += ['', '## 跨帧间隔 1/2/3', '', '| 间隔（帧） | 边数 | 学到余弦列 AUC | 学到逻辑回归 | 15 列手工 |',
          '|---|---|---|---|---|']
    for k, q in rep.get('by_gap', {}).items():
        L.append(f"| {k} | {q['n']} | {q['learned_col']:.4f} | {q['learned_logistic']:.4f} | "
                 f"{q['hand15_logistic']:.4f} |")
    L += ['', '## 边类型', '', '| 类型 | 边数 | 学到余弦列 AUC | 15 列手工 |', '|---|---|---|---|']
    for k, q in rep.get('by_type', {}).items():
        L.append(f"| {k} | {q['n']} | {q['learned_col']:.4f} | {q['hand15_logistic']:.4f} |")
    (out / 'summary.md').write_text('\n'.join(L) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--head', default=None, help='可选：只用于 --tag 默认值（原始层 / 描述子探针不给 head）')
    p.add_argument('--feats', default=None, help='碎片特征缓存目录；默认 <phase4.feat_cache>/holdout')
    p.add_argument('--mode', default=None, choices=['linkage', 'calib', 'two_step'])
    p.add_argument('--rule', default='p2a', choices=['p2a', 'p2b'])
    p.add_argument('--intra-threshold', type=float, default=0.65)
    p.add_argument('--cross-threshold', type=float, default=0.5)
    p.add_argument('--scenes', nargs='*', default=None)
    p.add_argument('--purity', type=float, default=0.6)
    p.add_argument('--cache-dir', default=None)
    p.add_argument('--out', default=None)
    p.add_argument('--tag', default=None, help='默认用 head 的上一级目录名（seed0 / drop_xxx）')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    dev = a.device if torch.cuda.is_available() else 'cpu'
    cache_dir = Path(a.cache_dir) if a.cache_dir else resolve_path(cfg['cache_root']) / 'seq' / 'train'
    feats = Path(a.feats) if a.feats else resolve_path(cfg['phase4']['feat_cache']) / 'holdout'
    if a.mode:
        variant = Variant(a.mode, a.rule, a.intra_threshold, a.cross_threshold)
    else:
        sel = resolve_path(cfg['stream_v2']['out_root']) / 'phase1' / 'selected.json'
        if not sel.exists():
            raise SystemExit(f'{sel} missing: run the holdout sweep first, or pass --mode')
        d = json.loads(sel.read_text())['variant']
        variant = Variant(d['mode'], d['rule'], d['intra_threshold'], d['cross_threshold'])
    tag = a.tag or (Path(a.head).parent.name if a.head else 'probe')
    out = Path(a.out) if a.out else resolve_path(cfg['phase4']['out_root']) / 'probe' / tag
    out.mkdir(parents=True, exist_ok=True)
    scenes = a.scenes if a.scenes else list(cfg['stream_v2']['holdout_scenes'])
    model, std, _ = load_group_merge_net(resolve_path('work_dirs/merge_net/full/model.pt'), dev)
    print(f'probe {tag}: variant {variant.as_dict()} | head {a.head} | feats {feats}', flush=True)

    parts, used, t0 = [], [], time.time()
    for scene in scenes:
        if not sorted(cache_dir.glob(f'{scene}_*.npz')):
            continue
        frames = load_scene_frames(cache_dir, scene, gt='2d')
        if len(frames) < 2:
            continue
        parts.append(collect_phase4_pairs(frames, variant, model, std, cfg, dev, feats, purity=a.purity))
        used.append(scene)
    if not parts:
        raise SystemExit('no holdout scene had a cached sequence')
    pairs = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    rep = phase4_report(pairs, purity=a.purity)
    rep.update(tag=tag, head=a.head, feats=str(feats), variant=variant.as_dict(), scenes=used,
               seconds=time.time() - t0)
    (out / 'probe.json').write_text(json.dumps(rep, indent=2, default=float))
    write_summary(out, rep)
    print((out / 'summary.md').read_text())


if __name__ == '__main__':
    main()
