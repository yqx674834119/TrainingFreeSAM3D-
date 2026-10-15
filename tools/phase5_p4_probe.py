"""阶段 5 · P4 门槛探针（不训练）：最难配对（reobserve vs adjacent，与阶段 4b 同口径）上
15 列手工 vs 新 44 列成对几何的信息量。

口径与 work_dirs/stream_v2/phase4b/probe/*/probe.json 完全一致（two_step/p2a、纯度 0.6、
60 个留出场景、同一批配对）；配对与标签是 model-free 的，重跑必须复现 hand15 = 0.792345。
门槛：新特征（44 列单独或 15+44）在该子集上的留一场景逻辑回归 AUC ≥ hand15 − 0.03（0.7623）。

    source tools/env.sh
    python tools/phase5_p4_probe.py --out work_dirs/stream_v2/phase5/p4/probe
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
from freesam3d.streaming.phase4_probe import collect_phase4_pairs, phase5_geom_report  # noqa: E402
from freesam3d.streaming.window_scores import load_group_merge_net  # noqa: E402

RECORDED_HAND15 = 0.7923449630676874   # phase4b desc/probe.json, hardest_reobserve_vs_adjacent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--mode', default=None, choices=['linkage', 'calib', 'two_step'])
    p.add_argument('--rule', default='p2a', choices=['p2a', 'p2b'])
    p.add_argument('--intra-threshold', type=float, default=0.65)
    p.add_argument('--cross-threshold', type=float, default=0.2)
    p.add_argument('--purity', type=float, default=0.6)
    p.add_argument('--feats', default=None, help='碎片特征缓存目录；默认 <phase4.feat_cache>/holdout')
    p.add_argument('--scenes', nargs='*', default=None)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--out', default=None)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    dev = a.device if torch.cuda.is_available() else 'cpu'
    if str(dev).startswith('cuda'):
        print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    if a.mode:
        variant = Variant(a.mode, a.rule, a.intra_threshold, a.cross_threshold)
    else:
        sel = resolve_path(cfg['stream_v2']['out_root']) / 'phase1' / 'selected.json'
        d = json.loads(sel.read_text())['variant']
        variant = Variant(d['mode'], d['rule'], d['intra_threshold'], d['cross_threshold'])
    feats = Path(a.feats) if a.feats else resolve_path(cfg['phase4']['feat_cache']) / 'holdout'
    out = Path(a.out) if a.out else resolve_path(cfg['phase5']['p4']['out']) / 'probe'
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = resolve_path(cfg['cache_root']) / 'seq' / 'train'
    scenes = a.scenes if a.scenes else list(cfg['stream_v2']['holdout_scenes'])
    if a.limit:
        scenes = scenes[:a.limit]
    model, std, _ = load_group_merge_net(resolve_path('work_dirs/merge_net/full/model.pt'), dev)
    print(f'P4 probe: variant {variant.as_dict()} | purity {a.purity} | feats {feats} | '
          f'{len(scenes)} scenes -> {out}', flush=True)

    acc, t0 = {}, time.time()
    for i, scene in enumerate(scenes):
        frames = load_scene_frames(cache_dir, scene, gt='2d')
        pairs = collect_phase4_pairs(frames, variant, model, std, cfg, dev, feats, a.purity, with_geom=True)
        for k, v in pairs.items():
            acc.setdefault(k, []).append(v)
        if (i + 1) % 10 == 0 or i + 1 == len(scenes):
            print(f'  [{i + 1}/{len(scenes)}] {scene} ({time.time() - t0:.0f}s)', flush=True)
    pairs = {k: np.concatenate(v) for k, v in acc.items()}
    rep = phase5_geom_report(pairs, purity=a.purity)
    hard = rep['hardest_reobserve_vs_adjacent']
    best = max(hard['new44_logistic'], hard['hand15_plus_new44_logistic'])
    rep['gate'] = dict(recorded_hand15=RECORDED_HAND15, bar=RECORDED_HAND15 - 0.03,
                       hand15_reproduced=float(hard['hand15_logistic']),
                       best_new=float(best), passed=bool(best >= RECORDED_HAND15 - 0.03))
    rep.update(tag='p4_geom', variant=variant.as_dict(), purity=float(a.purity),
               feats=str(feats), scenes=len(scenes), seconds=time.time() - t0)
    (out / 'probe.json').write_text(json.dumps(rep, indent=1, default=float))
    L = ['# 阶段 5 · P4 门槛探针（成对几何边特征，不训练）', '',
         f"- 口径：{variant.as_dict()}，纯度 {a.purity}，{len(scenes)} 个留出场景；"
         f"复现 hand15（hardest）= {hard['hand15_logistic']:.6f}（记录值 {RECORDED_HAND15:.6f}）", '',
         '| 配对集 | n | 正例 | hand15 | new44 | hand15+new44 |', '|---|---|---|---|---|---|']
    for key, label in (('cross', '全部跨帧'), ('hardest_reobserve_vs_adjacent', '最难（reobserve/adjacent）'),
                       ('touching_negative', '接触负样本')):
        q = rep[key]
        L.append(f"| {label} | {q['n']} | {q['n_pos']} | {q['hand15_logistic']:.4f} | "
                 f"{q['new44_logistic']:.4f} | {q['hand15_plus_new44_logistic']:.4f} |")
    L += ['', f"- 门槛：新特征最好 AUC {best:.4f} ≥ {RECORDED_HAND15 - 0.03:.4f} → "
              f"**{'通过' if rep['gate']['passed'] else '不通过'}**"]
    (out / 'summary.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L), flush=True)


if __name__ == '__main__':
    main()
