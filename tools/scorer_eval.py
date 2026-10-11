"""Evaluate proposal scoring + nested NMS on the val proposal cache (ESAM 3D GT, thing, >= 100 pts).

    source tools/env.sh
    python tools/scorer_eval.py --scorers scorer scorer_scalars
Reports, per scorer and per candidate set (raw / merged / raw∪merged): mean best IoU, recall@0.5, AP50/AP25,
masks per frame; plus training-free baselines (FastSAM raw conf, GroupMergeNet objectness, size) and the
oracle-score ceiling. Writes work_dirs/mask_scorer/eval/{summary.md,summary.json}.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.evaluation.instance import GTSet, class_agnostic_ap, coverage_stats, fragments_per_gt  # noqa: E402
from freesam3d.merging.proposal_data import ScalarStandardizer, list_proposal_cache, load_proposals  # noqa: E402
from freesam3d.merging.proposals import nested_nms, scalar_names  # noqa: E402
from freesam3d.merging.scorer import MaskScorer  # noqa: E402


def load_scorer(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = MaskScorer(ck['scalar_dim'], ck['feat_dim'], ck['hidden']).to(dev).eval()
    m.load_state_dict(ck['state_dict'])
    return m, ScalarStandardizer.from_state(ck['standardizer']), ck['feat_dim']


def frame_metrics(masks, scores, gts):
    cov = coverage_stats(masks, gts)
    return dict(iou=cov['mean_best_iou'], r50=cov['recall@0.5'], r25=cov['recall@0.25'],
                ap50=class_agnostic_ap(masks, scores, gts, .5), ap25=class_agnostic_ap(masks, scores, gts, .25),
                frags=float(fragments_per_gt(masks, gts).mean()), masks=float(len(masks)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--scorers', nargs='*', default=['scorer'])
    p.add_argument('--iou-thr', type=float, default=0.5)
    p.add_argument('--contain-thr', type=float, default=0.8)
    p.add_argument('--tag', default='eval')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    stuff = set(cfg['eval']['stuff_raw'])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    scorers = {t: load_scorer(resolve_path('work_dirs/mask_scorer') / t / 'model.pt', dev) for t in a.scorers}
    files = list_proposal_cache(resolve_path(cfg['cache_root']) / 'proposals' / 'val')
    names = scalar_names()
    col = {n: i for i, n in enumerate(names)}
    rows = []
    for f in files:
        d = load_proposals(f, with_masks=True)
        gt, sem = d['gt_instance'], d['gt_sem']
        ids, c = np.unique(gt[gt > 0], return_counts=True)
        keep = [k for k in ids[c >= cfg['eval']['min_gt_points']] if np.bincount(sem[gt == k]).argmax() not in stuff]
        if not keep or len(d['iou']) == 0:
            continue
        gts = GTSet.from_labels(gt, keep)
        scores = {'oracle': d['iou'], 'size': d['scalars'][:, col['log_size']], 'objectness': d['scalars'][:, col['obj_mean']],
                  'fastsam_conf': np.where(d['is_raw'], d['scalars'][:, col['raw_conf']], d['scalars'][:, col['obj_mean']])}
        X = torch.as_tensor(d['scalars'], device=dev)
        Fe = torch.as_tensor(d['feat'], device=dev)
        for t, (m, std, fd) in scorers.items():
            with torch.no_grad():
                scores[t] = m.score(std.apply(X), Fe if fd else Fe[:, :0]).cpu().numpy()
        row = dict(name=f.stem, num_gt=len(keep))
        for cand, sel in (('raw', d['is_raw']), ('merged', ~d['is_raw']), ('raw+merged', np.ones(len(d['iou']), bool))):
            for sname, s in scores.items():
                idx = np.flatnonzero(sel)
                if not len(idx):
                    continue
                k = idx[nested_nms(d['masks'][idx], s[idx], a.iou_thr, a.contain_thr)]
                row[f'{cand}|{sname}'] = frame_metrics(d['masks'][k], s[k], gts)
        rows.append(row)
    keys = [k for k in rows[0] if '|' in k]
    means = {k: {m: float(np.nanmean([r[k][m] for r in rows if k in r])) for m in ('iou', 'r50', 'r25', 'ap50', 'ap25', 'frags', 'masks')} for k in keys}
    out = resolve_path('work_dirs/mask_scorer') / a.tag
    out.mkdir(parents=True, exist_ok=True)
    lines = [f'# Proposal scoring + nested NMS (iou_thr={a.iou_thr}, contain_thr={a.contain_thr})', '',
             f'{len(rows)} val frames, ESAM 3D GT, thing instances >= 100 pts, frame-averaged.', '',
             '| candidates | score | masks/frame | mean best IoU | recall@0.5 | AP50 | AP25 | frags/GT |', '|---|---|---|---|---|---|---|---|']
    for k in keys:
        cand, s = k.split('|')
        r = means[k]
        lines.append(f"| {cand} | {s} | {r['masks']:.1f} | {r['iou']:.3f} | {r['r50']:.3f} | {r['ap50']:.3f} | {r['ap25']:.3f} | {r['frags']:.2f} |")
    ref = resolve_path('work_dirs/esam_eval/granularity/summary.json')
    if ref.exists():
        s = json.loads(ref.read_text())['summary']
        for m in ('ESAM-snap', 'ESAM'):
            r = s[f'thing/{m}']
            lines.append(f"| ESAM study | {m} | {r['num_pred']:.1f} | {r['mean_best_iou']:.3f} | {r['recall@0.5']:.3f} | {r['AP50']:.3f} | {r['AP25']:.3f} | {r['mean_fragments']:.2f} |")
    md = '\n'.join(lines) + '\n'
    (out / 'summary.md').write_text(md)
    (out / 'summary.json').write_text(json.dumps(dict(means=means, frames=len(rows), iou_thr=a.iou_thr, contain_thr=a.contain_thr), indent=2))
    print(md)


if __name__ == '__main__':
    main()
