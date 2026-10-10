"""Run training-free Query Lifting on RGB-D frames and write diagnostics.

    python tools/run_query_lifting.py --config configs/query_lifting.yaml \
        [--inputs frame.npz ...] [--output work_dirs/query_lifting/run] \
        [--opts features=[neck_p3] lifting.seed=1]
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.analysis.report import aggregate, to_markdown  # noqa: E402
from freesam3d.analysis.similarity import evaluate_frame  # noqa: E402
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.io.frames import list_frames, load_npz_frame  # noqa: E402
from freesam3d.io.scannet_gt import PointGT, ScanNetSceneGT  # noqa: E402
from freesam3d.lifting.geometry import backproject  # noqa: E402
from freesam3d.lifting.query_lifting import QueryLifting  # noqa: E402
from freesam3d.vis import frame_panels, heatmaps, pointcloud  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/query_lifting.yaml')
    p.add_argument('--inputs', nargs='*', help='npz frames or a directory (default: config inputs)')
    p.add_argument('--output', help='output directory (default: config output_dir)')
    p.add_argument('--opts', nargs='*', default=[], help='config overrides, e.g. lifting.seed=1')
    p.add_argument('--no-vis', action='store_true', help='skip figures and PLY files')
    return p.parse_args()


def dense_gt(scene_gt, frame, lc):
    """GT instance id for every valid depth pixel (HxW, 0 = none)."""
    xyz, px = backproject(frame.depth, frame.intrinsic, frame.pose_c2w, lc['depth_scale'], lc['min_depth_raw'])
    lab = scene_gt.label_points(xyz)
    dense = np.zeros(frame.depth.shape, np.int64)
    dense[px[:, 1], px[:, 0]] = lab.instance
    dist = np.full(frame.depth.shape, np.nan, np.float32)
    dist[px[:, 1], px[:, 0]] = lab.knn_distance
    return dense, dist


def save_frame(lifted, gt, out_dir):
    arrays = dict(xyz=lifted.xyz, rgb=lifted.rgb, pixels=lifted.pixels, group_ids=lifted.group_ids,
                  group_mask_index=lifted.group_mask_index, group_map=lifted.group_map.astype(np.int16),
                  fastsam_scores=lifted.fastsam.scores)
    arrays.update({f'group_feats_{n}': f.numpy() for n, f in lifted.group_feats.items()})
    if gt is not None:
        arrays['gt_instance'] = gt.instance
    np.savez_compressed(out_dir / 'lifted.npz', **arrays)


def visualize(lifted, gt, gt_dense, views, pairs, metrics, out_dir, cfg_pool):
    name = lifted.frame.name
    frame_panels.overview(lifted, gt_dense, gt.categories, out_dir / '01_overview.png')
    frame_panels.feature_pca(lifted, out_dir / '02_feature_pca.png')
    for feat, v in views.items():
        g, m = v['gt'], v['mask']
        heatmaps.similarity_heatmap(
            g, f'{name} | {feat} | GT instances pooled with {cfg_pool}, cosine', out_dir / f'03_gt_similarity_{feat}.png',
            major_keys=g.categories)
        pm = metrics['features'][feat]['mask']
        heatmaps.similarity_heatmap(
            m, f'{name} | {feat} | FastSAM masks sorted by majority GT  '
               f'(AUC={pm["auc"]:.3f}, NN-hit={pm["nn_hit_rate"]:.2f}, pooling={cfg_pool})\nlabel = category #GT-id | m<FastSAM idx> (purity)',
            out_dir / f'04_mask_similarity_{feat}.png', minor_keys=list(m.gt_ids), major_keys=m.categories)
    heatmaps.pair_distributions(pairs, metrics['features'], out_dir / '05_mask_pair_similarity.png',
                                f'{name}: FastSAM mask pairs on the same vs different GT instance')
    pointcloud.export_lifted(lifted, gt.instance, out_dir)


def main():
    args = parse_args()
    cfg = load_config(resolve_path(args.config), args.opts)
    out_root = resolve_path(args.output or cfg['output_dir'])
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / 'config.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    lifting = QueryLifting(cfg)
    print(f'device={lifting.runner.device} features={cfg["features"]}')
    gt_cache, per_frame = {}, {}
    for path in list_frames(args.inputs or cfg['inputs']):
        t0 = time.time()
        frame = load_npz_frame(path)
        lifted = lifting(frame)
        out_dir = out_root / frame.name
        out_dir.mkdir(exist_ok=True)

        gt = None
        if cfg['gt']['enabled']:
            if frame.scene not in gt_cache:
                g = cfg['gt']
                gt_cache[frame.scene] = ScanNetSceneGT(frame.scene, g['scans_roots'], g['label_tsv'], g['bbox_margin'])
            scene_gt = gt_cache[frame.scene]
            gt_dense, gt_dist = dense_gt(scene_gt, frame, cfg['lifting'])
            px = lifted.pixels
            gt = PointGT(gt_dense[px[:, 1], px[:, 0]], gt_dist[px[:, 1], px[:, 0]], scene_gt.categories)
            ac = cfg['analysis']
            metrics, views, pairs = evaluate_frame(lifted, gt, ac['min_points_per_group'], ac['min_purity'],
                                                   cfg['pooling']['method'])
            per_frame[frame.name] = metrics
            (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
            if not args.no_vis:
                visualize(lifted, gt, gt_dense, views, pairs, metrics, out_dir, cfg['pooling']['method'])
        save_frame(lifted, gt, out_dir)
        auc = {f: round(v['mask_pure']['auc'], 3) for f, v in per_frame.get(frame.name, {}).get('features', {}).items()}
        print(f'{frame.name}: {len(lifted.fastsam.masks)} masks -> {lifted.num_groups} groups, '
              f'pure mask-pair AUC {auc}  ({time.time() - t0:.1f}s)')

    if per_frame:
        summary = aggregate(per_frame)
        (out_root / 'summary.json').write_text(json.dumps(dict(summary=summary, frames=per_frame), indent=2))
        (out_root / 'summary.md').write_text(to_markdown(summary, per_frame))
        if not args.no_vis:
            heatmaps.summary_bars(summary['features'], out_root / 'summary_features.png')
        print((out_root / 'summary.md').read_text())


if __name__ == '__main__':
    main()
