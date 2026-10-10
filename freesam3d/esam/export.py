"""Run the official ESAM-E test loop and record what its decoder outputs per frame.

Nothing in ESAM is modified. We wrap ``OnlineMerge.merge``: its first three
arguments are exactly the per-frame decoder result after ESAM's own
post-processing (top-k, matrix NMS, ``inst_score_thr``, ``npoint_thr``), i.e.
the per-frame masks before cross-frame merging. The final scene result
(after online merging, mapped to the mesh and smoothed) is recorded as well.

Output per scene: ``<out>/<scene>.npz`` with
  frame_masks_<i>   (n_pred_i, 20000) bool, per-frame ESAM masks
  frame_scores_<i>  (n_pred_i,)
  scene_masks       (n_scene_pred, V) bool on reconstructed mesh vertices
  scene_scores      (n_scene_pred,)
"""
import os
from pathlib import Path
import sys

import numpy as np


def export_predictions(esam_root, esam_config, checkpoint, workspace, out_dir):
    """Scenes = those in the workspace info pkl (see ``ESAMDataBuilder.write_infos``)."""
    esam_root, workspace, out_dir = Path(esam_root), Path(workspace).resolve(), Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(esam_root))
    os.chdir(workspace)          # ESAM loads reconstruction data from the relative path data/scannet200/

    import torch
    from mmengine.config import Config
    from mmengine.hooks import Hook
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    init_default_scope('mmdet3d')
    import oneformer3d  # noqa: F401  registers ESAM modules
    from oneformer3d.instance_merge import OnlineMerge

    frames = []
    original_merge = OnlineMerge.merge

    def recording_merge(self, masks, labels, scores, *args, **kwargs):
        frames.append((masks.detach().bool().cpu().numpy(), scores.detach().float().cpu().numpy()))
        return original_merge(self, masks, labels, scores, *args, **kwargs)
    OnlineMerge.merge = recording_merge

    class DumpHook(Hook):
        def before_test_iter(self, runner, batch_idx, data_batch=None):
            frames.clear()

        def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
            for sample in outputs:
                scene = sample.img_metas['lidar_idx']
                pred = sample.pred_pts_seg
                arrays = dict(scene_masks=np.asarray(pred.pts_instance_mask[0]).astype(bool),
                              scene_scores=np.asarray(pred.instance_scores, np.float32))
                for i, (m, s) in enumerate(frames):
                    arrays[f'frame_masks_{i}'] = m
                    arrays[f'frame_scores_{i}'] = s
                np.savez_compressed(out_dir / f'{scene}.npz', num_frames=len(frames), **arrays)
                print(f'{scene}: {len(frames)} frames, {len(arrays["scene_scores"])} scene instances', flush=True)

    cfg = Config.fromfile(str(esam_root / esam_config))
    ds = cfg.test_dataloader.dataset
    ds.data_root = str(workspace / 'data/scannet200-mv_fast') + '/'
    cfg.test_dataloader.update(batch_size=1, num_workers=0, persistent_workers=False)
    cfg.val_dataloader = cfg.test_dataloader
    cfg.load_from = str(checkpoint)
    cfg.work_dir = str(out_dir / 'mmengine')
    cfg.randomness = dict(seed=0, deterministic=False)
    cfg.custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
    runner = Runner.from_cfg(cfg)
    runner.register_hook(DumpHook(), priority='LOWEST')
    metrics = runner.test()
    torch.cuda.empty_cache()
    return metrics
