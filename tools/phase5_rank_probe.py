"""阶段 5 · P1：掩码质量打分的排序信息上限（不训练，任务书 §2）。

    source tools/env.sh
    python tools/phase5_rank_probe.py --stage collect     # 60 个留出场景（GPU，可 --shard/--limit）
    python tools/phase5_rank_probe.py --stage report      # CPU：读缓存写 summary.md/json
    python tools/phase5_rank_probe.py --stage all
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.analysis import rank_probe as rp  # noqa: E402
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.models.instance_head import FastSAMEncoder, load_head  # noqa: E402
from freesam3d.streaming.unified_train import load_unified_checkpoint  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--stage', default='all', choices=['collect', 'report', 'all'])
    p.add_argument('--scenes', nargs='*', default=None)
    p.add_argument('--shard', default='0/1')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--out', default=None)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    p5 = cfg['phase5']['rank_probe']
    out = Path(a.out) if a.out else resolve_path(cfg['phase5']['out_root']) / 'rank_probe'
    if a.stage in ('collect', 'all'):
        dev = a.device if torch.cuda.is_available() else 'cpu'
        if str(dev).startswith('cuda'):
            print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
        model, std, ck = load_unified_checkpoint(resolve_path(p5['checkpoint']), dev)
        fast = cfg.get('fastsam') or load_config(resolve_path(cfg['lifting_config']))['fastsam']
        enc = FastSAMEncoder(fast['checkpoint'], dev, fast.get('imgsz', 640), raw=True)
        head = load_head(resolve_path(p5['head']), dev)[0]
        cache = resolve_path(cfg['cache_root']) / 'seq' / 'train'
        thr = [float(t) for t in p5['thresholds']]
        over = dict(threshold=thr[0], cross_threshold=thr[1], bo_threshold=thr[1], oo_threshold=thr[2])
        scenes = a.scenes or list(cfg['stream_v2']['holdout_scenes'])
        i, n = map(int, a.shard.split('/'))
        scenes = [s for k, s in enumerate(scenes) if k % n == i]
        if a.limit:
            scenes = scenes[:a.limit]
        print(f'{len(scenes)} holdout scenes, thresholds {thr} -> {out / "cache"}', flush=True)
        t0 = time.time()
        for si, scene in enumerate(scenes):
            dst = out / 'cache' / f'{scene}.npz'
            if dst.exists() and not a.overwrite:
                print(f'  {scene}: cached', flush=True)
                continue
            d = rp.collect_scene(scene, cfg, model, std, enc, head, dev, cache,
                                 cfg['dataset']['root'], mem_overrides=over)
            rp.save_scene(dst, d)
            m = d['metrics']
            print(f'[{a.shard}] {si + 1}/{len(scenes)} {scene}: {len(d["iou"])} cands | '
                  f'quality {m["quality"]:.3f} oracle {m["oracle"]:.3f} nobs {m["nobs"]:.3f} | '
                  f'{d["seconds"]:.1f}s ({(time.time() - t0) / 60:.1f} min)', flush=True)
    if a.stage in ('report', 'all'):
        rp.write_report(out, cfg)
    print('done', flush=True)


if __name__ == '__main__':
    main()
