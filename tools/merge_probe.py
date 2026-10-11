"""Feature probe: is the FastSAM feature worth learning from, before any training?

    source tools/env.sh
    python tools/merge_probe.py                  # on the train cache (scene-grouped CV)
    python tools/merge_probe.py --split val      # on the 16 val scenes only
Writes work_dirs/merge_probe/<split>/{report.md,report.json}.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.merging.data import FrameSet, list_cache  # noqa: E402
from freesam3d.merging.probe import probe, to_markdown  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--split', default='train', choices=['train', 'val'])
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--gt', choices=['2d', '3d'], default='2d', help='label source (3d needs attached ESAM GT)')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    files = list_cache(resolve_path(cfg['cache_root']) / a.split)
    if a.max_frames:
        files = files[:: max(1, len(files) // a.max_frames)][:a.max_frames]
    lb = cfg['labels']
    fs = FrameSet(files, keep_points=(a.gt == '3d'), purity=lb['purity'], min_labeled=lb['min_labeled'])
    if a.gt == '3d':
        from freesam3d.merging.labels import edge_labels
        for fr in fs.frames:
            fr['y'], fr['valid'] = edge_labels(fr['groups'], fr['gt_instance_3d'], fr['edges'], lb['purity'], lb['min_labeled'])
    x, y, scene = fs.edge_table()
    report = probe(x, y, scene, seed=cfg['training']['seed'])
    report['frames'] = len(fs)
    out = resolve_path('work_dirs/merge_probe') / f'{a.split}_{a.gt}'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    md = to_markdown(report, f'Feature probe on {a.split} cache ({len(fs)} frames, {a.gt} GT, purity {lb["purity"]})')
    (out / 'report.md').write_text(md)
    print(md)


if __name__ == '__main__':
    main()
