"""Prepare ESAM-format data (official preprocessing) for the scenes in configs/esam_eval.yaml.

    source tools/env.sh
    python tools/esam_prepare.py [--scenes scene0690_00 ...]
"""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.esam.prepare import ESAMDataBuilder  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/esam_eval.yaml')
    p.add_argument('--scenes', nargs='*')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    scenes = a.scenes or cfg['scenes']
    b = ESAMDataBuilder(cfg['esam_root'], resolve_path(cfg['workspace']), cfg['fastsam_checkpoint'],
                        cfg['raw_roots'], cfg['sens_stride'], cfg['seed'])
    for s in scenes:
        if b.done(s):
            print(f'{s}: already prepared, skip', flush=True)
            continue
        t = time.time()
        b.prepare(s)
        print(f'{s}: prepared in {time.time() - t:.0f}s', flush=True)
    b.write_infos(scenes)
    print('infos written for', len(scenes), 'scenes', flush=True)


if __name__ == '__main__':
    main()
