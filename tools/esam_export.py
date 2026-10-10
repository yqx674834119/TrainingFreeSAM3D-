"""Run official ESAM-E on the prepared workspace; dump per-frame and scene predictions.

    source tools/env.sh
    python tools/esam_export.py            # -> work_dirs/esam_eval/predictions/<scene>.npz
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.esam.export import export_predictions  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/esam_eval.yaml')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    ws = resolve_path(cfg['workspace'])
    metrics = export_predictions(cfg['esam_root'], cfg['esam_config'], cfg['checkpoint'], ws, ws / 'predictions')
    (ws / 'predictions' / 'esam_official_metrics.json').write_text(json.dumps(metrics, indent=2, default=float))
    print(json.dumps(metrics, indent=2, default=float))


if __name__ == '__main__':
    main()
