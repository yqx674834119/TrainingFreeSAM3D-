"""Train the stage-2 unified merge model on the collected graphs (three seeds by default).

    source tools/env.sh
    python tools/window_unified_train.py --round A                  # round A graphs only
    python tools/window_unified_train.py --round AB --extra intra_extra

Writes <graphs_root>/../model/<round>/seed<k>/model.pt plus train_summary.json (per-seed loss
history and per-edge-type AUC / node-quality diagnostics on the training graphs).
"""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.streaming.unified_train import (GraphSet, diagnostics, save_unified_checkpoint,  # noqa: E402
                                               train)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--round', default='A', help='graph sub-directory name (A, B, ...)')
    p.add_argument('--extra', nargs='*', default=None, help='further graph sub-directories (e.g. intra_extra)')
    p.add_argument('--graphs', nargs='*', default=None, help='explicit graph directories (overrides --round)')
    p.add_argument('--out', default=None)
    p.add_argument('--seeds', nargs='+', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--frames-per-step', type=int, default=None)
    p.add_argument('--no-init', action='store_true', help='skip the GroupMergeNet warm start')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    dev = a.device if torch.cuda.is_available() else 'cpu'
    if str(dev).startswith('cuda'):
        print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    root = resolve_path(cfg['phase2']['graphs_root'])
    dirs = [Path(d) for d in a.graphs] if a.graphs else [root / a.round] + [root / e for e in (a.extra or [])]
    if a.no_init:
        cfg['phase2']['init_from'] = None
    out = Path(a.out) if a.out else root.parent / 'model' / a.round
    seeds = a.seeds if a.seeds else [int(s) for s in cfg['phase2']['seeds']]
    graphs = GraphSet(dirs)
    print(f'dirs {[str(d) for d in dirs]} -> {out}', flush=True)
    summary = {'dirs': [str(d) for d in dirs], 'seeds': {}, 'config': cfg['phase2']}
    for seed in seeds:
        model, std, info = train(graphs, cfg, dev, seed=seed, epochs=a.epochs,
                                 frames_per_step=a.frames_per_step,
                                 log=lambda *a: print(*a, flush=True))
        info['diagnostics'] = diagnostics(model, std, graphs, dev)
        save_unified_checkpoint(out / f'seed{seed}' / 'model.pt', model, std, cfg, extra=dict(seed=seed))
        summary['seeds'][str(seed)] = info
        d = info['diagnostics']
        print(f'seed {seed}: loss {info["history"][-1]["loss"]:.4f} | '
              f'AUC intra {d["edge"]["intra"]["auc"]:.3f} '
              f'block_obj {d["edge"]["block_object"]["auc"]:.3f} '
              f'obj_obj {d["edge"]["object_object"]["auc"]:.3f} | '
              f'node spearman {d["node"].get("spearman")}', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'train_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'wrote {out}/train_summary.json', flush=True)


if __name__ == '__main__':
    main()
