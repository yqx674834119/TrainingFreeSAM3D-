"""Stage-3 training: the unified model with the stage-3 relational node inputs and the
scene-internal pairwise ranking loss (three seeds by default).

    source tools/env.sh
    python tools/window_phase3_train.py --rel --rank-weight 0.5          # rel + rank loss, seeds 0-2
    python tools/window_phase3_train.py                                  # YAML 默认关：无 rel、无排序损失
    python tools/window_phase3_train.py --attn-layers 1 --out work_dirs/stream_v2/phase3/model/rel_attn

Graphs: the re-collected A round (``phase3.graphs_root/A``, carrying the 8 relational columns)
plus the phase-2 ``intra_extra`` cache (no cross nodes; its relational columns are zeros).
Writes <out>/seed<k>/model.pt plus train_summary.json.
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
    p.add_argument('--graphs', nargs='*', default=None,
                   help='graph directories (default: phase3 A + phase2 intra_extra)')
    p.add_argument('--out', default=None)
    p.add_argument('--seeds', nargs='+', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--frames-per-step', type=int, default=None)
    p.add_argument('--rel', dest='rel', action='store_true', default=None)
    p.add_argument('--no-rel', dest='rel', action='store_false')
    p.add_argument('--rank-weight', type=float, default=None)
    p.add_argument('--attn-layers', type=int, default=None)
    p.add_argument('--no-init', action='store_true', help='skip the GroupMergeNet warm start')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    p3 = cfg.setdefault('phase3', {})
    if a.rel is not None:
        p3['rel'] = bool(a.rel)
    if a.rank_weight is not None:
        p3['rank_weight'] = float(a.rank_weight)
    if a.attn_layers is not None:
        p3['attn_layers'] = int(a.attn_layers)
    if a.no_init:
        cfg['phase2']['init_from'] = None
    dev = a.device if torch.cuda.is_available() else 'cpu'
    if str(dev).startswith('cuda'):
        print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    dirs = [Path(d) for d in a.graphs] if a.graphs else [
        resolve_path(p3['graphs_root']) / 'A',
        resolve_path(cfg['phase2']['graphs_root']) / 'intra_extra']
    out = Path(a.out) if a.out else resolve_path(p3['out_root']) / 'model' / 'rel'
    seeds = a.seeds if a.seeds else [int(s) for s in p3.get('seeds', [0, 1, 2])]
    graphs = GraphSet(dirs)
    print(f'dirs {[str(d) for d in dirs]} -> {out}', flush=True)
    summary = {'dirs': [str(d) for d in dirs], 'seeds': {}, 'config': cfg['phase2'], 'phase3': p3}
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
