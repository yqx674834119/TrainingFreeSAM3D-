"""Train the cross-frame MatchScorer on the dense stream cache.

    source tools/env.sh
    python tools/stream_train_matcher.py --scenes 400         # subsample of train scenes (pair collection is the slow part)
Writes work_dirs/stream/matcher/<tag>/{model.pt,train.json}.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.streaming.matcher import MatchScorer, collect_pairs  # noqa: E402
from freesam3d.streaming.memory import MatchFeatures  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--scenes', type=int, default=400)
    p.add_argument('--scorer', default='scorer')
    p.add_argument('--tag', default='matcher')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    dev = a.device
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scorer_eval import load_scorer
    scorer, std, fd = load_scorer(resolve_path('work_dirs/mask_scorer') / a.scorer / 'model.pt', dev)
    root = resolve_path(cfg['cache_root']) / 'stream' / 'train'
    scene_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    rng = np.random.default_rng(a.seed)
    scene_dirs = [scene_dirs[i] for i in rng.permutation(len(scene_dirs))[:a.scenes]]
    out = resolve_path('work_dirs/stream/matcher') / a.tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    X, Y, S = [], [], []
    for si, sd in enumerate(scene_dirs):
        frames = []
        for f in sorted(sd.glob('*.npz'), key=lambda q: int(q.stem)):
            with np.load(f) as z:
                d = {k: z[k] for k in z.files}
            with torch.no_grad():
                d['scores'] = scorer.score(std.apply(torch.as_tensor(d['scalars'], device=dev)),
                                           torch.as_tensor(d['feat'].astype(np.float32), device=dev) if fd else torch.zeros((len(d['scalars']), 0), device=dev)).cpu().numpy()
            frames.append(d)
        x, y = collect_pairs(frames, dev)
        X.append(x); Y.append(y); S += [sd.name] * len(y)
        if (si + 1) % 50 == 0:
            print(f'{si + 1}/{len(scene_dirs)} scenes, {sum(len(v) for v in Y)} pairs, {time.time() - t0:.0f}s', flush=True)
    X, Y, S = np.concatenate(X), np.concatenate(Y).astype(np.float32), np.array(S)
    print(f'pairs {len(Y)} positive rate {Y.mean():.3f} from {len(scene_dirs)} scenes ({time.time() - t0:.0f}s)', flush=True)
    np.savez_compressed(out / 'pairs.npz', X=X, Y=Y, S=S)
    # held-out scenes for a sanity AUC
    scenes = np.unique(S); rng.shuffle(scenes); hold = set(scenes[:len(scenes) // 10].tolist())
    te = np.array([s in hold for s in S]); tr = ~te
    mu, sd_ = X[tr].mean(0), X[tr].std(0) + 1e-6
    torch.manual_seed(a.seed)
    model = MatchScorer(X.shape[1], 32).to(dev)
    Xt = torch.as_tensor((X - mu) / sd_, device=dev, dtype=torch.float32); Yt = torch.as_tensor(Y, device=dev)
    tr_idx = torch.as_tensor(np.flatnonzero(tr), device=dev); te_idx = torch.as_tensor(np.flatnonzero(te), device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    pos = float(Y[tr].mean()); pw = torch.tensor([(1 - pos) / max(pos, 1e-3)], device=dev)
    for ep in range(a.epochs):
        model.train(); perm = tr_idx[torch.randperm(len(tr_idx), device=dev)]
        for s0 in range(0, len(perm), 8192):
            idx = perm[s0:s0 + 8192]
            loss = F.binary_cross_entropy_with_logits(model(Xt[idx]), Yt[idx], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pte = torch.sigmoid(model(Xt[te_idx])).cpu().numpy()
    auc = roc_auc_score(Y[te], pte) if te.any() and len(np.unique(Y[te])) == 2 else float('nan')
    rule = roc_auc_score(Y[te], np.maximum(X[te, 1], X[te, 2])) if te.any() else float('nan')
    print(f'held-out scenes ({len(hold)}): matcher AUC {auc:.3f}  vs rule max_contain AUC {rule:.3f}  vs voxel_iou {roc_auc_score(Y[te], X[te, 0]):.3f}')
    torch.save(dict(state_dict=model.state_dict(), dim=X.shape[1], hidden=32, mu=mu, sd=sd_, names=MatchFeatures.names()), out / 'model.pt')
    (out / 'train.json').write_text(json.dumps(dict(pairs=int(len(Y)), positive_rate=float(Y.mean()), scenes=len(scene_dirs),
                                                     heldout_auc=auc, rule_auc=rule, params=model.num_params(), seconds=time.time() - t0), indent=2))
    print('saved', out / 'model.pt')


if __name__ == '__main__':
    main()
