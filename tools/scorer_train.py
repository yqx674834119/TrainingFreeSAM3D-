"""Train the mask-quality scorer on the train proposal cache.

    source tools/env.sh
    python tools/scorer_train.py --tag scorer            # scalars + 640-d feature projection
    python tools/scorer_train.py --tag scorer_scalars --no-feat
Writes work_dirs/mask_scorer/<tag>/{model.pt,train.json}.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.merging.proposal_data import ProposalSet, ScalarStandardizer, list_proposal_cache  # noqa: E402
from freesam3d.merging.scorer import MaskScorer  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--tag', default='scorer')
    p.add_argument('--no-feat', action='store_true')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=4096)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--bce-weight', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = resolve_path('work_dirs/mask_scorer') / a.tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ps = ProposalSet(list_proposal_cache(resolve_path(cfg['cache_root']) / 'proposals' / 'train'))
    std = ScalarStandardizer(ps.scalars)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = std.apply(torch.as_tensor(ps.scalars, device=dev))
    Fe = torch.as_tensor(ps.feat, device=dev)
    Y = torch.as_tensor(ps.iou, device=dev)
    feat_dim = 0 if a.no_feat else Fe.shape[1]
    model = MaskScorer(X.shape[1], feat_dim, a.hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(Y) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1))
    pos = float((ps.iou >= 0.5).mean())
    pos_w = torch.tensor([(1 - pos) / max(pos, 1e-3)], device=dev)
    print(f'{len(ps)} proposals from {len(ps.frames)} frames ({ps.is_raw.mean():.0%} raw); IoU>=0.5 rate {pos:.3f}; '
          f'params {model.num_params()}; loaded in {time.time() - t0:.0f}s', flush=True)
    hist = []
    for ep in range(a.epochs):
        model.train()
        order = torch.as_tensor(rng.permutation(len(Y)), device=dev)
        tot = n = 0
        for s in range(0, len(Y) - a.batch + 1, a.batch):
            idx = order[s:s + a.batch]
            o = model(X[idx], Fe[idx] if feat_dim else Fe[idx, :0])
            loss = F.binary_cross_entropy_with_logits(o[:, 0], Y[idx]) + \
                a.bce_weight * F.binary_cross_entropy_with_logits(o[:, 1], (Y[idx] >= 0.5).float(), pos_weight=pos_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
            n += 1
        hist.append(tot / n)
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f'epoch {ep + 1:3d}/{a.epochs} loss {hist[-1]:.4f}', flush=True)
    torch.save(dict(state_dict=model.state_dict(), standardizer=std.state(), scalar_dim=X.shape[1], feat_dim=feat_dim,
                    hidden=a.hidden, history=hist, args=vars(a)), out / 'model.pt')
    (out / 'train.json').write_text(json.dumps(dict(tag=a.tag, params=model.num_params(), proposals=len(ps),
                                                     frames=len(ps.frames), history=hist, seconds=time.time() - t0), indent=2))
    print(f'saved {out / "model.pt"} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
