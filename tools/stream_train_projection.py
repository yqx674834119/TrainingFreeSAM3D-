"""Train the contrastive ProjectionHead on the dense stream cache.

    source tools/env.sh
    python tools/stream_train_projection.py --tag proj --epochs 40 --max-gap 3
Writes work_dirs/stream/projection/<tag>/{model.pt,train.json}. Held-out scenes report pair AUC at
frame gaps 1 and 4 for raw-feature cosine vs projected cosine.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.streaming.projection import ProjectionHead, load_scene_frames, multi_positive_infonce  # noqa: E402


def pair_auc(fn, scenes, gap):
    A, B, Y = [], [], []
    for sc in scenes:
        for i in range(len(sc) - gap):
            fa, ga, _ = sc[i]
            fb, gb, _ = sc[i + gap]
            if len(ga) and len(gb):
                A.append(np.repeat(fa, len(gb), 0))
                B.append(np.tile(fb, (len(ga), 1)))
                Y.append((ga[:, None] == gb[None, :]).reshape(-1))
    if not Y:
        return float('nan')
    A, B, Y = np.concatenate(A), np.concatenate(B), np.concatenate(Y)
    return float(roc_auc_score(Y, fn(A, B))) if 0 < Y.mean() < 1 else float('nan')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--tag', default='proj')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--max-gap', type=int, default=3, help='train on frame pairs up to this many cached frames apart')
    p.add_argument('--out-dim', type=int, default=128)
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--tau', type=float, default=0.07)
    p.add_argument('--min-iou', type=float, default=0.5)
    p.add_argument('--holdout', type=int, default=60)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    root = resolve_path(cfg['cache_root']) / 'stream' / 'train'
    dirs = sorted(d for d in root.iterdir() if d.is_dir())
    dirs = [dirs[i] for i in rng.permutation(len(dirs))]
    t0 = time.time()
    scenes = [load_scene_frames(d, a.min_iou) for d in dirs]
    te, tr = scenes[:a.holdout], scenes[a.holdout:]
    print(f'{len(tr)} train / {len(te)} held-out scenes, loaded in {time.time() - t0:.0f}s', flush=True)
    model = ProjectionHead(640, a.hidden, a.out_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    cos = lambda A, B: (F.normalize(torch.as_tensor(A), dim=1) * F.normalize(torch.as_tensor(B), dim=1)).sum(1).numpy()
    base = {g: pair_auc(cos, te, g) for g in (1, 4)}
    print(f'raw cosine held-out AUC: gap1 {base[1]:.3f} gap4 {base[4]:.3f}', flush=True)
    hist = []
    for ep in range(a.epochs):
        model.train()
        tot = n = 0
        order = rng.permutation(len(tr))
        for si in order:
            sc = tr[si]
            for i in range(len(sc)):
                for g in range(1, a.max_gap + 1):
                    if i + g >= len(sc):
                        break
                    fa, ga, _ = sc[i]
                    fb, gb, _ = sc[i + g]
                    if len(ga) < 2 or len(gb) < 2:
                        continue
                    same = torch.as_tensor(ga[:, None] == gb[None, :], device=dev)
                    if not same.any():
                        continue
                    za = model(torch.as_tensor(fa, device=dev))
                    zb = model(torch.as_tensor(fb, device=dev))
                    loss = multi_positive_infonce(za, zb, same, a.tau) + multi_positive_infonce(zb, za, same.T, a.tau)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    tot += float(loss)
                    n += 1
        sched.step()
        hist.append(tot / max(n, 1))
        if ep == 0 or (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                pc = lambda A, B: (model(torch.as_tensor(A, device=dev)) * model(torch.as_tensor(B, device=dev))).sum(1).cpu().numpy()
                r = {g: pair_auc(pc, te, g) for g in (1, 4)}
            print(f'epoch {ep + 1:3d}/{a.epochs} loss {hist[-1]:.4f}  held-out AUC gap1 {r[1]:.3f} gap4 {r[4]:.3f}', flush=True)
    out = resolve_path('work_dirs/stream/projection') / a.tag
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), in_dim=640, hidden=a.hidden, out_dim=a.out_dim, tau=a.tau), out / 'model.pt')
    (out / 'train.json').write_text(json.dumps(dict(tag=a.tag, params=model.num_params(), epochs=a.epochs, history=hist,
                                                     raw_auc=base, proj_auc=r, train_scenes=len(tr), holdout_scenes=len(te),
                                                     seconds=time.time() - t0), indent=2))
    print(f'saved {out / "model.pt"} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
