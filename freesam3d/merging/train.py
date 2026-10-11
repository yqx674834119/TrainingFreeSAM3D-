"""Training loop, threshold calibration and checkpoint I/O for GroupMergeNet."""
import numpy as np
import torch
from torch.nn import functional as F

from ..evaluation.instance import GTSet, coverage_stats, groups_to_masks
from .cluster import merge_groups
from .data import FrameSet, Standardizer
from .graph import EDGE_DIM, NODE_GEOM
from .model import build_model


def predict_frame(model, std, fr, device, with_objectness=False):
    """Edge merge probabilities (and per-group objectness) for one cached frame."""
    g = torch.from_numpy(fr['node_geom']).to(device)
    f = torch.from_numpy(fr['node_feat']).to(device)
    e = torch.from_numpy(fr['edges']).to(device)
    ef = torch.from_numpy(fr['edge_feat']).to(device)
    g, ef = std.apply(g, ef)
    with torch.no_grad():
        logit, obj = model(g, f, e, ef, return_objectness=True)
    p = torch.sigmoid(logit).cpu().numpy()
    return (p, torch.sigmoid(obj).cpu().numpy()) if with_objectness else p


def frame_iou(fr, merged, min_gt_points=100):
    gt = fr['gt_instance']
    ids, counts = np.unique(gt[gt > 0], return_counts=True)
    keep = ids[counts >= min_gt_points]
    if len(keep) == 0:
        return np.nan
    masks, _ = groups_to_masks(merged)
    return coverage_stats(masks, GTSet.from_labels(gt, keep))['mean_best_iou']


def calibrate_threshold(model, std, frames, device, linkage='average', grid=np.arange(0.3, 0.96, 0.05),
                        min_gt_points=100):
    """Pick the clustering threshold that maximises mean best IoU on ``frames``."""
    probs = [predict_frame(model, std, fr, device) for fr in frames]
    scores = {}
    for t in grid:
        ious = [frame_iou(fr, merge_groups(fr['groups'], fr['edges'], p, float(t), linkage), min_gt_points)
                for fr, p in zip(frames, probs)]
        scores[round(float(t), 2)] = float(np.nanmean(ious))
    best = max(scores, key=scores.get)
    return best, scores


def train(train_set: FrameSet, cfg, device, log=print, calib_frames=None):
    tc, mc = cfg['training'], cfg['model']
    torch.manual_seed(tc['seed'])
    rng = np.random.default_rng(tc['seed'])
    std = Standardizer.fit(train_set)
    feat_dim = train_set.frames[0]['node_feat'].shape[1]
    model = build_model(mc['kind'], NODE_GEOM, feat_dim, EDGE_DIM, mc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc['lr'], weight_decay=tc['weight_decay'])
    steps_per_epoch = max(1, len(train_set) // tc['frames_per_step'])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc['lr'], total_steps=tc['epochs'] * steps_per_epoch)
    # Class balance: weight the minority so a threshold of 0.5 is meaningful.
    ally = np.concatenate([f['y'][f['valid']] for f in train_set.frames])
    pos_w = torch.tensor([(1 - ally.mean()) / max(ally.mean(), 1e-3)], device=device)
    allo = np.concatenate([f['obj_y'][f['obj_valid']] for f in train_set.frames])
    obj_w = torch.tensor([(1 - allo.mean()) / max(allo.mean(), 1e-3)], device=device)
    obj_lambda = float(tc.get('objectness_weight', 0.5)) if model.has_objectness else 0.0
    log(f'model={mc["kind"]} params={model.num_params()} train_frames={len(train_set)} '
        f'edges={len(ally)} pos_rate={ally.mean():.3f} pos_weight={pos_w.item():.2f} | '
        f'objectness groups={len(allo)} thing_rate={allo.mean():.3f} weight={obj_lambda}')
    history = []
    for epoch in range(tc['epochs']):
        model.train()
        order = rng.permutation(len(train_set))
        tot, n = 0.0, 0
        for s in range(steps_per_epoch):
            idx = order[s * tc['frames_per_step']:(s + 1) * tc['frames_per_step']]
            geom, feat, edges, ef, y, valid, oy, ov = train_set.stack(idx, device)
            geom, ef = std.apply(geom, ef)
            logits, obj = model(geom, feat, edges, ef, return_objectness=True)
            loss = F.binary_cross_entropy_with_logits(logits[valid], y[valid], pos_weight=pos_w)
            if obj_lambda > 0 and ov.any():
                loss = loss + obj_lambda * F.binary_cross_entropy_with_logits(obj[ov], oy[ov], pos_weight=obj_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
            n += 1
        history.append(tot / n)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == tc['epochs']:
            log(f'epoch {epoch + 1:3d}/{tc["epochs"]} loss {history[-1]:.4f}')
    model.eval()
    threshold, sweep = (None, {})
    if calib_frames:
        threshold, sweep = calibrate_threshold(model, std, calib_frames, device, cfg['merge']['linkage'],
                                               min_gt_points=cfg['eval']['min_gt_points'])
        log(f'calibrated threshold {threshold} (mean best IoU on {len(calib_frames)} held-out train frames: '
            f'{sweep[threshold]:.3f})')
    return model, std, dict(history=history, threshold=threshold, threshold_sweep=sweep)


def save_checkpoint(path, model, std, cfg, extra):
    torch.save(dict(state_dict=model.state_dict(), standardizer=std.state(), config=cfg,
                    feat_dim=model.proj.in_features if getattr(model, 'proj', None) is not None else 0, **extra), path)


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    mc = ck['config']['model']
    model = build_model(mc['kind'], NODE_GEOM, ck['feat_dim'], EDGE_DIM, mc).to(device)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model, Standardizer.from_state(ck['standardizer']), ck
