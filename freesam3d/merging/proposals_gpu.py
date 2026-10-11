"""GPU proposal construction: the same scalars / features as ``proposals.build_proposals``,
all as (P, ·) and (P, N) tensor ops on the device where the masks already live.

Per-mask geometry comes from first / second moments ((P,N) @ (N,·)) and a masked
amin/amax; composition from (P,N) @ (N,G) one-hot; hierarchy from (R,R) and
(P,R) containment matmuls. 0.4 ms vs 42 ms in numpy for a typical frame.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .proposals import NUM_SCALARS, Proposals


def _containment_t(mf):
    return (mf @ mf.T) / mf.sum(1)[None, :].clamp_min(1)        # [i, j] = frac of j inside i


@torch.no_grad()
def build_proposals_gpu(xyz, groups, group_feat, point_proto, raw_masks_pts, raw_scores, mask_index, objectness,
                        merged_masks=None, min_points=100, nest_thr=0.8, device='cuda'):
    """Tensor-in, ``Proposals``-out (numpy, like the CPU builder). Inputs may be numpy or tensors."""
    dev = torch.device(device)

    def T(x, dt=torch.float32):
        return x.to(device=dev, dtype=dt) if torch.is_tensor(x) else torch.as_tensor(np.asarray(x), dtype=dt, device=dev)
    xyz = T(xyz)
    groups = T(groups, torch.long)
    N, G = len(xyz), int(groups.max()) + 1
    gfeat = T(group_feat)
    proto = F.normalize(T(point_proto), dim=1)
    raw = T(raw_masks_pts, torch.bool)
    rscore = T(raw_scores)
    mi = T(mask_index, torch.long)
    obj = T(objectness)
    conf_g = torch.zeros(G, device=dev)
    has = mi >= 0
    conf_g[has] = rscore[mi[has]]
    z0 = xyz[:, 2].min()

    keep_raw = raw.sum(1) >= min_points
    raw = raw[keep_raw]
    raw_idx = torch.nonzero(keep_raw).squeeze(1)
    parts = [raw]
    is_raw = [torch.ones(len(raw), dtype=torch.bool, device=dev)]
    src = [raw_idx]
    rconf = [rscore[raw_idx]]
    if merged_masks is not None:
        mm = T(merged_masks, torch.bool)
        mm = mm[mm.sum(1) >= min_points]
        parts.append(mm)
        is_raw.append(torch.zeros(len(mm), dtype=torch.bool, device=dev))
        src.append(torch.full((len(mm),), -1, dtype=torch.long, device=dev))
        rconf.append(torch.zeros(len(mm), device=dev))
    masks = torch.cat(parts)
    is_raw, src, rconf = torch.cat(is_raw), torch.cat(src), torch.cat(rconf)
    P = len(masks)
    if P == 0:
        return Proposals(np.zeros((0, N), bool), np.zeros((0, NUM_SCALARS), np.float32),
                         np.zeros((0, gfeat.shape[1]), np.float32), np.zeros(0, bool), np.zeros(0, int))
    mf = masks.float()
    cnt = mf.sum(1, keepdim=True)

    # geometry: moments + masked extents
    mean = (mf @ xyz) / cnt.clamp_min(1)
    sq = (mf @ (xyz[:, :, None] * xyz[:, None, :]).reshape(N, 9)).reshape(P, 3, 3) / cnt.clamp_min(1)[:, :, None]
    cov = (sq - mean[:, :, None] * mean[:, None, :]) * (cnt / (cnt - 1).clamp_min(1))[:, :, None]
    ev = torch.linalg.eigvalsh(cov).clamp_min(0).sqrt().flip(1)
    ev[cnt[:, 0] <= 3] = 0
    big = masks[:, :, None]
    ext = torch.where(big, xyz[None], torch.full((1,), -1e9, device=dev)).amax(1) - \
        torch.where(big, xyz[None], torch.full((1,), 1e9, device=dev)).amin(1)

    # composition
    onehot = F.one_hot(groups, G).float()
    pg = mf @ onehot
    w = pg / pg.sum(1, keepdim=True).clamp_min(1)
    obj_mean = w @ obj
    obj_min = torch.where(pg > 0, obj[None, :], torch.full_like(pg, float('inf'))).amin(1)
    n_groups = (pg > 0).sum(1).float()
    conc = (w ** 2).sum(1)
    conf_mean = w @ conf_g
    feat = F.normalize(w @ gfeat, dim=1)
    coh = ((mf @ proto) / cnt.clamp_min(1)).norm(dim=1)

    # hierarchy among raw masks and raw masks inside each candidate
    R = len(raw)
    n_parent = torch.zeros(P, device=dev)
    n_child = torch.zeros(P, device=dev)
    n_inside = torch.zeros(P, device=dev)
    if R:
        rf = raw.float()
        c_rr = _containment_t(rf)
        n_parent[:R] = (c_rr >= nest_thr).sum(0).float() - 1
        n_child[:R] = (c_rr >= nest_thr).sum(1).float() - 1
        inside = (mf @ rf.T) / rf.sum(1)[None, :].clamp_min(1)
        n_inside = (inside >= nest_thr).sum(1).float() - is_raw.float()

    scal = torch.stack([torch.log(cnt[:, 0]), *torch.log1p(ext).T, *ev.T, ev[:, 2] / ev[:, 0].clamp_min(1e-6),
                        mean[:, 2] - z0, obj_mean, obj_min, n_groups, conc, conf_mean, is_raw.float(), rconf, coh,
                        n_parent, n_child, n_inside], 1)
    return Proposals(masks.cpu().numpy(), scal.cpu().numpy().astype(np.float32), feat.cpu().numpy().astype(np.float32),
                     is_raw.cpu().numpy(), src.cpu().numpy())


@torch.no_grad()
def nested_nms_gpu(masks, scores, iou_thr=0.5, contain_thr=0.8, device='cuda'):
    """Same greedy rule as ``proposals.nested_nms`` with the pairwise matrices on the GPU;
    the sequential selection itself is a tiny loop over P ≈ 30 candidates."""
    mt = torch.as_tensor(np.asarray(masks), device=device).float() if not torch.is_tensor(masks) else masks.to(device).float()
    inter = mt @ mt.T
    area = mt.sum(1)
    iou = (inter / (area[:, None] + area[None, :] - inter).clamp_min(1)).cpu().numpy()
    frac_in = (inter / area[None, :].clamp_min(1)).cpu().numpy()
    order = np.argsort(-np.asarray(scores), kind='stable')
    keep, sup = [], np.zeros(len(order), bool)
    for i in order:
        if sup[i]:
            continue
        keep.append(i)
        sup |= (iou[i] > iou_thr) | (frac_in[i] >= contain_thr) | (frac_in[:, i] >= contain_thr)
    return np.array(keep, int)
