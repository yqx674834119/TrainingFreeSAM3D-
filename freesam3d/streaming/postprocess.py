"""Output-side post-processing of scene object masks (stage 0b): ranking scores, greedy NMS,
hard filters, and the wrong-answer destination decomposition.

All functions are pure (numpy in / numpy out) and matrix-based: the (K,K) mask-IoU matrix is
one float32 matmul, the per-mask GT IoU comes from ``evaluation.instance.iou_matrix``.
"""
import numpy as np

from ..evaluation.instance import iou_matrix


def iou_pair_matrix(masks, device='cpu'):
    """(K,K) IoU between the masks themselves (one float32 matmul; bool input).

    ``device`` != 'cpu' runs it on the GPU. The K~1500-mask scenes stage 0c produces make this
    ~4.8e11 flops per call (~39 s over 4 numpy threads, tens of ms on a 5090).
    """
    m = np.asarray(masks)
    if len(m) == 0:
        return np.zeros((0, 0), np.float32)
    if not (isinstance(device, str) and device == 'cpu'):    # torch.device('cpu') -> torch path
        import torch
        t = torch.from_numpy(np.ascontiguousarray(m)).to(device).to(torch.float32)
        inter = t @ t.t()
        n = t.sum(1)
        iou = inter / torch.clamp(n[:, None] + n[None, :] - inter, min=1)
        return iou.to('cpu').numpy().astype(np.float32)
    m = m.astype(np.float32)
    inter = m @ m.T
    n = m.sum(1)
    union = n[:, None] + n[None, :] - inter
    return (inter / np.maximum(union, 1)).astype(np.float32)


def greedy_nms(masks, scores, iou_thr=0.5, iou=None, device='cpu'):
    """Scene-level NMS: walk masks in score order, drop any whose IoU with a kept mask exceeds
    ``iou_thr``. Returns the kept indices in score order.

    Pass ``iou`` (a (K,K) matrix from :func:`iou_pair_matrix`) to reuse one matrix across several
    score/cleaning combinations instead of recomputing it for each.
    """
    masks = np.asarray(masks)
    scores = np.asarray(scores, np.float32)
    if len(masks) == 0:
        return np.zeros(0, np.int64)
    iou = iou_pair_matrix(masks, device) if iou is None else iou
    order = np.argsort(-scores, kind='stable')
    suppressed = np.zeros(len(masks), bool)
    keep = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        suppressed |= iou[i] > iou_thr
    return np.asarray(keep, np.int64)


def best_iou(masks, gts, device='cpu'):
    """(K,) best IoU of each mask against the GT instances (oracle ranking only; GT used)."""
    if len(masks) == 0 or len(gts.ids) == 0:
        return np.zeros(len(masks), np.float32)
    return iou_matrix(np.asarray(masks), gts, device).max(1).astype(np.float32)


def rank_scores(mode, masks=None, conf=None, n_obs=None, iou_gt=None, quality=None):
    """Ranking score per mask.

    ``conf`` = FastSAM confidence mean (the current default), ``size`` = vertex count,
    ``nobs`` = distinct observation frames with ties broken by the larger mask (the size
    term is < 1 by construction), ``quality`` = the stage-2 model's node quality
    (``WindowMemory.object_attrs()['quality']``), ``oracle`` = best GT IoU (ceiling only, uses GT).
    """
    if mode == 'conf':
        return np.asarray(conf, np.float32)
    if mode == 'quality':
        return np.asarray(quality, np.float32)
    if mode == 'size':
        return np.asarray(masks.sum(1), np.float32)
    if mode == 'nobs':
        size = np.asarray(masks.sum(1), np.float32)
        return np.asarray(n_obs, np.float32) + size / (2 * max(float(size.max()), 1.0))
    if mode == 'oracle':
        return np.asarray(iou_gt, np.float32)
    raise ValueError(f'unknown ranking mode {mode!r}')


def filter_impure(keep, impure_frac, thr=0.5):
    """Drop objects whose member points come mostly (fraction > thr) from impure fragments (GT used)."""
    return keep[np.asarray(impure_frac)[keep] <= thr]


def filter_single(keep, n_obs):
    """Drop objects observed in exactly one frame (no GT)."""
    return keep[np.asarray(n_obs)[keep] != 1]


def apply_topk(keep, scores, k):
    """First ``k`` of an already score-ordered index list (``k`` = 0 means keep all)."""
    if not k:
        return keep
    return keep[:k]


def classify_masks(masks, scores, gts, iou_thr=0.5, part_lo=0.25):
    """Wrong-answer destination per mask, in the same greedy order as ``class_agnostic_ap``.

    0 = correct (some *unmatched* GT reaches IoU >= iou_thr; that GT is then consumed),
    1 = duplicate (only already-matched GTs reach iou_thr), 2 = partial (best IoU in
    [part_lo, iou_thr)), 3 = junk (best IoU < part_lo).
    """
    out = np.full(len(masks), 3, np.int8)
    if len(masks) == 0 or len(gts.ids) == 0:
        return out
    iou = iou_matrix(np.asarray(masks), gts)
    order = np.argsort(-np.asarray(scores), kind='stable')
    matched = np.zeros(len(gts.ids), bool)
    for i in order:
        row = iou[i]
        cand = np.where(~matched & (row >= iou_thr), row, -1.0)
        j = int(cand.argmax())
        if cand[j] >= 0:
            matched[j] = True
            out[i] = 0
        elif row.max() >= iou_thr:
            out[i] = 1
        elif row.max() >= part_lo:
            out[i] = 2
    return out


def overmerge_flags(masks, gts, thr=0.5):
    """(K,) bool: the mask covers >= ``thr`` of two or more GT instances ("two objects glued into one").

    The per-mask counterpart of ``evaluation.instance.association_stats``'s ``overmerge_rate``;
    uses GT, so it is a diagnostic only (stage 0c reports it, no combination filters on it).
    """
    masks = np.asarray(masks)
    if len(masks) == 0 or len(gts.ids) == 0:
        return np.zeros(len(masks), bool)
    p = masks & gts.labeled[None, :]
    inter = p.astype(np.float32) @ gts.onehot.astype(np.float32)
    return ((inter / np.maximum(gts.sizes[None, :], 1)) >= thr).sum(1) >= 2
