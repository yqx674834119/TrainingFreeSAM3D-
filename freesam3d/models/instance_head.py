"""Phase-4 instance feature head: a small trainable 2D head on the *frozen* FastSAM maps.

Six feature maps (``HEAD_INPUTS``: backbone L9 + the three neck scales L15/L18/L21 +
``proto_hidden`` + ``proto``) each go through a 1x1 conv to 64 channels, are upsampled to the
finest stride (4) and summed FPN-style; 3x3 convs then produce a 32-d per-pixel feature that is
L2-normalised. ~0.24 M parameters, ~5 ms/frame on the RTX 5090.

Training is a PartField-style contrastive objective (arXiv 2504.11451): multi-positive InfoNCE
with a learnable temperature. Positives are points of the same instance, same-frame and 1-3
cached frames apart; negatives mix uniform sampling, 3D-hard points (< ``hard_3d_dist``, mined
from the frame's full labelled point complement) and feature-hard points (most similar
different-instance points, same complement). FastSAM stays frozen: the maps are always detached,
only the head receives gradients.

The learned feature lives on *points*; fragments/blocks/objects aggregate as the mean of their
member points' unit features, and the norm of that mean (<= 1) is the "internal consistency"
signal the phase-4 quality head reads. Fragment partitioning is untouched: point features are
sampled at exactly the pixels the merge cache already stores (``merge_cache/seq/*``).
"""
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import load_config, resolve_path
from .fastsam import FastSAMRunner, _load_fastsam, resolve_device

try:  # ultralytics ships LetterBox under two layouts depending on the build
    from ultralytics.data.augment import LetterBox
except ImportError:  # pragma: no cover - depends on the installed build
    from ultralytics.yolo.data.augment import LetterBox

# the six head inputs: name -> channels (strides: 32 / 8 / 16 / 32 / 4 / 4)
HEAD_INPUTS = (('backbone', 640), ('neck_p3', 320), ('neck_p4', 640), ('neck_p5', 640),
               ('proto_hidden', 320), ('proto', 32))
HEAD_NAMES = tuple(n for n, _ in HEAD_INPUTS)


# ------------------------------------------------------------------ geometry / sampling
@dataclass(frozen=True)
class SampleGeom:
    """Letterbox geometry of one network input: original -> input scale and padding (left, top)."""
    gain: float
    pad: tuple
    input_hw: tuple


def encode_image(rgb, imgsz=640, stride=32, device='cpu'):
    """RGB HxWx3 uint8 -> ((1,3,H',W') float tensor in [0,1], ``SampleGeom``).

    Uses ultralytics' own ``LetterBox`` and the exact ``BasePredictor.preprocess``
    sequence (BGR flip in, RGB CHW out, ``from_numpy(...).float().div_(255)``).
    Hand-rolling the same maths is NOT good enough: the network amplifies a
    1-ulp input difference into ~1e-2 activation differences (measured on
    neck_p4), so only a bit-identical input reproduces ``FastSAMRunner``'s maps
    (verified: max abs diff 0.0 against the runner path).
    """
    h, w = rgb.shape[:2]
    box = LetterBox(imgsz if isinstance(imgsz, tuple) else (imgsz, imgsz), auto=True, stride=stride)
    im = box(image=np.ascontiguousarray(rgb[..., ::-1]))          # ultralytics expects BGR
    im = np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))   # back to RGB, CHW
    # exact ultralytics order: uint8 -> device -> float -> in-place /255 (bit-identical
    # only in this order; a CPU-side division differs by 1 ulp, which the network amplifies)
    x = torch.from_numpy(im).to(device).float().div_(255)
    h2, w2 = im.shape[1:]
    gain = min(h2 / h, w2 / w)
    pad = (int(round((w2 - w * gain) / 2 - 0.1)), int(round((h2 - h * gain) / 2 - 0.1)))
    return x[None], SampleGeom(gain, pad, (h2, w2))


def _as_tensor(x, dtype, device):
    """Accept numpy arrays or tensors alike; return a tensor of ``dtype`` on ``device``."""
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def sample_features(fmap, pixels_xy, geom):
    """Bilinearly sample ``fmap`` (1,C,h,w) at original-image pixels (x, y).

    Same take as ``models.fastsam.sample_point_features`` (align_corners=False,
    padding_mode='border'), but the letterbox geometry is passed explicitly so this works for
    any map of the frozen network, not only ``FastSAMOutput``s.
    """
    xy = _as_tensor(pixels_xy, torch.float32, fmap.device)
    x_in = (xy[:, 0] + 0.5) * geom.gain + geom.pad[0]
    y_in = (xy[:, 1] + 0.5) * geom.gain + geom.pad[1]
    grid = torch.stack([2 * x_in / geom.input_hw[1] - 1, 2 * y_in / geom.input_hw[0] - 1], -1)
    feats = F.grid_sample(fmap, grid[None, :, None], mode='bilinear', padding_mode='border',
                          align_corners=False)
    return feats[0, :, :, 0].T.contiguous()


def fragment_mean_features(point_feat, groups, num_groups):
    """(G,C) mean of the member points' (unit) features per fragment; matrix op, no loop.

    Deliberately *not* re-normalised: the row norm (<= 1 for unit point features) is the
    fragment's internal consistency, which the phase-4 merge model reads.
    """
    dev = point_feat.device
    g = _as_tensor(groups, torch.int64, dev)
    m = torch.zeros((len(g), int(num_groups)), dtype=point_feat.dtype, device=dev)
    m[torch.arange(len(g), device=dev), g] = 1
    cnt = m.sum(0).clamp_min(1)
    return (m.T @ point_feat) / cnt[:, None]


# ------------------------------------------------------------------ the head
class InstanceHead(nn.Module):
    """1x1 per layer -> upsample to the finest stride -> sum -> 3x3 convs -> 32-d unit features."""

    def __init__(self, in_channels=None, width=64, out_dim=32, conv_layers=3):
        super().__init__()
        chans = dict(HEAD_INPUTS if in_channels is None else in_channels)
        self.names = tuple(chans)
        self.proj = nn.ModuleDict({k: nn.Conv2d(int(c), width, 1) for k, c in chans.items()})
        layers = []
        for _ in range(int(conv_layers) - 1):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(width, out_dim, 3, padding=1))
        self.body = nn.Sequential(*layers)
        self.out_dim = int(out_dim)

    def forward(self, maps):
        size = max((maps[n].shape[-2:] for n in self.names), key=lambda s: s[0] * s[1])
        acc = None
        for n in self.names:
            y = self.proj[n](maps[n])
            if y.shape[-2:] != size:
                y = F.interpolate(y, size=size, mode='bilinear', align_corners=False)
            acc = y if acc is None else acc + y
        out = self.body(acc)
        return out / out.norm(dim=1, keepdim=True).clamp_min(1e-6)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class FastSAMEncoder:
    """Frozen FastSAM-x feature maps for the six head inputs - no NMS, no mask decoding.

    ``raw=True`` calls the inner network directly on a self-preprocessed tensor (the training
    path: skips ultralytics' postprocessing, ~2x faster); ``raw=False`` runs the full
    ``FastSAMRunner`` predict so the maps come from the exact segmentation call. Both paths share
    the same preprocessing; a smoke check asserts the maps agree.
    """

    def __init__(self, checkpoint, device='auto', imgsz=640, raw=True):
        self.device = resolve_device(device)
        self.imgsz, self.raw = int(imgsz), bool(raw)
        self._captured = {}
        if self.raw:
            self.model = _load_fastsam(checkpoint)
            self.inner = self.model.model
            self.inner.to(self.device).eval()
            # ultralytics' predictor fuses Conv+BN when it wraps the model (AutoBackend); the
            # raw path must do the same or the last bits of every activation differ (measured
            # max abs diff 0.095 unfused). After fuse(): bit-identical to FastSAMRunner.
            if hasattr(self.inner, 'fuse'):
                self.inner.fuse()
            self.net = self.inner.model
        else:
            self.runner = FastSAMRunner(checkpoint, self.device, imgsz, features=HEAD_NAMES,
                                        keep_on_device=True)
            self.net = self.runner.model.model.model
        self._register_hooks()

    def _register_hooks(self):
        net, proto = self.net, self.net[-1].proto

        def save(key):
            def hook(_m, _inp, out):
                self._captured[key] = out.detach().float()
            return hook
        for name, src in (('backbone', 9), ('neck_p3', 15), ('neck_p4', 18), ('neck_p5', 21)):
            net[src].register_forward_hook(save(name))
        proto.cv2.register_forward_hook(save('proto_hidden'))
        proto.register_forward_hook(save('proto'))

    def _run_inner(self, x):
        inner = self.inner
        for name in ('_forward_once', '_predict_once'):
            fn = getattr(inner, name, None)
            if callable(fn):
                return fn(x)
        return inner(x)

    def encode(self, rgb):
        """rgb (H,W,3) uint8 RGB -> (maps {name: (1,C,h,w) float32 on device}, ``SampleGeom``).

        Runs under ``torch.no_grad`` (not ``inference_mode``): the captured maps are ordinary
        tensors, so the head can later build an autograd graph on top of them (the ``raw=False``
        runner path keeps its own inference tensors and is meant for the no-grad cache build).
        """
        self._captured.clear()
        with torch.no_grad():
            if self.raw:
                x, geom = encode_image(rgb, self.imgsz, device=self.device)
                self._run_inner(x.to(self.device))
                return {n: self._captured[n] for n in HEAD_NAMES}, geom
            out = self.runner(rgb)
            return {n: out.feature_maps[n] for n in HEAD_NAMES}, SampleGeom(out.gain, out.pad, out.input_hw)


# ------------------------------------------------------------------ contrastive pieces
def multi_positive_infonce(feat, ids, tau):
    """InfoNCE over in-step positives/negatives: loss_i = -log(sum_pos sim / sum_all sim).

    ``feat`` (N,C) rows need not be exactly unit (bilinear sampling blurs the normalisation);
    ``ids`` (N,) instance ids. Anchors without a positive or without a negative are masked out.
    Returns ``(loss, stats)``.
    """
    ids_t = _as_tensor(ids, torch.int64, feat.device)
    same = ids_t[:, None] == ids_t[None, :]
    eye = torch.eye(len(feat), dtype=torch.bool, device=feat.device)
    pos_m, neg_m = same & ~eye, ~same
    sim = (feat @ feat.t()) / tau
    sim = sim - sim.max(1, keepdim=True).values.detach()
    p = (sim.exp() * pos_m).sum(1)
    n = (sim.exp() * neg_m).sum(1)
    ok = pos_m.any(1) & neg_m.any(1)
    if not bool(ok.any()):
        return feat.sum() * 0.0, dict(n_anchor=0, pos_cos=float('nan'), neg_cos=float('nan'))
    cos = feat @ feat.t()
    stats = dict(n_anchor=int(ok.sum()),
                 pos_cos=float(cos[pos_m].mean()) if bool(pos_m.any()) else float('nan'),
                 neg_cos=float(cos[neg_m].mean()) if bool(neg_m.any()) else float('nan'))
    return -(p[ok] / (p[ok] + n[ok]).clamp_min(1e-12)).log().mean(), stats


def nearest_different_instance(xyz_anchor, ids_anchor, xyz_pool, ids_pool, max_dist, k):
    """(A,K) indices into ``pool`` of the K nearest different-instance points within ``max_dist``.

    Padding is -1 (no candidate). PartField's 3D-hard strategy: closer negatives are the
    informative ones, so this takes the nearest, not a random subset.
    """
    d = torch.cdist(xyz_anchor, xyz_pool)
    ok = (ids_anchor[:, None] != ids_pool[None, :]) & (d < float(max_dist))
    d = d.masked_fill(~ok, float('inf'))
    k = min(int(k), d.shape[1])
    vals, idx = d.topk(k, dim=1, largest=False)
    return torch.where(torch.isfinite(vals), idx, torch.full_like(idx, -1))


def most_similar_different_instance(feat_anchor, ids_anchor, feat_pool, ids_pool, k):
    """(A,K) indices into ``pool`` of the most cosine-similar different-instance points (-1 none)."""
    fa = feat_anchor / feat_anchor.norm(dim=1, keepdim=True).clamp_min(1e-6)
    fp = feat_pool / feat_pool.norm(dim=1, keepdim=True).clamp_min(1e-6)
    sim = fa @ fp.t()
    ok = ids_anchor[:, None] != ids_pool[None, :]
    sim = sim.masked_fill(~ok, float('-inf'))
    k = min(int(k), sim.shape[1])
    vals, idx = sim.topk(k, dim=1)
    return torch.where(torch.isfinite(vals), idx, torch.full_like(idx, -1))


def _collect_mined(rows, cap):
    """Union of mined indices (A,K) with -1 padding, capped, as a sorted numpy array."""
    flat = np.asarray(rows.detach().cpu(), np.int64).reshape(-1)
    flat = np.unique(flat[flat >= 0])
    return flat[:cap]


def fragment_pool(dense, geom, frame, sel, min_points=8, purity=0.6):
    """(F,C) fragment features of one frame (mean of member sampled point features, with grad).

    Only fragments with >= ``min_points`` sampled points, cached purity >= ``purity`` and a
    labelled majority instance take part; ids are the cached majority instances.
    """
    pf = sample_features(dense, frame.pixels[sel], geom)
    g = _as_tensor(frame.groups[sel], torch.int64, pf.device)
    K = int(len(frame.group_major))
    cnt = torch.bincount(g, minlength=K)
    acc = torch.zeros(K, pf.shape[1], dtype=pf.dtype, device=pf.device).index_add_(0, g, pf)
    mean = acc / cnt.clamp_min(1)[:, None]
    keep = np.flatnonzero((cnt.cpu().numpy() >= min_points) & (frame.group_purity >= purity)
                          & (frame.group_major > 0))
    if len(keep) == 0:
        return mean[:0], np.zeros(0, np.int64)
    return mean[torch.as_tensor(keep, device=pf.device)], frame.group_major[keep]


# ------------------------------------------------------------------ data plumbing
@dataclass
class CachedFrame:
    """One merge-cache frame, reduced to what head training needs (points, fragments, 2D GT)."""
    scene: str
    frame_id: int
    xyz: np.ndarray            # (N,3) float32, world coords
    pixels: np.ndarray         # (N,2) int
    groups: np.ndarray         # (N,) int64 compact fragment ids
    gt: np.ndarray             # (N,) int64 instance id (0 = unlabelled)
    group_major: np.ndarray    # (G,) majority instance per fragment
    group_purity: np.ndarray   # (G,) purity among labelled points


def load_cached_frame(path):
    with np.load(path) as z:
        scene, fid = Path(path).stem.rsplit('_', 1)
        return CachedFrame(scene, int(fid), np.asarray(z['xyz'], np.float32),
                           np.asarray(z['pixels'], np.int64), np.asarray(z['groups'], np.int64),
                           np.asarray(z['gt_instance'], np.int64),
                           np.asarray(z['group_major'], np.int64),
                           np.asarray(z['group_purity'], np.float32))


def load_rgb(image_root, scene, frame_id):
    """The frame's colour image from the ESAM 2D export (RGB uint8, same source as the cache)."""
    im = cv2.imread(str(Path(image_root) / scene / 'color' / f'{frame_id}.jpg'), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f'{image_root}/{scene}/color/{frame_id}.jpg')
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def sample_points(rng, gt, per_frame):
    """Up to ``per_frame`` labelled point indices of one frame (uniform over labelled points)."""
    lab = np.flatnonzero(gt > 0)
    if len(lab) <= per_frame:
        return lab
    return np.sort(rng.choice(lab, size=int(per_frame), replace=False))


def pick_frames(rng, fids, k, max_gap=3):
    """k distinct frame ids of one scene: a chain of 1..``max_gap`` jumps in cache order."""
    n = len(fids)
    k = min(int(k), n)
    for _ in range(20):
        chain = [int(rng.integers(n))]
        while len(chain) < k:
            cand = [j for j in range(n) if 1 <= abs(j - chain[-1]) <= max_gap and j not in chain]
            if not cand:
                break
            chain.append(cand[int(rng.integers(len(cand)))])
        if len(chain) == k:
            return [int(fids[i]) for i in chain]
    return [int(fids[i]) for i in rng.choice(n, size=k, replace=False)]


# ------------------------------------------------------------------ training
def train_instance_head(cfg, device='cuda', seed=0, scenes=None, cache_dir=None, image_root=None,
                        log=print, epochs=None, steps_per_epoch=None, raw=None, in_channels=None):
    """Train one head with the PartField-style objective. Returns ``(head, log_tau, info)``."""
    p4 = cfg['phase4']
    tr = p4['train']
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cache_dir = Path(cache_dir or Path(cfg['cache_root']) / 'seq' / 'train')
    image_root = Path(image_root or cfg['dataset']['root'])
    epochs = int(tr['epochs'] if epochs is None else epochs)
    steps = int(tr['steps_per_epoch'] if steps_per_epoch is None else steps_per_epoch)
    raw = bool(tr.get('raw_forward', True) if raw is None else raw)

    by_scene = defaultdict(list)
    for f in sorted(cache_dir.glob('*.npz')):
        by_scene[f.stem.rsplit('_', 1)[0]].append(f)
    scenes = [s for s in (scenes if scenes is not None else sorted(by_scene)) if s in by_scene]
    fids_of = {s: sorted(int(p.stem.rsplit('_', 1)[1]) for p in by_scene[s]) for s in scenes}
    path_of = {(s, fid): p for s in scenes for p, fid in zip(by_scene[s], fids_of[s])}

    head = InstanceHead(in_channels=in_channels, width=int(p4['head']['width']),
                        out_dim=int(p4['head']['out_dim']),
                        conv_layers=int(p4['head']['conv_layers'])).to(device)
    log_tau = nn.Parameter(torch.tensor(float(np.log(tr['tau_init'])), device=device)) \
        if tr.get('learn_tau', True) else None
    params = list(head.parameters()) + ([log_tau] if log_tau is not None else [])
    opt = torch.optim.AdamW(params, lr=float(tr['lr']), weight_decay=float(tr['weight_decay']))
    fast = cfg.get('fastsam') or load_config(resolve_path(cfg['lifting_config']))['fastsam']
    enc = FastSAMEncoder(fast['checkpoint'], device, fast.get('imgsz', 640), raw=raw)
    log(f'head train: {len(scenes)} scenes, {epochs} epochs x {steps} steps, k={tr["frames_per_step"]} '
        f'frames/step, {tr["points_per_frame"]} pts/frame, params {head.num_params()}, raw={raw}')

    history = []
    for epoch in range(epochs):
        head.train()
        t0 = time.perf_counter()
        agg = defaultdict(float)
        n_ok = 0
        for _ in range(steps):
            scene = scenes[int(rng.integers(len(scenes)))]
            fids = fids_of[scene]
            k = int(np.clip(int(rng.integers(2, 5)), 1, len(fids)))
            picks = pick_frames(rng, fids, k, int(tr['max_gap']))
            frames = [load_cached_frame(path_of[(scene, fid)]) for fid in picks]
            with torch.no_grad():
                encs = [enc.encode(load_rgb(image_root, scene, fid)) for fid in picks]
            dense = [head(m) for m, _ in encs]

            feats, ids, frag_feats, frag_ids = [], [], [], []
            for fi, (fr, (_, geom), dm) in enumerate(zip(frames, encs, dense)):
                sel = sample_points(rng, fr.gt, int(tr['points_per_frame']))
                if len(sel) == 0:
                    continue
                pf = sample_features(dm, fr.pixels[sel], geom)
                lab = np.flatnonzero(fr.gt > 0)
                pool_pix = fr.pixels[lab]
                with torch.no_grad():
                    pool_feat = sample_features(dm.detach(), pool_pix, geom)
                    a_xyz = torch.as_tensor(fr.xyz[sel], device=device)
                    p_xyz = torch.as_tensor(fr.xyz[lab], device=device)
                    a_ids = torch.as_tensor(fr.gt[sel], device=device)
                    p_ids = torch.as_tensor(fr.gt[lab], device=device)
                    h3 = nearest_different_instance(a_xyz, a_ids, p_xyz, p_ids,
                                                    tr['hard_3d_dist'], tr['hard_neg_k'])
                    hf = most_similar_different_instance(pf.detach(), a_ids, pool_feat, p_ids,
                                                         tr['hard_neg_k'])
                    mined = _collect_mined(torch.cat([h3, hf], 1), int(tr.get('mined_cap', 1024)))
                if len(mined):
                    extra = sample_features(dm, pool_pix[mined], geom)
                    pf = torch.cat([pf, extra], 0)
                    sel = np.concatenate([sel, lab[mined]])
                feats.append(pf)
                ids.append(fr.gt[sel])
                if float(tr.get('frag_weight', 0.0)) > 0:
                    ff, fi_ids = fragment_pool(dm, geom, fr, sel)
                    frag_feats.append(ff)
                    frag_ids.append(fi_ids)
            if not feats:
                continue
            feat = torch.cat(feats, 0)
            tau = log_tau.exp().clamp(0.02, 0.5) if log_tau is not None else float(tr['tau_init'])
            loss, st = multi_positive_infonce(feat, np.concatenate(ids), tau)
            wf = float(tr.get('frag_weight', 0.0))
            if wf > 0 and frag_feats and sum(len(f) for f in frag_feats) > 1:
                lf, _ = multi_positive_infonce(torch.cat(frag_feats, 0), np.concatenate(frag_ids), tau)
                loss = loss + wf * lf
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            agg['loss'] += float(loss)
            for key in ('pos_cos', 'neg_cos'):
                agg[key] += st[key] if st[key] == st[key] else 0.0
            agg['pool'] += len(feat)
            n_ok += 1
        nb = max(n_ok, 1)
        rec = dict(epoch=epoch + 1, loss=agg['loss'] / nb, pos_cos=agg['pos_cos'] / nb,
                   neg_cos=agg['neg_cos'] / nb, pool=agg['pool'] / nb, steps=n_ok,
                   tau=float(tau) if torch.is_tensor(tau) else float(tau),
                   sec=time.perf_counter() - t0)
        history.append(rec)
        log(f'  epoch {rec["epoch"]:3d}/{epochs} loss {rec["loss"]:.4f} pos_cos {rec["pos_cos"]:.3f} '
            f'neg_cos {rec["neg_cos"]:.3f} pool {rec["pool"]:.0f} tau {rec["tau"]:.3f} '
            f'{rec["sec"]:.0f}s')
    head.eval()
    info = dict(history=history, seed=seed, params=head.num_params(), scenes=len(scenes),
                config={k: v for k, v in p4.items()}, raw=raw)
    return head, log_tau, info


# ------------------------------------------------------------------ checkpoints / feature cache
def save_head(path, head, log_tau=None, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=head.state_dict(), out_dim=head.out_dim,
                    width=head.proj[head.names[0]].out_channels,
                    conv_layers=len([m for m in head.body if isinstance(m, nn.Conv2d)]),
                    in_channels={n: head.proj[n].in_channels for n in head.names},
                    log_tau=None if log_tau is None else float(log_tau.detach()),
                    **(extra or {})), path)


def load_head(path, device='cpu'):
    """Rebuild an ``InstanceHead`` from a checkpoint (architecture stored alongside the weights)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    head = InstanceHead(in_channels=ck.get('in_channels'), width=int(ck['width']),
                        out_dim=int(ck['out_dim']), conv_layers=int(ck['conv_layers'])).to(device)
    head.load_state_dict(ck['state_dict'])
    head.eval()
    return head, ck


def build_feature_cache(head, encoder, files, image_root, out_dir, log=print, overwrite=False,
                        limit=0, device='cuda', center=False):
    """Per-frame fragment instance features -> ``<out_dir>/<scene>_<fid>.npz {feat (G,32) fp16}``.

    Reuses the cache's pixels/groups exactly, so the fragment partition is untouched. Resumable.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(files)[:limit or None]
    head.eval()
    done = skipped = 0
    t0 = time.perf_counter()
    for f in files:
        dst = out_dir / f.name
        if dst.exists() and not overwrite:
            skipped += 1
            continue
        with np.load(f) as z:
            scene, fid = Path(f).stem.rsplit('_', 1)
            pixels, groups, num = np.asarray(z['pixels'], np.int64), np.asarray(z['groups'], np.int64), len(z['sizes'])
        rgb = load_rgb(image_root, scene, fid)
        maps, geom = encoder.encode(rgb)
        with torch.inference_mode():
            dense = head(maps)
            pf = sample_features(dense, pixels, geom)
            if center:   # F1.1: FastSAM-derived features carry a large per-frame common component
                pf = pf - pf.mean(0, keepdim=True)
                pf = pf / pf.norm(dim=1, keepdim=True).clamp_min(1e-6)
            frag = fragment_mean_features(pf, groups, num)
        np.savez_compressed(dst, feat=frag.half().cpu().numpy())
        done += 1
        if done % 500 == 0 or done == len(files):
            el = time.perf_counter() - t0
            log(f'  feats {done}/{len(files)} (+{skipped} skipped) {el / max(done, 1) * 1e3:.0f} ms/frame')
    stats = dict(frames=done, skipped=skipped, seconds=time.perf_counter() - t0,
                 disk_bytes=sum(p.stat().st_size for p in out_dir.glob('*.npz')))
    log(f'feature cache: {done} built, {skipped} skipped, {stats["seconds"] / 60:.1f} min, '
        f'{stats["disk_bytes"] / 2**20:.1f} MiB')
    return stats


def load_feature_cache(path):
    """(G,32) float32 fragment features of one frame (unnormalised means of unit point features)."""
    with np.load(path) as z:
        return np.asarray(z['feat'], np.float32)
