"""Stage-2 training: one small model for every merge decision (intra-frame, block <-> object,
object <-> object) plus a node quality head, trained on the graphs the collector saved.

Data: ``tools/window_unified_collect.py`` writes one npz per scene holding the intra graph (the
current frame's fragments) and the cross graph (blocks + history objects) as concatenated per-frame
tensors with ``node_ptr`` / ``edge_ptr`` offsets. A training step samples *frames* across scenes and
concatenates them into one disjoint graph - edges are local to their frame, so shifting them by the
frame's node offset is all the batching needs - and hands the model exactly what inference hands it.

Losses:
* edge BCE, balanced *across the three edge types* (each type contributes equally however many edges
  it has), masked to the edges whose both ends are pure and share a majority instance;
* node quality, the MSE of ``sigmoid(quality_logit)`` against the final voxel best IoU for history
  objects and against the objectness label for fragments and blocks (one head: "IoU with my GT
  instance", which is 0 for a fragment that is not a thing).

Three seeds; the checkpoint holds the model, the fitted standardizer and the training config, and
``load_unified_checkpoint`` mirrors ``window_scores.load_group_merge_net``'s contract.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..merging.data import Standardizer
from ..merging.graph import NODE_GEOM
from ..merging.model import UnifiedMergeNet, init_from_group_merge_net
from .pair_features import rel_feature_names
from .unified_flow import N_PAIR

TAGS = ('i', 'c')
_KEYS = ('geom', 'feat', 'n_frames', 'age', 'kind', 'edges', 'pf', 'etype', 'ey', 'ev',
         'ny', 'nv', 'niou', 'niou_valid', 'rel')


@dataclass
class FrameRef:
    """One frame of one graph: where it lives and how big it is."""
    file: int
    frame: int
    tag: str
    n_node: int
    n_edge: int


class GraphSet:
    """The collected per-frame graphs, read from the scene npz files (one file cached at a time).

    Frames are the unit of sampling: ``batches`` walks the scenes in a shuffled order but keeps a
    scene's frames together, so consecutive batches reuse the same decompressed file.
    """

    def __init__(self, dirs, log=print):
        self.files = sorted(str(p) for d in dirs for p in Path(d).glob('*.npz'))
        if not self.files:
            raise SystemExit(f'no graphs in {[str(d) for d in dirs]}')
        self.refs = []
        for fi, path in enumerate(self.files):
            with np.load(path) as z:
                nf = int(z['frames'])
                ptr_i, ptr_c = z['node_ptr_i'], z['node_ptr_c']
                ep_i, ep_c = z['edge_ptr_i'], z['edge_ptr_c']
                self.feat_dim = int(z['feat_dim'])
                pd = int(z['pf_i'].shape[1])
                if fi == 0:
                    self.pair_dim = pd
                elif pd != self.pair_dim:
                    raise SystemExit(f'pair feature width differs: {path} has {pd}, expected {self.pair_dim}')
            for t in range(nf):
                for tag, ptr, ep in (('i', ptr_i, ep_i), ('c', ptr_c, ep_c)):
                    if t + 1 >= len(ptr):        # an empty side (the intra-only cache) has no frames
                        continue
                    n_node = int(ptr[t + 1] - ptr[t])
                    if n_node == 0:              # nothing in this frame: no node, no edge, no signal
                        continue
                    self.refs.append(FrameRef(fi, t, tag, n_node, int(ep[t + 1] - ep[t])))
        self._cache_key = -1
        self._cache = {}
        cross = sum(1 for r in self.refs if r.tag == 'c' and r.n_node)
        log(f'graphs: {len(self.files)} scenes, {len(self.refs)} frames ({cross} cross frames with nodes), '
            f'feat_dim {self.feat_dim}')

    def __len__(self):
        return len(self.refs)

    def _load(self, fi):
        if self._cache_key != fi:
            with np.load(self.files[fi]) as z:
                self._cache = {k: z[k] for k in z.files}
            self._cache_key = fi
        return self._cache

    def stack(self, refs, device='cpu'):
        """One disjoint graph holding every frame of ``refs``.

        Returns tensors plus ``node_off`` / ``edge_off`` (per frame) and the assembled node target:
        the final voxel IoU for history objects, the objectness label for fragments and blocks.
        """
        parts = {k: [] for k in _KEYS}
        node_off, edge_off = [], []
        base = 0
        for r in refs:
            z = self._load(r.file)
            n0, n1 = int(z[f'node_ptr_{r.tag}'][r.frame]), int(z[f'node_ptr_{r.tag}'][r.frame + 1])
            e0, e1 = int(z[f'edge_ptr_{r.tag}'][r.frame]), int(z[f'edge_ptr_{r.tag}'][r.frame + 1])
            g = {k: z[f'{k}_{r.tag}'][n0:n1] for k in ('geom', 'feat', 'age', 'kind', 'ny', 'nv')}
            g['n_frames'] = z[f'nframes_{r.tag}'][n0:n1]        # the saved key drops the underscore
            g.update({k: z[f'{k}_{r.tag}'][e0:e1] for k in ('edges', 'pf', 'etype', 'ey', 'ev')})
            rel_key = f'rel_{r.tag}'                            # stage-3 columns; older files lack them
            if rel_key in z and len(z[rel_key]) == len(z[f'geom_{r.tag}']):
                g['rel'] = z[rel_key][n0:n1]
            else:
                g['rel'] = np.zeros((n1 - n0, len(rel_feature_names())), np.float32)
            if r.tag == 'c':
                g['niou'], g['niou_valid'] = z['niou_c'][n0:n1], z['niou_valid_c'][n0:n1]
            else:
                g['niou'] = np.full(n1 - n0, np.nan, np.float32)
                g['niou_valid'] = np.zeros(n1 - n0, bool)
            node_off.append(base)
            edge_off.append(np.full(e1 - e0, base, np.int64))
            base += n1 - n0
            for k in _KEYS:
                parts[k].append(g[k])
        cat = lambda k, dt: torch.as_tensor(np.concatenate(parts[k]), dtype=dt, device=device)
        out = dict(
            geom=cat('geom', torch.float32), feat=cat('feat', torch.float32),
            n_frames=cat('n_frames', torch.float32), age=cat('age', torch.float32),
            kind=cat('kind', torch.int64), pf=cat('pf', torch.float32),
            etype=cat('etype', torch.int64), ey=cat('ey', torch.float32),
            ev=cat('ev', torch.bool), niou=cat('niou', torch.float32),
            niou_valid=cat('niou_valid', torch.bool), rel=cat('rel', torch.float32),
            node_off=torch.as_tensor(node_off, dtype=torch.int64, device=device),
        )
        counts = np.diff(np.r_[node_off, base]).astype(np.int64)
        out['frame_id'] = torch.repeat_interleave(
            torch.arange(len(node_off), dtype=torch.int64, device=device),
            torch.as_tensor(counts, dtype=torch.int64, device=device))
        eoff = np.concatenate(edge_off) if edge_off else np.zeros(0, np.int64)
        out['edges'] = cat('edges', torch.int64) + torch.as_tensor(eoff, dtype=torch.int64,
                                                                  device=device)[:, None]
        obj = out['kind'] == 2
        out['target'] = torch.where(obj, out['niou'], cat('ny', torch.float32))
        out['nvalid'] = torch.where(obj, out['niou_valid'], cat('nv', torch.bool))
        return out

    def batches(self, frames_per_step, seed=0):
        """Lists of ``FrameRef``: scenes shuffled, a scene's frames kept together."""
        by_file = {}
        for r in self.refs:
            by_file.setdefault(r.file, []).append(r)
        order = np.random.default_rng(seed).permutation(len(self.files))
        buf = []
        for fi in order:
            buf.extend(by_file[int(fi)])
            while len(buf) >= frames_per_step:
                yield buf[:frames_per_step]
                buf = buf[frames_per_step:]
        if buf:
            yield buf


# ------------------------------------------------------------------ losses
def edge_loss(logit, y, valid, etype, num_types=3):
    """BCE balanced across edge types: every type contributes equally, whatever its edge count."""
    if len(logit) == 0:
        return logit.sum() * 0.0
    bce = F.binary_cross_entropy_with_logits(logit, y, reduction='none')
    terms = [bce[valid & (etype == t)].mean() for t in range(num_types)
             if bool((valid & (etype == t)).any())]
    return torch.stack(terms).mean() if terms else logit.sum() * 0.0


def node_loss(q_logit, target, valid):
    """MSE of the predicted quality (sigmoid) against the node target."""
    if len(q_logit) == 0 or not bool(valid.any()):
        return q_logit.sum() * 0.0
    return F.mse_loss(torch.sigmoid(q_logit[valid]), target[valid])


def rank_loss(q_logit, target, valid, frame_id):
    """Scene-internal pairwise ranking loss (stage 3).

    Within one frame, a node with a higher true IoU must score higher than one with a lower IoU:
    RankNet on every ordered pair of valid nodes (``softplus`` of the negated margin), ties
    skipped. Frames are a batch's scene units, marked by ``frame_id``; pairs never cross frames.
    """
    if len(q_logit) == 0 or not bool(valid.any()):
        return q_logit.sum() * 0.0
    same = frame_id[:, None] == frame_id[None, :]
    ok = same & valid[:, None] & valid[None, :]
    diff = target[:, None] - target[None, :]
    ok = ok & (diff != 0)
    if not bool(ok.any()):
        return q_logit.sum() * 0.0
    margin = (q_logit[:, None] - q_logit[None, :]) * torch.sign(diff)
    return F.softplus(-margin[ok]).mean()


# ------------------------------------------------------------------ diagnostics
def _auc(score, y):
    """Rank-based AUC with tie averaging (no sklearn)."""
    score, y = np.asarray(score, np.float64), np.asarray(y, bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(score, kind='stable')
    s, ranks = score[order], np.empty(len(score), np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _rank(x):
    order = np.argsort(x, kind='stable')
    r = np.empty(len(x), np.float64)
    r[order] = np.arange(len(x))
    return r


@torch.no_grad()
def diagnostics(model, std, graphs, device='cpu', frames_per_step=32, n_batches=8, seed=1234):
    """Per-edge-type AUC and the node-quality correlation on a fixed slice of the training graphs."""
    model.eval()
    logits, ys, valids, etypes, qs, tgt, qvalid = [], [], [], [], [], [], []
    for i, refs in enumerate(graphs.batches(frames_per_step, seed)):
        if i >= n_batches:
            break
        b = graphs.stack(refs, device)
        geom, ef = std.apply(b['geom'], b['pf'])
        logit, q = model(geom, b['feat'], b['edges'], ef, b['kind'], b['n_frames'], b['age'],
                         b['etype'], b['rel'], b['frame_id'])
        logits.append(logit.cpu().numpy()); ys.append(b['ey'].cpu().numpy())
        valids.append(b['ev'].cpu().numpy()); etypes.append(b['etype'].cpu().numpy())
        qs.append(q.cpu().numpy()); tgt.append(b['target'].cpu().numpy())
        qvalid.append(b['nvalid'].cpu().numpy())
    if not logits:
        return {}
    prob = 1 / (1 + np.exp(-np.concatenate(logits)))
    y, valid, etype = np.concatenate(ys), np.concatenate(valids), np.concatenate(etypes)
    q, t, qv = np.concatenate(qs), np.concatenate(tgt), np.concatenate(qvalid)
    names = ['intra', 'block_object', 'object_object']
    out = {'edge': {}}
    for k, name in enumerate(names):
        m = valid & (etype == k)
        out['edge'][name] = dict(n=int(m.sum()), pos_rate=float(y[m].mean()) if m.any() else None,
                                 mean_prob=float(prob[m].mean()) if m.any() else None,
                                 auc=_auc(prob[m], y[m]))
    qp = 1 / (1 + np.exp(-q))
    out['node'] = dict(n=int(qv.sum()), mse=float(np.mean((qp[qv] - t[qv]) ** 2)) if qv.any() else None)
    if int(qv.sum()) > 2:
        out['node']['spearman'] = float(np.corrcoef(_rank(qp[qv]), _rank(t[qv]))[0, 1])
    out['node']['target_mean'] = float(t[qv].mean()) if qv.any() else None
    return out


# ------------------------------------------------------------------ training
def fit_standardizer(graphs, limit=120, seed=0, log=print):
    """Per-column mean/std of the node geometry and the 29 edge columns, on a sample of the graphs."""
    idx = np.random.default_rng(seed).permutation(len(graphs.files))[:limit]
    g, e, r = [], [], []
    for fi in idx:
        z = graphs._load(int(fi))
        g.append(np.concatenate([z['geom_i'], z['geom_c']], 0))
        e.append(np.concatenate([z['pf_i'], z['pf_c']], 0))
        if 'rel_c' in z and len(z['rel_c']) == len(z['geom_c']):
            r.append(z['rel_c'])
    std = Standardizer(np.concatenate(g, 0), np.concatenate(e, 0),
                       np.concatenate(r, 0) if r else None)
    log(f'standardizer fitted on {len(idx)} scenes ({sum(len(a) for a in g)} nodes, '
        f'{sum(len(a) for a in e)} edges)')
    return std


def build_unified_model(cfg, feat_dim, device='cpu', log=print, pair_dim=N_PAIR):
    pc = cfg['phase2']
    p3 = cfg.get('phase3', {})
    rel_dim = len(rel_feature_names()) if p3.get('rel', False) else 0
    model = UnifiedMergeNet(NODE_GEOM, feat_dim, pair_dim, hidden=int(pc['hidden']),
                            feature_proj=int(pc['feature_proj']),
                            dropout=float(pc['dropout']), rel_dim=rel_dim,
                            attn_layers=int(p3.get('attn_layers', 0))).to(device)
    init_path = pc.get('init_from')
    if init_path and Path(init_path).exists():
        ck = torch.load(init_path, map_location='cpu', weights_only=False)
        copied = init_from_group_merge_net(model, ck['state_dict'], log=log)
        log(f'initialised {len(copied)} shared tensors from {init_path}')
    elif init_path:
        log(f'no {init_path}; training from scratch')
    return model


def train(graphs, cfg, device='cpu', seed=0, log=print, epochs=None, frames_per_step=None):
    """Train one model. Returns ``(model, standardizer, info)``."""
    pc = cfg['phase2']
    rank_weight = float(cfg.get('phase3', {}).get('rank_weight', 0.0))
    epochs = int(pc['epochs'] if epochs is None else epochs)
    frames_per_step = int(pc['frames_per_step'] if frames_per_step is None else frames_per_step)
    torch.manual_seed(seed)
    std = fit_standardizer(graphs, limit=int(pc.get('fit_files', 120)), seed=seed, log=log)
    model = build_unified_model(cfg, graphs.feat_dim, device, log=log, pair_dim=graphs.pair_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=float(pc['lr']), weight_decay=float(pc['weight_decay']))
    steps = max(1, -(-len(graphs) // frames_per_step))   # batches() always yields ceil(n / fps)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(pc['lr']), total_steps=epochs * steps)
    we, wn = float(pc['edge_weight']), float(pc['node_weight'])
    log(f'seed {seed}: params={model.num_params()} frames={len(graphs)} steps/epoch={steps} '
        f'epochs={epochs} frames/step={frames_per_step} rank_weight={rank_weight}')
    history = []
    for epoch in range(epochs):
        model.train()
        tot, ne, nn, nr, nb = 0.0, 0.0, 0.0, 0.0, 0
        for refs in graphs.batches(frames_per_step, seed + 1000 * epoch):
            b = graphs.stack(refs, device)
            geom, ef = std.apply(b['geom'], b['pf'])
            logit, q = model(geom, b['feat'], b['edges'], ef, b['kind'], b['n_frames'], b['age'],
                             b['etype'], b['rel'], b['frame_id'])
            le = edge_loss(logit, b['ey'], b['ev'], b['etype'])
            ln = node_loss(q, b['target'], b['nvalid'])
            lr_rank = rank_loss(q, b['target'], b['nvalid'], b['frame_id']) if rank_weight else \
                q.sum() * 0.0
            loss = we * le + wn * ln + rank_weight * lr_rank
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched.last_epoch < epochs * steps:        # OneCycleLR refuses to step past its total
                sched.step()
            tot += float(loss); ne += float(le); nn += float(ln); nr += float(lr_rank); nb += 1
        history.append(dict(loss=tot / max(nb, 1), edge=ne / max(nb, 1), node=nn / max(nb, 1),
                            rank=nr / max(nb, 1)))
        if epoch == 0 or (epoch + 1) % 2 == 0 or epoch + 1 == epochs:
            log(f'  epoch {epoch + 1:3d}/{epochs} loss {history[-1]["loss"]:.4f} '
                f'(edge {history[-1]["edge"]:.4f}, node {history[-1]["node"]:.4f}, '
                f'rank {history[-1]["rank"]:.4f})')
    model.eval()
    return model, std, dict(history=history, seed=seed, params=model.num_params(),
                            frames=len(graphs), feat_dim=graphs.feat_dim, pair_dim=graphs.pair_dim)


# ------------------------------------------------------------------ checkpoints
def save_unified_checkpoint(path, model, std, cfg, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), standardizer=std.state(), config=cfg,
                    feat_dim=int(model.proj.in_features),
                    pair_dim=int(model.edge_dim),
                    rel_dim=int(getattr(model, 'rel_dim', 0)),
                    attn_layers=int(getattr(model, 'attn_layers', 0)), **(extra or {})), path)


def load_unified_checkpoint(path, device='cpu'):
    """The stage-2 model plus its standardizer - same contract as ``load_group_merge_net``."""
    ck = torch.load(path, map_location=device, weights_only=False)
    pc = ck['config']['phase2']
    model = UnifiedMergeNet(NODE_GEOM, int(ck['feat_dim']), int(ck.get('pair_dim', N_PAIR)), hidden=int(pc['hidden']),
                            feature_proj=int(pc['feature_proj']),
                            dropout=float(pc['dropout']), rel_dim=int(ck.get('rel_dim', 0)),
                            attn_layers=int(ck.get('attn_layers', 0))).to(device)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model, Standardizer.from_state(ck['standardizer']), ck
