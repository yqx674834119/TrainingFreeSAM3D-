"""Unified per-frame grouping (stage 2): one model scores the intra-frame edges, the block <-> object
edges and the object <-> object edges, and predicts a per-node *quality* that ranks the output.

The flow (all decisions come from ``merging.model.UnifiedMergeNet``; thresholds are calibrated on
the 60 holdout scenes and never on the 16 val scenes):

1. the current frame's cached intra-frame edges are scored and average-linked into *blocks*
   (``WindowConfig.threshold``);
2. every block joins the touching object with the highest edge probability at or above
   ``bo_threshold`` (several blocks may join one object); a block that matches nothing opens a new
   object when it holds at least ``admit_min_points`` points;
3. two touching objects merge when their edge probability passes ``oo_threshold``;
4. the model's node head gives every object a quality (predicted IoU with the GT instance); the
   memory keeps it and ``object_attrs`` exposes it, so the output stage can rank by it and keep
   only NMS (no mask is dropped).

The same weights serve steps 1 and 2/3; step 1 sees the current fragments alone (node type 0) with
the cached intra-frame edges, steps 2/3 see the blocks (type 1) and the memory's objects (type 2)
with the dilated-contact pairs between them. Block <-> block pairs are step 1's business and are
deliberately not edges of the second graph. Everything is tensor/matrix based (AGENTS.md §2);
``WindowMemory`` owns the state and this module only computes.

``assign`` returns the per-node assignment plus an ``info`` dict holding every product of the frame
(blocks, both graphs exactly as the model saw them, edge probabilities, node qualities, the object
vote tables and the merge events). The stage-2 data collector and the analysis tools read that dict;
``RuleAdapter`` lets stage 1's rule drive the same machinery, so round A of the data collection
produces graphs from the identical code path.
"""
import time

import numpy as np
import torch

from ..merging.graph import edge_feature_names
from . import two_step
from .pair_features import (NodeStats, geom_pair_names, node_relations, pair_feature_names,
                            pair_features, rel_feature_names)
from .voxel import dilate_sets, overlap_matrix

# node types: 0 current-frame fragment, 1 current-frame block, 2 history object
FRAGMENT, BLOCK, OBJECT = 0, 1, 2
# edge types: 0 intra-frame, 1 block <-> object, 2 object <-> object
INTRA, BLOCK_OBJECT, OBJECT_OBJECT = 0, 1, 2
N_INTRA = len(edge_feature_names())          # 14 cached intra-frame columns, zero off intra edges
N_PAIR = len(pair_feature_names())           # 14 + 15
_CONTAIN_A_IN_B = pair_feature_names().index('contain_a_in_b')
_CONTAIN_B_IN_A = pair_feature_names().index('contain_b_in_a')


def block_stats(block_labels, stats, num_blocks):
    """(num_blocks,) ``NodeStats``: a block is the sum of its member fragments' statistics.

    Every field is reduced with its own operator (counts/feature sums add, bboxes take the
    extremes), so a block is featurised exactly like any other node. Matrix ops only.
    """
    dev = stats.n.device
    blk = torch.as_tensor(np.asarray(block_labels, np.int64), dtype=torch.int64, device=dev)

    def red(name, op, init):
        src = getattr(stats, name)
        out = torch.full((int(num_blocks),) + tuple(src.shape[1:]), init, dtype=src.dtype, device=dev)
        if op == 'sum':
            return out.zero_().index_add_(0, blk, src)
        idx = blk.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
        return out.scatter_reduce_(0, idx, src, op)

    return NodeStats(
        n=red('n', 'sum', 0.0), sum_xyz=red('sum_xyz', 'sum', 0.0), sum_xx=red('sum_xx', 'sum', 0.0),
        sum_rgb=red('sum_rgb', 'sum', 0.0), lo=red('lo', 'amin', float('inf')),
        hi=red('hi', 'amax', float('-inf')), feat_sum=red('feat_sum', 'sum', 0.0),
        n_feat=red('n_feat', 'sum', 0.0), first=red('first', 'amin', float('inf')),
        last=red('last', 'amax', float('-inf')), n_frames=red('n_frames', 'sum', 0.0),
        conf_sum=red('conf_sum', 'sum', 0.0))


def object_state(mem, with_votes=False):
    """(obj_ids, stats, vox[, votes]) of the memory's objects in ``render`` row order.

    An object's extent is its frozen part *plus* the fragments of the revisable window that belong
    to it - the same extent ``render`` / ``object_voxel_sets`` use, so a node's geometry, features
    and cells all describe what the object currently covers. ``votes`` (optional) is the merged GT
    vote table (frozen part plus window members); it feeds labels only, never a merge decision.
    """
    obj_ids = sorted(mem.objects)
    if not obj_ids:
        empty = ([], NodeStats.zeros(0, mem.feat_dim, mem.device), [])
        return empty + ([],) if with_votes else empty
    stats, vox, votes = [], [], []
    for oid in obj_ids:
        rec = mem.objects[oid]
        acc, parts = rec['stats'], [rec['vox']]
        v = dict(rec['votes_gt'])
        for e in mem.window:
            for fr in e['frags']:
                if fr['obj'] == oid:
                    acc = acc.add(fr['stats'])
                    parts.append(fr['vox'])
                    for k, c in fr['votes'].items():
                        v[k] = v.get(k, 0) + c
        stats.append(acc)
        vox.append(torch.unique(torch.cat(parts)))
        votes.append(v)
    out = (obj_ids, NodeStats.cat(stats), vox)
    return out + (votes,) if with_votes else out


class UnifiedGrouping:
    """Steps 1-3 above, driven by one ``UnifiedMergeNet``.

    ``model`` is any callable with ``UnifiedMergeNet``'s forward signature (tests pass a stub,
    round A passes a ``RuleAdapter``); ``std`` is the feature standardizer fitted on the training
    graphs. Thresholds are read from ``WindowConfig`` at call time, so a sweep only has to change
    the config.
    """

    def __init__(self, model, std, device='cuda'):
        self.model, self.std, self.device = model, std, device

    @torch.no_grad()
    def forward_graph(self, stats, z_ref, node_type, edges, edge_feat, edge_type, n_frames, age, rel=None):
        """One model call on a (nodes, edges) selection.

        Returns ``(edge probability, node quality, node_geom, node_feat)`` - the last two are the
        *unstandardised* node inputs, which the data collector stores verbatim. ``rel`` (G, 8) is
        the stage-3 relational node block; it is standardised and passed only to models that take
        it (``rel_dim > 0``), so the stage-2 adapters keep their eight-argument contract.
        """
        dev = self.device
        geom, feat = stats.geom(z_ref), stats.feat()
        geom_s, ef_s = self.std.apply(geom, edge_feat)
        t = lambda x, dt: (x.to(device=dev, dtype=dt) if torch.is_tensor(x)
                           else torch.as_tensor(np.asarray(x), dtype=dt, device=dev))
        args = (geom_s, feat, t(edges, torch.int64), ef_s, t(node_type, torch.int64),
                t(n_frames, torch.float32), t(age, torch.float32), t(edge_type, torch.int64))
        rel_s = None
        if rel is not None and int(getattr(self.model, 'rel_dim', 0)) > 0:
            rel_s = self.std.apply_rel(rel) if hasattr(self.std, 'apply_rel') else rel
        logit, q = self.model(*args, rel_s) if rel_s is not None else self.model(*args)
        return torch.sigmoid(logit), torch.sigmoid(q), geom, feat

    def assign(self, mem, graph, node_prev, node_n, num_cur, n_own, n_vox, f_vox_list, stats_cur, pos):
        """Per-node object ids (same contract as ``WindowMemory._assign``) plus the frame's products."""
        cfg, dev = mem.cfg, mem.device
        n_pair = N_PAIR + (len(geom_pair_names()) if cfg.geom_pair else 0)
        t_bo = cfg.bo_threshold if cfg.bo_threshold is not None else cfg.cross_threshold
        t_oo = cfg.oo_threshold if cfg.oo_threshold is not None else cfg.cross_threshold
        info = {}

        # ---- step 1: the current fragments' cached intra-frame edges -> blocks
        t0 = time.perf_counter()
        sel = two_step.as_bool_np(graph.has_intra)
        intra = np.asarray(graph.edges, np.int64).reshape(-1, 2)[sel]
        geom_i, feat_i = stats_cur.geom(graph.z_ref), stats_cur.feat()
        if len(intra):
            pf_i = graph.pair_feat[torch.as_tensor(sel, device=graph.pair_feat.device)].to(dev)
            prob_i, q_frag, geom_i, feat_i = self.forward_graph(
                stats_cur, graph.z_ref, np.zeros(num_cur, np.int64), intra, pf_i,
                np.zeros(len(intra), np.int64), np.ones(num_cur, np.float32), np.zeros(num_cur, np.float32))
            prob_i = prob_i.cpu().numpy()
        else:
            pf_i, prob_i = torch.zeros((0, n_pair), device=dev), np.zeros(0, np.float64)
            q_frag = torch.zeros(num_cur, device=dev)
        block = (two_step.intra_blocks(num_cur, intra, prob_i, cfg.threshold, cfg.linkage)
                 if num_cur else np.zeros(0, np.int64))
        num_blocks = int(block.max()) + 1 if num_cur else 0
        times = {'score_ms': (time.perf_counter() - t0) * 1e3}
        intra_info = dict(edges=intra, pair_feat=pf_i.cpu().numpy(), node_geom=geom_i.cpu().numpy(),
                          node_feat=feat_i.cpu().numpy(), n_frames=np.ones(num_cur, np.float32),
                          age=np.zeros(num_cur, np.float32))

        # ---- the cross graph: blocks (type 1) + the memory's objects (type 2)
        t0 = time.perf_counter()
        obj_ids, o_stats, o_vox, o_votes = object_state(mem, with_votes=True)
        num_obj = len(obj_ids)
        b_own, b_vox = two_step.block_voxel_sets(block, f_vox_list, dev) if num_blocks else \
            (torch.zeros(0, dtype=torch.int64, device=dev), torch.zeros(0, dtype=torch.int64, device=dev))
        parts = ([block_stats(block, stats_cur, num_blocks)] if num_blocks else []) + ([o_stats] if num_obj else [])
        if not parts:                                       # nothing to score: no fragments, no objects
            times['cluster_ms'] = (time.perf_counter() - t0) * 1e3
            times['assign_ms'] = 0.0
            info.update(times=times, block=block, num_blocks=num_blocks, prob_intra=prob_i,
                        q_frag=q_frag.cpu().numpy(), obj_ids=obj_ids, obj_votes=o_votes,
                        block_n=np.zeros(0), edges=np.zeros((0, 2), np.int64), edge_type=np.zeros(0, np.int64),
                        prob_cross=np.zeros(0), q_cross=np.zeros(0), block_obj=np.full(0, -1, np.int64),
                        new_obj=np.full(0, -1, np.int64), remap={}, merged=[], intra=intra_info,
                        cross=dict(edges=np.zeros((0, 2), np.int64), pair_feat=np.zeros((0, n_pair), np.float32),
                                   node_geom=np.zeros((0, 11), np.float32), node_feat=np.zeros((0, mem.feat_dim), np.float32),
                                   node_type=np.zeros(0, np.int64), n_frames=np.zeros(0, np.float32),
                                   age=np.zeros(0, np.float32),
                                   node_rel=np.zeros((0, len(rel_feature_names())), np.float32)))
            return np.full(graph.num_nodes, -1, np.int64), info
        cross_stats = parts[0] if len(parts) == 1 else NodeStats.cat(parts)
        node_type_c = np.r_[np.full(num_blocks, BLOCK, np.int64), np.full(num_obj, OBJECT, np.int64)]
        n_frames_c = np.r_[np.ones(num_blocks, np.float32),
                           o_stats.n_frames.cpu().numpy().astype(np.float32) if num_obj else np.zeros(0, np.float32)]
        age_c = np.r_[np.zeros(num_blocks, np.float32),
                      (pos - o_stats.last).clamp_min(0).cpu().numpy().astype(np.float32) if num_obj
                      else np.zeros(0, np.float32)]
        owners, voxels = [], []
        if num_blocks:
            owners.append(b_own); voxels.append(b_vox)
        if num_obj:
            owners.append(torch.repeat_interleave(
                torch.arange(num_obj, device=dev) + num_blocks,
                torch.tensor([len(v) for v in o_vox], device=dev)))
            voxels.append(torch.cat(o_vox))
        c_own, c_vox = torch.cat(owners), torch.cat(voxels)
        n_cross = num_blocks + num_obj
        inter0, size_c, _ = overlap_matrix(c_own, c_vox, c_own, c_vox, n_cross, n_cross)
        d_own, d_vox = dilate_sets(c_own, c_vox, cfg.touch_dilation)
        s1, _, _ = overlap_matrix(c_own, c_vox, d_own, d_vox, n_cross, n_cross)
        inter1 = torch.maximum(s1, s1.t())
        iu = torch.triu_indices(n_cross, n_cross, 1, device=dev)
        a, b = iu[0], iu[1]
        keep = inter1[a, b] > 0
        if cfg.cand_dist is not None and cfg.cand_cos is not None:
            # stage 3 candidate expansion: a block <-> object pair counts as a candidate even
            # without voxel contact when the centroids are close and the appearance matches (the
            # "the object exists but is not a candidate" failure of the stage-2 probe)
            cent, feat_c = cross_stats.centroid(), cross_stats.feat()
            near = ((cent[a] - cent[b]).norm(dim=1) < float(cfg.cand_dist)) & \
                   ((feat_c[a] * feat_c[b]).sum(1) > float(cfg.cand_cos))
            near &= (a < num_blocks) != (b < num_blocks)   # block <-> object pairs only
            keep |= near
        if num_blocks:                                       # block <-> block: decided by step 1
            keep &= ~((a < num_blocks) & (b < num_blocks))
        a, b = a[keep], b[keep]
        is_block = (a < num_blocks) if num_blocks else torch.zeros(len(a), dtype=torch.bool, device=dev)
        etype = np.where(is_block.cpu().numpy() != (b < num_blocks).cpu().numpy(), BLOCK_OBJECT, OBJECT_OBJECT)
        age_t = torch.as_tensor(age_c, device=dev)
        gm = cross_stats.geom(graph.z_ref) if cfg.geom_pair else None
        pf_c = pair_features(cross_stats, a, b, inter0[a, b], inter1[a, b], size_c, size_c,
                             (age_t[a] - age_t[b]).abs(),
                             torch.zeros(len(a), dtype=torch.bool, device=dev),
                             torch.zeros(len(a), dtype=torch.bool, device=dev),
                             (a >= num_blocks).float() + (b >= num_blocks).float(),
                             geom_a=None if gm is None else gm[a], geom_b=None if gm is None else gm[b])
        join = torch.zeros((n_cross, 3), device=dev)         # stage-3 join statistics per node
        for k, oid in enumerate(obj_ids):
            rec = mem.objects[oid]
            join[num_blocks + k, 0] = float(rec.get('join_n', 0))
            join[num_blocks + k, 1] = float(rec.get('join_sum', 0.0))
            join[num_blocks + k, 2] = float(rec.get('join_min', 1.0))
        rel = node_relations(cross_stats, a, b, inter0[a, b], size_c,
                             torch.as_tensor(node_type_c, device=dev), join)
        prob_c, q_c, geom_c, feat_c = self.forward_graph(
            cross_stats, graph.z_ref, node_type_c, torch.stack([a, b], 1), pf_c, etype, n_frames_c, age_c,
            rel=rel)
        prob_c, q_c = prob_c.cpu().numpy(), q_c.cpu().numpy()
        times['score_ms'] += (time.perf_counter() - t0) * 1e3

        # ---- step 2: every block joins its best touching object above the threshold
        t0 = time.perf_counter()
        b_n = np.bincount(block, weights=node_n[:num_cur], minlength=num_blocks).astype(np.float64) \
            if num_blocks else np.zeros(0)
        block_obj = np.full(num_blocks, -1, np.int64)
        if num_blocks and num_obj:
            b2o = np.full((num_blocks, num_obj), -1.0)
            m = etype == BLOCK_OBJECT
            b2o[a.cpu().numpy()[m], b.cpu().numpy()[m] - num_blocks] = prob_c[m]
            best = b2o.argmax(1)
            best_p = b2o.max(1)
            ok = best_p >= t_bo
            block_obj[ok] = np.asarray(obj_ids, np.int64)[best[ok]]
            for bi in np.flatnonzero(ok):                    # stage-3 join statistics (a node input)
                rec = mem.objects[int(block_obj[bi])]
                p = float(best_p[bi])
                rec['join_n'] = int(rec.get('join_n', 0)) + 1
                rec['join_sum'] = float(rec.get('join_sum', 0.0)) + p
                rec['join_min'] = min(float(rec.get('join_min', 1.0)), p)
        # ---- step 3: touching objects whose edge passes the threshold merge into the smallest id
        remap = {int(o): int(o) for o in obj_ids}
        merged = []
        if num_obj > 1:
            parent = list(range(num_obj))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            m = (etype == OBJECT_OBJECT) & (prob_c >= t_oo)
            for i in np.flatnonzero(m):
                u, v = find(int(a[i]) - num_blocks), find(int(b[i]) - num_blocks)
                if u != v:
                    parent[v] = u
            groups = {}
            for k in range(num_obj):
                groups.setdefault(find(k), []).append(k)
            for members in groups.values():
                if len(members) < 2:
                    continue
                keep_id = min(int(obj_ids[k]) for k in members)
                w = np.asarray([float(o_stats.n[k]) for k in members])
                q = float((np.asarray([q_c[num_blocks + k] for k in members]) * w).sum() / max(w.sum(), 1e-9))
                for k in members:
                    oid = int(obj_ids[k])
                    if oid != keep_id:
                        mem._merge_objects(keep_id, oid)
                    remap[oid] = keep_id
                mem.objects[keep_id]['quality'] = q
                merged.append(dict(keep=keep_id, members=[int(obj_ids[k]) for k in members], quality=q))
        # ---- blocks that matched nothing: a new object when they hold enough points
        new_obj = np.full(num_blocks, -1, np.int64)
        for bi in np.flatnonzero((block_obj < 0) & (b_n >= cfg.admit_min_points)):
            new_obj[bi] = mem._new_object()
            mem.objects[int(new_obj[bi])]['quality'] = float(q_c[bi])
        times['cluster_ms'] = (time.perf_counter() - t0) * 1e3

        # ---- per-node assignment (fragments -> their block's object; past nodes keep theirs)
        t0 = time.perf_counter()
        out = np.full(graph.num_nodes, -1, np.int64)
        if num_cur:
            chosen = np.where(block_obj >= 0, [remap.get(int(o), int(o)) for o in block_obj], new_obj)
            out[:num_cur] = chosen[block]
        kind = np.asarray(graph.node_type, np.int64)
        prev = np.asarray(node_prev, np.int64)
        for i in np.flatnonzero((kind == 1) | (kind == 2)):
            out[i] = remap.get(int(prev[i]), int(prev[i]))
        if num_obj:                                          # refresh the stored quality of every survivor
            for k, oid in enumerate(obj_ids):
                if int(oid) in mem.objects and int(oid) == remap[int(oid)]:
                    mem.objects[int(oid)]['quality'] = float(q_c[num_blocks + k])
        times['assign_ms'] = (time.perf_counter() - t0) * 1e3

        info.update(times=times, block=block, num_blocks=num_blocks, prob_intra=prob_i,
                    q_frag=q_frag.cpu().numpy(), obj_ids=obj_ids, obj_votes=o_votes, block_n=b_n,
                    edges=np.stack([a.cpu().numpy(), b.cpu().numpy()], 1), edge_type=etype,
                    prob_cross=prob_c, q_cross=q_c, block_obj=block_obj, new_obj=new_obj,
                    remap=remap, merged=merged, intra=intra_info,
                    cross=dict(edges=np.stack([a.cpu().numpy(), b.cpu().numpy()], 1),
                               pair_feat=pf_c.cpu().numpy(), node_geom=geom_c.cpu().numpy(),
                               node_feat=feat_c.cpu().numpy(), node_type=node_type_c,
                               n_frames=n_frames_c, age=age_c, node_rel=rel.cpu().numpy()))
        return out, info


class IdentityStandardizer:
    """Passthrough standardizer for adapters that bring their own (``RuleAdapter``)."""

    def apply(self, geom, ef):
        return geom, ef

    def apply_rel(self, rel):
        return rel

    def state(self):
        return {}


class RuleAdapter:
    """Stage 1's rule dressed as a 'model', so round A of the data collection drives the *same*
    ``WindowMemory`` / graph construction as inference (the brief's "A 轮由第 1 阶段的规则驱动记忆").

    Intra-frame edges get the trained ``GroupMergeNet`` probability (the same call the per-frame
    pipeline makes); block <-> object edges get stage 1's ``p2a`` score,
    ``max(contain_a_in_b, contain_b_in_a)`` of the pair features; object <-> object edges get 0,
    i.e. objects never merge - the property that separates stage 1's two-step grouping from
    stage 2's. Pass ``IdentityStandardizer()`` as the flow's ``std``: this adapter
    applies ``GroupMergeNet``'s own standardizer to the 11 geometry columns and the first 14 edge
    columns it was trained on.
    """

    def __init__(self, group_model, group_std, device='cuda'):
        self.model, self.std, self.device = group_model, group_std, device

    @torch.no_grad()
    def __call__(self, geom, feat, edges, edge_feat, node_type, n_frames, age, edge_type):
        dev = geom.device
        prob = torch.zeros(len(edges), device=dev)
        intra = edge_type == 0
        if bool(intra.any()):
            g, ef = self.std.apply(geom, edge_feat[:, :N_INTRA])
            prob[intra] = torch.sigmoid(self.model(g, feat, edges[intra], ef[intra]))
        bo = edge_type == BLOCK_OBJECT
        if bool(bo.any()):
            # stage 1's p2a: max(contain_a_in_b, contain_b_in_a) over the (block, object) pair
            prob[bo] = torch.maximum(edge_feat[bo, _CONTAIN_A_IN_B], edge_feat[bo, _CONTAIN_B_IN_A])
        # OBJECT_OBJECT stays 0: stage 1 never merges two objects
        p = prob.clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p)), torch.zeros(len(geom), device=dev)


class SingleFrameAdapter:
    """A ``UnifiedMergeNet`` dressed as ``GroupMergeNet`` for the per-frame protocol.

    ``merging.evaluate.evaluate`` scores one frame's cached intra-frame graph at a time and expects
    ``model(geom, feat, edges, edge_feat, return_objectness=True)``. The unified model needs the rest
    of its inputs, so this fills them with their single-frame values: node type 0 (current fragment),
    one observation, age 0, intra-frame edges. ``has_objectness`` maps onto the quality head, which
    is trained on the objectness label for exactly these fragment nodes, so the stage-1 protocol
    reads it unchanged - this is what makes the degradation check (history off) a fair comparison.
    """

    has_objectness = True

    def __init__(self, model):
        self.model = model

    def num_params(self):
        return self.model.num_params()

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def to(self, *a, **kw):
        return self

    def __call__(self, geom, feat, edges, edge_feat, return_objectness=False):
        dev = geom.device
        logit, quality = self.model(
            geom, feat, edges, edge_feat,
            torch.zeros(len(geom), dtype=torch.int64, device=dev),   # node type: fragment
            torch.ones(len(geom), device=dev),                       # one observation
            torch.zeros(len(geom), device=dev),                      # age 0
            torch.zeros(len(edges), dtype=torch.int64, device=dev))  # intra-frame edges
        return (logit, quality) if return_objectness else logit

    forward = __call__


def make_unified_memory(cfg, model, std, device, **overrides):
    """A ``WindowMemory`` in ``grouping='unified'`` mode from the ``stream_v2`` config section."""
    from .window_memory import WindowConfig, WindowMemory
    wc = cfg['stream_v2']
    kw = dict(window=1, past_unit=wc['past_unit'], voxel=wc['voxel'], touch_dilation=wc['touch_dilation'],
              linkage=wc['linkage'], admit_min_points=wc['admit_min_points'],
              object_merge=bool(wc['object_merge']), purity=wc['purity'], grouping='unified',
              threshold=wc['threshold'], cross_threshold=wc['threshold'],
              cand_dist=wc.get('cand_dist'), cand_cos=wc.get('cand_cos'))
    kw.update(overrides)
    return WindowMemory(WindowConfig(**kw), unified=UnifiedGrouping(model, std, device), device=device)
