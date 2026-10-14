"""WindowMemory: intra-frame merging and cross-frame merging as one task.

Frame t's fragments and the *recent* past form a single graph:
nodes = [current fragments] + [window nodes: the fragments of the last ``window-1`` frames,
either as one block per object or one node per fragment] + [frozen objects touching them];
edges = the cached intra-frame edges plus every pair whose voxel sets touch (after dilation);
a scorer (stage 0: the GT oracle; later: GroupMergeNet) gives each edge a glue probability,
average linkage glues nodes into groups, and every group becomes / joins one object.
Frames that slide out of the window are frozen: their fragments' voxels enter the object's
frozen table and can no longer be reassigned. With ``window = 1`` the past is frozen
immediately and the memory degenerates to "current fragments + frozen objects".
"""
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..merging.cluster import LINKAGE
from ..merging.graph import edge_feature_names
from ..merging.labels import group_gt_votes
from . import two_step
from .pair_features import NodeStats, pair_features
from .voxel import dilate_sets, overlap_matrix, render_voxel_table, voxelize


@dataclass
class WindowConfig:
    window: int = 5                 # frames that stay revisable, counting the current one
    past_unit: str = 'block'        # 'block' = one node per (frame, object); 'fragment' = one node per fragment
    voxel: float = 0.05
    touch_dilation: int = 1         # two nodes are connected when their voxel sets touch after this dilation
    threshold: float = 0.5
    linkage: str = 'average_sparse'
    admit_min_points: int = 100     # all-current group: below this many points no new object is opened
    object_merge: bool = True       # allow two frozen objects to merge inside one group
    purity: float = 0.8             # oracle scoring threshold
    grouping: str = 'linkage'       # 'linkage' (stage 0/0c) | 'two_step' (stage 1) | 'unified' (stage 2)
    cross_rule: str = 'p2a'         # two_step step 2: 'p2a' = block containment, 'p2b' = that x clamp(cos,0,1)
    cross_threshold: float = 0.5    # two_step step 2 threshold (step 1 reuses ``threshold``)
    # 'unified' (stage 2, streaming/unified_flow.py): one model scores the intra-frame edges, the
    # block <-> object edges and the object <-> object edges. Steps 2/3 fall back to
    # ``cross_threshold`` when left at None, so a single shared threshold is the default reading.
    bo_threshold: float = None      # unified step 2: block -> object join threshold
    oo_threshold: float = None      # unified step 3: object <-> object merge threshold


@dataclass
class FragmentFrame:
    """One frame of the stream: the sampled points, their fragment ids, and the cached intra-frame graph."""
    scene: str
    frame_id: int
    xyz: np.ndarray            # (N,3) float32
    rgb: np.ndarray            # (N,3) 0-255
    groups: np.ndarray         # (N,) int64, compact 0..G-1
    node_feat: np.ndarray      # (G,C) float32
    edges: np.ndarray          # (E,2) int64, intra-frame adjacency (cache)
    edge_feat: np.ndarray      # (E,14) float32
    conf: np.ndarray           # (G,) float32, FastSAM confidence (0 for KMeans fill groups)
    gt: np.ndarray = None      # (N,) int64, 0 = unlabeled (oracle / training only)
    sem: np.ndarray = None     # (N,) int64, semantic id in the same space as ``gt`` (stuff filter)


@dataclass
class PairGraph:
    """Everything the scorer sees: nodes, edges, features, and the oracle/training labels."""
    edges: np.ndarray          # (E,2)
    pair_feat: torch.Tensor    # (E, 29) = 14 intra + 15 cross
    has_intra: torch.Tensor    # (E,) bool, edge comes from the cached intra-frame graph
    node_major: np.ndarray     # (G,) majority GT id (0 = none)
    node_purity: np.ndarray    # (G,) purity among labeled points
    node_n: np.ndarray         # (G,) point counts
    node_type: np.ndarray      # (G,) 0 = current fragment, 1 = past window node, 2 = frozen object
    node_prev_obj: np.ndarray  # (G,) previous object id (-1 for current fragments)
    node_geom: np.ndarray      # (G,11)
    node_feat: np.ndarray      # (G,C) member-mean, L2-normalised
    num_nodes: int
    z_ref: float
    frame_pos: int
    scene: str
    frame_id: int


def stats_from_frame(frame, pos, device):
    """Per-fragment NodeStats of one frame (vectorised: one-hot matmul + scatter_reduce)."""
    xyz = torch.as_tensor(np.asarray(frame.xyz, np.float32), device=device)
    rgb = torch.as_tensor(np.asarray(frame.rgb, np.float32), device=device) / 255
    groups = torch.as_tensor(np.asarray(frame.groups, np.int64), device=device)
    num = int(len(frame.node_feat))
    m = torch.zeros((len(xyz), num), device=device)
    m[torch.arange(len(xyz), device=device), groups] = 1
    n = m.sum(0)
    idx3 = groups[:, None].expand(-1, 3)
    out = NodeStats.zeros(num, int(frame.node_feat.shape[1]), device)
    out.n = n
    out.sum_xyz = m.T @ xyz
    out.sum_xx = m.T @ (xyz[:, :, None] * xyz[:, None, :]).reshape(-1, 9)
    out.sum_rgb = m.T @ rgb
    out.lo = torch.full((num, 3), float('inf'), device=device).scatter_reduce(0, idx3, xyz, 'amin')
    out.hi = torch.full((num, 3), float('-inf'), device=device).scatter_reduce(0, idx3, xyz, 'amax')
    out.feat_sum = torch.as_tensor(np.asarray(frame.node_feat, np.float32), device=device)
    out.n_feat = (n > 0).float()
    out.conf_sum = torch.as_tensor(np.asarray(frame.conf, np.float32), device=device)
    out.first = out.last = torch.full((num,), float(pos), device=device)
    out.n_frames = torch.ones(num, device=device)
    return out


def oracle_scores(graph, purity=0.8):
    """Stage-0 scorer: prob = 1 iff both ends are pure (>= purity) and share their majority GT."""
    pure = (graph.node_purity >= purity) & (graph.node_major > 0)
    a, b = graph.edges[:, 0], graph.edges[:, 1]
    ok = pure[a] & pure[b] & (graph.node_major[a] == graph.node_major[b])
    return torch.as_tensor(ok.astype(np.float32))


def _sum_stats(parts):
    """One row per node: a block's statistics are the *sum* of its fragments' (``NodeStats`` is additive).

    ``NodeStats.cat`` concatenates rows and is only right where every part is already a single node.
    """
    out = parts[0]
    for p in parts[1:]:
        out = out.add(p)
    return out


def _fragment_voxels(groups, vox):
    """(owner, voxel) pairs deduplicated per fragment, sorted by fragment id."""
    pairs = torch.unique(torch.stack([groups, vox], 1), dim=0)
    return pairs[:, 0], pairs[:, 1]


def _union_votes(old_vox, old_votes, new_vox):
    """Union of two voxel sets; a voxel keeps the number of contributions (votes)."""
    if len(new_vox) == 0:
        return old_vox, old_votes
    vox = torch.cat([old_vox, new_vox])
    votes = torch.cat([old_votes, torch.ones(len(new_vox), dtype=old_votes.dtype, device=old_votes.device)])
    uniq, inv = torch.unique(vox, return_inverse=True)
    return uniq, torch.zeros(len(uniq), dtype=old_votes.dtype, device=old_votes.device).index_add_(0, inv, votes)


class WindowMemory:
    """The unified memory. ``update`` consumes one frame and returns its point-level object ids."""

    def __init__(self, cfg=None, score_fn=None, device='cuda', unified=None):
        self.cfg = cfg or WindowConfig()
        self.score_fn = score_fn
        self.device = device
        self.unified = unified      # UnifiedGrouping for cfg.grouping == 'unified' (stage 2)
        self.feat_dim = 0
        self.reset()

    def reset(self):
        self.frame_index = -1
        self.window = []            # [{'pos', 'frame_id', 'frags': [{'idx','obj','vox','stats','votes'}]}] oldest first
        self.objects = {}           # id -> {'vox', 'votes', 'stats' (1-row NodeStats), 'votes_gt', 'quality'}
        self.next_obj = 0
        self.last_graph = None
        self.last_unified = None    # products of the last 'unified' grouping (blocks, edges, quality)
        self.times = {}

    # ---------------------------------------------------------------- update
    @torch.no_grad()
    def update(self, frame):
        cfg, dev = self.cfg, self.device
        pos = self.frame_index = self.frame_index + 1
        self.feat_dim = int(frame.node_feat.shape[1])
        sync = torch.cuda.synchronize if str(dev).startswith('cuda') else (lambda: None)

        # 1. frames that slid out of the window are frozen before anything else
        sync(); t0 = time.perf_counter()
        while self.window and self.window[0]['pos'] <= pos - cfg.window:
            self._freeze(self.window.pop(0))
        sync(); self.times['freeze_ms'] = (time.perf_counter() - t0) * 1e3

        sync(); t0 = time.perf_counter()
        xyz = torch.as_tensor(np.asarray(frame.xyz, np.float32), device=dev)
        groups = torch.as_tensor(np.asarray(frame.groups, np.int64), device=dev)
        num_cur = int(len(frame.node_feat))
        stats_cur = stats_from_frame(frame, pos, dev)
        f_owner, f_vox = _fragment_voxels(groups, voxelize(xyz, cfg.voxel))
        f_vox_list = torch.split(f_vox, torch.bincount(f_owner, minlength=num_cur).tolist())
        votes_cur = [dict() for _ in range(num_cur)]
        for n_i, g_i, c_i in zip(*group_gt_votes(np.asarray(frame.groups), np.asarray(frame.gt), num_cur)):
            votes_cur[int(n_i)][int(g_i)] = int(c_i)

        # 2. nodes = current fragments + past window nodes + frozen objects touching them
        past = self._past_nodes()
        node_kind, node_stats, node_vox, node_votes, node_prev = [], [], [], [], []
        for i in range(num_cur):
            node_kind.append(0); node_stats.append(stats_cur.select(i)); node_vox.append(f_vox_list[i])
            node_votes.append(votes_cur[i]); node_prev.append(-1)
        for p in past:
            node_kind.append(1); node_stats.append(p['stats']); node_vox.append(p['vox'])
            node_votes.append(p['votes']); node_prev.append(p['prev_obj'])
        frozen_ids = []
        if self.objects:
            base_own = [f_owner] + [torch.full((len(p['vox']),), num_cur + i, dtype=torch.int64, device=dev)
                                    for i, p in enumerate(past)]
            base_vox = [f_vox] + [p['vox'] for p in past]
            num_base = num_cur + len(past)
            d_own, d_vox = dilate_sets(torch.cat(base_own), torch.cat(base_vox), cfg.touch_dilation)
            obj_ids = sorted(self.objects)
            f_own, f_vox_all = self._frozen_records(obj_ids)
            inter_f, _, _ = overlap_matrix(d_own, d_vox, f_own, f_vox_all, num_base, len(obj_ids))
            frozen_ids = [obj_ids[i] for i in torch.nonzero(inter_f.sum(0) > 0).flatten().tolist()]
        for oid in frozen_ids:
            rec = self.objects[oid]
            node_kind.append(2); node_stats.append(rec['stats']); node_vox.append(rec['vox'])
            node_votes.append(rec['votes_gt']); node_prev.append(oid)
        num_nodes = len(node_kind)
        stats = NodeStats.cat(node_stats)
        z_ref = float(torch.median(xyz[:, 2]))

        # 3. GT majority / purity per node (oracle scoring and training labels)
        vi, gi, ci = [], [], []
        for i, v in enumerate(node_votes):
            for k, c in v.items():
                vi.append(i); gi.append(k); ci.append(c)
        major, purity = np.zeros(num_nodes, np.int64), np.zeros(num_nodes, np.float32)
        if vi:
            vi, gi, ci = np.asarray(vi, np.int64), np.asarray(gi, np.int64), np.asarray(ci, np.float64)
            order = np.lexsort((gi, -ci, vi))
            vs, first = vi[order], np.r_[True, vi[order][1:] != vi[order][:-1]]
            major[vs[first]] = gi[order][first]
            purity[vs[first]] = (ci[order][first] / np.bincount(vi, weights=ci, minlength=num_nodes)[vs[first]]).astype(np.float32)

        # 4. edges: cached intra-frame edges + every pair whose voxel sets touch
        e_cur = np.asarray(frame.edges, np.int64).reshape(-1, 2)
        row_of = torch.full((num_nodes, num_nodes), -1, dtype=torch.int64, device=dev)
        if len(e_cur):
            row_of[torch.as_tensor(e_cur[:, 0], device=dev), torch.as_tensor(e_cur[:, 1], device=dev)] = \
                torch.arange(len(e_cur), device=dev)
        n_own = torch.repeat_interleave(torch.arange(num_nodes, device=dev),
                                        torch.tensor([len(v) for v in node_vox], device=dev))
        n_vox = torch.cat(node_vox)
        inter0, va, vb = overlap_matrix(n_own, n_vox, n_own, n_vox, num_nodes, num_nodes)
        d_own, d_vox = dilate_sets(n_own, n_vox, cfg.touch_dilation)
        s1, _, _ = overlap_matrix(n_own, n_vox, d_own, d_vox, num_nodes, num_nodes)
        inter1 = torch.maximum(s1, s1.t())
        iu = torch.triu_indices(num_nodes, num_nodes, 1, device=dev)
        a, b = iu[0], iu[1]
        kind_t = torch.as_tensor(np.asarray(node_kind), device=dev)
        keep = (inter1[a, b] > 0) | (row_of[a, b] >= 0)
        if not cfg.object_merge:
            keep &= ~((kind_t[a] == 2) & (kind_t[b] == 2))
        a, b = a[keep], b[keep]
        rows = row_of[a, b]
        ef = torch.zeros((len(a), len(edge_feature_names())), device=dev)
        sel = rows >= 0
        if bool(sel.any()):
            ef[sel] = torch.as_tensor(np.asarray(frame.edge_feat, np.float32), device=dev)[rows[sel]]
        age = pos - stats.last
        graph = PairGraph(
            edges=torch.stack([a, b], 1).cpu().numpy(), pair_feat=pair_features(
                stats, a, b, inter0[a, b], inter1[a, b], va, vb,
                (age[a] - age[b]).abs(), (kind_t[a] == 0) & (kind_t[b] == 0), sel,
                (kind_t[a] == 2).float() + (kind_t[b] == 2).float(), edge_feat=ef),
            has_intra=sel, node_major=major, node_purity=purity, node_n=stats.n.cpu().numpy(),
            node_type=np.asarray(node_kind), node_prev_obj=np.asarray(node_prev),
            node_geom=stats.geom(z_ref).cpu().numpy(), node_feat=stats.feat().cpu().numpy(),
            num_nodes=num_nodes, z_ref=z_ref, frame_pos=pos, scene=frame.scene, frame_id=frame.frame_id)
        self.last_graph = graph
        sync(); self.times['build_ms'] = (time.perf_counter() - t0) * 1e3

        # 5. score + cluster + assign
        if cfg.grouping == 'unified':
            # stage 2: one model scores the intra-frame edges, the block <-> object edges and the
            # object <-> object edges, and predicts the node quality the output stage ranks by
            # (streaming/unified_flow.py). It owns its own stages and timing.
            if self.unified is None:
                raise RuntimeError("grouping='unified' needs a UnifiedGrouping passed as unified=")
            sync(); t0 = time.perf_counter()
            node_n = stats.n.cpu().numpy()
            assign, info = self.unified.assign(self, graph, node_prev, node_n, num_cur, n_own, n_vox,
                                               f_vox_list, stats_cur, pos)
            sync()
            self.times.update(info['times'])
            self.last_unified = info
        else:
            if self.score_fn is None:
                raise RuntimeError('WindowMemory.update needs a score_fn (stage 0: oracle_scores)')
            sync(); t0 = time.perf_counter()
            prob = np.asarray(self.score_fn(graph).detach().cpu(), np.float64).reshape(-1)
            sync(); self.times['score_ms'] = (time.perf_counter() - t0) * 1e3
            sync(); t0 = time.perf_counter()
            cluster = None if cfg.grouping == 'two_step' else \
                LINKAGE[cfg.linkage](num_nodes, graph.edges, prob, cfg.threshold)
            sync(); self.times['cluster_ms'] = (time.perf_counter() - t0) * 1e3

            sync(); t0 = time.perf_counter()
            node_n = stats.n.cpu().numpy()
            assign = (self._two_step_assign(graph, prob, node_prev, node_n, num_cur, n_own, n_vox,
                                            f_vox_list, stats_cur)
                      if cfg.grouping == 'two_step' else self._assign(cluster, node_kind, node_prev, node_n))
        for i, p in enumerate(past):                              # revise the window: re-glue past fragments
            obj = int(assign[num_cur + i])
            for wpos, fidx in p['members']:
                self._entry(wpos)['frags'][fidx]['obj'] = obj
        obj_of_frag = assign[:num_cur]
        entry = {'pos': pos, 'frame_id': frame.frame_id, 'frags': []}
        for i in range(num_cur):
            if obj_of_frag[i] < 0:
                continue
            n_i = int(node_n[i])          # node_n is already on the host (no per-fragment sync)
            entry['frags'].append({'idx': i, 'obj': int(obj_of_frag[i]), 'vox': f_vox_list[i],
                                   'stats': stats_cur.select(i), 'votes': votes_cur[i],
                                   'n': n_i, 'imp': n_i if purity[i] < cfg.purity else 0})
        self.window.append(entry)
        self._drop_empty()
        sync(); self.times['assign_ms'] = (time.perf_counter() - t0) * 1e3
        return np.asarray(obj_of_frag, np.int64)[np.asarray(frame.groups, np.int64)]

    # ---------------------------------------------------------------- internals
    def _entry(self, wpos):
        for e in self.window:
            if e['pos'] == wpos:
                return e
        raise KeyError(wpos)

    def _past_nodes(self):
        """Window nodes, oldest frame first: one per (frame, object) block, or one per fragment.

        ``members`` holds *positions* in the window entry's fragment list (fragments that were
        never assigned an object are not stored, so the position is not the fragment id).
        """
        out = []
        for e in self.window:
            if self.cfg.past_unit == 'fragment':
                for pos_in_list, fr in enumerate(e['frags']):
                    out.append(dict(vox=fr['vox'], stats=fr['stats'], votes=fr['votes'],
                                    prev_obj=int(fr['obj']), members=[(e['pos'], pos_in_list)]))
                continue
            by_obj = {}
            for pos_in_list, fr in enumerate(e['frags']):
                by_obj.setdefault(int(fr['obj']), []).append((pos_in_list, fr))
            for obj in sorted(by_obj):
                members = by_obj[obj]
                votes = {}
                for _, m in members:
                    for k, c in m['votes'].items():
                        votes[k] = votes.get(k, 0) + c
                out.append(dict(vox=torch.unique(torch.cat([m['vox'] for _, m in members])),
                                stats=_sum_stats([m['stats'] for _, m in members]), votes=votes,
                                prev_obj=obj, members=[(e['pos'], pos_in_list) for pos_in_list, _ in members]))
        return out

    def _frozen_records(self, obj_ids):
        """Frozen (object index, voxel) pairs; ``obj_ids`` maps the index back to the object id."""
        own = [torch.full((len(self.objects[o]['vox']),), i, dtype=torch.int64, device=self.device)
               for i, o in enumerate(obj_ids)]
        return torch.cat(own), torch.cat([self.objects[o]['vox'] for o in obj_ids])

    def _new_object(self):
        oid = self.next_obj
        self.next_obj += 1
        self.objects[oid] = {'vox': torch.empty(0, dtype=torch.int64, device=self.device),
                             'votes': torch.empty(0, dtype=torch.int32, device=self.device),
                             'stats': NodeStats.zeros(1, self.feat_dim, self.device), 'votes_gt': {},
                             'quality': float('nan'),      # stage-2 'unified': model's node quality
                             'fz_n': 0, 'fz_imp': 0, 'fz_pos': set()}
        return oid

    def _merge_objects(self, keep, drop):
        if keep == drop:
            return
        ka, kb = self.objects[keep], self.objects[drop]
        ka['vox'], ka['votes'] = _union_votes(ka['vox'], ka['votes'], kb['vox'])
        ka['stats'] = ka['stats'].add(kb['stats'])
        ka['fz_n'] += kb['fz_n']; ka['fz_imp'] += kb['fz_imp']; ka['fz_pos'] |= kb['fz_pos']
        for k, c in kb['votes_gt'].items():
            ka['votes_gt'][k] = ka['votes_gt'].get(k, 0) + c
        del self.objects[drop]

    def _two_step_assign(self, graph, prob, node_prev, node_n, num_cur, n_own, n_vox, f_vox_list, stats_cur):
        """Two-step grouping (``cfg.grouping == 'two_step'``): intra-frame blocks, then block -> object.

        Step 1 clusters the current fragments with the cached intra-frame edges alone (the
        GroupMergeNet probabilities at the per-frame threshold) into blocks; step 2 lets every block
        join the object whose cells contain it best, when that containment passes
        ``cfg.cross_threshold``. Past window nodes keep their object — there is no object-object
        merge and no history revision — so with ``window = 1`` this is exactly "the blocks of this
        frame go to the objects they touch". See ``streaming/two_step.py`` for why.
        """
        cfg, dev = self.cfg, self.device
        sel = two_step.as_bool_np(graph.has_intra)
        edges = np.asarray(graph.edges, np.int64).reshape(-1, 2)[sel]
        block = two_step.intra_blocks(num_cur, edges, np.asarray(prob)[sel], cfg.threshold, cfg.linkage)
        num_blocks = int(block.max()) + 1 if num_cur else 0
        b_own, b_vox = two_step.block_voxel_sets(block, f_vox_list, dev)
        b_n = np.bincount(block, weights=node_n[:num_cur], minlength=num_blocks)
        b_feat = two_step.block_features(block, num_blocks, stats_cur.feat().cpu().numpy(),
                                         node_n[:num_cur], dev)
        cand_ids, c_own, c_vox, c_feat = two_step.object_candidates(node_prev, n_own, n_vox,
                                                                    graph.node_feat, node_n, dev)
        score = two_step.block_object_scores(b_own, b_vox, b_feat, c_own, c_vox, c_feat, cfg.cross_rule)
        block_obj = two_step.assign_blocks(score, cand_ids, b_n, cfg.cross_threshold,
                                           cfg.admit_min_points, self._new_object)
        prev = np.asarray(node_prev, np.int64)
        out = np.full(len(prev), -1, np.int64)
        out[prev >= 0] = prev[prev >= 0]
        out[:num_cur] = block_obj[block]
        return out

    def _assign(self, cluster, node_kind, node_prev, node_n):
        """One object id per node: frozen wins (smallest id), else past-majority, else a new object."""
        cfg = self.cfg
        kind = np.asarray(node_kind)
        prev = np.asarray(node_prev, np.int64)
        n = np.asarray(node_n, np.float64)
        num_g = int(cluster.max()) + 1 if len(cluster) else 0
        out = np.full(len(cluster), -1, np.int64)
        if num_g == 0:
            return out
        # 1) clusters holding a frozen object: it wins, and several frozen objects merge into the
        #    smallest of their ids. Per-cluster aggregates come from one pass, not a Python loop.
        fz = kind == 2
        NONE = np.iinfo(np.int64).max
        fz_min = np.full(num_g, NONE, np.int64)
        if fz.any():
            np.minimum.at(fz_min, cluster[fz], prev[fz])
        has_fz = fz_min != NONE
        # 2) otherwise the point-weighted majority of the cluster's past nodes (tie -> smaller id).
        #    Point counts are exact integers in float64, so the arg-max is not a rounding question.
        past = kind == 1
        past_obj = np.zeros(num_g, np.int64)
        has_past = np.zeros(num_g, bool)
        if past.any():
            ids, inv = np.unique(prev[past], return_inverse=True)
            tally = np.zeros((num_g, len(ids)))
            np.add.at(tally, (cluster[past], inv), n[past])
            past_obj, has_past = ids[tally.argmax(1)], tally.max(1) > 0
        # 3) everything else: a new object when the group is big enough, else unassigned
        obj = np.where(has_fz, fz_min, np.where(has_past, past_obj, -1))
        total = np.bincount(cluster, weights=n, minlength=num_g)
        for g in np.flatnonzero(~has_fz & ~has_past):
            obj[g] = self._new_object() if total[g] >= cfg.admit_min_points else -1
        if fz.any():
            order = np.argsort(cluster[fz], kind='stable')
            fc, fp = cluster[fz][order], prev[fz][order]
            starts = np.r_[0, np.flatnonzero(fc[1:] != fc[:-1]) + 1, len(fc)]
            for s, e in zip(starts[:-1], starts[1:]):
                ids = np.unique(fp[s:e])
                if len(ids) > 1:                      # a single frozen object is already the target
                    for other in ids:
                        self._merge_objects(int(obj[fc[s]]), int(other))
        out[:] = obj[cluster]
        return out

    def _freeze(self, entry):
        for fr in entry['frags']:
            oid = int(fr['obj'])
            if oid < 0 or oid not in self.objects:
                continue
            rec = self.objects[oid]
            rec['vox'], rec['votes'] = _union_votes(rec['vox'], rec['votes'], fr['vox'])
            rec['stats'] = rec['stats'].add(fr['stats'])
            rec['fz_n'] += fr['n']; rec['fz_imp'] += fr['imp']; rec['fz_pos'].add(entry['pos'])
            for k, c in fr['votes'].items():
                rec['votes_gt'][k] = rec['votes_gt'].get(k, 0) + c

    def _drop_empty(self):
        live = {int(fr['obj']) for e in self.window for fr in e['frags']}
        for oid in list(self.objects):
            if len(self.objects[oid]['vox']) == 0 and oid not in live:
                del self.objects[oid]

    # ---------------------------------------------------------------- output
    @torch.no_grad()
    def render(self, xyz):
        """(K,V) masks over arbitrary points + per-object scores (mean confidence of the members)."""
        obj_ids = sorted(self.objects)
        xyz_t = torch.as_tensor(np.asarray(xyz, np.float32), device=self.device)
        if not obj_ids:
            return np.zeros((0, len(xyz_t)), bool), np.zeros(0, np.float32)
        rec_vox, rec_own, n_tot, conf_tot = [], [], [], []
        for i, oid in enumerate(obj_ids):
            vox = [self.objects[oid]['vox']]
            n = float(self.objects[oid]['stats'].n.sum())
            c = float(self.objects[oid]['stats'].conf_sum.sum())
            for e in self.window:
                for fr in e['frags']:
                    if fr['obj'] == oid:
                        vox.append(fr['vox'])
                        n += float(fr['stats'].n.sum())
                        c += float(fr['stats'].conf_sum.sum())
            vox = torch.unique(torch.cat(vox))
            rec_vox.append(vox)
            rec_own.append(torch.full((len(vox),), i, dtype=torch.int64, device=self.device))
            n_tot.append(n); conf_tot.append(c)
        rv, ro = torch.cat(rec_vox), torch.cat(rec_own)
        order = torch.argsort(rv)
        masks = render_voxel_table(rv[order], ro[order], len(obj_ids), voxelize(xyz_t, self.cfg.voxel))
        scores = np.asarray(conf_tot, np.float32) / np.maximum(np.asarray(n_tot, np.float32), 1)
        return masks.cpu().numpy(), scores

    @torch.no_grad()
    def object_attrs(self):
        '''Per-object attributes aligned with ``render`` row order (stage 0b post-processing):
        point count, points from impure (< purity) member fragments, distinct observation
        frames, the mean FastSAM confidence (the render score), and - under ``grouping='unified'``
        - the model's node quality (the stage-2 ranking score; NaN otherwise).'''
        ids, n_pts, imp_pts, n_obs, conf, quality = [], [], [], [], [], []
        for oid in sorted(self.objects):
            rec = self.objects[oid]
            n, imp = float(rec['fz_n']), float(rec['fz_imp'])
            c_sum, n_tot = float(rec['stats'].conf_sum.sum()), float(rec['stats'].n.sum())
            positions = set(rec['fz_pos'])
            for e in self.window:
                for fr in e['frags']:
                    if fr['obj'] != oid:
                        continue
                    n += fr['n']; imp += fr['imp']
                    c_sum += float(fr['stats'].conf_sum.sum()); n_tot += float(fr['stats'].n.sum())
                    positions.add(e['pos'])
            ids.append(oid); n_pts.append(int(n)); imp_pts.append(int(imp))
            n_obs.append(len(positions)); conf.append(c_sum / max(n_tot, 1.0))
            quality.append(float(rec.get('quality', float('nan'))))
        return dict(ids=np.asarray(ids, np.int64), n_pts=np.asarray(n_pts, np.int64),
                    imp_pts=np.asarray(imp_pts, np.int64), n_obs=np.asarray(n_obs, np.int64),
                    conf=np.asarray(conf, np.float32), quality=np.asarray(quality, np.float32))

    def stats(self):
        return dict(objects=len(self.objects), window_frames=len(self.window),
                    frozen_voxels=int(sum(len(r['vox']) for r in self.objects.values())),
                    **self.times)

    def window_objects(self):
        """[(frame_pos, fragment_idx, object_id)] of the revisable window (tests / debugging)."""
        return [(e['pos'], fr['idx'], int(fr['obj'])) for e in self.window for fr in e['frags']]


class GtAssocMemory:
    """``gt_assoc`` ceiling: a fragment with purity >= 0.8 joins its majority GT's object from any
    frame (no window, no revision); impure fragments get singleton objects. Upper bound of
    "associate fragments to GT objects" without any model."""

    def __init__(self, purity=0.8, voxel=0.05, device='cuda'):
        self.purity = purity
        self.voxel = voxel
        self.device = device
        self.objects = {}          # id -> [voxel tensors] ; scores accumulated separately
        self.conf_sum, self.n_tot = {}, {}
        self.next_single = -1

    @torch.no_grad()
    def update(self, frame):
        num = int(len(frame.node_feat))
        xyz = torch.as_tensor(np.asarray(frame.xyz, np.float32), device=self.device)
        groups = torch.as_tensor(np.asarray(frame.groups, np.int64), device=self.device)
        owner, vox = _fragment_voxels(groups, voxelize(xyz, self.voxel))
        chunks = torch.split(vox, torch.bincount(owner, minlength=num).tolist())
        vi, gi, ci = group_gt_votes(np.asarray(frame.groups), np.asarray(frame.gt), num)
        major, purity = np.zeros(num, np.int64), np.zeros(num, np.float32)
        if len(vi):
            order = np.lexsort((gi, -ci, vi))
            vs, first = vi[order], np.r_[True, vi[order][1:] != vi[order][:-1]]
            major[vs[first]] = gi[order][first]
            purity[vs[first]] = (ci[order][first] / np.bincount(vi, weights=ci, minlength=num)[vs[first]]).astype(np.float32)
        n_frag = torch.bincount(groups, minlength=num).cpu().numpy()
        labels = np.full(len(groups), -1, np.int64)
        for f in range(num):
            if purity[f] >= self.purity and major[f] > 0:
                oid = int(major[f])
            else:
                oid = self.next_single
                self.next_single -= 1
            self.objects.setdefault(oid, []).append(chunks[f])
            self.conf_sum[oid] = self.conf_sum.get(oid, 0.0) + float(frame.conf[f]) * n_frag[f]
            self.n_tot[oid] = self.n_tot.get(oid, 0.0) + n_frag[f]
            labels[groups.cpu().numpy() == f] = oid
        return labels

    @torch.no_grad()
    def render(self, xyz):
        ids = sorted(self.objects)
        if not ids:
            return np.zeros((0, len(np.asarray(xyz))), bool), np.zeros(0, np.float32)
        rec_vox = [torch.unique(torch.cat(self.objects[o])) for o in ids]
        rec_own = [torch.full((len(v),), i, dtype=torch.int64, device=self.device) for i, v in enumerate(rec_vox)]
        rv, ro = torch.cat(rec_vox), torch.cat(rec_own)
        order = torch.argsort(rv)
        xyz_t = torch.as_tensor(np.asarray(xyz, np.float32), device=self.device)
        masks = render_voxel_table(rv[order], ro[order], len(ids), voxelize(xyz_t, self.voxel))
        scores = np.asarray([self.conf_sum[o] for o in ids], np.float32) / np.maximum(
            np.asarray([self.n_tot[o] for o in ids], np.float32), 1)
        return masks.cpu().numpy(), scores
