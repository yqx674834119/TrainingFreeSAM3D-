"""QueryMemory: scene-level instance queries maintained as voxel occupancy, updated per frame.

Counterpart of ESAM's ``OnlineMerge`` (ESAM/oneformer3d/instance_merge.py:180) with one
structural difference: ESAM keeps a (K, 20000·F) boolean mask over *all points ever seen*,
so its memory grows with the number of frames. Here each query owns a set of voxel ids, and
the whole memory is one sparse (voxel, query) table that grows with the *scene volume* only.

Per frame:
  1. proposals -> (owner, voxel) pairs                              (torch.unique)
  2. (P, K) overlap with the memory                                   (sparse join, matmul)
  3. match score per pair: a rule (voxel IoU) or a trained MatchScorer
  4. Hungarian assignment on the (P, K) score matrix, threshold       (scipy, P,K <= 200)
  5. matched -> union voxels, count-weighted running feature mean (as ESAM), score max
     unmatched -> new query if it passes the admission gate
  6. maintenance: merge queries that came to overlap, evict stale singletons, cap K

Everything the memory stores per query: voxels, feat (un-centred, 640-d, running mean),
score, n_obs, first/last seen. Centroid / bbox are derived from the voxels on demand.
"""
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .voxel import containment_from_counts, devoxelize, dilate_sets, iou_from_counts, mask_voxel_sets, overlap_matrix, voxelize


@dataclass
class MemoryConfig:
    voxel: float = 0.05
    match_threshold: float = 0.3       # minimum match score to accept an assignment
    admit_score: float = 0.3           # proposal score needed to open a new query
    admit_min_voxels: int = 20
    admit_support: float = 0.8         # fraction of a new query's voxels that must lie near seen scene voxels
    support_radius_voxels: int = 3     # "near" = within this many voxels (Chebyshev) of an observed voxel
    max_queries: int = 200             # ESAM inscat_topk_insts
    evict_after: int = 30              # frames unseen before a single-observation query may be evicted
    overlap_merge_iou: float = 0.5     # queries whose voxel IoU exceeds this are merged in maintenance
    feature_dim: int = 640
    assignment: str = 'hungarian'      # 'hungarian': ESAM-style 1:1; 'many_to_one': each proposal -> its best query
    match_score: str = 'max_contain'   # rule score: 'iou' | 'max_contain' | 'iou_or_contain' | 'contain_x_feat' (geometry x projected cosine)
    feat_weight: float = 1.0           # contain_x_feat: score = max_contain * (clamp(cos,0,1) ** feat_weight)
    merge_rule: str = 'iou'            # maintenance merge: 'iou' (> overlap_merge_iou) | 'contain' (either containment > overlap_merge_contain)
    overlap_merge_contain: float = 0.5
    merge_feat_cos: float = -1.0       # if >= 0: geometric merge pairs must also have query-feature cosine >= this
    merge_touch_cos: float = -1.0      # if >= 0: also merge queries that merely touch (share a voxel after merge_dilation) and have cosine >= this
    match_dilation: int = 1           # dilate the MEMORY side by r voxels before overlap (tolerates a one-voxel shift between views)
    merge_dilation: int = 0            # same for the query-vs-query maintenance merge
    render_min_votes: int = 1          # render a (voxel, query) record only if >= this many frames voted for it ...
    render_min_vote_frac: float = 0.0  # ... or >= this fraction of the query's observations (whichever is stricter, if > 0)
    render_exclusive: bool = False     # each voxel rendered for its single best query (votes, then vote fraction) -> exclusive masks
    keep_points: bool = False          # also store every frame's points with their query ids (ESAM-style KNN render; memory grows with frames)


@dataclass
class FrameInput:
    """What one frame contributes: proposals on the frame's points."""
    xyz: np.ndarray            # (N,3) world coordinates
    masks: np.ndarray          # (P,N) bool proposals (may overlap)
    scores: np.ndarray         # (P,) proposal quality (scorer IoU prediction)
    feat: np.ndarray           # (P,C) un-centred, L2-normalised appearance feature
    frame_id: int = 0


@dataclass
class MatchFeatures:
    """Pairwise cues for the (P, K) candidate pairs; all (P, K) float tensors on ``device``."""
    voxel_iou: torch.Tensor
    contain_p_in_k: torch.Tensor
    contain_k_in_p: torch.Tensor
    centroid_dist: torch.Tensor
    bbox_iou: torch.Tensor
    log_size_ratio: torch.Tensor
    feat_cos: torch.Tensor        # cosine after subtracting the memory's running global mean
    feat_cos_raw: torch.Tensor    # cosine of the un-centred features
    proposal_score: torch.Tensor
    query_score: torch.Tensor
    query_n_obs: torch.Tensor
    frames_since_seen: torch.Tensor

    def stack(self):
        return torch.stack([self.voxel_iou, self.contain_p_in_k, self.contain_k_in_p, self.centroid_dist, self.bbox_iou,
                            self.log_size_ratio, self.feat_cos, self.feat_cos_raw, self.proposal_score, self.query_score,
                            self.query_n_obs, self.frames_since_seen], -1)

    @staticmethod
    def names():
        return ['voxel_iou', 'contain_p_in_k', 'contain_k_in_p', 'centroid_dist', 'bbox_iou', 'log_size_ratio',
                'feat_cos', 'feat_cos_raw', 'proposal_score', 'query_score', 'query_n_obs', 'frames_since_seen']


class QueryMemory:
    def __init__(self, cfg: MemoryConfig = None, device='cuda', scorer=None, projection=None):
        self.cfg = cfg or MemoryConfig()
        self.device = torch.device(device)
        self.scorer = scorer                      # callable(MatchFeatures) -> (P,K) scores, or None for the rule
        self.projection = projection              # nn.Module mapping raw (P,640) -> L2 embedding; applied on update
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        dev = self.device
        self.owner = torch.empty(0, dtype=torch.int64, device=dev)     # (M,) query index per occupied voxel
        self.voxel = torch.empty(0, dtype=torch.int64, device=dev)     # (M,) voxel id
        self.votes = torch.empty(0, dtype=torch.int32, device=dev)     # (M,) frames that put this voxel into this query
        self.feat = torch.empty((0, self.cfg.feature_dim), device=dev)
        self.score = torch.empty(0, device=dev)
        self.n_obs = torch.empty(0, dtype=torch.int64, device=dev)
        self.first_seen = torch.empty(0, dtype=torch.int64, device=dev)
        self.last_seen = torch.empty(0, dtype=torch.int64, device=dev)
        self.scene_voxels = torch.empty(0, dtype=torch.int64, device=dev)   # every voxel observed so far
        self.feat_sum = torch.zeros(self.cfg.feature_dim, device=dev)       # running global mean of features
        self.feat_count = 0
        self.frame = -1
        self.pending = None                        # proposals that failed admission last frame
        self.stats = []
        self.pt_xyz = []                           # keep_points: list of (N,3) per frame
        self.pt_owner = []                         # keep_points: list of (N,) query id per point (-1 none), remapped on merge/evict

    @property
    def num_queries(self):
        return len(self.score)

    @property
    def num_records(self):
        return len(self.owner)

    def sizes(self):
        return torch.bincount(self.owner, minlength=self.num_queries).float()

    def centroids_bboxes(self, owner=None, voxel=None, K=None):
        owner = self.owner if owner is None else owner
        voxel = self.voxel if voxel is None else voxel
        K = self.num_queries if K is None else K
        xyz = devoxelize(voxel, self.cfg.voxel)
        cnt = torch.bincount(owner, minlength=K).float().clamp_min(1)[:, None]
        cent = torch.zeros((K, 3), device=self.device).index_add_(0, owner, xyz) / cnt
        lo = torch.full((K, 3), float('inf'), device=self.device).scatter_reduce(0, owner[:, None].expand(-1, 3), xyz, 'amin')
        hi = torch.full((K, 3), float('-inf'), device=self.device).scatter_reduce(0, owner[:, None].expand(-1, 3), xyz, 'amax')
        return cent, lo, hi

    # ------------------------------------------------------------------ per-frame update
    @torch.no_grad()
    def update(self, fr: FrameInput):
        self.frame = fr.frame_id
        dev = self.device
        xyz = torch.as_tensor(fr.xyz, dtype=torch.float32, device=dev)
        masks = torch.as_tensor(fr.masks, device=dev)
        pscore = torch.as_tensor(fr.scores, dtype=torch.float32, device=dev)
        pfeat = torch.as_tensor(fr.feat, dtype=torch.float32, device=dev)
        if self.projection is not None and len(pfeat):
            pfeat = self.projection(pfeat)
        if self.feat.shape[1] != pfeat.shape[1]:          # first frame: size the store to the embedding
            self.feat = torch.empty((0, pfeat.shape[1]), device=dev)
            self.feat_sum = torch.zeros(pfeat.shape[1], device=dev)
        vox = voxelize(xyz, self.cfg.voxel)
        P = len(masks)
        p_owner, p_vox = mask_voxel_sets(masks, vox)
        p_size = torch.bincount(p_owner, minlength=P).float()
        frame_voxels = torch.unique(vox)

        matched_p = torch.full((P,), -1, dtype=torch.int64, device=dev)
        if self.num_queries and P:
            feats = self.match_features(p_owner, p_vox, p_size, pscore, pfeat)
            if self.scorer is not None:
                S = self.scorer(feats)
            elif self.cfg.match_score == 'max_contain':
                S = torch.maximum(feats.contain_p_in_k, feats.contain_k_in_p)
            elif self.cfg.match_score == 'iou_or_contain':
                S = torch.maximum(feats.voxel_iou, torch.maximum(feats.contain_p_in_k, feats.contain_k_in_p))
            elif self.cfg.match_score == 'contain_x_feat':
                S = torch.maximum(feats.contain_p_in_k, feats.contain_k_in_p) * feats.feat_cos_raw.clamp(0, 1) ** self.cfg.feat_weight
            elif self.cfg.match_score == 'feat_x_bbox':                        # ESAM's OnlineMerge score: cos x bbox IoU
                S = feats.feat_cos_raw.clamp(0, 1) * feats.bbox_iou
            elif self.cfg.match_score == 'feat_gate':                          # feature ranks, geometry only gates
                S = feats.feat_cos_raw.clamp(0, 1)
            else:
                S = feats.voxel_iou
            # never match disjoint pairs (bbox contact for the ESAM-style score, voxel contact otherwise)
            gate = feats.bbox_iou > 0 if self.cfg.match_score == 'feat_x_bbox' else feats.voxel_iou > 0
            S = torch.where(gate, S, torch.zeros_like(S))
            if self.cfg.assignment == 'hungarian':
                rows, cols = linear_sum_assignment(-S.cpu().numpy())
                ok = S[rows, cols] >= self.cfg.match_threshold
                matched_p[torch.as_tensor(rows[ok.cpu().numpy()], device=dev)] = torch.as_tensor(cols[ok.cpu().numpy()], device=dev)
            else:
                best, arg = S.max(1)
                ok = best >= self.cfg.match_threshold
                matched_p[ok] = arg[ok]
        # ---- matched: union voxels, running mean feature, max score
        m = matched_p >= 0
        if m.any():
            q = matched_p[m]
            sel = m[p_owner]
            self._add_records(matched_p[p_owner[sel]], p_vox[sel])
            K = self.num_queries
            hits = torch.bincount(q, minlength=K).float()                       # proposals per query this frame
            fsum = torch.zeros_like(self.feat).index_add_(0, q, pfeat[m])
            smax = torch.zeros(K, device=dev).scatter_reduce(0, q, pscore[m], 'amax')
            hit = hits > 0
            n = self.n_obs.float()[:, None]
            # one observation per query per frame; the frame's feature = mean over its matched proposals
            self.feat[hit] = (self.feat[hit] * n[hit] + fsum[hit] / hits[hit][:, None]) / (n[hit] + 1)
            self.score = torch.maximum(self.score, smax)
            self.n_obs[hit] += 1
            self.last_seen[hit] = self.frame
        # ---- unmatched: admission gate -> new queries (or pending)
        new = torch.nonzero(~m).squeeze(1)
        admitted = self._admit(new, p_owner, p_vox, p_size, pscore)
        for p in admitted.tolist():
            self._new_query(p_vox[p_owner == p], pfeat[p], pscore[p])
        # ---- bookkeeping
        self.scene_voxels = torch.unique(torch.cat([self.scene_voxels, frame_voxels]))
        if P:
            self.feat_sum += pfeat.sum(0)
            self.feat_count += P
        if self.cfg.keep_points:
            # point -> query: the query of the proposal that covers it (highest score wins); unmatched/unadmitted -> -1
            prop_q = matched_p.clone()
            adm = torch.full((P,), -1, dtype=torch.int64, device=dev)
            adm[admitted] = torch.arange(self.num_queries - len(admitted), self.num_queries, device=dev)
            prop_q = torch.where(prop_q >= 0, prop_q, adm)
            order = torch.argsort(pscore)                       # paint low score first, high score last
            owner_pt = torch.full((len(xyz),), -1, dtype=torch.int64, device=dev)
            for p_ in order.tolist():
                if prop_q[p_] >= 0:
                    owner_pt[masks[p_]] = prop_q[p_]
            self.pt_xyz.append(xyz); self.pt_owner.append(owner_pt)
        self.maintain()
        self.stats.append(dict(frame=self.frame, proposals=int(P), matched=int(m.sum()), admitted=int(len(admitted)),
                               queries=self.num_queries, records=self.num_records, scene_voxels=int(len(self.scene_voxels))))
        return matched_p.cpu().numpy()

    def match_features(self, p_owner, p_vox, p_size, pscore, pfeat):
        K = self.num_queries
        P = len(p_size)
        q_owner, q_vox = dilate_sets(self.owner, self.voxel, self.cfg.match_dilation)
        inter, _, q_size_d = overlap_matrix(p_owner, p_vox, q_owner, q_vox, P, K)
        q_size = torch.bincount(self.owner, minlength=K).float()
        # containment / IoU use the undilated query size so that dilation only adds tolerance, not volume
        iou = inter / (p_size[:, None] + q_size[None, :] - inter).clamp_min(1)
        iou = iou.clamp(max=1.0)
        c_pk, c_kp = inter / p_size[:, None].clamp_min(1), inter / q_size[None, :].clamp_min(1)
        c_pk, c_kp = c_pk.clamp(max=1.0), c_kp.clamp(max=1.0)
        pc, plo, phi = self.centroids_bboxes(p_owner, p_vox, P)
        qc, qlo, qhi = self.centroids_bboxes()
        dist = torch.cdist(pc, qc)
        lo = torch.maximum(plo[:, None], qlo[None]); hi = torch.minimum(phi[:, None], qhi[None])
        inter_v = (hi - lo).clamp_min(0).prod(-1)
        vol_p = (phi - plo).clamp_min(0).prod(-1); vol_q = (qhi - qlo).clamp_min(0).prod(-1)
        bbox_iou = inter_v / (vol_p[:, None] + vol_q[None] - inter_v).clamp_min(1e-9)
        mean = self.feat_sum / max(self.feat_count, 1)
        pf = torch.nn.functional.normalize(pfeat - mean, dim=1); qf = torch.nn.functional.normalize(self.feat - mean, dim=1)
        return MatchFeatures(iou, c_pk, c_kp, dist, bbox_iou, torch.log(p_size[:, None] / q_size[None].clamp_min(1)).abs(),
                             pf @ qf.T,
                             torch.nn.functional.normalize(pfeat, dim=1) @ torch.nn.functional.normalize(self.feat, dim=1).T,
                             pscore[:, None].expand(P, K), self.score[None].expand(P, K),
                             self.n_obs[None].float().expand(P, K), (self.frame - self.last_seen)[None].float().expand(P, K))

    def _admit(self, cand, p_owner, p_vox, p_size, pscore):
        """Admission gate for unmatched proposals; returns the admitted proposal indices."""
        if len(cand) == 0:
            return cand
        ok = (pscore[cand] >= self.cfg.admit_score) & (p_size[cand] >= self.cfg.admit_min_voxels)
        if len(self.scene_voxels):
            # support: fraction of the proposal's voxels within r voxels (Chebyshev) of an already-observed voxel.
            # Dilate the observed set by r via integer offsets (27 neighbours for r=1; r=3 -> 343): cheap, matrix-only.
            r = self.cfg.support_radius_voxels
            offs = torch.stack(torch.meshgrid(*[torch.arange(-r, r + 1, device=self.device)] * 3, indexing='ij'), -1).reshape(-1, 3)
            from .voxel import BITS
            shift = (offs[:, 0] << (2 * BITS)) + (offs[:, 1] << BITS) + offs[:, 2]
            dil = torch.unique((self.scene_voxels[:, None] + shift[None]).reshape(-1))
            near = torch.isin(p_vox, dil)
            sup = torch.zeros(len(p_size), device=self.device).index_add_(0, p_owner, near.float()) / p_size.clamp_min(1)
            ok &= sup[cand] >= self.cfg.admit_support
        return cand[ok]

    def _new_query(self, voxels, feat, score):
        k = self.num_queries
        self.owner = torch.cat([self.owner, torch.full((len(voxels),), k, dtype=torch.int64, device=self.device)])
        self.voxel = torch.cat([self.voxel, voxels])
        self.votes = torch.cat([self.votes, torch.ones(len(voxels), dtype=torch.int32, device=self.device)])
        self.feat = torch.cat([self.feat, feat[None]])
        self.score = torch.cat([self.score, score[None]])
        self.n_obs = torch.cat([self.n_obs, torch.ones(1, dtype=torch.int64, device=self.device)])
        self.first_seen = torch.cat([self.first_seen, torch.full((1,), self.frame, dtype=torch.int64, device=self.device)])
        self.last_seen = torch.cat([self.last_seen, torch.full((1,), self.frame, dtype=torch.int64, device=self.device)])

    def _add_records(self, owner, voxel):
        # new records vote once each (dedupe within the frame first); existing records accumulate
        new = torch.unique(torch.stack([owner, voxel], 1), dim=0)
        allp = torch.cat([torch.stack([self.owner, self.voxel], 1), new])
        votes = torch.cat([self.votes, torch.ones(len(new), dtype=torch.int32, device=self.device)])
        pairs, inv = torch.unique(allp, dim=0, return_inverse=True)
        self.votes = torch.zeros(len(pairs), dtype=torch.int32, device=self.device).index_add_(0, inv, votes)
        self.owner, self.voxel = pairs[:, 0], pairs[:, 1]

    # ------------------------------------------------------------------ maintenance
    def maintain(self):
        K = self.num_queries
        if K == 0:
            return
        # merge queries that have come to overlap heavily (same object admitted twice from different views)
        d_owner, d_vox = dilate_sets(self.owner, self.voxel, self.cfg.merge_dilation)
        inter, s, _ = overlap_matrix(self.owner, self.voxel, d_owner, d_vox, K, K)
        inter = torch.minimum(inter, s[:, None])              # dilation can only add tolerance, never volume
        iou = (inter / (s[:, None] + s[None, :] - inter).clamp_min(1)).clamp(max=1.0)
        iou.fill_diagonal_(0)
        if self.cfg.merge_rule == 'contain':
            ca, cb = (inter / s[:, None].clamp_min(1)).clamp(max=1.0), (inter / s[None, :].clamp_min(1)).clamp(max=1.0)
            crit = torch.maximum(ca, cb)
            crit.fill_diagonal_(0)
            pairs = torch.nonzero(torch.triu(crit > self.cfg.overlap_merge_contain))
        else:
            pairs = torch.nonzero(torch.triu(iou > self.cfg.overlap_merge_iou))
        if self.cfg.merge_feat_cos >= 0 or self.cfg.merge_touch_cos >= 0:
            qn = torch.nn.functional.normalize(self.feat, dim=1)
            cos = qn @ qn.T
            if self.cfg.merge_feat_cos >= 0 and len(pairs):            # geometric pairs must also agree in appearance
                pairs = pairs[cos[pairs[:, 0], pairs[:, 1]] >= self.cfg.merge_feat_cos]
            if self.cfg.merge_touch_cos >= 0:                            # touching + same appearance -> same object
                touch = torch.triu(inter > 0, diagonal=1) & (cos >= self.cfg.merge_touch_cos)
                pairs = torch.cat([pairs, torch.nonzero(touch)])
        if len(pairs):
            parent = torch.arange(K, device=self.device)
            for a, b in pairs.tolist():               # tiny: a handful of pairs per frame
                ra, rb = int(parent[a]), int(parent[b])
                if ra != rb:
                    parent[parent == rb] = ra
            self._relabel(parent)
        # evict stale single-observation queries, then cap K by score
        K = self.num_queries
        stale = (self.n_obs == 1) & (self.frame - self.last_seen > self.cfg.evict_after)
        keep = ~stale
        if keep.sum() > self.cfg.max_queries:
            order = torch.argsort(self.score, descending=True)
            keep = torch.zeros(K, dtype=torch.bool, device=self.device)
            keep[order[:self.cfg.max_queries]] = True
        if not keep.all():
            self._select(keep)

    def _relabel(self, parent):
        """Merge queries by union-find parent map (K,) -> compact ids."""
        roots, new = torch.unique(parent, return_inverse=True)
        if self.cfg.keep_points:
            self.pt_owner = [torch.where(o >= 0, new[o.clamp_min(0)], o) for o in self.pt_owner]
        Knew = len(roots)
        n = self.n_obs.float()
        feat = torch.zeros((Knew, self.feat.shape[1]), device=self.device).index_add_(0, new, self.feat * n[:, None])
        n_new = torch.zeros(Knew, device=self.device).index_add_(0, new, n)
        self.feat = feat / n_new.clamp_min(1)[:, None]
        self.score = torch.zeros(Knew, device=self.device).scatter_reduce(0, new, self.score, 'amax')
        self.n_obs = n_new.long()
        self.first_seen = torch.full((Knew,), 1 << 60, dtype=torch.int64, device=self.device).scatter_reduce(0, new, self.first_seen, 'amin')
        self.last_seen = torch.zeros(Knew, dtype=torch.int64, device=self.device).scatter_reduce(0, new, self.last_seen, 'amax')
        self._add_records_relabel(new)

    def _add_records_relabel(self, new):
        pairs, inv = torch.unique(torch.stack([new[self.owner], self.voxel], 1), dim=0, return_inverse=True)
        self.votes = torch.zeros(len(pairs), dtype=torch.int32, device=self.device).index_add_(0, inv, self.votes)
        self.owner, self.voxel = pairs[:, 0], pairs[:, 1]

    def _select(self, keep):
        idx = torch.nonzero(keep).squeeze(1)
        remap = torch.full((self.num_queries,), -1, dtype=torch.int64, device=self.device)
        remap[idx] = torch.arange(len(idx), device=self.device)
        rec = keep[self.owner]
        self.owner, self.voxel, self.votes = remap[self.owner[rec]], self.voxel[rec], self.votes[rec]
        if self.cfg.keep_points:
            self.pt_owner = [torch.where(o >= 0, remap[o.clamp_min(0)], o) for o in self.pt_owner]
        for name in ('feat', 'score', 'n_obs', 'first_seen', 'last_seen'):
            setattr(self, name, getattr(self, name)[idx])

    # ------------------------------------------------------------------ output
    @torch.no_grad()
    def render(self, xyz):
        """Scene-level masks (K, V) for arbitrary query points (e.g. mesh vertices): a point belongs to every
        query occupying its voxel. Returns (masks bool numpy, scores numpy)."""
        xyz = torch.as_tensor(np.asarray(xyz), dtype=torch.float32, device=self.device)
        v = voxelize(xyz, self.cfg.voxel)
        K, V = self.num_queries, len(v)
        out = torch.zeros((K, V), dtype=torch.bool, device=self.device)
        if K and len(self.voxel):
            keep = self.votes >= self.cfg.render_min_votes
            if self.cfg.render_min_vote_frac > 0:
                keep &= self.votes.float() >= self.cfg.render_min_vote_frac * self.n_obs[self.owner].float()
            if self.cfg.render_exclusive:
                # per voxel keep the record with the most votes (tie-break: higher vote fraction of its query)
                frac = self.votes.float() / self.n_obs[self.owner].float().clamp_min(1)
                key = self.votes.float() * 1000 + frac
                vox_k = self.voxel[keep]; key_k = key[keep]
                _, inv = torch.unique(vox_k, return_inverse=True)
                best = torch.full((int(inv.max()) + 1,), -1e9, device=self.device).scatter_reduce(0, inv, key_k, 'amax')
                win = key_k >= best[inv]
                idx_keep = torch.nonzero(keep).squeeze(1)[win]
                keep = torch.zeros_like(keep); keep[idx_keep] = True
            order = torch.argsort(self.voxel[keep])
            sv, so = self.voxel[keep][order], self.owner[keep][order]
            lo = torch.searchsorted(sv, v, right=False)
            hi = torch.searchsorted(sv, v, right=True)
            has = hi > lo
            # expand ranges (a voxel may be owned by several queries)
            idx_pt = torch.repeat_interleave(torch.nonzero(has).squeeze(1), (hi - lo)[has])
            starts = lo[has]; lens = (hi - lo)[has]
            offs = torch.arange(int(lens.sum()), device=self.device) - torch.repeat_interleave(torch.cumsum(lens, 0) - lens, lens)
            rec = torch.repeat_interleave(starts, lens) + offs
            out[so[rec], idx_pt] = True
        return out.cpu().numpy(), self.score.cpu().numpy()


    @torch.no_grad()
    def render_knn(self, xyz, chunk=4096, candidates=64):
        """ESAM-style render (nearest stored point) without a dense distance matrix: candidates are the stored
        points sharing the query point's voxel or a 26-neighbour voxel; the exact nearest among them wins.
        Points with no stored neighbour within one voxel get no instance (as in a KNN with a distance cut)."""
        assert self.cfg.keep_points and self.pt_xyz, 'render_knn needs keep_points=True'
        from .voxel import BITS
        xyz = torch.as_tensor(np.asarray(xyz), dtype=torch.float32, device=self.device)
        pts = torch.cat(self.pt_xyz); own = torch.cat(self.pt_owner)
        keep = own >= 0
        pts, own = pts[keep], own[keep]
        pv = voxelize(pts, self.cfg.voxel)
        order = torch.argsort(pv); pv, pts, own = pv[order], pts[order], own[order]
        r = torch.arange(-1, 2, device=self.device)
        offs = torch.stack(torch.meshgrid(r, r, r, indexing='ij'), -1).reshape(-1, 3)
        shift = (offs[:, 0] << (2 * BITS)) + (offs[:, 1] << BITS) + offs[:, 2]
        K = self.num_queries
        out = torch.zeros((K, len(xyz)), dtype=torch.bool, device=self.device)
        for s0 in range(0, len(xyz), chunk):
            q = xyz[s0:s0 + chunk]
            qv = voxelize(q, self.cfg.voxel)
            nb = (qv[:, None] + shift[None])                                   # (c, 27) neighbour voxel ids
            lo = torch.searchsorted(pv, nb, right=False); hi = torch.searchsorted(pv, nb, right=True)
            cnt = (hi - lo).clamp(max=candidates)                               # cap candidates per voxel
            total = int(cnt.sum())
            if total == 0:
                continue
            row = torch.repeat_interleave(torch.arange(len(q), device=self.device)[:, None].expand(-1, 27).reshape(-1), cnt.reshape(-1))
            base = torch.repeat_interleave(lo.reshape(-1), cnt.reshape(-1))
            within = torch.arange(total, device=self.device) - torch.repeat_interleave(torch.cumsum(cnt.reshape(-1), 0) - cnt.reshape(-1), cnt.reshape(-1))
            cand = base + within
            d = (pts[cand] - q[row]).norm(dim=1)
            best_d = torch.full((len(q),), float('inf'), device=self.device).scatter_reduce(0, row, d, 'amin')
            win = d <= best_d[row]
            out[own[cand[win]], row[win] + s0] = True
        return out.cpu().numpy(), self.score.cpu().numpy()
