"""Stage-2 training data: stream a scene's consecutive frames through the unified flow and save the
two graphs the model consumes, with GT labels (see ``tools/window_unified_collect.py``).

Round A drives the memory with stage 1's rule (``unified_flow.RuleAdapter``: the trained
``GroupMergeNet`` on the intra-frame edges, containment for block -> object, no object-object
merge); round B drives it with the model trained on round A. Both rounds build the graphs through
the same ``UnifiedGrouping`` code path, so the training input is exactly the inference input.

Labels (GT enters labels only, never a decision):
* edge: both ends pure (>= ``purity`` = 0.6, the stage-1 probe's convention) and the same majority
  instance -> 1, both pure and different -> 0, anything else is masked out of the loss (a
  merge-only model cannot express a split);
* node: history objects get their *final* voxel best IoU against the scene's thing GT (the ranking
  target); fragments and blocks keep the objectness label of ``merging.labels.objectness_labels``
  (pure, labeled enough, majority semantic is a thing).

One scene -> one ``.npz`` holding, for the intra graph and the cross graph separately, the
concatenated per-frame node/edge tensors plus ``node_ptr`` / ``edge_ptr`` offsets.
"""
import numpy as np
import torch

from ..merging.cache import majority_semantic
from ..merging.labels import group_gt_votes, group_purity, objectness_labels
from .pair_features import pair_feature_names
from .unified_flow import N_PAIR, make_unified_memory, object_state
from .voxel import dilate_sets, overlap_matrix, voxelize
from .window_memory import stats_from_frame

POS_PURITY = 0.6
MIN_LABELED = 0.3


# ------------------------------------------------------------------ per-frame GT statistics
def fragment_stats(fr):
    """(major, purity, labeled_frac, sem_major) per fragment of one frame (2D instance GT)."""
    groups = np.asarray(fr.groups, np.int64)
    gt = np.asarray(fr.gt, np.int64)
    num = int(len(fr.node_feat))
    major, purity, labfrac = group_purity(groups, gt, num)
    sem = np.zeros(len(gt), np.int64) if fr.sem is None else np.asarray(fr.sem, np.int64)
    return (major, purity.astype(np.float32), labfrac.astype(np.float32),
            majority_semantic(groups, gt, sem, num))


def fragment_votes(fr):
    """Per fragment, its GT vote table {instance id: labeled points}."""
    num = int(len(fr.node_feat))
    vi, gi, ci = group_gt_votes(np.asarray(fr.groups, np.int64), np.asarray(fr.gt, np.int64), num)
    out = [dict() for _ in range(num)]
    for i, k, c in zip(vi, gi, ci):
        out[int(i)][int(k)] = int(c)
    return out


def votes_stats(votes):
    """(major, purity, labeled_n) per node from its GT vote table; ties go to the smaller id."""
    num = len(votes)
    major = np.zeros(num, np.int64)
    purity = np.zeros(num, np.float32)
    labeled = np.zeros(num, np.float64)
    for i, v in enumerate(votes):
        if not v:
            continue
        tot = float(sum(v.values()))
        k = min(v, key=lambda g: (-v[g], g))
        major[i], purity[i], labeled[i] = k, v[k] / max(tot, 1.0), tot
    return major, purity, labeled


def edge_labels(major_a, purity_a, major_b, purity_b, purity=POS_PURITY):
    """(y, valid): 1 when both ends are pure and share the majority instance, 0 when both are pure
    and differ, masked otherwise."""
    valid = (purity_a >= purity) & (major_a > 0) & (purity_b >= purity) & (major_b > 0)
    return (major_a == major_b).astype(np.float32), valid


# ------------------------------------------------------------------ the intra graph
def intra_frame_graph(fr, voxel=0.05, dilation=1, device='cpu'):
    """One frame's intra graph exactly as ``UnifiedGrouping`` builds it.

    Node geometry from the frame's fragment statistics, the cached intra-frame edges carrying their
    14 cached columns plus the 15 cross columns (``same_frame`` / ``has_intra`` set, ``frame_gap``
    and ``frozen_count`` zero), node type 0, one observation, age 0. Used for the extra per-frame
    training cache (``work_dirs/merge_cache/train``) and asserted against the flow in the tests.
    """
    xyz = torch.as_tensor(np.asarray(fr.xyz, np.float32), device=device)
    groups = torch.as_tensor(np.asarray(fr.groups, np.int64), device=device)
    num = int(len(fr.node_feat))
    stats = stats_from_frame(fr, 0, device)
    z_ref = float(torch.median(xyz[:, 2]))
    edges = np.asarray(fr.edges, np.int64).reshape(-1, 2)
    owner, vox = torch.unique(torch.stack([groups, voxelize(xyz, voxel)], 1), dim=0).T
    counts = torch.bincount(owner, minlength=num)
    n_own = torch.repeat_interleave(torch.arange(num, device=device), counts)
    inter0, va, vb = overlap_matrix(n_own, vox, n_own, vox, num, num)
    d_own, d_vox = dilate_sets(n_own, vox, dilation)
    s1, _, _ = overlap_matrix(n_own, vox, d_own, d_vox, num, num)
    inter1 = torch.maximum(s1, s1.t())
    a = torch.as_tensor(edges[:, 0], dtype=torch.int64, device=device)
    b = torch.as_tensor(edges[:, 1], dtype=torch.int64, device=device)
    from .pair_features import pair_features
    pf = pair_features(stats, a, b, inter0[a, b], inter1[a, b], va, vb,
                       torch.zeros(len(a), device=device),
                       torch.ones(len(a), dtype=torch.bool, device=device),
                       torch.ones(len(a), dtype=torch.bool, device=device),
                       torch.zeros(len(a), device=device),
                       edge_feat=torch.as_tensor(np.asarray(fr.edge_feat, np.float32), device=device))
    return dict(node_geom=stats.geom(z_ref).cpu().numpy(), node_feat=stats.feat().cpu().numpy(),
                n_frames=np.ones(num, np.float32), age=np.zeros(num, np.float32), edges=edges,
                pair_feat=pf.cpu().numpy())


def collect_intra_frame(fr, stuff, purity=POS_PURITY, min_labeled=MIN_LABELED, voxel=0.05,
                        dilation=1, device='cpu'):
    """The intra graph of one frame plus its labels (extra training cache; no memory involved)."""
    g = intra_frame_graph(fr, voxel=voxel, dilation=dilation, device=device)
    major, pur, labfrac, sem = fragment_stats(fr)
    a, b = (g['edges'][:, 0], g['edges'][:, 1]) if len(g['edges']) else (np.zeros(0, np.int64),) * 2
    y, valid = edge_labels(major[a], pur[a], major[b], pur[b], purity)
    ny, nv = objectness_labels(major, pur, labfrac, sem, stuff, purity, min_labeled)
    return dict(**g, node_type=np.zeros(len(g['node_geom']), np.int64),
                edge_type=np.zeros(len(g['edges']), np.int64), edge_y=y, edge_valid=valid,
                node_y=ny.astype(np.float32), node_valid=nv)


# ------------------------------------------------------------------ scene-end object labels
def final_object_labels(mem, frames, voxel, stuff, min_gt_voxels, merges, device='cpu'):
    """(id -> final object id, final object id -> voxel best IoU) at the end of a scene.

    The GT is the voxel proxy of the holdout protocol (5 cm cells, cross-frame majority instance,
    thing and >= ``min_gt_voxels`` cells). An object's IoU is measured on its *final* extent
    (frozen part plus the window members) - what the memory would render. The merge events recorded
    by the flow fold every historical object id onto the object that survived.
    """
    from .evaluate import voxel_majority_gt
    cells, inst, sem = voxel_majority_gt(frames, voxel)
    ids, counts = np.unique(inst[inst > 0], return_counts=True)
    stuff = set(stuff)
    keep = []
    for k, n in zip(ids, counts):
        if n < min_gt_voxels:
            continue
        s = sem[inst == k]
        s = s[s > 0]
        if len(s) and int(np.bincount(s).argmax()) in stuff:
            continue
        keep.append(int(k))
    obj_ids, _, o_vox = object_state(mem)
    iou_of_final = {}
    if keep:
        slot = np.zeros(len(cells), np.int64)
        for i, k in enumerate(keep):
            slot[inst == k] = i + 1
        sizes = np.bincount(slot, minlength=len(keep) + 1)[1:].astype(np.float64)
        cells_t = torch.as_tensor(cells, device=device)
        slot_t = torch.as_tensor(slot, device=device)
        for oid, v in zip(obj_ids, o_vox):
            pos = torch.searchsorted(cells_t, v).clamp_max(len(cells_t) - 1)
            hit = cells_t[pos] == v
            if not bool(hit.any()):
                iou_of_final[int(oid)] = 0.0
                continue
            cnt = torch.bincount(slot_t[pos[hit]], minlength=len(keep) + 1)[1:].float().cpu().numpy()
            union = float(len(v)) + sizes - cnt.astype(np.float64)
            iou_of_final[int(oid)] = float((cnt / np.maximum(union, 1.0)).max())
    else:
        iou_of_final = {int(o): 0.0 for o in obj_ids}
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ev in merges:
        members = [int(m) for m in ev['members']]
        for m in members[1:]:
            ra, rb = find(members[0]), find(m)
            if ra != rb:
                parent[rb] = ra
    live = {int(o) for o in obj_ids}
    groups = {}
    for oid in set(parent) | live:
        groups.setdefault(find(oid), []).append(oid)
    final_of = {}
    for members in groups.values():
        alive = [m for m in members if m in live]
        f = alive[0] if alive else members[0]
        for m in members:
            final_of[m] = f
    return final_of, iou_of_final


# ------------------------------------------------------------------ one scene
def collect_scene(frames, cfg, model, std, device, stuff, purity=POS_PURITY, min_labeled=MIN_LABELED,
                  voxel=None, min_gt_voxels=20, overrides=None):
    """Stream one scene through the unified flow and return the per-frame graphs with labels."""
    wc = cfg['stream_v2']
    voxel = float(wc['voxel'] if voxel is None else voxel)
    dilation = int(wc['touch_dilation'])
    mem = make_unified_memory(cfg, model, std, device, **(overrides or {}))
    rows_i, rows_c, obj_ids_frames, num_blocks_frames, merges = [], [], [], [], []
    for fr in frames:
        mem.update(fr)
        info = mem.last_unified
        merges.extend(info['merged'])
        major_f, pur_f, labfrac_f, sem_f = fragment_stats(fr)
        sizes_f = np.bincount(np.asarray(fr.groups, np.int64), minlength=len(fr.node_feat)).astype(np.float64)
        votes_f = fragment_votes(fr)

        ia = info['intra']
        a, b = (ia['edges'][:, 0], ia['edges'][:, 1]) if len(ia['edges']) else (np.zeros(0, np.int64),) * 2
        y_i, v_i = edge_labels(major_f[a], pur_f[a], major_f[b], pur_f[b], purity)
        ny_i, nv_i = objectness_labels(major_f, pur_f, labfrac_f, sem_f, stuff, purity, min_labeled)
        rows_i.append(dict(**ia, node_type=np.zeros(len(ia['node_geom']), np.int64),
                           edge_type=np.zeros(len(ia['edges']), np.int64), edge_y=y_i, edge_valid=v_i,
                           node_y=ny_i.astype(np.float32), node_valid=nv_i))

        cg, nb, n_obj = info['cross'], int(info['num_blocks']), len(info['obj_ids'])
        block_votes = [dict() for _ in range(nb)]
        block_sem = [dict() for _ in range(nb)]
        block_lab, block_tot = np.zeros(nb), np.zeros(nb)
        for i, bi in enumerate(np.asarray(info['block'], np.int64)):
            for k, c in votes_f[i].items():
                block_votes[bi][k] = block_votes[bi].get(k, 0) + c
            if sem_f[i] > 0:
                block_sem[bi][int(sem_f[i])] = block_sem[bi].get(int(sem_f[i]), 0.0) + sizes_f[i]
            block_lab[bi] += labfrac_f[i] * sizes_f[i]
            block_tot[bi] += sizes_f[i]
        major_c, pur_c, _ = votes_stats(block_votes + [dict(v) for v in info['obj_votes']])
        ca, cb = (cg['edges'][:, 0], cg['edges'][:, 1]) if len(cg['edges']) else (np.zeros(0, np.int64),) * 2
        y_c, v_c = edge_labels(major_c[ca], pur_c[ca], major_c[cb], pur_c[cb], purity)
        sem_b = np.asarray([max(v, key=v.get) if v else 0 for v in block_sem], np.int64)
        ny_b, nv_b = objectness_labels(major_c[:nb], pur_c[:nb], block_lab / np.maximum(block_tot, 1),
                                       sem_b, stuff, purity, min_labeled)
        rows_c.append(dict(**cg, edge_type=np.asarray(info['edge_type'], np.int64),
                           edge_y=y_c, edge_valid=v_c,
                           node_y=np.concatenate([ny_b.astype(np.float32), np.full(n_obj, np.nan, np.float32)]),
                           node_valid=np.concatenate([nv_b, np.zeros(n_obj, bool)])))
        obj_ids_frames.append(np.asarray(info['obj_ids'], np.int64))
        num_blocks_frames.append(nb)

    final_of, iou_of_final = final_object_labels(mem, frames, voxel, stuff, min_gt_voxels, merges, device)
    n_cross = int(sum(len(r['node_geom']) for r in rows_c))
    niou = np.full(n_cross, np.nan, np.float32)
    niou_valid = np.zeros(n_cross, bool)
    off = 0
    for row, oids, nb in zip(rows_c, obj_ids_frames, num_blocks_frames):
        for k, oid in enumerate(oids):
            f = final_of.get(int(oid))
            if f is not None and f in iou_of_final:
                niou[off + nb + k] = iou_of_final[f]
                niou_valid[off + nb + k] = True
        off += len(row['node_geom'])
    feat_dim = int(frames[0].node_feat.shape[1]) if frames else 0
    return pack(rows_i, rows_c, niou, niou_valid, scene=frames[0].scene if frames else '',
                feat_dim=feat_dim)


# ------------------------------------------------------------------ storage
def pack(rows_i, rows_c, niou, niou_valid, scene='', feat_dim=0, frames=None):
    """Concatenate per-frame rows into the saved layout (pointers + fp16 appearance features)."""
    out = dict(scene=np.asarray(scene), frames=np.int64(len(rows_i)), feat_dim=np.int64(feat_dim))
    for tag, rows in (('i', rows_i), ('c', rows_c)):
        node_ptr, edge_ptr = np.zeros(len(rows) + 1, np.int64), np.zeros(len(rows) + 1, np.int64)
        for j, r in enumerate(rows):
            node_ptr[j + 1] = node_ptr[j] + len(r['node_geom'])
            edge_ptr[j + 1] = edge_ptr[j] + len(r['edges'])
        nf = len(rows_i) if frames is None else int(frames)
        if len(node_ptr) < nf + 1:              # an empty side still gets one pointer per frame
            node_ptr = np.concatenate([node_ptr, np.full(nf + 1 - len(node_ptr), node_ptr[-1], np.int64)])
            edge_ptr = np.concatenate([edge_ptr, np.full(nf + 1 - len(edge_ptr), edge_ptr[-1], np.int64)])

        def cat(key, dtype, dim):
            parts = [np.asarray(r[key], dtype) for r in rows]
            parts = [p for p in parts if p.ndim > 1 or p.size]
            if not parts:
                return np.zeros((0, dim), dtype) if dim else np.zeros(0, dtype)
            return np.concatenate(parts, 0).astype(dtype)

        out[f'node_ptr_{tag}'], out[f'edge_ptr_{tag}'] = node_ptr, edge_ptr
        out[f'geom_{tag}'] = cat('node_geom', np.float32, 11)
        out[f'feat_{tag}'] = cat('node_feat', np.float16, feat_dim)
        out[f'nframes_{tag}'] = cat('n_frames', np.float32, 0)
        out[f'age_{tag}'] = cat('age', np.float32, 0)
        out[f'kind_{tag}'] = cat('node_type', np.int8, 0)
        out[f'edges_{tag}'] = cat('edges', np.int32, 2)
        out[f'pf_{tag}'] = cat('pair_feat', np.float32, N_PAIR)
        out[f'etype_{tag}'] = cat('edge_type', np.int8, 0)
        out[f'ey_{tag}'] = cat('edge_y', np.float32, 0)
        out[f'ev_{tag}'] = cat('edge_valid', bool, 0)
        out[f'ny_{tag}'] = cat('node_y', np.float32, 0)
        out[f'nv_{tag}'] = cat('node_valid', bool, 0)
    out['niou_c'] = np.asarray(niou, np.float32)
    out['niou_valid_c'] = np.asarray(niou_valid, bool)
    return out


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
