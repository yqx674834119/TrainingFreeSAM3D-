"""Where the frames' mixed ("impure") fragments end up once a real pipeline has run (stage 0c).

Merging never consults GT. This module only *labels* a fragment as mixed — its labeled points do
not come from one GT instance at >= ``purity`` — and then asks which final object swallowed it.
That is reporting, the same role GT plays elsewhere in ``freesam3d/analysis`` and
``freesam3d/evaluation``; nothing here feeds a merge decision.

The fragment-vs-object lookup is one ``overlap_matrix`` call (matrix ops only, AGENTS.md §2);
object order matches ``WindowMemory.render`` so a row index is directly a rendered mask row.
"""
import numpy as np
import torch

from ..merging.labels import group_gt_votes
from .voxel import overlap_matrix, voxelize

FATES = ('own_single', 'glued_overmerge', 'glued_clean', 'dropped')


def frame_fragment_purity(groups, gt, num_groups):
    """(major_gt, purity, labeled_n) per fragment of one frame.

    ``purity`` is the share of the fragment's *labeled* points that belong to its majority GT
    instance; a fragment with no labeled point at all gets purity 0 and ``labeled_n`` 0, which
    callers must read as "unlabeled", not as "mixed".
    """
    num = int(num_groups)
    major = np.zeros(num, np.int64)
    purity = np.zeros(num, np.float32)
    vi, gi, ci = group_gt_votes(np.asarray(groups), np.asarray(gt), num)
    labeled_n = np.zeros(num, np.int64)
    if len(vi):
        order = np.lexsort((gi, -ci, vi))
        vs, first = vi[order], np.r_[True, vi[order][1:] != vi[order][:-1]]
        major[vs[first]] = gi[order][first]
        tot = np.bincount(vi, weights=ci, minlength=num)
        purity[vs[first]] = (ci[order][first] / np.maximum(tot[vs[first]], 1)).astype(np.float32)
        labeled_n = tot.astype(np.int64)
    return major, purity, labeled_n


def impure_fragments(frames, purity=0.8):
    """[(frame_pos, frag_id, major_gt, purity, n_points)] of every mixed fragment of the stream.

    Only fragments with at least one labeled point can be mixed; KMeans fill groups (no proposal,
    no GT) are skipped rather than counted as mixed.
    """
    out = []
    for pos, fr in enumerate(frames):
        num = int(len(fr.node_feat))
        major, pur, lab_n = frame_fragment_purity(fr.groups, fr.gt, num)
        n_pts = np.bincount(np.asarray(fr.groups, np.int64), minlength=num)
        for f in np.flatnonzero((lab_n > 0) & (pur < purity)):
            out.append((pos, int(f), int(major[f]), float(pur[f]), int(n_pts[f])))
    return out


def object_voxel_sets(mem):
    """Per-object voxel tensors in ``render`` row order (frozen table plus the revisable window)."""
    out = []
    for oid in sorted(mem.objects):
        vox = [mem.objects[oid]['vox']]
        for e in mem.window:
            for fr in e['frags']:
                if fr['obj'] == oid:
                    vox.append(fr['vox'])
        out.append(torch.unique(torch.cat(vox)))
    return out


def frag_object_rows(frame, frag_ids, obj_voxels, voxel=0.05):
    """(F,) index into ``obj_voxels`` of the object holding the majority of each fragment's voxels.

    ``-1`` when no object holds >= 50% of them, i.e. the fragment's object no longer exists
    (evicted, or never admitted). Objects only ever merge, so the fragment sits in exactly one
    of them and the arg-max containment is ~1.
    """
    frag_ids = np.asarray(frag_ids, np.int64)
    F, O = len(frag_ids), len(obj_voxels)
    if F == 0 or O == 0:
        return np.full(F, -1, np.int64)
    dev = obj_voxels[0].device
    groups = torch.as_tensor(np.asarray(frame.groups, np.int64), device=dev)
    xyz = torch.as_tensor(np.asarray(frame.xyz, np.float32), device=dev)
    rank = torch.full((int(groups.max()) + 1,), -1, dtype=torch.int64, device=dev)
    rank[torch.as_tensor(frag_ids, device=dev)] = torch.arange(F, device=dev)
    fr_of_pt = rank[groups]
    sel = fr_of_pt >= 0
    if not bool(sel.any()):
        return np.full(F, -1, np.int64)
    pairs = torch.unique(torch.stack([fr_of_pt[sel], voxelize(xyz[sel], voxel)], 1), dim=0)
    oo = torch.repeat_interleave(torch.arange(O, device=dev),
                                 torch.tensor([len(v) for v in obj_voxels], device=dev))
    inter, size_f, _ = overlap_matrix(pairs[:, 0], pairs[:, 1], oo, torch.cat(obj_voxels), F, O)
    contain = inter / size_f[:, None].clamp_min(1)
    best, rows = contain.max(1).values, contain.argmax(1)
    rows = rows.clone()
    rows[best < 0.5] = -1
    return rows.cpu().numpy()


def classify_fates(frag_rows, frag_n, obj_overmerge, obj_n_obs):
    """Point-weighted share of each destination, plus the raw point totals.

    * ``own_single``: the object it joined was seen in exactly one frame (the free finishing's
      "drop n_obs == 1" candidate);
    * ``glued_overmerge``: the object it joined covers >= 50% of two or more GT instances;
    * ``glued_clean``: glued into an object that is a single, coherent instance;
    * ``dropped``: its object no longer exists (evicted / never admitted).
    """
    rows = np.asarray(frag_rows, np.int64)
    w = np.asarray(frag_n, np.float64)
    overmerge, n_obs = np.asarray(obj_overmerge, bool), np.asarray(obj_n_obs, np.int64)
    totals = dict.fromkeys(FATES, 0.0)
    for r, weight in zip(rows, w):
        if r < 0:
            totals['dropped'] += weight
        elif overmerge[r]:
            totals['glued_overmerge'] += weight
        elif n_obs[r] <= 1:
            totals['own_single'] += weight
        else:
            totals['glued_clean'] += weight
    tot = max(sum(totals.values()), 1.0)
    return {k: v / tot for k, v in totals.items()}, totals
