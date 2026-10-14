"""Phase-4 gate probe: does the learned instance feature beat the hand features on the F1.7 pairs?

Diagnostic only — GT labels the pairs, nothing here feeds a merge. Pairs are collected with the
*exact* stage-1 driver (window=1, the holdout-selected ``two_step`` variant, 640-d cached
features), so the cross-frame numbers are directly comparable with F1.7 (logistic on the 15 hand
columns 0.8198; re-observe vs adjacent 0.7923).

The learned feature of a current-frame fragment is its cached (G,32) row; a frozen object's is
the point-count-weighted running mean of its member fragments' features (mirrored through the
memory's object merges via a wrapper on ``_merge_objects``). Score = cosine. Objects are scored
on their *past* contributions only, exactly like the memory scores them.
"""
from pathlib import Path

import numpy as np

from .pair_features import cross_feature_names, pair_feature_names
from .window_probe import POS_PURITY, RELATION_PAIRS, column_auc, logistic_auc, relation_of

CROSS_N = len(cross_feature_names())


def collect_phase4_pairs(frames, variant, model, std, cfg, device, feats_dir, purity=POS_PURITY):
    """Stream the frames through the variant's memory; keep every labeled cross-frame edge
    (learned cosine + the 15 hand columns + gap + relation) and every labeled intra-frame edge
    (learned cosine + the 14 cached columns + the driver's probability)."""
    from ..models.instance_head import load_feature_cache
    from .holdout import make_memory
    from .two_step import as_bool_np

    mem = make_memory(variant, model, std, cfg, device)
    names = pair_feature_names()
    ca, cb = names.index('contain_a_in_b'), names.index('contain_b_in_a')
    keys = ('cos', 'hand', 'y', 'valid', 'rel', 'scene', 'fi', 'gap', 'types',
            'intra_cos', 'intra_hand', 'intra_prob', 'intra_y', 'intra_valid', 'intra_scene', 'intra_fi')
    acc = {k: [] for k in keys}
    obj_sum, obj_cnt, obj_last = {}, {}, {}
    orig_merge = mem._merge_objects

    def merge(keep, drop):
        orig_merge(keep, drop)
        if drop in obj_sum:
            obj_sum[keep] = obj_sum.get(keep, 0.0) + obj_sum.pop(drop)
            obj_cnt[keep] = obj_cnt.get(keep, 0.0) + obj_cnt.pop(drop)
        obj_last.pop(drop, None)
    mem._merge_objects = merge

    feats_dir = Path(feats_dir)
    for fi, fr in enumerate(frames):
        feat = load_feature_cache(feats_dir / f'{fr.scene}_{fr.frame_id}.npz')
        pt_obj = np.asarray(mem.update(fr), np.int64)
        g = mem.last_graph
        prob = np.asarray(mem.score_fn(g).detach().cpu(), np.float64).reshape(-1)
        kind, prev = np.asarray(g.node_type), np.asarray(g.node_prev_obj)
        major, pur, node_n = np.asarray(g.node_major), np.asarray(g.node_purity), np.asarray(g.node_n)
        num_cur = int(np.count_nonzero(kind == 0))
        # node instance features: current fragments = cached rows, frozen objects = past mean
        nf = np.zeros((len(kind), feat.shape[1]), np.float64)
        nf[:num_cur] = feat
        for j in range(num_cur, len(kind)):
            o = int(prev[j])
            if kind[j] == 2 and obj_cnt.get(o, 0.0) > 0:
                nf[j] = obj_sum[o] / obj_cnt[o]
        nf /= np.maximum(np.linalg.norm(nf, axis=1, keepdims=True), 1e-9)
        cos = nf @ nf.T
        e = np.asarray(g.edges, np.int64)
        if len(e):
            pf = g.pair_feat.cpu().numpy()
            intra = as_bool_np(g.has_intra)
            a, b = e[:, 0], e[:, 1]
            pure = (pur >= purity) & (major > 0)
            same, valid = major[a] == major[b], pure[a] & pure[b]
            last = np.array([obj_last.get(int(prev[j]), -10 ** 9) if kind[j] == 2 else fi
                             for j in range(len(kind))])
            gap = fi - np.minimum(last[a], last[b])
            contain = np.maximum(pf[:, ca], pf[:, cb])
            rel = np.asarray([relation_of(bool(s), float(c)) for s, c in zip(same, contain)])
            cosv = cos[a, b]
            sel = ~intra
            if sel.any():
                acc['cos'].append(cosv[sel]); acc['hand'].append(pf[sel][:, 14:])
                acc['y'].append(same[sel]); acc['valid'].append(valid[sel]); acc['rel'].append(rel[sel])
                acc['scene'].append(np.full(int(sel.sum()), fr.scene, dtype=object))
                acc['fi'].append(np.full(int(sel.sum()), fi)); acc['gap'].append(gap[sel])
                acc['types'].append(np.stack([kind[a[sel]], kind[b[sel]]], 1))
            sel = intra
            if sel.any():
                acc['intra_cos'].append(cosv[sel]); acc['intra_hand'].append(pf[sel][:, :14])
                acc['intra_prob'].append(prob[sel]); acc['intra_y'].append(same[sel])
                acc['intra_valid'].append(valid[sel])
                acc['intra_scene'].append(np.full(int(sel.sum()), fr.scene, dtype=object))
                acc['intra_fi'].append(np.full(int(sel.sum()), fi))
        # accumulate this frame's contributions *after* scoring (objects are scored on the past)
        frag_obj = np.full(num_cur, -1, np.int64)
        frag_obj[np.asarray(fr.groups, np.int64)] = pt_obj
        for i in np.flatnonzero(frag_obj >= 0):
            o = int(frag_obj[i])
            obj_sum[o] = obj_sum.get(o, 0.0) + feat[i].astype(np.float64) * float(node_n[i])
            obj_cnt[o] = obj_cnt.get(o, 0.0) + float(node_n[i])
            obj_last[o] = fi
    return {k: (np.concatenate(v) if v else np.zeros((0,), np.float64)) for k, v in acc.items()}


def _cos_logistic(cos, y, scene):
    return logistic_auc(np.asarray(cos, np.float64).reshape(-1, 1), y, scene)


def _cross_block(cos, hand, y, scene, hand_names):
    """One labeled pair set: learned-cosine AUC, the 15-col hand control, and the phase-4b
    complementary rows — the 10 hand *geometry* columns alone, learned+contact, and
    hand(15) + learned cosine + the 4 contact columns (the "special approval" condition of §五)."""
    out = dict(n=int(len(y)), n_pos=int(np.asarray(y).sum()),
               learned_col=float(column_auc(cos, y)),
               learned_logistic=float(_cos_logistic(cos, y, scene)),
               hand15_logistic=float(logistic_auc(hand, y, scene)),
               hand_geom10_logistic=float(logistic_auc(np.asarray(hand)[:, :10], y, scene)))
    idx = [hand_names.index(n) for n in ('inter0_frac_min', 'inter1_frac_min',
                                         'contain_a_in_b', 'contain_b_in_a')]
    c = np.asarray(cos, np.float64).reshape(-1, 1)
    out['learned_plus_contact'] = float(logistic_auc(np.c_[c, hand[:, idx]], y, scene))
    out['hand15_plus_learned_contact'] = float(logistic_auc(np.c_[hand, c, hand[:, idx]], y, scene))
    return out


def phase4_report(pairs, purity=POS_PURITY):
    """The gate numbers: (a) intra vs group_cos / GroupMergeNet prob, (b) cross vs 15-col logistic,
    (c) re-observe vs adjacent, (d) learned + contact, and AUC by cross-frame gap 1/2/3."""
    names = pair_feature_names()
    hand_names = list(cross_feature_names())
    rep = dict(purity=purity)

    v = pairs['intra_valid'].astype(bool)
    yi = pairs['intra_y'][v].astype(bool)
    gi = names.index('group_cos')
    assert gi < 14, 'group_cos is not an intra-frame column'
    rep['intra'] = dict(
        n=int(len(yi)), n_pos=int(yi.sum()),
        learned_col=float(column_auc(pairs['intra_cos'][v], yi)),
        learned_logistic=float(_cos_logistic(pairs['intra_cos'][v], yi, pairs['intra_scene'][v])),
        group_cos_col=float(column_auc(pairs['intra_hand'][v][:, gi], yi)),
        group_cos_logistic=float(logistic_auc(pairs['intra_hand'][v][:, gi:gi + 1], yi, pairs['intra_scene'][v])),
        groupmerge_prob_col=float(column_auc(pairs['intra_prob'][v], yi)),
        groupmerge_prob_logistic=float(logistic_auc(pairs['intra_prob'][v][:, None], yi, pairs['intra_scene'][v])))

    v = pairs['valid'].astype(bool)
    cos, hand, y, scene = pairs['cos'][v], pairs['hand'][v], pairs['y'][v].astype(bool), pairs['scene'][v]
    rel, gap = pairs['rel'][v], pairs['gap'][v]
    rep['cross'] = _cross_block(cos, hand, y, scene, hand_names)
    m = (rel == 'reobserve') | (rel == 'adjacent')
    rep['hardest_reobserve_vs_adjacent'] = _cross_block(cos[m], hand[m], y[m], scene[m], hand_names)
    i0 = hand_names.index('inter0_frac_min')
    mneg = (~y) & (hand[:, i0] > 0)
    mt = y | mneg
    rep['touching_negative'] = _cross_block(cos[mt], hand[mt], y[mt], scene[mt], hand_names)
    rep['by_gap'] = {}
    for g in (1, 2, 3):
        mg = gap == g
        if mg.sum() >= 50:
            rep['by_gap'][str(g)] = _cross_block(cos[mg], hand[mg], y[mg], scene[mg], hand_names)
    rep['by_type'] = {}
    ty = pairs['types'][v]
    for tag, mt2 in (('frag_object', ty[:, 1] == 2), ('object_object', (ty[:, 0] == 2) & (ty[:, 1] == 2))):
        if mt2.sum() >= 50:
            rep['by_type'][tag] = _cross_block(cos[mt2], hand[mt2], y[mt2], scene[mt2], hand_names)
    rep['gate'] = dict(
        hardest_ge_0_85=bool(rep['hardest_reobserve_vs_adjacent']['learned_logistic'] >= 0.85),
        intra_not_below_groupmerge=bool(rep['intra']['learned_col'] >= rep['intra']['groupmerge_prob_col']))
    rep['gate']['passed'] = bool(rep['gate']['hardest_ge_0_85'] and rep['gate']['intra_not_below_groupmerge'])
    return rep
