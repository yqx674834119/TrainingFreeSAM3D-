"""Stage-2 failure analysis: why did a block end up with no object?

Step 2 of the unified flow gives every block either the best-scoring object above the threshold or a
fresh object (when it holds enough points). The blocks left unassigned are the ones the brief asks
about: was the correct object *not among the candidates*, or did it not exist at all? That
distinction says whether stage 2's remaining error lives in the memory (the instance was never
opened, or never touched the block) or in the scorer (it was a candidate and scored too low).

The 2D GT enters only here, in the labels: no verdict changes a merge decision.
"""
import numpy as np

from .unified_flow import BLOCK_OBJECT
from .window_data import fragment_votes, votes_stats

FATES = ('no_object_yet', 'not_a_candidate', 'scored_below', 'unlabeled')


def block_vote_stats(info, fr):
    """(block majority instance, block purity) from the frame's GT and the flow's fragment->block map."""
    nb = int(info['num_blocks'])
    votes_f = fragment_votes(fr)
    block_votes = [dict() for _ in range(nb)]
    for i, bi in enumerate(np.asarray(info['block'], np.int64)):
        for k, c in votes_f[i].items():
            block_votes[bi][k] = block_votes[bi].get(k, 0) + c
    major, purity, _ = votes_stats(block_votes)
    return major, purity


def object_major(info):
    """(n_obj,) majority instance per object of the frame, from the flow's merged vote tables."""
    return votes_stats([dict(v) for v in info['obj_votes']])[0]


def unassigned_block_fates(info, fr, purity=0.6, admit_min_points=100):
    """Counts and point sums per fate over the frame's unassigned blocks.

    * ``no_object_yet``: no existing object's majority instance is the block's - the instance has not
      been opened yet in this stream (its first appearance);
    * ``not_a_candidate``: such an object exists but does not touch the block, so no edge between them
      ever reached the model (the memory missed the contact);
    * ``scored_below``: it touched the block and was a candidate, but the score stayed under the
      threshold (the scorer's mistake);
    * ``unlabeled``: the block's own GT is impure or absent, so no verdict is possible.

    ``small`` counts the unassigned blocks below the point floor; they could still have joined an
    object, so they stay in the three classes above.
    """
    out = {k: 0 for k in FATES}
    out.update(points={k: 0.0 for k in FATES}, unassigned=0, small=0)
    nb = int(info['num_blocks'])
    if nb == 0:
        return out
    block_obj = np.asarray(info['block_obj'], np.int64)
    block_n = np.asarray(info['block_n'], np.float64)
    todo = np.flatnonzero(block_obj < 0)
    if len(todo) == 0:
        return out
    b_major, b_purity = block_vote_stats(info, fr)
    o_major = object_major(info)
    edges = np.asarray(info['edges'], np.int64).reshape(-1, 2)
    etype = np.asarray(info['edge_type'], np.int64)
    cand = {}
    for a, b in edges[etype == BLOCK_OBJECT]:
        cand.setdefault(int(a), set()).add(int(b) - nb)
    for bi in todo:
        out['unassigned'] += 1
        if block_n[bi] < admit_min_points:
            out['small'] += 1
        gt = int(b_major[bi])
        if gt <= 0 or b_purity[bi] < purity:
            fate = 'unlabeled'
        elif any(int(o_major[k]) == gt for k in cand.get(int(bi), ()) if k < len(o_major)):
            fate = 'scored_below'
        elif any(int(o_major[k]) == gt for k in range(len(o_major))):
            fate = 'not_a_candidate'
        else:
            fate = 'no_object_yet'
        out[fate] += 1
        out['points'][fate] += float(block_n[bi])
    return out


def add_fates(acc, new):
    """Accumulate one frame's fates into a running total (in place, returns ``acc``)."""
    for k in FATES:
        acc[k] = acc.get(k, 0) + new[k]
        acc.setdefault('points', {}).setdefault(k, 0.0)
        acc['points'][k] += new['points'][k]
    acc['unassigned'] = acc.get('unassigned', 0) + new['unassigned']
    acc['small'] = acc.get('small', 0) + new['small']
    return acc
