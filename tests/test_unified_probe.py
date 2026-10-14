"""Block-fate analysis (CPU): first appearance, missed contact, low score, unlabeled.

One point per fragment, so each fragment's GT is unambiguous: fragments 0/1/2 -> instances 1/2/3,
3/4/5 -> block 3 (instances 2, 1, 3: three different, purity 1/3, impure), 6/7 -> block 4
(instance 1 twice). The vote table only counts labeled points, so purity is over labeled points only.
Blocks 0..4; with two objects the object nodes are the graph nodes 5 and 6, so an edge to object 10
(instance 1) is the pair (block, 5) and to object 11 (instance 2) is (block, 6).
"""
import numpy as np

from freesam3d.streaming.unified_probe import add_fates, unassigned_block_fates
from freesam3d.streaming.window_memory import FragmentFrame

GROUPS = np.arange(8)
GT = np.array([1, 2, 3, 2, 1, 3, 1, 1])
BLOCK = np.array([0, 1, 2, 3, 3, 3, 4, 4])
BLOCK_OBJ = [-1, -1, -1, -1, 10]              # only block 4 joined an object
BLOCK_N = [200.0, 150.0, 90.0, 120.0, 300.0]
OBJ_IDS = (10, 11)
OBJ_VOTES = [{1: 5}, {2: 4}]                  # object 10 -> instance 1, object 11 -> instance 2


def _frame():
    xyz = np.zeros((8, 3), np.float32)
    return FragmentFrame('s', 0, xyz, np.zeros((8, 3), np.uint8), GROUPS, np.zeros((8, 4), np.float32),
                         np.zeros((0, 2), np.int64), np.zeros((0, 14), np.float32),
                         np.ones(8, np.float32), GT, np.full(8, 5, np.int64))


def _info(block_obj=BLOCK_OBJ, edges=((0, 5),), etype=(1,)):
    return dict(num_blocks=5, block=BLOCK, block_obj=np.asarray(block_obj, np.int64),
                block_n=np.asarray(BLOCK_N, np.float64), obj_ids=np.asarray(OBJ_IDS, np.int64),
                obj_votes=OBJ_VOTES, edges=np.asarray(edges, np.int64).reshape(-1, 2),
                edge_type=np.asarray(etype, np.int64))


def test_fates_split_first_appearance_missed_contact_and_low_score():
    # block 0 (instance 1): object 10 holds it and touches the block -> the score stayed too low
    # block 1 (instance 2): object 11 holds it but does not touch the block -> never a candidate
    # block 2 (instance 3): no object holds instance 3 -> it has not appeared yet
    # block 3 (three different instances, purity 1/3): impure, no verdict
    f = unassigned_block_fates(_info(), _frame())
    assert f['unassigned'] == 4 and f['small'] == 1        # block 2 is below the point floor
    assert (f['scored_below'], f['not_a_candidate'], f['no_object_yet'], f['unlabeled']) == (1, 1, 1, 1)
    assert f['points']['scored_below'] == 200 and f['points']['not_a_candidate'] == 150
    assert f['points']['no_object_yet'] == 90 and f['points']['unlabeled'] == 120


def test_an_impure_block_is_unlabeled_even_when_the_object_is_a_candidate():
    """Block 3's majority *is* object 11's instance, and here it is a candidate - still no verdict."""
    f = unassigned_block_fates(_info(edges=((3, 6),)), _frame())
    assert f['unlabeled'] == 1 and f['scored_below'] == 0
    assert f['not_a_candidate'] == 2 and f['no_object_yet'] == 1     # blocks 0 and 1, block 2


def test_assigned_blocks_and_empty_frames_are_ignored():
    f = unassigned_block_fates(_info(block_obj=[10, 10, 10, 10, 10]), _frame())
    assert f['unassigned'] == 0
    assert sum(f[k] for k in ('scored_below', 'not_a_candidate', 'no_object_yet', 'unlabeled')) == 0
    empty = dict(num_blocks=0, block=np.zeros(0, np.int64), block_obj=np.zeros(0, np.int64),
                 block_n=np.zeros(0), obj_ids=np.zeros(0, np.int64), obj_votes=[],
                 edges=np.zeros((0, 2), np.int64), edge_type=np.zeros(0, np.int64))
    assert unassigned_block_fates(empty, _frame())['unassigned'] == 0


def test_add_fates_accumulates_counts_and_points():
    acc = add_fates({}, unassigned_block_fates(_info(), _frame()))
    acc = add_fates(acc, unassigned_block_fates(_info(), _frame()))
    assert acc['unassigned'] == 8 and acc['scored_below'] == 2 and acc['points']['scored_below'] == 400
    assert acc['small'] == 2
