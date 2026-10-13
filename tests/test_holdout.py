"""Stage-1 holdout plumbing: the variant grid, the calibrated scorer, and one run end to end (CPU)."""
import numpy as np
import torch

from freesam3d.streaming.holdout import Variant, default_variants, evaluate_scene, make_memory, run_scene
from freesam3d.streaming.window_memory import FragmentFrame
from freesam3d.streaming.window_scores import CROSS_OFFSET, calibrated_score_fn

CFG = dict(stream_v2=dict(past_unit='block', voxel=0.05, touch_dilation=1, linkage='average_sparse',
                          admit_min_points=100, object_merge=True, purity=0.8))


def _graph(pair_feat, edges, has_intra, node_type):
    from freesam3d.streaming.window_memory import PairGraph
    num = int(np.max(node_type)) + 1
    return PairGraph(edges=np.asarray(edges, np.int64).reshape(-1, 2),
                     pair_feat=torch.as_tensor(np.asarray(pair_feat), dtype=torch.float32),
                     has_intra=torch.as_tensor(np.asarray(has_intra), dtype=torch.bool),
                     node_major=np.zeros(num, np.int64), node_purity=np.ones(num, np.float32),
                     node_n=np.ones(num, np.int64), node_type=np.asarray(node_type, np.int64),
                     node_prev_obj=np.full(num, -1, np.int64), node_geom=np.zeros((num, 11), np.float32),
                     node_feat=np.zeros((num, 4), np.float32), num_nodes=num, z_ref=0.0,
                     frame_pos=0, scene='s', frame_id=0)


def _frame(frag_pts, edges=(), feat=None, fid=0):
    xyz = np.concatenate([np.asarray(p, np.float32) for p in frag_pts])
    groups = np.concatenate([np.full(len(p), i, np.int64) for i, p in enumerate(frag_pts)])
    num = len(frag_pts)
    feat = np.zeros((num, 4), np.float32) if feat is None else np.asarray(feat, np.float32)
    e = np.asarray(edges, np.int64).reshape(-1, 2)
    return FragmentFrame('scene', fid, xyz, np.full_like(xyz, 128), groups, feat, e,
                         np.zeros((len(e), 14), np.float32), np.full(num, 0.9, np.float32),
                         np.zeros(len(xyz), np.int64), np.zeros(len(xyz), np.int64))


def test_calibrated_score_fn_binarises_each_scale_at_its_own_threshold():
    pf = np.zeros((3, CROSS_OFFSET + 15), np.float32)
    pf[1, CROSS_OFFSET + 3] = 0.7                    # contain_a_in_b of the cross edge
    g = _graph(pf, [[0, 1], [1, 2], [0, 2]], [True, False, False], [0, 0, 2])

    class _Model:
        def __call__(self, geom, feat, edges, ef):
            return torch.full((len(edges),), 0.0)    # sigmoid(0) = 0.5

    class _Std:
        def apply(self, geom, ef):
            return geom, ef

    fn = calibrated_score_fn(_Model(), _Std(), 'p2a', 'cpu', intra_threshold=0.4, cross_threshold=0.6)
    p = fn(g).numpy()
    assert p[0] == 1.0                               # intra edge: 0.5 >= 0.4
    assert p[1] == 1.0 and p[2] == 0.0               # cross edges: 0.7 >= 0.6, 0.0 < 0.6
    fn = calibrated_score_fn(_Model(), _Std(), 'p2a', 'cpu', intra_threshold=0.6, cross_threshold=0.6)
    assert fn(g).numpy()[0] == 0.0                   # the intra threshold moved, the cross one did not
    # without a model the rule scores every edge, intra ones included
    assert calibrated_score_fn(None, None, 'p2a', 'cpu', cross_threshold=0.6)(g).numpy().tolist() == [0.0, 1.0, 0.0]


def test_variant_validation_tag_and_grid():
    assert Variant('two_step', 'p2b', 0.65, 0.4).tag == 'two_step_p2b_0.4'
    for args in (('nope', 'p2a'), ('linkage', 'p2c')):
        try:
            Variant(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'{args} should be rejected')
    grid = default_variants([0.2, 0.5], modes=('linkage', 'two_step'), rules=('p2a',))
    assert [v.tag for v in grid] == ['linkage_p2a_0.2', 'linkage_p2a_0.5',
                                     'two_step_p2a_0.2', 'two_step_p2a_0.5']
    assert all(v.intra_threshold == 0.65 for v in grid)


def test_make_memory_wires_each_mode_to_its_own_knobs():
    link = make_memory(Variant('linkage', 'p2a', 0.65, 0.3), None, None, CFG, 'cpu')
    assert link.cfg.grouping == 'linkage' and link.cfg.threshold == 0.3
    calib = make_memory(Variant('calib', 'p2a', 0.65, 0.3), None, None, CFG, 'cpu')
    assert calib.cfg.grouping == 'linkage' and calib.cfg.threshold == 0.5
    two = make_memory(Variant('two_step', 'p2b', 0.65, 0.3), None, None, CFG, 'cpu')
    assert two.cfg.grouping == 'two_step' and two.cfg.threshold == 0.65
    assert two.cfg.cross_rule == 'p2b' and two.cfg.cross_threshold == 0.3
    assert all(m.cfg.window == 1 for m in (link, calib, two))


def test_run_scene_and_evaluate_scene_report_metrics():
    rng = np.random.default_rng(0)
    box = np.stack([rng.uniform(0, 0.2, 400), rng.uniform(0, 0.2, 400), rng.uniform(0, 0.2, 400)], 1)
    far = rng.uniform(5.0, 5.2, (200, 3))
    frames = [_frame([box, far], [[0, 1]], fid=0),
              _frame([box, far], [[0, 1]], fid=1)]
    for f in frames:                                  # one GT instance per fragment region
        f.gt = np.concatenate([np.ones(400, np.int64), np.full(200, 2, np.int64)])
    obj, attrs, _ = run_scene(frames, Variant('two_step', 'p2a', 0.65, 0.5), None, None, CFG, 'cpu')
    met = evaluate_scene(obj, attrs, frames, stuff=set(), voxel=0.05, min_gt_voxels=5,
                         min_pred_voxels=5, device='cpu')
    assert met['gt_voxels'] == 2 and met['objects'] == 2
    assert abs(met['ap50'] - 1.0) < 1e-9 and met['masks'] == 2
    assert 'dup_rate' in met and met['single_frame_objects'] == 0


def test_load_scene_frames_globs_the_scene_and_sorts_by_frame_id(tmp_path):
    """Regression: the cache lookup must glob (a bare ``dir/name_*.npz`` Path is not iterable)."""
    from freesam3d.merging import cache as mc
    from freesam3d.streaming.holdout import load_scene_frames
    n = 50
    for fid in (80, 0, 40):
        mc.save(dict(xyz=np.zeros((n, 3), np.float32), rgb=np.zeros((n, 3), np.uint8),
                     groups=np.zeros(n, np.int32), node_feat=np.zeros((1, 4), np.float32),
                     edges=np.zeros((0, 2), np.int32), edge_feat=np.zeros((0, 14), np.float32),
                     sizes=np.array([n], np.int32), group_mask_index=np.zeros(0, np.int32),
                     fastsam_scores=np.zeros(0, np.float32), gt_instance=np.ones(n, np.int32),
                     gt_nyu40=np.full(n, 5, np.uint8)),
                tmp_path / f'scene0000_00_{fid}.npz')
    mc.save(dict(xyz=np.zeros((n, 3), np.float32), rgb=np.zeros((n, 3), np.uint8),
                 groups=np.zeros(n, np.int32), node_feat=np.zeros((1, 4), np.float32),
                 edges=np.zeros((0, 2), np.int32), edge_feat=np.zeros((0, 14), np.float32),
                 sizes=np.array([n], np.int32), group_mask_index=np.zeros(0, np.int32),
                 fastsam_scores=np.zeros(0, np.float32), gt_instance=np.ones(n, np.int32),
                 gt_nyu40=np.full(n, 5, np.uint8)),
            tmp_path / 'scene9999_00_0.npz')            # another scene must not leak in
    frames = load_scene_frames(tmp_path, 'scene0000_00', gt='2d')
    assert [f.frame_id for f in frames] == [0, 40, 80]
    assert frames[0].gt.tolist() == [1] * n and frames[0].sem.tolist() == [5] * n
