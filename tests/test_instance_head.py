"""CPU unit tests for the phase-4 instance head (no GPU, no FastSAM weights)."""
import numpy as np
import pytest
import torch

from freesam3d.models.instance_head import (
    HEAD_INPUTS, HEAD_NAMES, CachedFrame, InstanceHead, SampleGeom, _as_tensor, _collect_mined,
    encode_image, fragment_mean_features, fragment_pool, load_cached_frame, load_head,
    most_similar_different_instance, multi_positive_infonce, nearest_different_instance,
    pick_frames, sample_features, sample_points, save_head,
)


def test_head_inputs_match_documented_channels():
    assert dict(HEAD_INPUTS) == {'backbone': 640, 'neck_p3': 320, 'neck_p4': 640,
                                 'neck_p5': 640, 'proto_hidden': 320, 'proto': 32}


def test_encode_image_identity_640x480():
    rgb = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    x, geom = encode_image(rgb)
    assert x.shape == (1, 3, 480, 640) and x.dtype == torch.float32
    assert (geom.gain, tuple(geom.pad), tuple(geom.input_hw)) == (1.0, (0, 0), (480, 640))
    ref = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255
    assert torch.allclose(x[0], ref, atol=1.3e-7, rtol=0)


def test_encode_image_is_rgb_not_bgr():
    rgb = np.zeros((480, 640, 3), np.uint8)
    rgb[..., 0] = 255
    x, _ = encode_image(rgb)
    assert float(x[0, 0].min()) > 0.999 and float(x[0, 1:].max()) < 1e-6


def test_encode_image_letterbox_pads_and_geometry():
    rgb = np.full((500, 700, 3), 200, np.uint8)
    x, geom = encode_image(rgb)
    assert x.shape == (1, 3, 480, 640) and geom.input_hw == (480, 640)
    assert geom.gain == pytest.approx(min(480 / 500, 640 / 700))
    assert tuple(geom.pad) == (0, 11)
    assert float(x[0, :, 5, :].mean()) == pytest.approx(114 / 255, abs=1e-6)   # top border
    assert float(x[0, :, 12, :].mean()) == pytest.approx(200 / 255, abs=1e-6)  # first content rows


def test_sample_features_exact_at_pixel_centres():
    fmap = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    geom = SampleGeom(1.0, (0, 0), (2, 2))
    f = sample_features(fmap, np.array([[0, 1], [1, 0]]), geom)
    assert torch.allclose(f[0], fmap[0, :, 1, 0]) and torch.allclose(f[1], fmap[0, :, 0, 1])


def test_sample_features_averages_at_half_pixel():
    fmap = torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]])
    geom = SampleGeom(1.0, (0, 0), (2, 2))
    f = sample_features(fmap, np.array([[0.5, 0.5]]), geom)
    assert float(f[0, 0]) == pytest.approx(0.25)


def test_sample_features_accepts_tensor_pixels():
    fmap = torch.zeros(1, 1, 2, 2)
    geom = SampleGeom(1.0, (0, 0), (2, 2))
    f = sample_features(fmap, torch.tensor([[0, 0], [1, 1]]), geom)
    assert f.shape == (2, 1) and torch.isfinite(f).all()


def test_fragment_mean_features():
    pf = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    m = fragment_mean_features(pf, np.array([0, 0, 1, 1]), 2)
    assert torch.allclose(m[0], torch.tensor([1.0, 0.0])) and torch.allclose(m[1], torch.tensor([0.0, 1.0]))
    assert float(m[0].norm()) == pytest.approx(1.0)
    m2 = fragment_mean_features(pf, torch.tensor([0, 1, 2, 3]), 4)   # one point per group
    assert torch.allclose(m2, pf)


def test_instance_head_forward_shape_norm_and_budget():
    maps = {'backbone': torch.randn(1, 640, 8, 8), 'neck_p3': torch.randn(1, 320, 32, 32),
            'neck_p4': torch.randn(1, 640, 16, 16), 'neck_p5': torch.randn(1, 640, 8, 8),
            'proto_hidden': torch.randn(1, 320, 64, 64), 'proto': torch.randn(1, 32, 64, 64)}
    head = InstanceHead(width=64, out_dim=32, conv_layers=3)
    out = head(maps)
    assert out.shape == (1, 32, 64, 64)
    assert torch.allclose(out.norm(dim=1), torch.ones(1, 64, 64), atol=1e-5)
    assert head.num_params() < 1_000_000
    assert list(head.names) == list(HEAD_NAMES)


def test_infonce_matches_hand_computed():
    feat = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    feat = feat / feat.norm(dim=1, keepdim=True)
    ids = np.array([1, 1, 2, 2])
    loss, st = multi_positive_infonce(feat, ids, torch.tensor(0.5))
    same = torch.as_tensor(ids)[:, None] == torch.as_tensor(ids)[None, :]
    pos, neg = same & ~torch.eye(4, dtype=torch.bool), ~same
    sim = (feat @ feat.t()) / 0.5
    sim = sim - sim.max(1, keepdim=True).values
    p, n = (sim.exp() * pos).sum(1), (sim.exp() * neg).sum(1)
    assert float(loss) == pytest.approx(float(-(p / (p + n)).log().mean()), rel=1e-6)
    assert st['n_anchor'] == 4
    loss_t, _ = multi_positive_infonce(feat, torch.as_tensor(ids), torch.tensor(0.5))
    assert torch.allclose(loss, loss_t)


def test_infonce_without_negatives_is_zero():
    feat = torch.randn(3, 4)
    loss, st = multi_positive_infonce(feat, np.array([1, 1, 1]), torch.tensor(0.1))
    assert float(loss) == 0.0 and st['n_anchor'] == 0


def test_nearest_different_instance():
    xyz_a, ids_a = torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([1])
    xyz_p = torch.tensor([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [5.0, 0.0, 0.0]])
    ids_p = torch.tensor([2, 1, 2])
    idx = nearest_different_instance(xyz_a, ids_a, xyz_p, ids_p, 0.15, 4)
    assert idx.shape == (1, 3)
    assert idx[0, 0].item() == 0 and (idx[0, 1:] == -1).all()


def test_most_similar_different_instance():
    fa, ia = torch.tensor([[1.0, 0.0]]), torch.tensor([1])
    fp = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    ip = torch.tensor([2, 1, 2])
    idx = most_similar_different_instance(fa, ia, fp, ip, 2)
    assert idx[0].tolist() == [0, 2]


def test_collect_mined_dedupes_caps_and_drops_padding():
    rows = torch.tensor([[3, -1], [1, 3], [-1, -1]])
    assert _collect_mined(rows, cap=2).tolist() == [1, 3]
    assert _collect_mined(rows, cap=99).tolist() == [1, 3]


def test_sample_points_labelled_only():
    gt = np.array([0, 1, 1, 2, 0, 2])
    sel = sample_points(np.random.default_rng(0), gt, 2)
    assert len(sel) == 2 and (gt[sel] > 0).all() and len(np.unique(sel)) == 2
    assert sorted(sample_points(np.random.default_rng(1), gt, 10).tolist()) == [1, 2, 3, 5]


def test_pick_frames_chain_gaps_and_determinism():
    fids = [0, 20, 40, 60, 80, 100]
    picks = pick_frames(np.random.default_rng(3), fids, 3, max_gap=3)
    assert len(picks) == 3 and len(set(picks)) == 3 and set(picks) <= set(fids)
    idx = sorted(fids.index(p) for p in picks)
    assert all(1 <= b - a <= 3 for a, b in zip(idx, idx[1:]))
    assert pick_frames(np.random.default_rng(3), fids, 3, max_gap=3) == picks


def test_load_cached_frame_roundtrip(tmp_path):
    path = tmp_path / 'scene0000_00_20.npz'
    np.savez(path, xyz=np.zeros((3, 3), np.float32), pixels=np.zeros((3, 2), np.int64),
             groups=np.array([0, 0, 1]), gt_instance=np.array([1, 1, 2]),
             group_major=np.array([1, 2]), group_purity=np.array([1.0, 1.0], np.float32))
    fr = load_cached_frame(path)
    assert fr.scene == 'scene0000_00' and fr.frame_id == 20
    assert fr.groups.tolist() == [0, 0, 1] and fr.gt.tolist() == [1, 1, 2]
    assert fr.xyz.shape == (3, 3) and fr.group_major.tolist() == [1, 2]


def test_save_load_head_roundtrip(tmp_path):
    head = InstanceHead(width=16, out_dim=8, conv_layers=2)
    save_head(tmp_path / 'head.pt', head, log_tau=torch.tensor(-2.0))
    back, ck = load_head(tmp_path / 'head.pt')
    assert back.out_dim == 8 and ck['width'] == 16 and ck['conv_layers'] == 2
    assert ck['log_tau'] == pytest.approx(-2.0)
    maps = {n: torch.randn(1, c, 8, 8) for n, c in HEAD_INPUTS}
    assert torch.allclose(head(maps), back(maps), atol=1e-6)


def test_fragment_pool_means_purity_and_grad():
    frame = CachedFrame(scene='s', frame_id=0, xyz=np.zeros((4, 3), np.float32),
                        pixels=np.array([[0, 0], [1, 0], [2, 0], [3, 0]]),
                        groups=np.array([0, 0, 1, 1]), gt=np.array([1, 1, 2, 2]),
                        group_major=np.array([1, 2]), group_purity=np.array([1.0, 1.0], np.float32))
    dense = torch.zeros(1, 1, 1, 4)
    dense[0, 0, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    dense.requires_grad_(True)
    geom = SampleGeom(1.0, (0, 0), (1, 4))
    ff, ids = fragment_pool(dense, geom, frame, np.arange(4), min_points=2, purity=0.6)
    assert ff.shape == (2, 1) and ff.requires_grad
    assert ff[:, 0].tolist() == pytest.approx([0.5, 2.5])
    assert ids.tolist() == [1, 2]
    ff2, ids2 = fragment_pool(dense, geom, frame, np.arange(4), min_points=3, purity=0.6)
    assert ff2.shape[0] == 0 and len(ids2) == 0


def test_save_load_head_roundtrip_ablated(tmp_path):
    chans = {k: c for k, c in HEAD_INPUTS if k != 'proto'}
    head = InstanceHead(in_channels=chans, width=16, out_dim=8, conv_layers=2)
    save_head(tmp_path / 'head.pt', head)
    back, ck = load_head(tmp_path / 'head.pt')
    assert list(back.names) == list(chans) and ck['in_channels'] == chans
    maps = {n: torch.randn(1, c, 8, 8) for n, c in chans.items()}
    assert torch.allclose(head(maps), back(maps), atol=1e-6)
