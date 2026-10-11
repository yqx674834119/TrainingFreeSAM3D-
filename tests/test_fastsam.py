from pathlib import Path

import numpy as np
import pytest
import torch

from freesam3d.models.fastsam import FastSAMOutput, sample_point_features

CKPT = Path('/home/yqx/SegmentAnything3D/AutoSeg3D/data/FastSAM-x.pt')
FRAME = Path('/home/yqx/SegmentAnything3D/ESAM/work_dirs/feature_benchmark/inputs/scene0690_00_000000.npz')


@pytest.mark.parametrize('stride', [4, 8, 32])
def test_point_sampling_hits_the_right_cell_with_letterbox(stride):
    # Original 100x200 image into a 160x256 input: gain 1.28 plus 16 px top/bottom pad.
    # Border pixels must clamp to the edge cell, not blend with zero padding.
    h, w, H, W = 100, 200, 160, 256
    gain = min(H / h, W / w)
    pad = (int(round((W - w * gain) / 2 - 0.1)), int(round((H - h * gain) / 2 - 0.1)))
    fh, fw = H // stride, W // stride
    yy, xx = torch.meshgrid(torch.arange(fh, dtype=torch.float32), torch.arange(fw, dtype=torch.float32), indexing='ij')
    fmap = torch.stack([xx, yy])[None]
    out = FastSAMOutput(None, None, {}, (H, W), gain, pad)
    px = np.array([[0, 0], [199, 99], [100, 50]])
    f = sample_point_features(fmap, px, out).numpy()
    centre_x = ((px[:, 0] + 0.5) * gain + pad[0]) / stride - 0.5
    centre_y = ((px[:, 1] + 0.5) * gain + pad[1]) / stride - 0.5
    np.testing.assert_allclose(f[:, 0], np.clip(centre_x, 0, fw - 1), atol=1e-4)
    np.testing.assert_allclose(f[:, 1], np.clip(centre_y, 0, fh - 1), atol=1e-4)


@pytest.mark.skipif(not (CKPT.exists() and FRAME.exists()), reason='FastSAM checkpoint / frame not available')
def test_runner_features_and_masks():
    from freesam3d.io.frames import load_npz_frame
    from freesam3d.models.fastsam import FEATURE_SPECS, FastSAMRunner
    frame = load_npz_frame(FRAME)
    runner = FastSAMRunner(CKPT, 'cpu', features=tuple(FEATURE_SPECS))
    out = runner(frame.rgb)
    assert out.masks.shape[1:] == frame.rgb.shape[:2] and len(out.masks) > 0
    H, W = out.input_hw
    for name, (_, stride, _) in FEATURE_SPECS.items():
        assert out.feature_maps[name].shape[-2:] == (H // stride, W // stride), name
    assert out.feature_maps['proto'].shape[1] == 32


@pytest.mark.skipif(not (CKPT.exists() and FRAME.exists()), reason='FastSAM checkpoint / frame not available')
def test_cpu_runner_does_not_clobber_cuda_visible_devices():
    import os
    from freesam3d.io.frames import load_npz_frame
    from freesam3d.models.fastsam import FastSAMRunner
    before = os.environ.get('CUDA_VISIBLE_DEVICES')
    FastSAMRunner(CKPT, 'cpu', features=('proto',))(load_npz_frame(FRAME).rgb)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == before
