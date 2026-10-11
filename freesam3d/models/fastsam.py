"""Frozen FastSAM-x: everything-mode masks + dense features from one forward pass.

FastSAM-x is YOLOv8x-seg (a CNN). It has no ViT blocks and no SAM-style
256-d neck. ``FEATURE_SPECS`` lists the tensors we expose and their closest
SAM counterpart; see ``docs/fastsam_features.zh-CN.md`` for the reasoning.

Features are captured with forward hooks during the *same* ``predict`` call
that produces the masks, so preprocessing (letterbox, RGB, /255) is exactly
the one FastSAM used for segmentation.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# name -> (source tensors, stride, SAM analogue)
FEATURE_SPECS = {
    'backbone':     (('L9',), 32, 'SAM 最后一个 Transformer block 输出（neck 之前）'),
    'neck_p3':      (('L15',), 8, 'SAM neck 输出（mask 解码器的输入）；P3 是 Proto 的输入'),
    'neck_fpn':     (('L15', 'L18', 'L21'), 8, 'neck 三尺度 P3/P4/P5 上采样到 stride 8 后拼接'),
    'proto_hidden': (('proto_hidden',), 4, 'mask 解码器内部（Proto.cv2 输出）'),
    'proto':        (('proto',), 4, 'mask 原型（32 维），实例 mask = 系数 · 原型'),
}


@dataclass
class FastSAMOutput:
    masks: np.ndarray          # (N, H, W) bool, on the original image grid
    scores: np.ndarray         # (N,) confidence
    feature_maps: dict         # name -> (1, C, h, w) float32 tensor (on ``device`` of the runner)
    input_hw: tuple            # letterboxed network input size
    gain: float                # original -> input scale
    pad: tuple                 # (left, top) letterbox padding in input pixels


def _load_fastsam(checkpoint):
    from fastsam import FastSAM
    # ultralytics 8.0.120 calls torch.load without weights_only; the checkpoint
    # pickles a full model, which torch>=2.6 refuses by default. It is a local
    # trusted file, so opt into the legacy loader only for this call.
    original = torch.load

    def trusted_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original(*args, **kwargs)
    torch.load = trusted_load
    try:
        return FastSAM(str(checkpoint))
    finally:
        torch.load = original


def resolve_device(device):
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


class FastSAMRunner:
    def __init__(self, checkpoint, device='auto', imgsz=640, conf=0.1, iou=0.9,
                 fallback_iou=0.7, features=('backbone', 'neck_p3'), keep_on_device=True):
        unknown = set(features) - set(FEATURE_SPECS)
        if unknown:
            raise KeyError(f'unknown features {sorted(unknown)}; available {sorted(FEATURE_SPECS)}')
        self.device = resolve_device(device)
        self.kwargs = dict(device=self.device, retina_masks=True, imgsz=imgsz, conf=conf, verbose=False)
        self.iou, self.fallback_iou = iou, fallback_iou
        self.features = tuple(features)
        self.keep_on_device = keep_on_device      # False: copy maps to CPU (legacy behaviour)
        self.model = _load_fastsam(checkpoint)
        self._captured = {}
        self._register_hooks()

    def _register_hooks(self):
        layers = self.model.model.model
        proto = layers[-1].proto

        def save(key):
            def hook(_m, _inp, out):
                t = out.detach().float()
                self._captured[key] = t if self.keep_on_device else t.cpu()
            return hook
        needed = {src for f in self.features for src in FEATURE_SPECS[f][0]}
        for key in needed:
            if key.startswith('L'):
                layers[int(key[1:])].register_forward_hook(save(key))
        if 'proto_hidden' in needed:
            proto.cv2.register_forward_hook(save('proto_hidden'))
        if 'proto' in needed:
            proto.register_forward_hook(save('proto'))
        layers[0].register_forward_pre_hook(
            lambda _m, inp: self._captured.__setitem__('input_hw', tuple(inp[0].shape[-2:])))

    @torch.inference_mode()
    def __call__(self, rgb):
        bgr = np.ascontiguousarray(rgb[..., ::-1])   # ultralytics expects cv2 BGR arrays
        # ultralytics select_device() rewrites CUDA_VISIBLE_DEVICES process-wide ('-1' for cpu)
        # and never restores it, which disables CUDA for everything else in the process.
        import os
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        try:
            self._captured.clear()
            results = self.model(bgr, iou=self.iou, **self.kwargs)
            if not results:                          # ESAM: retry with iou=0.7
                self._captured.clear()
                results = self.model(bgr, iou=self.fallback_iou, **self.kwargs)
        finally:
            if visible is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = visible
        h, w = rgb.shape[:2]
        if results and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy() == 1.0   # ESAM format_result
            scores = results[0].boxes.conf.cpu().numpy().astype(np.float32)
        else:
            masks, scores = np.zeros((0, h, w), bool), np.zeros(0, np.float32)

        input_hw = self._captured['input_hw']
        gain = min(input_hw[0] / h, input_hw[1] / w)
        # Same rounding as ultralytics LetterBox / scale_boxes.
        pad = (int(round((input_hw[1] - w * gain) / 2 - 0.1)), int(round((input_hw[0] - h * gain) / 2 - 0.1)))
        maps = {name: self._compose(name) for name in self.features}
        return FastSAMOutput(masks, scores, maps, input_hw, gain, pad)

    def _compose(self, name):
        sources, _, _ = FEATURE_SPECS[name]
        tensors = [self._captured[s] for s in sources]
        if len(tensors) == 1:
            return tensors[0]
        size = tensors[0].shape[-2:]
        return torch.cat([F.interpolate(t, size=size, mode='bilinear', align_corners=False) for t in tensors], 1)


def sample_point_features(fmap, pixels_xy, out: FastSAMOutput):
    """Bilinearly sample ``fmap`` (1,C,h,w) at original-image pixels ``(x, y)``.

    Pixel centres are mapped into the letterboxed input; ``fmap`` covers the
    whole input (stride divides it), so ``align_corners=False`` is exact.
    """
    xy = torch.as_tensor(np.asarray(pixels_xy), dtype=torch.float32, device=fmap.device)
    x_in = (xy[:, 0] + 0.5) * out.gain + out.pad[0]
    y_in = (xy[:, 1] + 0.5) * out.gain + out.pad[1]
    grid = torch.stack([2 * x_in / out.input_hw[1] - 1, 2 * y_in / out.input_hw[0] - 1], -1)
    feats = F.grid_sample(fmap, grid[None, :, None], mode='bilinear', padding_mode='border',
                          align_corners=False)
    return feats[0, :, :, 0].T.contiguous()
