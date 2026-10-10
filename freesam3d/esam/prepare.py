"""Build an ESAM-format data root for several ScanNet scenes with ESAM's own code.

Mirrors ``ESAM/local/prepare_subset.py`` (which handles one scene) but writes
into a separate workspace, so the user's ESAM reproduction data is untouched:

    <ws>/data/scannet200-mv_fast/   per-frame points / FastSAM super_points / GT
    <ws>/data/scannet200/           reconstructed mesh points + GT (``with_rec``)

Per-frame data comes from ESAM's unmodified
``data/scannet200-mv/load_scannet_mv_data_fast.process_cur_scan``.
"""
import importlib.util
import os
from pathlib import Path
import random
import shutil
import sys

import numpy as np


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_sens(source, dest, stride=20):
    """ScanNet .sens -> color/{id}.jpg (640x480), depth/{id}.png, pose/{id}.txt, every ``stride`` frames."""
    import io
    import json
    import struct
    import zlib
    from PIL import Image
    for folder in ('color', 'depth', 'pose', 'intrinsics'):
        (dest / folder).mkdir(parents=True, exist_ok=True)
    exported, invalid = [], []
    with open(source, 'rb') as f:
        def unpack(fmt):
            return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))
        assert unpack('I')[0] == 4, 'unsupported .sens version'
        f.seek(unpack('Q')[0], 1)                                   # sensor name
        for key in ('intrinsic_color', 'extrinsic_color', 'intrinsic_depth', 'extrinsic_depth'):
            np.savetxt(dest / 'intrinsics' / f'{key}.txt', np.array(unpack('16f'), np.float32).reshape(4, 4))
        color_comp, depth_comp, _, _, dw, dh = unpack('iiIIII')
        depth_shift, = unpack('f')
        assert color_comp == 2 and depth_comp == 1 and (dw, dh) == (640, 480) and depth_shift == 1000
        for frame_id in range(unpack('Q')[0]):
            pose = np.array(unpack('16f'), np.float32).reshape(4, 4)
            _, _, color_len, depth_len = unpack('QQQQ')
            if frame_id % stride:
                f.seek(color_len + depth_len, 1)
                continue
            color, depth = f.read(color_len), f.read(depth_len)
            Image.open(io.BytesIO(color)).convert('RGB').resize((640, 480), Image.Resampling.BILINEAR) \
                .save(dest / 'color' / f'{frame_id}.jpg', quality=95)
            Image.fromarray(np.frombuffer(zlib.decompress(depth), '<u2').reshape(dh, dw)).save(dest / 'depth' / f'{frame_id}.png')
            np.savetxt(dest / 'pose' / f'{frame_id}.txt', pose)
            exported.append(frame_id)
            if not np.isfinite(pose).all():
                invalid.append(frame_id)
    (dest / 'extraction.json').write_text(json.dumps(dict(source=str(source), export_stride=stride,
                                                          exported_frames=exported, invalid_pose_frames=invalid)))


def find_raw(scene, raw_roots):
    for root in raw_roots:
        d = Path(root) / scene
        if (d / f'{scene}.sens').exists():
            return d
    raise FileNotFoundError(f'{scene}: no .sens under {raw_roots}')


def _link(target, dest):
    if not dest.exists() and not dest.is_symlink():
        dest.symlink_to(target, target_is_directory=target.is_dir())


class ESAMDataBuilder:
    def __init__(self, esam_root, workspace, fastsam_checkpoint, raw_roots, sens_stride=20, seed=0):
        self.esam = Path(esam_root)
        self.ws = Path(workspace).resolve()
        self.mv = self.ws / 'data/scannet200-mv_fast'
        self.rec = self.ws / 'data/scannet200'
        self.raw_roots, self.stride, self.seed = raw_roots, sens_stride, seed
        self.fastsam_checkpoint = fastsam_checkpoint
        self.tsv = self.esam / 'data/scannet200/meta_data/scannetv2-labels.combined.tsv'
        for d in ('2D', '3D', 'meta_data', 'points', 'super_points', 'instance_mask',
                  'semantic_mask', 'axis_align_matrix'):
            (self.mv / d).mkdir(parents=True, exist_ok=True)
        for d in ('scans', 'points', 'super_points', 'instance_mask', 'semantic_mask', 'axis_align_matrix'):
            (self.rec / d).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.tsv, self.mv / 'meta_data' / self.tsv.name)
        self._fastsam = None

    def done(self, scene):
        return (self.rec / 'points' / f'{scene}.bin').exists() and any((self.mv / 'points' / scene).glob('*.bin'))

    def prepare(self, scene):
        raw = find_raw(scene, self.raw_roots)
        np.random.seed(self.seed)          # ESAM's sampling / KMeans use the global RNG
        random.seed(self.seed)
        import torch
        torch.manual_seed(self.seed)
        # 1. .sens -> color/depth/pose every `stride` frames
        frames_2d = self.mv / '2D' / scene
        if not (frames_2d / 'extraction.json').exists():
            self._extract_sens(raw / f'{scene}.sens', frames_2d)
        _link(raw, self.mv / '3D' / scene)
        # 2. FastSAM super points + 20k points + GT per frame (official preprocessing)
        sys.path.insert(0, str(self.esam / 'data/scannet200-mv'))
        mod = _load('esam_mv_fast', self.esam / 'data/scannet200-mv/load_scannet_mv_data_fast.py')
        cwd = os.getcwd()
        os.chdir(self.mv)
        try:
            mod.process_cur_scan(dict(scan_name_index=0, scan_name=scene, scan_num=1,
                                      path_dict=dict(DATA_PATH='./2D', INS_DATA_PATH='./2D',
                                                     AXIS_ALIGN_MATRIX_PATH='./3D')), self.fastsam(), 'val')
        finally:
            os.chdir(cwd)
        # 3. reconstructed mesh + mesh superpoints for scene-level evaluation (with_rec)
        _link(raw, self.rec / 'scans' / scene)
        export = _load('esam_scannet200_export', self.esam / 'data/scannet200/load_scannet_data.py').export
        vertices, semantics, instances, _, _, _, axis = export(
            str(raw / f'{scene}_vh_clean_2.ply'), str(raw / f'{scene}.aggregation.json'),
            str(raw / f'{scene}_vh_clean_2.0.010000.segs.json'), str(raw / f'{scene}.txt'),
            str(self.tsv), scannet200=True)
        import open3d as o3d
        import segmentator
        mesh = o3d.io.read_triangle_mesh(str(raw / f'{scene}_vh_clean_2.ply'))
        sp = segmentator.segment_mesh(torch.from_numpy(np.asarray(mesh.vertices).astype(np.float32)),
                                      torch.from_numpy(np.asarray(mesh.triangles).astype(np.int64))).numpy()
        for kind, arr, dtype in [('super_points', sp, np.int64), ('semantic_mask', semantics, np.int64),
                                 ('instance_mask', instances, np.int64), ('points', vertices, np.float32)]:
            arr.astype(dtype).tofile(self.rec / kind / f'{scene}.bin')      # points last = completion marker
        np.save(self.rec / 'axis_align_matrix' / f'{scene}.npy', axis)

    def _extract_sens(self, source, dest):
        """Same output as ESAM local/extract_sens.py, but keeps invalid-pose (-inf) frames.

        ESAM's process_cur_scan asserts contiguous frame ids and itself drops
        frames whose lifted points are NaN (invalid pose), as with the official
        ScanNet SensReader export it was written for.
        """
        export_sens(source, dest, self.stride)

    def fastsam(self):
        if self._fastsam is None:
            from fastsam import FastSAM
            self._fastsam = FastSAM(str(self.fastsam_checkpoint))
        return self._fastsam

    def write_infos(self, scenes):
        """Info pkl over ``scenes`` exactly as tools/create_data.py builds it (val split)."""
        (self.mv / 'meta_data/scannetv2_val.txt').write_text('\n'.join(scenes) + '\n')
        sys.path.insert(0, str(self.esam))
        import mmengine
        from tools.scannet_data_utils import ScanNetMVData
        from tools.update_infos_to_v2 import update_scannet200_mv_infos
        infos = ScanNetMVData(str(self.mv), split='val', scannet200=True).get_infos(num_workers=1)
        stage = self.ws / 'infos_stage'
        stage.mkdir(exist_ok=True)
        pkl = stage / 'scannet200_mv_oneformer3d_infos_val.pkl'
        mmengine.dump(infos, str(pkl))
        update_scannet200_mv_infos(str(pkl), str(self.mv))
