"""Build per-frame caches from the ESAM ScanNet 2D export for GroupMergeNet.

Each cached frame (``<cache>/<split>/<scene>_<frame>.npz``) holds the lifted
20k points, exclusive groups, the adjacency graph with all edge features,
GT edge labels and the per-point GT needed by the evaluator. Building the
graph here (not at train time) keeps training and the feature probe cheap.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from ..io.frames import RGBDFrame
from ..lifting.query_lifting import QueryLifting
from .graph import build_graph
from .graph_gpu import build_graph_gpu
from .labels import edge_labels, group_purity


def read_scene_list(path):
    return [s.strip() for s in Path(path).read_text().splitlines() if s.strip()]


def frame_ids(scene_dir, stride=2, per_scene=0):
    """Frame ids of a scene at ``stride`` x the export stride, evenly subsampled to ``per_scene``."""
    ids = sorted(int(p.stem) for p in (scene_dir / 'color').glob('*.jpg'))[::max(1, stride)]
    if per_scene and per_scene < len(ids):
        ids = [ids[i] for i in np.linspace(0, len(ids) - 1, per_scene).round().astype(int)]
    return ids


def load_frame(root, scene, fid, intrinsic):
    d = Path(root) / scene
    pose = np.loadtxt(d / 'pose' / f'{fid}.txt')
    if not np.isfinite(pose).all():
        return None, None
    rgb = np.asarray(Image.open(d / 'color' / f'{fid}.jpg').convert('RGB'))
    depth = np.asarray(Image.open(d / 'depth' / f'{fid}.png'))
    inst = np.asarray(Image.open(d / 'instance' / f'{fid}.png')).astype(np.int64)
    label = np.asarray(Image.open(d / 'label' / f'{fid}.png')).astype(np.int64)
    K = np.array([[intrinsic[0], 0, intrinsic[2]], [0, intrinsic[1], intrinsic[3]], [0, 0, 1]], np.float64)
    return RGBDFrame(f'{scene}_{fid}', scene, fid, rgb, depth, pose.astype(np.float64), K), (inst, label)


class CacheBuilder:
    def __init__(self, lifting_cfg, graph_cfg, label_cfg, feature='backbone', boundary_feature='proto'):
        lifting_cfg = dict(lifting_cfg)
        lifting_cfg['features'] = sorted({feature, boundary_feature})
        lifting_cfg['gt'] = dict(lifting_cfg.get('gt', {}), enabled=False)
        self.lifting = QueryLifting(lifting_cfg)
        self.feature, self.boundary_feature = feature, boundary_feature
        self.gcfg, self.lcfg = graph_cfg, label_cfg

    def build(self, frame, inst, label, seed):
        self.lifting.cfg['lifting']['seed'] = int(seed)
        lifted = self.lifting(frame)
        px = lifted.pixels
        gt = inst[px[:, 1], px[:, 0]]
        nyu = label[px[:, 1], px[:, 0]]
        pf = lifted.point_feats[self.boundary_feature]
        gf = lifted.group_feats[self.feature]
        if self.gcfg.get('backend', 'cpu') == 'gpu':
            g = build_graph_gpu(lifted.xyz, lifted.rgb, lifted.group_ids, gf, pf, lifted.fastsam.scores,
                                lifted.group_mask_index, lifted.fastsam.masks, self.gcfg['adjacency_eps'],
                                self.gcfg['voxel'], self.gcfg['boundary_k'], self.gcfg.get('feature_centering', True),
                                device=self.lifting.runner.device)
        else:
            g = build_graph(lifted.xyz, lifted.rgb, lifted.group_ids, gf.cpu().numpy(), pf.cpu().numpy(),
                            lifted.fastsam.scores, lifted.group_mask_index, lifted.fastsam.masks,
                            self.gcfg['adjacency_eps'], self.gcfg['voxel'], self.gcfg['boundary_k'],
                            self.gcfg.get('feature_centering', True))
        y, valid = edge_labels(lifted.group_ids, gt, g.edges, self.lcfg['purity'], self.lcfg['min_labeled'])
        major, pur, lab = group_purity(lifted.group_ids, gt, len(g.sizes))
        sem_major = majority_semantic(lifted.group_ids, gt, nyu, len(g.sizes))
        return dict(group_major=major.astype(np.int32), group_purity=pur, group_labeled_frac=lab,
                    group_sem_major=sem_major.astype(np.int32),xyz=lifted.xyz.astype(np.float32), rgb=lifted.rgb, pixels=px.astype(np.int16),
                    groups=lifted.group_ids.astype(np.int32), group_mask_index=lifted.group_mask_index.astype(np.int32),
                    gt_instance=gt.astype(np.int32), gt_nyu40=nyu.astype(np.uint8),
                    edges=g.edges.astype(np.int32), node_geom=g.node_geom, node_feat=g.node_feat.astype(np.float16),
                    edge_feat=g.edge_feat, sizes=g.sizes.astype(np.int32), edge_y=y, edge_valid=valid,
                    fastsam_scores=lifted.fastsam.scores.astype(np.float32))


def save(arrays, path):
    np.savez_compressed(path, **arrays)


def load(path):
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    d['node_feat'] = d['node_feat'].astype(np.float32)
    return d


def attach_esam_gt(cache_dir, mv_root):
    """Add ESAM's 3D-transferred per-frame GT (``instance_mask``/``semantic_mask`` .bin) to cached
    frames of the same scene/frame, by nearest neighbour on the 20k points (sub-cm)."""
    from scipy.spatial import cKDTree
    mv_root, done = Path(mv_root), 0
    for f in sorted(Path(cache_dir).glob('*.npz')):
        scene, fid = f.stem.rsplit('_', 1)
        p3 = mv_root / 'points' / scene / f'{fid}.bin'
        if not p3.exists():
            continue
        with np.load(f) as z:
            d = {k: z[k] for k in z.files}
        if 'gt_instance_3d' in d:
            done += 1
            continue
        pts3 = np.fromfile(p3, np.float32).reshape(-1, 6)[:, :3]
        dist, idx = cKDTree(pts3).query(d['xyz'], k=1)
        d['gt_instance_3d'] = np.fromfile(mv_root / 'instance_mask' / scene / f'{fid}.bin', np.int64)[idx].astype(np.int32)
        d['gt_sem_3d'] = np.fromfile(mv_root / 'semantic_mask' / scene / f'{fid}.bin', np.int64)[idx].astype(np.int32)
        d['gt3d_nn_dist'] = dist.astype(np.float32)
        np.savez_compressed(f, **d)
        done += 1
    return done


def majority_semantic(groups, gt, sem, num_groups):
    """Majority semantic id among each group's labeled points (0 = none)."""
    out = np.zeros(num_groups, np.int64)
    lab = gt > 0
    if not lab.any():
        return out
    pairs = np.stack([groups[lab], sem[lab]], 1)
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    best = np.zeros(num_groups, np.int64)
    for (g, k), c in zip(uniq, counts):
        if c > best[g]:
            best[g], out[g] = c, k
    return out
