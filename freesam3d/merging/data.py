"""Load cached frames and assemble them into batches for training / probing."""
from pathlib import Path

import numpy as np
import torch

from .cache import load, majority_semantic
from .labels import group_purity, objectness_labels, relabel


def list_cache(root, scenes=None):
    files = sorted(Path(root).glob('*.npz'))
    if scenes:
        keep = set(scenes)
        files = [f for f in files if f.stem.rsplit('_', 1)[0] in keep]
    return files


def scene_of(path):
    return Path(path).stem.rsplit('_', 1)[0]


class FrameSet:
    """All cached frames of a split held in memory (graph tensors only, no points)."""
    def __init__(self, files, keep_points=False, purity=None, min_labeled=0.3, stuff_nyu40=(1, 2, 22)):
        self.frames = []
        for f in files:
            d = load(f)
            edges = d['edges'].astype(np.int64)
            y, valid = d['edge_y'], d['edge_valid']
            if 'group_major' not in d:          # older cache: recompute per-group stats
                d['group_major'], d['group_purity'], d['group_labeled_frac'] = group_purity(
                    d['groups'], d['gt_instance'], len(d['sizes']))
            if 'group_sem_major' not in d:
                d['group_sem_major'] = majority_semantic(d['groups'], d['gt_instance'], d['gt_nyu40'], len(d['sizes']))
            if purity is not None:
                y, valid = relabel(d['group_major'], d['group_purity'], d['group_labeled_frac'], edges, purity, min_labeled)
            oy, ov = objectness_labels(d['group_major'], d['group_purity'], d['group_labeled_frac'],
                                       d['group_sem_major'], set(stuff_nyu40))
            item = dict(name=f.stem, scene=scene_of(f), edges=edges, node_geom=d['node_geom'],
                        node_feat=d['node_feat'], edge_feat=d['edge_feat'], y=y, valid=valid, sizes=d['sizes'],
                        obj_y=oy, obj_valid=ov)
            if keep_points:
                item.update(groups=d['groups'], gt_instance=d['gt_instance'], gt_nyu40=d['gt_nyu40'],
                            xyz=d['xyz'], pixels=d['pixels'], fastsam_scores=d['fastsam_scores'],
                            group_mask_index=d['group_mask_index'])
                if 'gt_instance_3d' in d:
                    item.update(gt_instance_3d=d['gt_instance_3d'], gt_sem_3d=d['gt_sem_3d'])
            self.frames.append(item)

    def __len__(self):
        return len(self.frames)

    def stack(self, indices, device):
        """Concatenate frames into one disjoint graph (edge indices offset per frame)."""
        geom, feat, edges, ef, y, valid, oy, ov, offset = [], [], [], [], [], [], [], [], 0
        for i in indices:
            fr = self.frames[i]
            geom.append(fr['node_geom'])
            feat.append(fr['node_feat'])
            edges.append(fr['edges'] + offset)
            ef.append(fr['edge_feat'])
            y.append(fr['y'])
            valid.append(fr['valid'])
            oy.append(fr['obj_y'])
            ov.append(fr['obj_valid'])
            offset += len(fr['node_geom'])
        t = lambda a, dt=torch.float32: torch.from_numpy(np.concatenate(a)).to(device=device, dtype=dt)
        return t(geom), t(feat), t(edges, torch.long), t(ef), t(y), t(valid, torch.bool), t(oy), t(ov, torch.bool)

    def edge_table(self):
        """(edge_feat, group_cos..., y) over all valid edges, for the linear probe."""
        x = np.concatenate([fr['edge_feat'][fr['valid']] for fr in self.frames])
        y = np.concatenate([fr['y'][fr['valid']] for fr in self.frames])
        scene = np.concatenate([[fr['scene']] * int(fr['valid'].sum()) for fr in self.frames])
        return x, y, scene


class Standardizer:
    """Per-column mean/std of node geometry and edge features, fitted on the training split.

    ``rel`` (optional) adds the stage-3 relational node columns: their statistics are stored as
    ``rm`` / ``rs`` and applied by ``apply_rel``; a standardizer built without them (every stage-2
    checkpoint) keeps ``rm = rs = None`` and ``apply_rel`` is the identity.
    """
    def __init__(self, geom, ef, rel=None):
        self.gm, self.gs = geom.mean(0), geom.std(0) + 1e-6
        self.em, self.es = ef.mean(0), ef.std(0) + 1e-6
        if rel is not None and len(rel):
            self.rm, self.rs = rel.mean(0), rel.std(0) + 1e-6
        else:
            self.rm = self.rs = None

    @classmethod
    def fit(cls, frameset):
        return cls(np.concatenate([f['node_geom'] for f in frameset.frames]),
                   np.concatenate([f['edge_feat'] for f in frameset.frames]))

    def state(self):
        s = dict(gm=self.gm, gs=self.gs, em=self.em, es=self.es)
        if self.rm is not None:
            s.update(rm=self.rm, rs=self.rs)
        return s

    @classmethod
    def from_state(cls, s):
        o = cls.__new__(cls)
        o.gm, o.gs, o.em, o.es = (np.asarray(s[k], np.float32) for k in ('gm', 'gs', 'em', 'es'))
        o.rm = np.asarray(s['rm'], np.float32) if 'rm' in s else None
        o.rs = np.asarray(s['rs'], np.float32) if 'rs' in s else None
        return o

    def apply(self, geom, ef):
        return (geom - torch.as_tensor(self.gm, device=geom.device)) / torch.as_tensor(self.gs, device=geom.device), \
               (ef - torch.as_tensor(self.em, device=ef.device)) / torch.as_tensor(self.es, device=ef.device)

    def apply_rel(self, rel):
        """Standardize the (G, R) relational columns; identity without fitted statistics."""
        if self.rm is None:
            return rel
        return (rel - torch.as_tensor(self.rm, device=rel.device)) / torch.as_tensor(self.rs, device=rel.device)
