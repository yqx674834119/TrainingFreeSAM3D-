"""Proposal-cache loading for the mask scorer."""
from pathlib import Path

import numpy as np
import torch


def list_proposal_cache(root):
    return sorted(Path(root).glob('*.npz'))


def scene_of(path):
    return Path(path).stem.rsplit('_', 1)[0]


def load_proposals(path, with_masks=False):
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    d['feat'] = d['feat'].astype(np.float32)
    if with_masks:
        d['masks'] = np.unpackbits(d['masks'], axis=1, count=int(d['num_points'])).astype(bool)
    else:
        d.pop('masks')
    return d


class ProposalSet:
    """All proposals of a split in flat arrays (frame index kept for grouping)."""
    def __init__(self, files, with_masks=False):
        self.files = list(files)
        self.frames = [load_proposals(f, with_masks) for f in self.files]
        self.scalars = np.concatenate([f['scalars'] for f in self.frames])
        self.feat = np.concatenate([f['feat'] for f in self.frames])
        self.iou = np.concatenate([f['iou'] for f in self.frames])
        self.is_raw = np.concatenate([f['is_raw'] for f in self.frames])
        self.frame_id = np.concatenate([np.full(len(f['iou']), i) for i, f in enumerate(self.frames)])
        self.scene = np.array([scene_of(p) for p in self.files])[self.frame_id]

    def __len__(self):
        return len(self.iou)


class ScalarStandardizer:
    def __init__(self, x):
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6

    def apply(self, x):
        return (x - torch.as_tensor(self.mu, device=x.device)) / torch.as_tensor(self.sd, device=x.device)

    def state(self):
        return dict(mu=self.mu, sd=self.sd)

    @classmethod
    def from_state(cls, s):
        o = cls.__new__(cls)
        o.mu, o.sd = np.asarray(s['mu'], np.float32), np.asarray(s['sd'], np.float32)
        return o
