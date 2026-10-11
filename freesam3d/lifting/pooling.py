"""Point features -> one feature per group (the lifted 3D query)."""
import numpy as np
import torch


def mean_l2(point_feats, group_ids, num_groups):
    """Baseline: average the point features inside each group, then L2-normalise."""
    point_feats = torch.as_tensor(point_feats, dtype=torch.float32)
    idx = (group_ids.to(point_feats.device, torch.long) if torch.is_tensor(group_ids)
           else torch.as_tensor(np.asarray(group_ids), dtype=torch.long, device=point_feats.device))
    sums = torch.zeros(num_groups, point_feats.shape[1], device=point_feats.device).index_add_(0, idx, point_feats)
    counts = torch.bincount(idx, minlength=num_groups).clamp(min=1).unsqueeze(1)
    return torch.nn.functional.normalize(sums / counts, dim=1)


def center_mean_l2(point_feats, group_ids, num_groups):
    """Subtract the frame-mean point feature first, then ``mean_l2``.

    FastSAM features follow SiLU, so they are mostly positive and share a large
    common component that pushes every cosine towards 1 (see README).
    """
    point_feats = torch.as_tensor(point_feats, dtype=torch.float32)
    return mean_l2(point_feats - point_feats.mean(0, keepdim=True), group_ids, num_groups)


POOLING = {'mean_l2': mean_l2, 'center_mean_l2': center_mean_l2}


def pool(method, point_feats, group_ids, num_groups):
    if method not in POOLING:
        raise KeyError(f'unknown pooling {method!r}; available {sorted(POOLING)}')
    return POOLING[method](point_feats, group_ids, num_groups)
