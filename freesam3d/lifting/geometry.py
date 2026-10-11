"""Depth back-projection to world coordinates and fixed-size point sampling."""
import numpy as np


def backproject(depth, intrinsic, pose_c2w, depth_scale=1000.0, min_depth_raw=0.1):
    """Valid depth pixels -> world xyz, following ESAM ``convert_from_uvd``.

    Returns ``xyz`` (M,3) in ScanNet unaligned world coordinates and the source
    ``pixels`` (M,2) as integer ``(x, y)`` = ``(u, v)``.
    """
    v, u = np.nonzero(depth > min_depth_raw)
    z = depth[v, u].astype(np.float64) / depth_scale
    uv1 = np.stack([u, v, np.ones_like(u)], 0).astype(np.float64)
    cam = (np.linalg.inv(intrinsic[:3, :3]) @ uv1) * z
    world = pose_c2w[:3, :3] @ cam + pose_c2w[:3, 3:4]
    return world.T.astype(np.float32), np.stack([u, v], 1).astype(np.int32)


def sample_indices(num_valid, num_points=20000, seed=0):
    """ESAM ``random_sampling``: without replacement unless too few points."""
    if num_valid == 0:
        raise ValueError('frame has no valid depth')
    rng = np.random.default_rng(seed)
    return rng.choice(num_valid, num_points, replace=num_valid < num_points)


def backproject_torch(depth, intrinsic, pose_c2w, depth_scale=1000.0, min_depth_raw=0.1, device='cuda'):
    """``backproject`` on the GPU. Returns (xyz (M,3) float32, pixels (M,2) int64 as (u, v))."""
    import torch
    # uint16 is not indexable on CUDA; upload as int32 (ScanNet depth < 65536 mm)
    d = torch.as_tensor(np.asarray(depth).astype(np.int32), device=device)
    v, u = torch.nonzero(d > min_depth_raw, as_tuple=True)
    z = d[v, u].to(torch.float64) / depth_scale
    K_inv = torch.as_tensor(np.linalg.inv(np.asarray(intrinsic, np.float64)[:3, :3]), device=device)
    P = torch.as_tensor(np.asarray(pose_c2w, np.float64), device=device)
    uv1 = torch.stack([u.double(), v.double(), torch.ones_like(z)], 0)
    cam = (K_inv @ uv1) * z
    world = P[:3, :3] @ cam + P[:3, 3:4]
    return world.T.float(), torch.stack([u, v], 1)
