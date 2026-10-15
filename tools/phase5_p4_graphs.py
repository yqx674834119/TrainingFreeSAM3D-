"""阶段 5 · P4：把 A 轮训练图的边特征离线扩成 73 列（不重采数据）。

对 <src>/<scene>.npz 的每一帧、每一侧（i=帧内 / c=跨帧），用存储的 geom_{tag} + edges_{tag}
（edges 是帧内局部下标）算 44 列 [e_a, e_b, e_a*e_b, |e_a-e_b|]（两个端点的 11 维 node_geom），
追加在 pf_{tag} 的 29 列之后，其余键原样写出。与推理端 phase5.p4.geom_pair 打开的在线计算
逐位一致（同一个 geom_pair_features、同一 float32 运算）。

    source tools/env.sh
    python tools/phase5_p4_graphs.py            # 默认 phase2/graphs/A -> phase5/p4/graphs/A
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freesam3d.config import load_config, resolve_path  # noqa: E402
from freesam3d.streaming.pair_features import geom_pair_features  # noqa: E402


def extend_pf(pf, geom, edges, node_ptr, edge_ptr):
    """(E,29) -> (E,73)：逐帧追加两个端点的成对几何编码（edges 为帧内局部下标）。"""
    out = np.zeros((len(pf), pf.shape[1] + 4 * geom.shape[1]), np.float32)
    out[:, :pf.shape[1]] = pf
    for t in range(len(node_ptr) - 1):
        e0, e1 = int(edge_ptr[t]), int(edge_ptr[t + 1])
        if e1 <= e0:
            continue
        n0, n1 = int(node_ptr[t]), int(node_ptr[t + 1])
        loc = np.asarray(edges[e0:e1], np.int64)
        assert loc.min() >= 0 and loc.max() < n1 - n0, f'edge index out of frame range at frame {t}'
        g = torch.as_tensor(geom[n0:n1], dtype=torch.float32)
        out[e0:e1, pf.shape[1]:] = geom_pair_features(g[loc[:, 0]], g[loc[:, 1]]).numpy()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/merge_net.yaml')
    p.add_argument('--src', default=None, help='源图目录；默认 <phase2.graphs_root>/A')
    p.add_argument('--dst', default=None, help='输出目录；默认 <phase5.p4.graphs>')
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    cfg = load_config(resolve_path(a.config))
    src = Path(a.src) if a.src else resolve_path(cfg['phase2']['graphs_root']) / 'A'
    dst = Path(a.dst) if a.dst else resolve_path(cfg['phase5']['p4']['graphs'])
    files = sorted(src.glob('*.npz'))
    if not files:
        raise SystemExit(f'no npz under {src}')
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, f in enumerate(files):
        out = dst / f.name
        if out.exists() and not a.overwrite:
            continue
        with np.load(f) as z:
            data = {k: z[k] for k in z.files}
        for tag in ('i', 'c'):
            key = f'pf_{tag}'
            if data[key].shape[1] != 29:
                raise SystemExit(f'{f.name}: {key} width {data[key].shape[1]} != 29 (already extended?)')
            data[key] = extend_pf(data[key], data[f'geom_{tag}'], data[f'edges_{tag}'],
                                  data[f'node_ptr_{tag}'], data[f'edge_ptr_{tag}'])
        tmp = out.with_name(out.name + '.part.npz')
        np.savez(tmp, **data)
        tmp.rename(out)
        if (i + 1) % 100 == 0 or i + 1 == len(files):
            print(f'[{i + 1}/{len(files)}] {f.name} ({time.time() - t0:.0f}s)', flush=True)
    print(f'done -> {dst}')


if __name__ == '__main__':
    main()
