"""阶段 5 · P4（CPU）：成对几何边特征的名称与数值、pair_features 的 opt-in 追加语义。"""
import torch

from freesam3d.streaming.pair_features import (geom_pair_features, geom_pair_names,
                                               pair_feature_names, pair_features)


class _Stats:
    """pair_features 用到的最小 NodeStats 替身。"""

    def __init__(self, G=3, C=4):
        torch.manual_seed(0)
        self.n = torch.full((G,), 5.0)
        self.hi = torch.rand(G, 3) + 1.0
        self.lo = torch.rand(G, 3)
        self._feat = torch.nn.functional.normalize(torch.rand(G, C), dim=1)

    def centroid(self):
        return (self.hi + self.lo) / 2

    def extent(self):
        return (self.hi - self.lo).clamp_min(1e-3)

    def feat(self):
        return self._feat


def test_geom_pair_names_and_dims():
    assert len(geom_pair_names()) == 44
    assert len(pair_feature_names()) == 29
    assert len(pair_feature_names(geom_pair=True)) == 73
    assert pair_feature_names(geom_pair=True)[:29] == pair_feature_names()


def test_geom_pair_features_values():
    ga = torch.arange(11, dtype=torch.float32)[None]
    gb = torch.arange(11, dtype=torch.float32)[None] * 2
    out = geom_pair_features(ga, gb)
    assert out.shape == (1, 44)
    assert torch.allclose(out[0, :11], ga[0])
    assert torch.allclose(out[0, 11:22], gb[0])
    assert torch.allclose(out[0, 22:33], ga[0] * gb[0])
    assert torch.allclose(out[0, 33:], (ga[0] - gb[0]).abs())


def test_pair_features_geom_pair_appends_only():
    stats = _Stats()
    a, b = torch.tensor([0, 1]), torch.tensor([1, 2])
    args = (stats, a, b, torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0]),
            torch.full((3,), 5.0), torch.full((3,), 5.0), torch.tensor([1.0, 2.0]),
            torch.tensor([True, False]), torch.tensor([True, True]), torch.tensor([0.0, 1.0]))
    base = pair_features(*args)
    assert base.shape == (2, 29)
    ga, gb = torch.rand(2, 11), torch.rand(2, 11)
    ext = pair_features(*args, geom_a=ga, geom_b=gb)
    assert ext.shape == (2, 73)
    assert torch.allclose(ext[:, :29], base)
    assert torch.allclose(ext[:, 29:], geom_pair_features(ga, gb))
