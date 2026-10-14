"""Holdout sweep helpers (CPU): the grid, the nan-safe aggregation and the pick."""
import numpy as np

from freesam3d.streaming.unified_sweep import aggregate, pick_best, triples


def test_triples_is_the_full_product():
    assert triples([0.5, 0.6], [0.3], [0.2, 0.4]) == [(0.5, 0.3, 0.2), (0.5, 0.3, 0.4),
                                                     (0.6, 0.3, 0.2), (0.6, 0.3, 0.4)]


def test_aggregate_is_nan_safe_and_keeps_the_mask_count():
    rows = [dict(ap50=0.4, ap25=0.6, iou=0.5, r50=0.7, masks=30),
            dict(ap50=float('nan'), ap25=0.8, iou=0.6, r50=0.9, masks=50)]
    m = aggregate(rows)
    assert abs(m['ap50'] - 0.4) < 1e-9 and abs(m['ap25'] - 0.7) < 1e-9
    assert abs(m['masks'] - 40.0) < 1e-9 and m['scenes'] == 2


def test_pick_best_prefers_ap50_then_fewer_masks():
    e = lambda ap, masks: dict(metrics=dict(ap50=ap, masks=masks))
    assert pick_best([e(0.3, 10), e(0.5, 80), e(0.5, 40)])['metrics']['masks'] == 40
    assert pick_best([e(float('nan'), 10), e(0.2, 90)])['metrics']['ap50'] == 0.2
