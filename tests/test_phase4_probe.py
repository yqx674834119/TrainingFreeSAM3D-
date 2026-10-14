"""CPU tests for the phase-4 probe report (no GPU, no weights, no caches).

Covers the two complementary columns added in phase 4b — the 10-column hand *geometry* slice, and
hand(15) + learned cosine + the 4 contact columns (the §五 "special approval" condition) — plus the
gate keys the report always emits. Synthetic pair sets with the signal placed in a known block of
columns pin down exactly which columns each row reads.
"""
import numpy as np

from freesam3d.streaming.pair_features import cross_feature_names, pair_feature_names
from freesam3d.streaming.phase4_probe import phase4_report

CROSS = list(cross_feature_names())
N_INTRA = len(pair_feature_names()) - len(CROSS)


def make_pairs(n_per_scene=200, n_scenes=3, geom_signal=False, cos_signal=False, seed=0):
    """A labeled pair set with the signal placed where the test wants it.

    ``geom_signal`` puts it in the first 10 hand columns (the hand-geometry slice); ``cos_signal``
    puts it in the learned cosine. Everything else is noise, so a row that reads the wrong block
    scores ~0.5.
    """
    rng = np.random.default_rng(seed)
    n = n_per_scene * n_scenes
    y = np.zeros(n, bool)
    scene = np.empty(n, dtype=object)
    for s in range(n_scenes):
        sl = slice(s * n_per_scene, (s + 1) * n_per_scene)
        y[sl] = np.arange(n_per_scene) % 2 == 0
        scene[sl] = f'scene{s:04d}_00'
    hand = rng.normal(size=(n, len(CROSS)))
    if geom_signal:
        hand[:, :10] += y[:, None] * 2.0
    cos = rng.normal(size=n)
    if cos_signal:
        cos += y * 3.0
    types = np.zeros((n, 2), np.int64)
    types[: n // 2, 1] = 2
    types[n // 2:, :] = 2
    return dict(
        cos=cos, hand=hand, y=y, valid=np.ones(n, bool), rel=np.full(n, 'reobserve', dtype=object),
        scene=scene, fi=np.zeros(n, np.int64), gap=np.ones(n, np.int64), types=types,
        intra_cos=cos.copy(), intra_hand=rng.normal(size=(n, N_INTRA)), intra_prob=rng.normal(size=n),
        intra_y=y.copy(), intra_valid=np.ones(n, bool), intra_scene=scene.copy(),
        intra_fi=np.zeros(n, np.int64))


def test_report_emits_the_gate_and_slice_keys():
    rep = phase4_report(make_pairs())
    for key in ('cross', 'hardest_reobserve_vs_adjacent', 'touching_negative', 'intra', 'by_gap',
                'by_type', 'gate', 'purity'):
        assert key in rep, key
    for key in ('n', 'n_pos', 'learned_col', 'learned_logistic', 'hand15_logistic',
                'hand_geom10_logistic', 'learned_plus_contact', 'hand15_plus_learned_contact'):
        assert key in rep['cross'], key
    assert set(rep['gate']) >= {'hardest_ge_0_85', 'intra_not_below_groupmerge', 'passed'}
    assert rep['cross']['n'] == 600 and rep['cross']['n_pos'] == 300
    assert '1' in rep['by_gap'] and 'frag_object' in rep['by_type']


def test_hand_geom10_reads_only_the_first_ten_columns():
    """Signal in columns 0-9 -> both the 10-col slice and the full 15 read it."""
    rep = phase4_report(make_pairs(geom_signal=True))
    c = rep['cross']
    assert c['hand_geom10_logistic'] > 0.9
    assert c['hand15_logistic'] > 0.9
    # noise-only cosine must stay near chance
    assert c['learned_logistic'] < 0.6


def test_hand_geom10_ignores_columns_ten_to_fourteen():
    """Signal in the learned cosine only -> the hand rows stay at chance, the +cosine rows do not."""
    rep = phase4_report(make_pairs(cos_signal=True))
    c = rep['cross']
    assert c['hand_geom10_logistic'] < 0.6          # the slice is the geometric prefix, not the tail
    assert c['hand15_logistic'] < 0.6
    assert c['learned_logistic'] > 0.9
    assert c['learned_plus_contact'] > 0.9          # cosine + the 4 contact columns
    assert c['hand15_plus_learned_contact'] > 0.9   # hand(15) + cosine + the 4 contact columns


def test_hardest_block_is_a_subset_of_the_cross_block():
    rep = phase4_report(make_pairs())
    assert rep['hardest_reobserve_vs_adjacent']['n'] <= rep['cross']['n']
    assert rep['touching_negative']['n'] <= rep['cross']['n']


def test_intra_block_reads_the_fourteen_intra_columns():
    rep = phase4_report(make_pairs())
    i = rep['intra']
    for key in ('learned_col', 'learned_logistic', 'group_cos_col', 'group_cos_logistic',
                'groupmerge_prob_col', 'groupmerge_prob_logistic', 'n', 'n_pos'):
        assert key in i, key
    assert i['n'] == 600
