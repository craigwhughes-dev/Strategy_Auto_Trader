"""Tests for the first cut's pass/fail machinery. These decide the result, so they must be exact."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Strategy_Auto_Trader.research import phase_scoring as ps
from Strategy_Auto_Trader.research.crash_phases import Episode


def series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2000-01-03", periods=len(values)))


@pytest.fixture
def setup():
    """Peak 100 at 0, trough 50 at 5, episode runs to 15."""
    px = series([100, 90, 80, 70, 60, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
    ep = Episode(peak_idx=0, trough_idx=5, end_idx=15, recovery_idx=15,
                 peak_close=100.0, trough_close=50.0, recovered=True)
    return px, ep


def fire_at(n: int, *idx: int) -> np.ndarray:
    f = np.zeros(n, dtype=bool)
    for i in idx:
        f[i] = True
    return f


def test_firing_just_after_the_trough_is_a_hit(setup):
    px, ep = setup
    sc = ps.score_episode(fire_at(len(px), 7), px, ep, hit_window=5)
    assert sc.hit and not sc.false_alarm


def test_firing_at_the_trough_itself_counts(setup):
    px, ep = setup
    assert ps.score_episode(fire_at(len(px), 5), px, ep, hit_window=5).hit


def test_firing_beyond_the_window_is_not_a_hit(setup):
    px, ep = setup
    assert not ps.score_episode(fire_at(len(px), 12), px, ep, hit_window=5).hit


def test_window_boundary_is_inclusive(setup):
    px, ep = setup
    assert ps.score_episode(fire_at(len(px), 10), px, ep, hit_window=5).hit
    assert not ps.score_episode(fire_at(len(px), 11), px, ep, hit_window=5).hit


def test_early_firing_with_a_big_further_fall_is_a_false_alarm(setup):
    px, ep = setup
    # Fires at index 1 (close 90); trough 50 is 44% lower, well past the 10% test.
    sc = ps.score_episode(fire_at(len(px), 1), px, ep, hit_window=5)
    assert sc.false_alarm and not sc.hit


def test_early_firing_with_only_a_small_further_fall_is_not_a_false_alarm():
    """A signal just before a bottom that barely dips further is not the failure mode being measured."""
    px = series([100, 54, 52, 51, 50, 60, 70, 80])
    ep = Episode(peak_idx=0, trough_idx=4, end_idx=7, recovery_idx=None,
                 peak_close=100.0, trough_close=50.0, recovered=False)
    # Fires at index 1 (close 54): trough 50 is only 7.4% below, under the 10% threshold.
    assert not ps.score_episode(fire_at(len(px), 1), px, ep, fa_decline=0.10).false_alarm
    # With a stricter 5% threshold the same firing does count.
    assert ps.score_episode(fire_at(len(px), 1), px, ep, fa_decline=0.05).false_alarm


def test_hit_and_false_alarm_are_independent_and_can_both_be_true(setup):
    px, ep = setup
    sc = ps.score_episode(fire_at(len(px), 1, 7), px, ep, hit_window=5)
    assert sc.hit and sc.false_alarm and sc.n_firings == 2


def test_invalid_window_is_rejected(setup):
    px, ep = setup
    with pytest.raises(ValueError):
        ps.score_episode(fire_at(len(px), 7), px, ep, hit_window=0)


def test_shock_weighting_stops_one_crowded_shock_dominating():
    """Twelve markets in one shock must not outvote a shock recorded by a single market."""
    rows = pd.DataFrame(
        [{"shock_id": 0, "hit": True, "false_alarm": False} for _ in range(12)]
        + [{"shock_id": 1, "hit": False, "false_alarm": False}]
    )
    sc = ps.aggregate(rows)
    assert sc.n_shocks == 2
    assert sc.hit_rate == pytest.approx(0.5)      # 1.0 and 0.0 averaged, not 12/13
    assert sc.edge == pytest.approx(0.5)


def test_partial_agreement_within_a_shock_averages(setup):
    rows = pd.DataFrame([
        {"shock_id": 0, "hit": True, "false_alarm": False},
        {"shock_id": 0, "hit": False, "false_alarm": True},
    ])
    sc = ps.aggregate(rows)
    assert sc.hit_rate == pytest.approx(0.5)
    assert sc.false_alarm_rate == pytest.approx(0.5)
    assert sc.edge == pytest.approx(0.0)


def test_aggregate_of_nothing_is_not_a_crash():
    sc = ps.aggregate(pd.DataFrame(columns=["shock_id", "hit", "false_alarm"]))
    assert sc.n_shocks == 0 and np.isnan(sc.hit_rate)


def test_rotation_preserves_the_firing_count():
    f = fire_at(20, 3, 4, 9)
    for shift in range(12):
        assert ps.rotate(f, 2, 13, shift).sum() == f.sum()


def test_rotation_leaves_bars_outside_the_episode_untouched():
    f = fire_at(20, 0, 5, 19)
    out = ps.rotate(f, 2, 13, 3)
    assert out[0] and out[19], "firings outside the rotated span must survive"


def test_rotation_by_the_span_length_is_the_identity():
    f = fire_at(20, 3, 4, 9)
    np.testing.assert_array_equal(ps.rotate(f, 2, 13, 12), f)


def test_rotation_handles_an_empty_span():
    f = fire_at(5, 1)
    np.testing.assert_array_equal(ps.rotate(f, 3, 2, 1), f)


def test_permutation_p_value_is_bounded_and_never_zero(setup):
    px, ep = setup
    payload = [(fire_at(len(px), 7), px, ep, 0)]
    p, null = ps.permutation_p_value(1.0, payload, n_permutations=200)
    assert 0.0 < p <= 1.0
    assert len(null) == 200


def test_an_indicator_no_better_than_chance_gets_a_large_p_value(setup):
    """A single firing placed at random inside the episode should not look significant."""
    px, ep = setup
    payload = [(fire_at(len(px), 7), px, ep, 0)]
    observed = ps.aggregate(pd.DataFrame([{"shock_id": 0, "hit": True, "false_alarm": False}])).edge
    p, _ = ps.permutation_p_value(observed, payload, n_permutations=500)
    assert p > 0.05, f"a lone firing in a 16-bar episode should not be significant, got p={p}"


def test_rotation_table_shift_zero_matches_direct_scoring(setup):
    """Pins the fast rotation tables to the slow direct scorer they replace."""
    px, ep = setup
    for firings in (fire_at(len(px), 7), fire_at(len(px), 1), fire_at(len(px), 1, 7), fire_at(len(px), 12)):
        h, fa = ps.rotation_tables(firings, px, ep, hit_window=5)
        direct = ps.score_episode(firings, px, ep, hit_window=5)
        assert bool(h[0]) == direct.hit
        assert bool(fa[0]) == direct.false_alarm


def test_rotation_table_every_shift_matches_direct_scoring(setup):
    px, ep = setup
    firings = fire_at(len(px), 2, 8)
    h, fa = ps.rotation_tables(firings, px, ep, hit_window=5)
    for shift in range(len(h)):
        rotated = ps.rotate(firings, ep.peak_idx, ep.end_idx, shift)
        direct = ps.score_episode(rotated, px, ep, hit_window=5)
        assert bool(h[shift]) == direct.hit, f"hit mismatch at shift {shift}"
        assert bool(fa[shift]) == direct.false_alarm, f"false-alarm mismatch at shift {shift}"


def test_rotation_table_of_a_silent_indicator_is_all_false(setup):
    px, ep = setup
    h, fa = ps.rotation_tables(np.zeros(len(px), dtype=bool), px, ep)
    assert not h.any() and not fa.any()


def test_permutation_with_no_episodes_returns_nan():
    p, null = ps.permutation_p_value(0.5, [], n_permutations=10)
    assert np.isnan(p) and len(null) == 0
