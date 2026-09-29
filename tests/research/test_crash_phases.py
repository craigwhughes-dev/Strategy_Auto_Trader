"""Tests for the P1 mechanical episode detector and phase labeller."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Strategy_Auto_Trader.research import crash_phases as cp


def series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2000-01-03", periods=len(values)))


def test_default_min_depth_is_the_ratified_20_percent():
    """A 0.25 floor excluded COVID (-21.6% on the GBP-unhedged asset), the best recovery event."""
    assert cp.MIN_DEPTH == 0.20
    assert len(cp.find_episodes(series([100, 78, 100]))) == 1


def test_no_episode_when_drawdown_stays_shallower_than_threshold():
    # -19% never reaches the -20% definition.
    assert cp.find_episodes(series([100, 81, 100])) == []


def test_boundary_drawdown_exactly_at_threshold_qualifies():
    # 80 is exactly (1 - 0.20) * 100, and the rule is inclusive.
    eps = cp.find_episodes(series([100, 80, 100]))
    assert len(eps) == 1
    assert eps[0].depth_pct == pytest.approx(-0.20)


def test_peak_is_the_high_before_the_fall_not_the_series_start():
    eps = cp.find_episodes(series([100, 120, 90, 120]))
    assert len(eps) == 1
    assert eps[0].peak_idx == 1
    assert eps[0].peak_close == 120.0


def test_trough_is_the_minimum_over_the_whole_episode():
    eps = cp.find_episodes(series([100, 75, 85, 60, 90, 100]))
    assert len(eps) == 1
    assert eps[0].trough_idx == 3
    assert eps[0].trough_close == 60.0
    assert eps[0].depth_pct == pytest.approx(-0.40)


def test_deeper_leg_inside_the_horizon_extends_the_episode_rather_than_opening_a_second():
    # One episode, not two: the -40% leg lands inside the first trough's horizon.
    eps = cp.find_episodes(series([100, 78, 82, 60, 70, 101, 99, 100]), horizon=5)
    assert len(eps) == 1
    assert eps[0].trough_idx == 3


def test_a_later_crash_from_a_lower_high_is_found_as_its_own_episode():
    """Regression: closing an episode only on recovery to the prior peak hid 2008 inside the
    2000-2014 Nasdaq drawdown. A second fall from a high well below the old peak must register."""
    px = series([100, 50] + [60] * 8 + [75, 74, 73, 55] + [60] * 8)
    eps = cp.find_episodes(px, horizon=5)
    assert len(eps) == 2
    assert eps[0].trough_idx == 1
    assert eps[1].peak_idx == 10 and eps[1].trough_idx == 13
    assert eps[1].peak_close == 75.0          # a local high, not the 100 all-time high
    assert eps[0].recovered is False          # never regains 100, yet does not block episode 2


def test_two_separate_episodes_when_the_first_fully_recovers():
    eps = cp.find_episodes(series([100, 70, 100, 105, 70, 106]), horizon=1)
    assert [e.trough_idx for e in eps] == [1, 4]
    assert [e.peak_idx for e in eps] == [0, 3]


def test_episode_ends_horizon_bars_after_the_trough_clamped_to_the_series():
    eps = cp.find_episodes(series([100, 70] + [72] * 10), horizon=3)
    assert eps[0].end_idx == 4
    eps = cp.find_episodes(series([100, 70, 72]), horizon=99)
    assert eps[0].end_idx == 2


def test_unrecovered_episode_is_flagged_and_has_no_recovery_index():
    eps = cp.find_episodes(series([100, 70, 75, 80]))
    assert len(eps) == 1
    assert eps[0].recovered is False
    assert eps[0].recovery_idx is None


def test_recovery_is_descriptive_and_may_fall_outside_the_episode():
    eps = cp.find_episodes(series([100, 70] + [72] * 10 + [101]), horizon=2)
    assert eps[0].end_idx == 3
    assert eps[0].recovery_idx == 12        # recorded even though it is past end_idx
    assert eps[0].recovered is True


def test_post_trough_days_drops_an_episode_whose_trough_is_too_close_to_the_end():
    px = series([100, 70, 75, 80])  # trough at index 1, three bars of data after it
    assert len(cp.find_episodes(px, post_trough_days=2)) == 1
    assert cp.find_episodes(px, post_trough_days=3) == []


def test_capitulation_window_is_centred_on_the_trough():
    px = series([100.0] + [90.0] * 5 + [70.0] + [90.0] * 5 + [101.0])
    ep = cp.find_episodes(px, horizon=20)[0]
    ph = cp.label_phases(ep, px, capit_days=5)
    assert ep.trough_idx == 6
    assert (ph.capitulation_start, ph.capitulation_end) == (4, 8)


def test_capitulation_window_clamps_at_the_series_edges():
    px = series([100, 70, 72, 74, 101])
    ep = cp.find_episodes(px, horizon=20)[0]
    ph = cp.label_phases(ep, px, capit_days=11)
    assert ph.capitulation_start == 0
    assert ph.capitulation_end == len(px) - 1


def test_retracement_markers_sit_at_the_first_close_regaining_each_level():
    # Peak 100, trough 50, range 50 -> stabilization at >= 60, confirmation at >= 75.
    px = series([100, 50, 55, 62, 70, 76, 101])
    ep = cp.find_episodes(px, horizon=20)[0]
    ph = cp.label_phases(ep, px, stab_retrace=0.20, confirm_retrace=0.50)
    assert ph.stabilization_idx == 3
    assert ph.confirmation_idx == 5


def test_a_level_regained_only_after_the_episode_ends_is_not_marked():
    px = series([100, 50, 51, 52, 53, 76, 101])
    ep = cp.find_episodes(px, horizon=2)[0]
    assert ep.end_idx == 3
    ph = cp.label_phases(ep, px, stab_retrace=0.20, confirm_retrace=0.50)
    assert ph.confirmation_idx is None


def test_retracement_markers_are_none_when_the_level_is_never_regained():
    px = series([100, 50, 52, 54])
    ep = cp.find_episodes(px, horizon=20)[0]
    ph = cp.label_phases(ep, px, stab_retrace=0.20, confirm_retrace=0.50)
    assert ph.stabilization_idx is None
    assert ph.confirmation_idx is None


def test_markers_are_searched_forward_from_the_trough_only():
    # 80 on the way down also clears the stabilization level, but only post-trough closes count.
    px = series([100, 80, 50, 51, 65, 101])
    ep = cp.find_episodes(px, horizon=20)[0]
    ph = cp.label_phases(ep, px, stab_retrace=0.20, confirm_retrace=0.50)
    assert ph.stabilization_idx == 4


def test_invalid_parameters_are_rejected():
    px = series([100, 50, 101])
    ep = cp.find_episodes(px, horizon=20)[0]
    with pytest.raises(ValueError):
        cp.find_episodes(px, min_depth=0.0)
    with pytest.raises(ValueError):
        cp.find_episodes(px, horizon=0)
    with pytest.raises(ValueError):
        cp.label_phases(ep, px, capit_days=0)
    with pytest.raises(ValueError):
        cp.label_phases(ep, px, stab_retrace=0.6, confirm_retrace=0.5)


def test_register_reports_dates_and_depth():
    px = series([100, 50, 55, 62, 70, 76, 101])
    reg = cp.episode_register(px, market="TEST", horizon=20)
    assert len(reg) == 1
    row = reg.iloc[0]
    assert row["market"] == "TEST"
    assert row["peak_date"] == px.index[0]
    assert row["trough_date"] == px.index[1]
    assert row["recovery_date"] == px.index[6]
    assert row["depth_pct"] == pytest.approx(-50.0)
    assert row["trough_to_recovery_days"] == 5
    assert row["recovered"] is np.True_ or row["recovered"] is True
