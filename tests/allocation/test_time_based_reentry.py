from __future__ import annotations

import numpy as np
import pytest

from Strategy_Auto_Trader.allocation import time_based_reentry as tbr


def flat(n: int, price: float = 100.0) -> np.ndarray:
    return np.full(n, price, dtype=float)


class TestOverrideFractions:

    def test_never_fires_while_base_rule_is_invested(self):
        base_cash = np.zeros(400, dtype=bool)
        frac = tbr.override_fractions(base_cash, flat(400), wait_days=21, ramp_days=1)
        assert not frac.any()

    def test_waits_the_full_wait_days_before_arming(self):
        base_cash = np.ones(100, dtype=bool)
        frac = tbr.override_fractions(base_cash, flat(100), wait_days=21, ramp_days=1)
        assert not frac[:21].any(), "must stay flat for the whole wait"
        assert frac[21] == pytest.approx(1.0), "arms on the day after the wait elapses"

    def test_ramp_is_linear_and_caps_at_one(self):
        base_cash = np.ones(100, dtype=bool)
        frac = tbr.override_fractions(base_cash, flat(100), wait_days=10, ramp_days=5)
        assert list(frac[10:16]) == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0, 1.0])

    def test_streak_resets_when_base_rule_re_enters(self):
        """A cash episode interrupted before wait_days elapses must not accumulate toward the next one."""
        base_cash = np.array([True] * 15 + [False] * 5 + [True] * 15, dtype=bool)
        frac = tbr.override_fractions(base_cash, flat(35), wait_days=21, ramp_days=1)
        assert not frac.any(), "two 15-day episodes are not one 30-day episode"

    def test_short_episodes_never_fire(self):
        """The base rule's median cash episode is a few days; the override must ignore those entirely."""
        base_cash = np.tile([True] * 3 + [False] * 3, 60).astype(bool)
        frac = tbr.override_fractions(base_cash, flat(360), wait_days=21, ramp_days=1)
        assert not frac.any()

    def test_trailing_stop_ends_the_override(self):
        base_cash = np.ones(60, dtype=bool)
        close = flat(60)
        close[30:] = 80.0  # -20% from the peak, well through the 8% stop
        frac = tbr.override_fractions(base_cash, close, wait_days=10, ramp_days=1, stop_pct=0.08)
        assert frac[29] > 0
        assert not frac[30:].any()

    def test_no_re_arm_within_the_same_cash_episode_after_a_stop(self):
        """Unlike the T1 triggers, a stopped-out time rule stays out until the base rule re-enters —
        re-arming on elapsed time alone would just be a fixed-interval retry."""
        base_cash = np.ones(120, dtype=bool)
        close = flat(120)
        close[30:] = 80.0
        close[40:] = 200.0  # a strong recovery the override must not chase
        frac = tbr.override_fractions(base_cash, close, wait_days=10, ramp_days=1)
        assert not frac[30:].any()

    def test_re_arms_on_a_new_cash_episode_after_a_stop(self):
        base_cash = np.array([True] * 40 + [False] * 5 + [True] * 40, dtype=bool)
        close = flat(85)
        close[20:40] = 80.0  # stops out inside the first episode
        frac = tbr.override_fractions(base_cash, close, wait_days=10, ramp_days=1)
        assert not frac[20:45].any()
        assert frac[55:].any(), "a fresh cash episode gets a fresh override"

    def test_fraction_stays_within_unit_interval(self):
        rng = np.random.default_rng(0)
        base_cash = rng.random(1000) < 0.4
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 1000)))
        frac = tbr.override_fractions(base_cash, close, wait_days=21, ramp_days=10)
        assert frac.min() >= 0.0 and frac.max() <= 1.0

    @pytest.mark.parametrize("wait,ramp", [(0, 1), (1, 0), (-5, 5)])
    def test_rejects_non_positive_parameters(self, wait, ramp):
        with pytest.raises(ValueError):
            tbr.override_fractions(np.ones(10, dtype=bool), flat(10), wait_days=wait, ramp_days=ramp)


class TestBarWeights:

    def test_base_rule_invested_always_means_full_weight(self):
        tiers = np.zeros(10, dtype=int)  # 0 = Nasdaq
        frac = np.ones(10)
        w = tbr.bar_weights(tiers, np.arange(10), frac, size=0.5)
        assert (w == 1.0).all(), "the override never reduces the base rule's own exposure"

    def test_override_weight_is_lagged_one_day_no_look_ahead(self):
        tiers = np.full(5, 3, dtype=int)  # 3 = cash
        frac = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
        w = tbr.bar_weights(tiers, np.arange(5), frac, size=1.0)
        assert w[0] == 0.0 and w[1] == 0.0, "a fraction set at day 1's close cannot be held during day 1"
        assert w[2] == 1.0

    def test_size_caps_the_deployed_fraction(self):
        tiers = np.full(4, 3, dtype=int)
        frac = np.array([1.0, 1.0, 1.0, 1.0])
        w = tbr.bar_weights(tiers, np.arange(4), frac, size=0.5)
        assert w[1:].max() == pytest.approx(0.5)


class TestEpisodes:

    def test_counts_contiguous_runs(self):
        frac = np.array([0, 0.5, 1, 0, 0, 0.2, 0, 1])
        assert tbr.episodes(frac) == [(1, 2), (5, 5), (7, 7)]

    def test_no_activity_gives_no_episodes(self):
        assert tbr.episodes(np.zeros(10)) == []

    def test_run_open_at_the_end_is_closed_off(self):
        assert tbr.episodes(np.array([0, 0, 1, 1])) == [(2, 3)]
