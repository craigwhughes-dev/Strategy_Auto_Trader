from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Strategy_Auto_Trader.allocation import intraday_engine as eng


def _inputs(log_ret: np.ndarray, day_codes: list[int], start: str = "2020-01-06") -> eng.Inputs:
    n = len(day_codes)
    days = pd.bdate_range(start, periods=max(day_codes) + 1)
    grid = pd.date_range(f"{start} 09:00", periods=n, freq="h", tz="UTC")
    return eng.Inputs(grid, log_ret, np.full(n, 15.0), np.full(n, 17.0), np.array(day_codes), days)


def _flat(n: int) -> np.ndarray:
    return np.zeros((n, 4))


class TestTiers:
    def test_vxn_vix_ladder(self):
        vix = np.array([14.0, 16.0, 18.0, 25.0])
        vxn = np.array([20.0, 20.0, 20.0, 20.0])
        assert eng.tiers_vxn_vix(vxn, vix, 18.0, 15.0, 17.5).tolist() == [1, 2, 3, 3]

    def test_low_vxn_overrides_vix_for_nasdaq(self):
        assert eng.tiers_vxn_vix(np.array([17.0]), np.array([30.0]), 18.0, 15.0, 17.5).tolist() == [0]

    def test_thresholds_are_inclusive(self):
        assert eng.tiers_vxn_vix(np.array([18.0, 99.0, 99.0]), np.array([99.0, 15.0, 17.5]), 18.0, 15.0, 17.5).tolist() == [0, 1, 2]

    def test_nan_vix_goes_to_cash_and_nan_vxn_skips_nasdaq(self):
        assert eng.tiers_vxn_vix(np.array([np.nan, np.nan]), np.array([np.nan, 14.0]), 18.0, 15.0, 17.5).tolist() == [3, 1]

    def test_vix_only_ladder(self):
        vix = np.array([12.0, 14.0, 16.0, 30.0])
        assert eng.tiers_vix_only(vix, 13.0, 15.0, 20.0).tolist() == [0, 1, 2, 3]


class TestDeadband:
    def test_width_zero_equals_the_plain_rule(self):
        vxn = np.array([30.0, 20.0, 24.0, 25.0, 23.0, 40.0, 10.0])
        plain = np.where(vxn <= 24.0, 0, 3)
        assert eng.tiers_vxn_deadband(vxn, 24.0, 24.0).tolist() == plain.tolist()

    def test_holds_position_while_inside_the_band(self):
        vxn = np.array([20.0, 23.0, 26.0, 27.0, 25.0, 22.0, 26.0])
        # enter <=22, exit >26: in at 20, still in at 23/26, out at 27, stays out at 25 (band), back at 22, then 26 stays in
        assert eng.tiers_vxn_deadband(vxn, 22.0, 26.0).tolist() == [0, 0, 0, 3, 3, 0, 0]

    def test_starts_in_cash_until_a_reading_qualifies(self):
        vxn = np.array([25.0, 25.0, 21.0])
        assert eng.tiers_vxn_deadband(vxn, 22.0, 26.0).tolist() == [3, 3, 0]

    def test_nan_reading_keeps_the_current_state(self):
        vxn = np.array([20.0, np.nan, np.nan, 30.0, np.nan])
        assert eng.tiers_vxn_deadband(vxn, 22.0, 26.0).tolist() == [0, 0, 0, 3, 3]

    def test_leading_nan_means_cash(self):
        assert eng.tiers_vxn_deadband(np.array([np.nan, 20.0]), 22.0, 26.0).tolist() == [3, 0]

    def test_wider_band_never_switches_more_often(self):
        rng = np.random.default_rng(0)
        vxn = 24 + np.cumsum(rng.normal(0, 0.6, 2000))
        flips = [int((np.diff(eng.tiers_vxn_deadband(vxn, 22.0, 22.0 + w)) != 0).sum()) for w in (0, 2, 4, 8)]
        assert flips == sorted(flips, reverse=True) and flips[0] > flips[-1]

    def test_inverted_band_rejected(self):
        with pytest.raises(ValueError):
            eng.tiers_vxn_deadband(np.array([20.0]), 26.0, 22.0)


class TestSimulate:
    def _run(self, tiers, log_ret, day_codes, fill="same_bar", cost=0.0):
        return eng.simulate(_inputs(log_ret, day_codes), np.array(tiers), fill, cost)

    def test_same_bar_earns_the_next_intervals_return_of_the_tier_chosen_now(self):
        lr = _flat(4)
        lr[1, 0] = 0.01  # Nasdaq gains in the interval ending at bar 1
        lr[2, 0] = 0.02
        # chosen at bar 0 -> holds Nasdaq through bar 1 and 2 intervals; back to cash at bar 3
        run = self._run([0, 0, 0, 3], lr, [0, 0, 0, 0])
        assert run.daily[0] == pytest.approx(np.expm1(0.03))

    def test_signal_on_the_last_bar_does_not_earn_that_bars_own_return(self):
        lr = _flat(3)
        lr[2, 0] = 0.05  # interval ending at bar 2
        run = self._run([3, 3, 0], lr, [0, 0, 0])  # Nasdaq only chosen AT bar 2 -> cannot capture it
        assert run.daily[0] == pytest.approx(0.0)

    def test_next_bar_fill_delays_the_position_by_one_bar(self):
        lr = _flat(4)
        lr[1, 0] = 0.01
        lr[2, 0] = 0.02
        same = self._run([0, 0, 0, 3], lr, [0, 0, 0, 0], "same_bar")
        late = self._run([0, 0, 0, 3], lr, [0, 0, 0, 0], "next_bar")
        assert late.daily[0] == pytest.approx(np.expm1(0.02))  # misses the first interval
        assert same.daily[0] > late.daily[0]

    def test_cost_is_charged_once_per_switch(self):
        run = self._run([3, 0, 0, 3], _flat(4), [0, 0, 0, 0], cost=13.0)
        assert run.switches[0] == 2
        assert run.daily[0] == pytest.approx(np.expm1(2 * np.log1p(-13e-4)))

    def test_no_switch_no_cost(self):
        run = self._run([3, 3, 3, 3], _flat(4), [0, 0, 1, 1], cost=13.0)
        assert run.switches.sum() == 0 and np.allclose(run.daily, 0.0)

    def test_returns_are_grouped_by_london_day(self):
        lr = _flat(4)
        lr[:, 3] = [0.0, 0.01, 0.01, 0.01]
        run = self._run([3, 3, 3, 3], lr, [0, 0, 1, 1])
        assert run.daily[0] == pytest.approx(np.expm1(0.01)) and run.daily[1] == pytest.approx(np.expm1(0.02))

    def test_unknown_fill_mode_rejected(self):
        with pytest.raises(ValueError):
            self._run([3, 3], _flat(2), [0, 0], "instant")

    def test_overnight_gap_is_earned_by_the_asset_held_overnight(self):
        lr = _flat(3)
        lr[2, 1] = -0.04  # first bar next morning: S&P gaps down
        run = self._run([1, 1, 3], lr, [0, 0, 1])  # holding S&P into the close of day 0, exit chosen at bar 2
        assert run.daily[1] == pytest.approx(np.expm1(-0.04))


class TestStats:
    def test_window_bounds_and_stats(self):
        n_days = 30
        days = pd.bdate_range("2019-12-16", periods=n_days)
        run = eng.Run(np.full(n_days, 0.001), np.zeros(n_days), days)
        stats = eng.window_stats(run, "test")  # 2020-01-01 onward
        assert stats["n_days"] == int((days >= "2020-01-01").sum())
        assert stats["ret_pct"] == pytest.approx(((1.001 ** stats["n_days"]) - 1) * 100)

    def test_switches_per_year_annualises_on_252_days(self):
        days = pd.bdate_range("2021-01-04", periods=252)
        sw = np.zeros(252)
        sw[:10] = 1
        stats = eng.window_stats(eng.Run(np.full(252, 0.0005), sw, days), "test")
        assert stats["sw_per_yr"] == pytest.approx(10.0)

    def test_empty_window_returns_nan_not_error(self):
        days = pd.bdate_range("2010-01-04", periods=10)
        assert np.isnan(eng.window_stats(eng.Run(np.zeros(10), np.zeros(10), days), "test")["sharpe"])

    def test_annual_returns_compounds_within_each_year(self):
        days = pd.DatetimeIndex(["2020-12-30", "2020-12-31", "2021-01-04"])
        out = eng.annual_returns(eng.Run(np.array([0.01, 0.01, 0.02]), np.zeros(3), days))
        assert out[2020] == pytest.approx(2.0, abs=0.05) and out[2021] == pytest.approx(2.0, abs=0.05)


def test_buy_and_hold_ignores_the_first_bar_and_tracks_the_asset():
    lr = _flat(3)
    lr[:, 0] = [0.5, 0.01, 0.01]  # bar 0 has no prior price, so its return must not count
    run = eng.buy_and_hold(_inputs(lr, [0, 0, 1]), "NASDAQ")
    assert run.daily[0] == pytest.approx(np.expm1(0.01)) and run.daily[1] == pytest.approx(np.expm1(0.01))
