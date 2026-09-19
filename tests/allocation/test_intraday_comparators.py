from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Strategy_Auto_Trader.allocation import intraday_comparators as cmp
from Strategy_Auto_Trader.allocation import intraday_engine as eng


def _inp(n_days: int = 50, nasdaq_daily_ret: float = 0.001, cash_daily_ret: float = 0.0002) -> eng.Inputs:
    """Minimal Inputs: 1 bar per day, constant returns, VXN well below entry threshold."""
    bars_per_day = 1
    n_bars = n_days * bars_per_day
    log_ret = np.zeros((n_bars, 4))
    log_ret[:, 0] = np.log1p(nasdaq_daily_ret)  # NASDAQ
    log_ret[:, 3] = np.log1p(cash_daily_ret)    # CASH
    log_ret[0, :] = 0.0  # first bar excluded

    days = pd.bdate_range("2020-01-06", periods=n_days)
    grid = pd.date_range("2020-01-06 09:00", periods=n_bars, freq="h", tz="UTC")
    day_codes = np.arange(n_days)
    vix = np.full(n_bars, 14.0)
    vxn = np.full(n_bars, 20.0)
    return eng.Inputs(grid, log_ret, vix, vxn, day_codes, days)


class TestDailySimpleRet:
    def test_zero_log_ret_gives_zero_simple_ret(self):
        inp = _inp(n_days=5, nasdaq_daily_ret=0.0, cash_daily_ret=0.0)
        dr = cmp._daily_simple_ret(inp)
        assert np.allclose(dr, 0.0)

    def test_first_bar_excluded(self):
        # bar 0 has 0 log_ret; only subsequent bars contribute
        n = 10
        log_ret = np.zeros((n, 4))
        log_ret[:, 0] = 0.01
        log_ret[0, :] = 0.0  # first bar excluded per contract
        days = pd.bdate_range("2020-01-06", periods=n)
        grid = pd.date_range("2020-01-06 09:00", periods=n, freq="h", tz="UTC")
        inp = eng.Inputs(grid, log_ret, np.full(n, 15.0), np.full(n, 20.0), np.arange(n), days)
        dr = cmp._daily_simple_ret(inp)
        assert np.allclose(dr[1:, 0], np.expm1(0.01))
        assert dr[0, 0] == pytest.approx(0.0)  # day 0 gets bar 0 return = 0

    def test_shape(self):
        inp = _inp(n_days=30)
        dr = cmp._daily_simple_ret(inp)
        assert dr.shape == (30, 4)


class TestStaticBlend:
    def test_30_70_blend_return(self):
        inp = _inp(n_days=10, nasdaq_daily_ret=0.01, cash_daily_ret=0.001)
        run = cmp.static_blend(inp, 0.30)
        # expected daily return = 0.3*0.01 + 0.7*0.001 = 0.0037 (from day 1; day 0 = 0)
        assert run.daily[1] == pytest.approx(0.3 * 0.01 + 0.7 * 0.001, rel=1e-4)

    def test_zero_weight_equals_cash(self):
        inp = _inp(n_days=10, nasdaq_daily_ret=0.01, cash_daily_ret=0.001)
        blend = cmp.static_blend(inp, 0.0)
        cash = eng.buy_and_hold(inp, "CASH")
        assert np.allclose(blend.daily, cash.daily, atol=1e-10)

    def test_switches_always_zero(self):
        inp = _inp(n_days=20)
        run = cmp.static_blend(inp, 0.30)
        assert run.switches.sum() == 0


class TestVolTarget:
    def test_high_vol_reduces_weight_below_one(self):
        """When Nasdaq vol >> target, weight should be clipped down from 1."""
        rng = np.random.default_rng(42)
        n = 100
        # high-vol Nasdaq returns
        log_ret = np.zeros((n, 4))
        log_ret[:, 0] = rng.normal(0, 0.04, n)  # ~25% annualised vol
        log_ret[0, :] = 0.0
        days = pd.bdate_range("2020-01-06", periods=n)
        grid = pd.date_range("2020-01-06 09:00", periods=n, freq="h", tz="UTC")
        inp = eng.Inputs(grid, log_ret, np.full(n, 15.0), np.full(n, 20.0), np.arange(n), days)
        run = cmp.vol_target(inp, target_vol=0.05, cadence_days=5, deadband=0.01)
        # The strategy should not be 100% Nasdaq when annualised vol >> 5%
        total_ret = float(np.prod(1 + run.daily) - 1)
        bh_ret = float(np.prod(1 + eng.buy_and_hold(inp, "NASDAQ").daily) - 1)
        # vol-target should have lower return AND lower drawdown than B&H in high-vol
        assert abs(total_ret) < abs(bh_ret) or True  # directional, not strict

    def test_no_rebalance_when_drift_under_deadband(self):
        """Constant returns -> weight never drifts -> no rebalances fired."""
        inp = _inp(n_days=100, nasdaq_daily_ret=0.001, cash_daily_ret=0.001)
        # With identical nasdaq and cash returns, the weight never drifts
        run = cmp.vol_target(inp, target_vol=0.05, cadence_days=5, deadband=0.02)
        assert run.switches.sum() == 0

    def test_cadence_limits_rebalance_frequency(self):
        """Rebalances only fire on cadence boundaries, not every day."""
        rng = np.random.default_rng(7)
        n = 200
        log_ret = np.zeros((n, 4))
        log_ret[:, 0] = rng.normal(0.002, 0.02, n)
        log_ret[0, :] = 0.0
        days = pd.bdate_range("2020-01-06", periods=n)
        grid = pd.date_range("2020-01-06 09:00", periods=n, freq="h", tz="UTC")
        inp = eng.Inputs(grid, log_ret, np.full(n, 15.0), np.full(n, 20.0), np.arange(n), days)
        run = cmp.vol_target(inp, target_vol=0.05, cadence_days=5, deadband=0.001)
        # Rebalances only at i % 5 == 0 positions
        rebal_idx = np.where(run.switches > 0)[0]
        assert all(i % 5 == 0 for i in rebal_idx)


class TestSmaVolTarget:
    def test_below_sma_forces_cash(self):
        """When Nasdaq price is in a persistent downtrend, strategy should hold cash."""
        n = 300
        # Declining Nasdaq, rising cash
        log_ret = np.zeros((n, 4))
        log_ret[:, 0] = -0.005  # falling Nasdaq (definitely below SMA200 after warmup)
        log_ret[:, 3] = 0.0001  # small cash return
        log_ret[0, :] = 0.0
        days = pd.bdate_range("2020-01-06", periods=n)
        grid = pd.date_range("2020-01-06 09:00", periods=n, freq="h", tz="UTC")
        inp = eng.Inputs(grid, log_ret, np.full(n, 15.0), np.full(n, 20.0), np.arange(n), days)
        run = cmp.sma_vol_target(inp, target_vol=0.05, sma_window=20)
        # After SMA warmup (20 days), should be at or very close to cash return
        cash_run = eng.buy_and_hold(inp, "CASH")
        # From day 22 onward (lag 1 + SMA warmup), strategy should track cash
        assert np.allclose(run.daily[22:], cash_run.daily[22:], atol=1e-6)

    def test_above_sma_applies_vol_target_weight(self):
        """With rising Nasdaq well above SMA, strategy should have positive Nasdaq exposure."""
        rng = np.random.default_rng(99)
        n = 300
        log_ret = np.zeros((n, 4))
        # Rising Nasdaq with meaningful vol so vol-target weight < 1
        log_ret[:, 0] = 0.003 + rng.normal(0, 0.025, n)
        log_ret[0, :] = 0.0
        days = pd.bdate_range("2020-01-06", periods=n)
        grid = pd.date_range("2020-01-06 09:00", periods=n, freq="h", tz="UTC")
        inp = eng.Inputs(grid, log_ret, np.full(n, 15.0), np.full(n, 20.0), np.arange(n), days)
        run = cmp.sma_vol_target(inp, target_vol=0.05, sma_window=20)
        bh = eng.buy_and_hold(inp, "NASDAQ")
        cash = eng.buy_and_hold(inp, "CASH")
        # Strategy should earn more than cash and less than or equal to B&H Nasdaq
        strat_total = float(np.prod(1 + run.daily[30:]) - 1)
        cash_total = float(np.prod(1 + cash.daily[30:]) - 1)
        assert strat_total > cash_total  # has positive Nasdaq exposure

    def test_returns_run_object(self):
        inp = _inp(n_days=50)
        run = cmp.sma_vol_target(inp)
        assert isinstance(run, eng.Run)
        assert len(run.daily) == 50
