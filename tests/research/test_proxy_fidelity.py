"""Tests for X1 proxy-fidelity (research/proxy_fidelity.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Strategy_Auto_Trader.allocation.intraday_engine import _CASH
from Strategy_Auto_Trader.research import proxy_fidelity as PF


def _days(n, start="2001-02-02"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC").normalize()


def test_realized_vol_constant_growth_zero_vol():
    d = _days(300)
    close = pd.Series(100.0 * 1.001 ** np.arange(300), index=d)  # constant log return -> zero std
    rv = PF.realized_vol(close, 21)
    assert rv.dropna().abs().max() < 1e-9


def test_realized_vol_scales_with_noise():
    rng = np.random.default_rng(0)
    d = _days(600)
    lo_noise = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.005, 600))), index=d)
    hi_noise = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 600))), index=d)
    assert PF.realized_vol(hi_noise, 21).mean() > PF.realized_vol(lo_noise, 21).mean()


def test_deadband_state_matches_engine_shape():
    vals = np.array([30.0, 30, 20, 20, 30, 30, 20], dtype=float)  # enter<=22, exit>25
    st = PF.deadband_state(vals, 22.0, 25.0)
    # starts cash (30>25), enters at idx2 (<=22), holds through idx4/5 (in deadband), stays until >25
    assert st[0] == _CASH and st[1] == _CASH
    assert st[2] == 0 and st[3] == 0
    assert st[4] == _CASH  # 30 > 25 exits


def test_calibrate_matches_train_cash_fraction():
    d = _days(800)
    rng = np.random.default_rng(1)
    # synthetic RV and VXN, correlated-ish; calibration should match deployed train cash fraction
    vxn = pd.Series(20 + 10 * np.abs(rng.normal(0, 1, 800)), index=d).to_numpy()
    rv = pd.Series(0.15 + 0.10 * np.abs(rng.normal(0, 1, 800)), index=d).to_numpy()
    train = np.asarray(d < pd.Timestamp("2004-01-01", tz="UTC"))
    lo, hi = PF.calibrate_thresholds(rv, vxn, train)
    assert lo <= hi
    from Strategy_Auto_Trader.allocation.intraday_engine import tiers_vxn_deadband
    dep = tiers_vxn_deadband(vxn, 23.0, 24.0)
    proxy = tiers_vxn_deadband(rv, lo, hi)
    target = np.mean(dep[train] == _CASH)
    got = np.mean(proxy[train] == _CASH)
    assert abs(got - target) < 0.02  # bisection tolerance


def test_first_exit_on_or_after():
    d = _days(10)
    state = np.array([0, 0, 0, 0, _CASH, _CASH, 0, 0, _CASH, 0])
    # peak at idx2 -> first cash at idx4
    assert PF.first_exit_on_or_after(state, d, d[2]) == 4
    # peak at idx6 -> first cash at idx8
    assert PF.first_exit_on_or_after(state, d, d[6]) == 8


def test_first_exit_none_when_never_cash():
    d = _days(5)
    state = np.zeros(5, dtype=int)
    assert PF.first_exit_on_or_after(state, d, d[0]) is None


def test_episode_timing_passes_when_close_and_before_trough():
    d = _days(40)
    dep = np.zeros(40, dtype=int); dep[10:] = _CASH        # deployed exits at 10
    proxy = np.zeros(40, dtype=int); proxy[13:] = _CASH    # proxy exits at 13 (3 bars later)
    peak, trough = d[5], d[20]
    t = PF.episode_timing(dep, proxy, d, peak, trough)
    assert t.deployed_exit_pos == 10 and t.proxy_exit_pos == 13
    assert t.lag_bars == 3 and t.proxy_precedes_trough
    assert t.passes(tol_bars=10)


def test_episode_timing_fails_when_lag_too_large():
    d = _days(60)
    dep = np.zeros(60, dtype=int); dep[10:] = _CASH
    proxy = np.zeros(60, dtype=int); proxy[25:] = _CASH   # 15 bars late
    peak, trough = d[5], d[40]
    t = PF.episode_timing(dep, proxy, d, peak, trough)
    assert not t.passes(tol_bars=10)


def test_episode_timing_fails_when_proxy_after_trough():
    d = _days(40)
    dep = np.zeros(40, dtype=int); dep[10:] = _CASH
    proxy = np.zeros(40, dtype=int); proxy[22:] = _CASH   # exits after the trough at 20
    peak, trough = d[5], d[20]
    t = PF.episode_timing(dep, proxy, d, peak, trough)
    assert not t.proxy_precedes_trough
    assert not t.passes(tol_bars=10)
