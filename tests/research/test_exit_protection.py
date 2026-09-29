"""Tests for X2 exit-protection metrics (research/exit_protection.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from Strategy_Auto_Trader.allocation.intraday_engine import _CASH
from Strategy_Auto_Trader.research import exit_protection as XP


def _series(vals, start="1990-01-01"):
    idx = pd.date_range(start, periods=len(vals), freq="B", tz="UTC").normalize()
    return pd.Series(np.asarray(vals, dtype=float), index=idx, name="^TEST")


def test_protection_exit_at_peak_full_protection():
    # price 100 -> 50 then recovers; gate exits right at the peak (state cash from peak on)
    close = _series([100, 90, 80, 70, 60, 50, 60, 70])
    state = np.array([0, _CASH, _CASH, _CASH, _CASH, _CASH, _CASH, _CASH])
    cat, raw, clipped, lag = XP.episode_protection(close, state, close.index[0], close.index[5])
    # state at peak (idx0) is invested (0); exits idx1 at price 90 -> protection=(90-50)/(100-50)=0.8
    assert cat == "scored"
    assert clipped == raw == 0.8
    assert lag == 1


def test_protection_already_out_at_peak():
    close = _series([100, 90, 80, 70, 60, 50])
    state = np.full(6, _CASH)
    cat, raw, clipped, lag = XP.episode_protection(close, state, close.index[0], close.index[5])
    assert cat == "already_out"
    assert raw is None and clipped is None and lag is None


def test_protection_never_exits_before_trough_is_zero():
    close = _series([100, 90, 80, 70, 60, 50, 60])
    state = np.zeros(7, dtype=int)  # stays invested throughout
    cat, raw, clipped, lag = XP.episode_protection(close, state, close.index[0], close.index[5])
    assert cat == "never_exited"
    assert clipped == 0.0


def test_protection_clips_overshoot():
    # exit price above the peak -> raw > 1, clipped to 1
    close = _series([100, 120, 80, 60, 40, 20])
    state = np.array([0, _CASH, _CASH, _CASH, _CASH, _CASH])
    cat, raw, clipped, lag = XP.episode_protection(close, state, close.index[0], close.index[5])
    assert cat == "scored"
    assert raw > 1.0 and clipped == 1.0


def test_percentile_thresholds_ordering():
    rv = _series(np.linspace(0.05, 0.60, 500))
    lo, hi = XP.percentile_thresholds(rv, 0.4556, 0.4985)
    assert lo <= hi
    assert 0.05 < lo < 0.60 and 0.05 < hi < 0.60


def test_market_idle_rate_excludes_episode_days():
    days = pd.date_range("1990-01-01", periods=20, freq="B", tz="UTC").normalize()
    state = np.array([_CASH] * 10 + [0] * 10)  # cash first half, invested second
    # one episode window covering the last 10 (invested) days -> non-episode = first 10, all cash
    windows = [(days[10], days[19])]
    assert XP.market_idle_rate(state, days, windows) == 1.0
    # episode over the first 10 (cash) days -> non-episode = last 10, all invested -> idle 0
    assert XP.market_idle_rate(state, days, [(days[0], days[9])]) == 0.0


def test_aggregate_by_shock_means_within_shock():
    eps = [
        XP.EpisodeProtection("^A", 0, True, pd.Timestamp("1987-01-01", tz="UTC"), pd.Timestamp("1987-06-01", tz="UTC"), "scored", 0.6, 0.6, 5),
        XP.EpisodeProtection("^B", 0, True, pd.Timestamp("1987-02-01", tz="UTC"), pd.Timestamp("1987-06-01", tz="UTC"), "scored", 0.8, 0.8, 3),
        XP.EpisodeProtection("^C", 5, False, pd.Timestamp("2008-01-01", tz="UTC"), pd.Timestamp("2009-01-01", tz="UTC"), "already_out", None, None, None),
    ]
    df = XP.aggregate_by_shock(eps)
    s0 = df[df["shock_id"] == 0].iloc[0]
    assert s0["protection"] == 0.7 and s0["n_scored"] == 2 and s0["is_holdout"]
    s5 = df[df["shock_id"] == 5].iloc[0]
    assert np.isnan(s5["protection"]) and s5["n_already_out"] == 1


def test_permutation_pvalue_identical_sets_high_p():
    a = np.array([0.5, 0.6, 0.55, 0.52, 0.58])
    b = np.array([0.51, 0.59, 0.54, 0.53, 0.57])
    assert XP.permutation_pvalue(a, b, n=2000) > 0.2  # same distribution -> not significant


def test_permutation_pvalue_separated_sets_low_p():
    # cleanly separated, non-overlapping, decent n -> median-permutation is decisive
    a = np.linspace(0.55, 1.0, 20)
    b = np.linspace(0.0, 0.45, 20)
    assert XP.permutation_pvalue(a, b, n=3000) < 0.05
