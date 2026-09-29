"""Tests for X0b register reconciliation (research/reconcile.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Strategy_Auto_Trader.research import reconcile as R


def _series(values, start="2000-01-03"):
    idx = pd.date_range(start, periods=len(values), freq="B", tz="UTC").normalize()
    return pd.Series(np.asarray(values, dtype=float), index=idx, name="^TEST")


def _clean(n=300, seed=0):
    rng = np.random.default_rng(seed)
    # geometric random walk, no defects
    steps = rng.normal(0.0, 0.01, n)
    return _series(100.0 * np.exp(np.cumsum(steps)))


# ---- scan_unit_shift ----

def test_unit_shift_detects_100x_cutover():
    s = _clean(200)
    px = s.to_numpy().copy()
    px[:80] *= 100.0  # pre-cutover units are 100x too large
    flagged = R.scan_unit_shift(_series(px))
    assert len(flagged) == 1
    assert flagged[0] == s.index[80]  # the step lands on the first correctly-scaled bar


def test_unit_shift_clean_series_no_flags():
    assert len(R.scan_unit_shift(_clean(300))) == 0


# ---- scan_spike ----

def test_spike_detects_isolated_bad_print():
    s = _clean(200, seed=1)
    px = s.to_numpy().copy()
    px[100] *= 3.0  # single bad print that reverts next bar
    flagged = R.scan_spike(_series(px))
    assert s.index[100] in flagged


def test_spike_clean_series_no_flags():
    assert len(R.scan_spike(_clean(300, seed=2))) == 0


def test_spike_ignores_real_crash_day_no_revert():
    # a big one-day drop that STAYS down (genuine capitulation) must not be flagged
    s = _clean(200, seed=7)
    px = s.to_numpy().copy()
    px[100:] *= 0.78  # -22% step down that persists, like a Tiananmen-style trough
    flagged = R.scan_spike(_series(px))
    assert s.index[100] not in flagged


# ---- scan_stale ----

def test_stale_detects_repeated_run():
    px = list(_clean(100, seed=3).to_numpy())
    for i in range(40, 46):
        px[i] = px[39]  # 7-bar flat run
    flagged = R.scan_stale(_series(px), run=5)
    assert len(flagged) >= 1


def test_stale_short_run_not_flagged():
    px = list(_clean(100, seed=4).to_numpy())
    px[50] = px[49]  # a single repeat, not a run
    assert len(R.scan_stale(_series(px), run=5)) == 0


# ---- _close_at ----

def test_close_at_exact_and_lagged():
    s = _clean(50)
    d = s.index[10]
    assert R._close_at(s, d)[0] == pytest.approx(s.iloc[10])
    # drop one bar, then query its date -> nearest prior trading day within tolerance
    dropped_date = s.index[10]
    gapped = s.drop(dropped_date)
    val, lag = R._close_at(gapped, dropped_date)
    assert val == pytest.approx(s.iloc[9]) and 0 < lag <= 4


def test_close_at_missing_beyond_tolerance():
    s = _clean(50)
    gap_date = s.index[-1] + pd.Timedelta(days=30)
    assert R._close_at(s, gap_date) == (None, -1)


# ---- reconcile_episode ----

def _row(market, peak_date, trough_date, depth_pct):
    return pd.Series(
        {"market": market, "peak_date": peak_date, "trough_date": trough_date, "depth_pct": depth_pct}
    )


def test_reconcile_clean_intrinsic_only_not_dropped():
    s = _series([100, 110, 90, 80, 85, 95], start="2000-01-03")
    peak, trough = s.index[1], s.index[3]  # 110 -> 80 = -27.27%
    row = _row("^TEST", peak.isoformat(), trough.isoformat(), round((80 / 110 - 1) * 100, 2))
    res = R.reconcile_episode(row, s, None, None)
    assert res.depth_ok
    assert res.confirmation == "intrinsic-only"
    assert not res.dropped


def test_reconcile_depth_mismatch_drops():
    s = _series([100, 110, 90, 80, 85, 95], start="2000-01-03")
    peak, trough = s.index[1], s.index[3]
    row = _row("^TEST", peak.isoformat(), trough.isoformat(), -50.0)  # stored depth wrong
    res = R.reconcile_episode(row, s, None, None)
    assert not res.depth_ok
    assert res.dropped


def test_reconcile_second_source_agreement():
    s = _series([100, 110, 90, 80, 85, 95], start="2000-01-03")
    peak, trough = s.index[1], s.index[3]
    fred = s.copy() * 1.001  # 0.1% off, inside 1% tol
    fred.name = "TESTFRED"
    row = _row("^TEST", peak.isoformat(), trough.isoformat(), round((80 / 110 - 1) * 100, 2))
    res = R.reconcile_episode(row, s, fred, s.index[0])
    assert res.second_source_ok is True
    assert res.confirmation == "second-source"
    assert not res.dropped


def test_reconcile_second_source_disagreement_drops():
    s = _series([100, 110, 90, 80, 85, 95], start="2000-01-03")
    peak, trough = s.index[1], s.index[3]
    fred = s.copy() * 1.10  # 10% off, outside tol
    fred.name = "TESTFRED"
    row = _row("^TEST", peak.isoformat(), trough.isoformat(), round((80 / 110 - 1) * 100, 2))
    res = R.reconcile_episode(row, s, fred, s.index[0])
    assert res.second_source_ok is False
    assert res.dropped


def test_reconcile_pre_source_start_is_intrinsic_only():
    s = _series([100, 110, 90, 80, 85, 95], start="2000-01-03")
    peak, trough = s.index[1], s.index[3]
    fred = s.copy()
    fred.name = "TESTFRED"
    # source only trustworthy from far after the episode -> stays intrinsic-only
    row = _row("^TEST", peak.isoformat(), trough.isoformat(), round((80 / 110 - 1) * 100, 2))
    res = R.reconcile_episode(row, s, fred, pd.Timestamp("2020-01-01", tz="UTC"))
    assert res.second_source_ok is None
    assert res.confirmation == "intrinsic-only"
