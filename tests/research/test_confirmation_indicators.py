"""Tests for the causal confirmation indicators. The critical property is no lookahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Strategy_Auto_Trader.research import confirmation_indicators as ci


def series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2000-01-03", periods=len(values)))


def test_fires_only_on_fresh_crossings_not_every_bar_it_stays_true():
    # Ramp up for long enough that price sits above a rising average for many bars.
    px = series([100.0] * 60 + [100.0 + i for i in range(1, 40)])
    f = ci.price_above_rising_sma(px, sma_window=10, slope_lookback=3)
    assert f.sum() == 1, "a persistent condition must register one firing, not one per bar"


def test_never_fires_on_the_first_bar():
    px = series([100.0 + i for i in range(80)])
    assert not ci.price_above_rising_sma(px, sma_window=5, slope_lookback=2)[0]


def test_indicator_is_causal_future_bars_cannot_change_an_earlier_firing():
    base = [100.0] * 30 + [100.0 + i for i in range(1, 30)]
    short = ci.price_above_rising_sma(series(base), sma_window=10, slope_lookback=3)
    extended = ci.price_above_rising_sma(series(base + [5.0] * 40), sma_window=10, slope_lookback=3)
    np.testing.assert_array_equal(short, extended[: len(short)])


def test_a_rising_average_is_required_not_just_price_above_it():
    """The distinguishing clause: a bounce inside a downtrend has price above a still-falling SMA."""
    down = [200.0 - 2 * i for i in range(40)]      # steady decline, SMA falling
    bounce = [down[-1] + 6 * i for i in range(1, 5)]  # short sharp bounce above the SMA
    f = ci.price_above_rising_sma(series(down + bounce), sma_window=10, slope_lookback=5)
    assert f.sum() == 0, "price above a falling average must not fire"


def test_spread_indicator_is_the_mirror_image():
    widening = [3.0 + 0.2 * i for i in range(40)]
    tightening = [widening[-1] - 0.5 * i for i in range(1, 30)]
    f = ci.spread_below_falling_average(series(widening + tightening), sma_window=10, slope_lookback=3)
    assert f.sum() == 1
    assert ci.spread_below_falling_average(series(widening), sma_window=10, slope_lookback=3).sum() == 0


def test_warmup_period_never_fires():
    px = series([100.0 + i for i in range(30)])
    f = ci.price_above_rising_sma(px, sma_window=20, slope_lookback=5)
    assert not f[:20].any(), "no firing is possible before the average has any value"


def test_elapsed_time_baseline_fires_once_at_the_declared_offset():
    px = series([100.0] * 5 + [70.0] + [75.0] * 40)   # -30% breach at index 5
    f = ci.elapsed_time_baseline(px, drawdown_depth=0.20, wait_bars=10)
    assert f.sum() == 1
    assert int(np.flatnonzero(f)[0]) == 15


def test_elapsed_time_baseline_is_silent_when_the_wait_runs_past_the_data():
    px = series([100.0] * 5 + [70.0] + [75.0] * 3)
    assert ci.elapsed_time_baseline(px, drawdown_depth=0.20, wait_bars=100).sum() == 0


def test_elapsed_time_baseline_is_silent_without_a_breach():
    px = series([100.0 + i for i in range(50)])
    assert ci.elapsed_time_baseline(px, drawdown_depth=0.20, wait_bars=5).sum() == 0


def test_invalid_parameters_are_rejected():
    px = series([100.0] * 30)
    with pytest.raises(ValueError):
        ci.price_above_rising_sma(px, sma_window=1)
    with pytest.raises(ValueError):
        ci.price_above_rising_sma(px, slope_lookback=0)
    with pytest.raises(ValueError):
        ci.spread_below_falling_average(px, sma_window=1)
