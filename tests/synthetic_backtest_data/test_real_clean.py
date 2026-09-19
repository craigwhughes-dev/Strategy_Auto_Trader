from __future__ import annotations

import numpy as np
import pandas as pd

from Strategy_Auto_Trader.synthetic_backtest_data.real_clean import clean_fund_bars, drop_spikes, remove_level_shifts


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2024-01-02 08:00", periods=len(values), freq="h", tz="UTC"))


def test_single_bar_unit_spike_is_dropped():
    s = _series([100, 101, 100, 10000, 101, 100, 102, 101, 100])
    out = drop_spikes(s)
    assert 10000 not in out.to_numpy()
    assert len(out) == len(s) - 1


def test_genuine_large_move_is_kept():
    s = _series([100, 100.5, 101, 93, 92.5, 92, 91.5, 92, 91])  # -8% in an hour is real (COVID-like)
    assert len(drop_spikes(s)) == len(s)


def test_persistent_two_for_one_shift_is_back_adjusted_to_latest_level():
    s = _series([100, 101, 100, 50, 50.5, 51, 50, 49.5])
    out = remove_level_shifts(s)
    assert out.iloc[-1] == s.iloc[-1]
    assert np.log(out).diff().abs().max() < 0.05


def test_clean_fund_bars_reports_counts():
    s = _series([100, 101, 100, 10000, 101, 100, 102, 101, 100])
    out, info = clean_fund_bars(s)
    assert info == {"dropped_spikes": 1, "level_shifts": 0}
    assert len(out) == 8
