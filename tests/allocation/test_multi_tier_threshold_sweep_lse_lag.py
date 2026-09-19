from __future__ import annotations

import numpy as np
import pandas as pd

from Strategy_Auto_Trader.allocation.multi_tier_threshold_sweep_lse_lag import (
    ALWAYS_VXN,
    NO_CASH,
    NO_VXN,
    default_grid,
    sweep,
    tiers_for,
)

DATES = pd.bdate_range("2018-01-02", periods=6)
READINGS = [(15.0, 12.0), (30.0, 12.0), (30.0, 16.0), (30.0, 19.0), (30.0, 40.0), (None, None)]


def test_tiers_for_default_thresholds():
    t = tiers_for(DATES, READINGS, 18.0, 15.0, 17.5)
    assert list(t) == [1, 2, 3, 4, 4, 4]


def test_extreme_thresholds_disable_tiers():
    # VXN always under threshold -> Nasdaq every day, except the last day where VXN is missing (-> cash fallback)
    assert list(tiers_for(DATES, READINGS, ALWAYS_VXN, 15.0, 17.5)) == [1, 1, 1, 1, 1, 4]
    no_nasdaq = tiers_for(DATES, READINGS, NO_VXN, 15.0, 17.5)
    assert 1 not in set(no_nasdaq)
    no_cash = tiers_for(DATES, READINGS, 18.0, 15.0, NO_CASH)
    assert list(no_cash)[3:5] == [3, 3]


def test_grid_only_has_ordered_vix_thresholds():
    assert all(vix1 < vix2 for _, vix1, vix2 in default_grid())


def test_sweep_row_per_threshold_and_filter():
    n = 300
    dates = pd.bdate_range("2017-06-01", periods=n)
    rng = np.random.default_rng(5)
    rets = pd.DataFrame(rng.normal(0.0002, 0.01, (n, 4)), index=dates, columns=[1, 2, 3, 4])
    readings = [(float(v), float(w)) for v, w in zip(rng.uniform(12, 35, n), rng.uniform(11, 30, n))]
    grid = [(18.0, 15.0, 17.5), (25.0, 15.0, 20.0)]
    df = sweep(dates, readings, rets, grid, bps=13.0)
    assert len(df) == 4 and set(df["filter"]) == {"raw", "asym10d"}
    raw = df[df["filter"] == "raw"]["sw/yr"].values
    filt = df[df["filter"] == "asym10d"]["sw/yr"].values
    assert (filt < raw).all()
