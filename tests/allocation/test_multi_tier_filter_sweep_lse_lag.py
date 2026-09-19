from __future__ import annotations

import numpy as np
import pandas as pd

from Strategy_Auto_Trader.allocation.multi_tier_filter_sweep_lse_lag import evaluate, filter_grid


def test_evaluate_filters_reduce_switches_and_report_every_filter():
    dates = pd.bdate_range("2018-06-01", periods=800)
    rng = np.random.default_rng(3)
    rets = pd.DataFrame(rng.normal(0.0002, 0.01, (len(dates), 4)), index=dates, columns=[1, 2, 3, 4])
    raw = pd.Series(rng.choice([1, 2, 3, 4], len(dates)), index=dates)

    df = evaluate(raw, rets, bps=13.0).set_index("filter")

    assert list(df.index) == list(filter_grid())
    base = df.loc["raw (no filter)", "switches/yr"]
    assert (df.drop("raw (no filter)")["switches/yr"] <= base).all()
    assert df.loc["hysteresis 60d", "switches/yr"] < base / 10
    assert (df["net_ret"] <= df["gross_ret"] + 1e-9).all()
