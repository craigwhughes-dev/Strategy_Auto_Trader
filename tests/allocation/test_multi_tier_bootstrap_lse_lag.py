"""Tests for multi_tier_bootstrap_lse_lag: bootstrap and annual-return helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Strategy_Auto_Trader.allocation.multi_tier_bootstrap_lse_lag import (
    annual_returns,
    block_bootstrap_sharpe,
    bootstrap_table,
    run,
)
from Strategy_Auto_Trader.allocation.multi_tier_comparators_lse_lag import sma_vol_target_returns


def _fake_tier_returns(n: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n)
    return pd.DataFrame(
        {
            1: rng.normal(0.0006, 0.013, n),
            2: rng.normal(0.0004, 0.010, n),
            3: rng.normal(0.0002, 0.008, n),
            4: rng.normal(0.0001, 0.001, n),
        },
        index=idx,
    )


def _fake_hourly(seed: int, n: int = 7000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-03 14:30", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"Close": np.clip(rng.normal(18, 4, n), 8, 60)}, index=idx)


@pytest.fixture
def tier_returns():
    return _fake_tier_returns()


# ---------------------------------------------------------------------------
# sma_vol_target_returns
# ---------------------------------------------------------------------------

def test_sma_vol_target_no_nan_after_warmup(tier_returns):
    r = sma_vol_target_returns(tier_returns, target_vol=0.05)
    assert r.iloc[210:].isna().sum() == 0


def test_sma_vol_target_weight_bounded(tier_returns):
    """All daily returns must be a convex combination of r1 and r4 (w in [0,1])."""
    r = sma_vol_target_returns(tier_returns, target_vol=0.05)
    valid = r.dropna()
    r_min = tier_returns[[1, 4]].min(axis=1).loc[valid.index]
    r_max = tier_returns[[1, 4]].max(axis=1).loc[valid.index]
    assert (valid >= r_min - 1e-10).all()
    assert (valid <= r_max + 1e-10).all()


# ---------------------------------------------------------------------------
# block_bootstrap_sharpe
# ---------------------------------------------------------------------------

def test_bootstrap_returns_correct_count():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.001, 0.01, 500))
    bs = block_bootstrap_sharpe(rets, block_size=21, n_iter=100, seed=0)
    assert len(bs) == 100


def test_bootstrap_positive_drift():
    """Strongly positive series should have p5 > 0."""
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0.002, 0.005, 1000))
    bs = block_bootstrap_sharpe(rets, block_size=21, n_iter=500, seed=1)
    assert np.nanpercentile(bs, 5) > 0


def test_bootstrap_zero_series_gives_nan_or_zero():
    rets = pd.Series(np.zeros(200))
    bs = block_bootstrap_sharpe(rets, block_size=21, n_iter=50, seed=0)
    assert np.all((bs == 0) | np.isnan(bs))


# ---------------------------------------------------------------------------
# bootstrap_table
# ---------------------------------------------------------------------------

def test_bootstrap_table_columns(tier_returns):
    r1 = tier_returns[1]
    r4 = tier_returns[4]
    strats = {"A": r1, "B": r4}
    df = bootstrap_table(strats, block_size=21, n_iter=100)
    assert set(df.columns) >= {"strategy", "full_SR", "p5", "p25", "p50", "p95", "prob_SR_gt_0"}
    assert len(df) == 2


def test_bootstrap_table_p5_le_p50(tier_returns):
    r1 = tier_returns[1]
    df = bootstrap_table({"A": r1}, block_size=21, n_iter=200)
    assert df.iloc[0]["p5"] <= df.iloc[0]["p50"]
    assert df.iloc[0]["p50"] <= df.iloc[0]["p95"]


# ---------------------------------------------------------------------------
# annual_returns
# ---------------------------------------------------------------------------

def test_annual_returns_shape(tier_returns):
    r = annual_returns(tier_returns[1])
    years = tier_returns.index.year.unique()
    assert set(r.index) == set(years)


def test_annual_returns_values_plausible(tier_returns):
    r = annual_returns(tier_returns[1])
    assert (r > -100).all()
    assert (r < 500).all()


# ---------------------------------------------------------------------------
# run() integration
# ---------------------------------------------------------------------------

def test_run_returns_two_dfs(tier_returns):
    vix = _fake_hourly(seed=7)
    vxn = _fake_hourly(seed=11)
    from datetime import time as dt_time
    bs_df, ann_df = run(tier_returns, vxn, vix, "America/New_York", dt_time(16, 30),
                        n_iter=50)
    assert len(bs_df) == 6
    assert "full_SR" in bs_df.columns
    assert "SMA200 + Vol-target 5%" in ann_df.columns


def test_run_sma_vt5_finite(tier_returns):
    vix = _fake_hourly(seed=7)
    vxn = _fake_hourly(seed=11)
    from datetime import time as dt_time
    bs_df, ann_df = run(tier_returns, vxn, vix, "America/New_York", dt_time(16, 30),
                        n_iter=50)
    row = bs_df[bs_df["strategy"] == "SMA200 + Vol-target 5%"]
    assert len(row) == 1
    assert np.isfinite(row.iloc[0]["full_SR"])
    assert np.isfinite(row.iloc[0]["p5"])
