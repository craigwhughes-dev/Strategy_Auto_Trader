"""Tests for multi_tier_comparators_lse_lag: check each strategy produces finite metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Strategy_Auto_Trader.allocation.multi_tier_comparators_lse_lag import (
    _SPLIT,
    _VXN_THRESHOLD,
    run,
    sma_signal_tiers,
    static_blend_returns,
    two_tier_vxn_tiers,
    vol_target_returns,
)


def _fake_tier_returns(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n)
    return pd.DataFrame(
        {
            1: rng.normal(0.0006, 0.013, n),   # Nasdaq-ish
            2: rng.normal(0.0004, 0.010, n),   # SPY-ish
            3: rng.normal(0.0002, 0.008, n),   # ISF-ish
            4: rng.normal(0.0001, 0.001, n),   # CSH2-ish
        },
        index=idx,
    )


def _fake_vix_hourly(n: int = 5000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-03 14:30", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"Close": np.clip(rng.normal(18, 4, n), 8, 50)}, index=idx)


def _fake_vxn_hourly(n: int = 5000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-03 14:30", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"Close": np.clip(rng.normal(20, 5, n), 10, 60)}, index=idx)


@pytest.fixture
def inputs():
    tr = _fake_tier_returns()
    vix = _fake_vix_hourly()
    vxn = _fake_vxn_hourly()
    return tr, vix, vxn


def test_static_blend_shape(inputs):
    tr, _, _ = inputs
    r = static_blend_returns(tr)
    assert len(r) == len(tr)
    assert r.isna().sum() == 0


def test_vol_target_no_nan_after_warmup(inputs):
    tr, _, _ = inputs
    for tv in (0.10, 0.15):
        r = vol_target_returns(tr, tv)
        # First window rows may be NaN due to rolling std; rest should be finite
        assert r.iloc[20:].isna().sum() == 0, f"NaN found after warmup for vol-target {tv}"


def test_sma_signal_binary(inputs):
    tr, _, _ = inputs
    sig = sma_signal_tiers(tr)
    valid = sig.dropna()
    assert set(valid.unique()).issubset({1, 4})


def test_two_tier_vxn_binary(inputs):
    tr, vix, vxn = inputs
    from Strategy_Auto_Trader.allocation.multi_tier_backtest_lse_lag import bars_by_end_time
    from datetime import time as dt_time
    vxn_end = bars_by_end_time(vxn)
    sig = two_tier_vxn_tiers(tr.index, vxn_end, "America/New_York", dt_time(16, 30), _VXN_THRESHOLD)
    assert set(sig.unique()).issubset({1, 4})


def test_run_produces_expected_rows(inputs):
    tr, vix, vxn = inputs
    from datetime import time as dt_time
    df = run(tr, vxn, vix, "America/New_York", dt_time(16, 30), cost_bps=13.0)
    # All rows should have finite sharpe and return
    assert not df["sharpe"].isna().any(), "NaN sharpe"
    assert not df["return%"].isna().any(), "NaN return%"
    # B&H Nasdaq should have higher return than CSH2 (over 600 days of positive drift)
    bh_nasdaq = df[df["strategy"].str.startswith("[B&H] Nasdaq")]["return%"].item()
    bh_csh2 = df[df["strategy"].str.startswith("[B&H] CSH2")]["return%"].item()
    assert bh_nasdaq > bh_csh2, "Nasdaq B&H should beat CSH2 in synthetic data"


def test_ref_rows_present(inputs):
    tr, vix, vxn = inputs
    from datetime import time as dt_time
    df = run(tr, vxn, vix, "America/New_York", dt_time(16, 30))
    strategies = df["strategy"].tolist()
    assert any("Run B raw" in s for s in strategies)
    assert any("Run B asym" in s for s in strategies)
    assert any("C2" in s for s in strategies)
    assert any("SMA" in s for s in strategies)
    assert any("Vol-target" in s for s in strategies)
    assert any("Static" in s for s in strategies)


def test_deployed_ref_matches_filter_sweep_approx(inputs):
    """Run B asym10d sharpe should be in plausible range for random data."""
    tr, vix, vxn = inputs
    from datetime import time as dt_time
    df = run(tr, vxn, vix, "America/New_York", dt_time(16, 30))
    ref = df[df["strategy"].str.contains("Run B asym")]
    assert len(ref) == 1
    sharpe = ref["sharpe"].item()
    assert -5.0 < sharpe < 5.0, f"Implausible sharpe {sharpe}"
