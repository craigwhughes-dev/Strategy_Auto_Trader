"""Test HMM-gated allocation: VIX + P(Bull) gating to avoid false alarms."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .hmm_gated import HMMGatedAllocator

_log = logging.getLogger(__name__)


def test_hmm_gated_allocation():
    """Test allocation with HMM regime gating on 2015-2024."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance required")
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("\n" + "=" * 100)
    print("PHASE 2: HMM-GATED ALLOCATION TEST")
    print("=" * 100)

    # Fetch daily data
    print("\nFetching daily data...")
    spy = yf.download("SPY", start="2015-01-01", end="2024-12-31", progress=False)
    shv = yf.download("SHV", start="2015-01-01", end="2024-12-31", progress=False)
    vix = yf.download("^VIX", start="2015-01-01", end="2024-12-31", progress=False)

    # Flatten columns
    for df in [spy, shv, vix]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df.columns = [col.upper() for col in df.columns]

    print(f"SPY: {len(spy)} bars, {spy.index[0]} to {spy.index[-1]}")
    print(f"SHV: {len(shv)} bars")
    print(f"VIX: {len(vix)} bars")

    # Load hourly SPY for HMM
    print("\nLoading hourly SPY for HMM...")
    hmm_cache_path = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "ibkr_hourly" / "SPY.csv"
    if hmm_cache_path.exists():
        hourly_df = pd.read_csv(hmm_cache_path, index_col=0, parse_dates=True)
        hmm_dates = hourly_df.index
        hmm_closes = hourly_df["Close"].values
        print(f"Hourly SPY: {len(hourly_df)} bars, {hmm_dates[0]} to {hmm_dates[-1]}")
    else:
        print(f"WARNING: Hourly SPY not found at {hmm_cache_path}")
        hmm_dates = None
        hmm_closes = None
        hourly_df = None

    # Test VIX-only allocation (baseline)
    print("\n" + "-" * 100)
    print("BASELINE: VIX-only gating (no HMM)")
    print("-" * 100)

    from .rotator import AllocationRotator

    rotator_vix_only = AllocationRotator(market_ticker="SPY", vix_threshold=15.0, mode="binary")
    result_vix_only = rotator_vix_only.backtest(spy, shv, vix, initial_cash=100_000)
    summary_vix = result_vix_only["summary"]

    print(f"Sharpe: {summary_vix['sharpe']:.2f}")
    print(f"Sortino: {summary_vix['sortino']:.2f}")
    print(f"Max DD: {summary_vix['max_drawdown_pct']:.2f}%")
    print(f"Return: {summary_vix['total_return_pct']:.2f}%")
    print(f"Time in market: {summary_vix['pct_time_in_market']:.1f}%")

    # Test HMM-gated allocation
    print("\n" + "-" * 100)
    print("HMM-GATED: VIX + P(Bull) gating")
    print("-" * 100)

    if hmm_dates is not None and hmm_closes is not None:
        from .extract_daily_pbull import extract_daily_pbull_from_hourly

        print("Extracting daily P(Bull) from hourly HMM...")
        pbull_series = extract_daily_pbull_from_hourly(
            pd.DataFrame({"Close": hourly_df["Close"].values}, index=hmm_dates),
            spy.index,
        )

        allocator = HMMGatedAllocator(
            market_ticker="SPY",
            vix_threshold=15.0,
            pbull_gate=0.4,
            hmm_dates=hmm_dates,
            hmm_closes=hourly_df["Close"].values,
        )

        print("Running HMM-gated allocation with daily P(Bull)...")
        result_hmm = allocator.backtest_with_hmm(spy, shv, vix, pbull_series=pbull_series, initial_cash=100_000)
        summary_hmm = result_hmm["summary"]

        print(f"Sharpe: {summary_hmm['sharpe']:.2f}")
        print(f"Sortino: {summary_hmm['sortino']:.2f}")
        print(f"Max DD: {summary_hmm['max_drawdown_pct']:.2f}%")
        print(f"Return: {summary_hmm['total_return_pct']:.2f}%")
        print(f"Time in market: {summary_hmm['pct_time_in_market']:.1f}%")

        # Comparison
        print("\n" + "-" * 100)
        print("COMPARISON: VIX-only vs HMM-gated")
        print("-" * 100)
        print(f"Sharpe improvement: {summary_hmm['sharpe'] - summary_vix['sharpe']:+.2f}")
        print(f"Sortino improvement: {summary_hmm['sortino'] - summary_vix['sortino']:+.2f}")
        print(f"DD improvement: {summary_hmm['max_drawdown_pct'] - summary_vix['max_drawdown_pct']:+.2f}%")
        print(f"Return difference: {summary_hmm['total_return_pct'] - summary_vix['total_return_pct']:+.2f}%")
        print(f"Time in market: {summary_hmm['pct_time_in_market']:.1f}% vs {summary_vix['pct_time_in_market']:.1f}%")

    else:
        print("Skipping HMM-gated test (hourly data not available)")

    print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    test_hmm_gated_allocation()
