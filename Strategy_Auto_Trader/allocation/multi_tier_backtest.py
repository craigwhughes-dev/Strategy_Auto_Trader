"""Phase B: Multi-tier allocation backtest and comparison.

Compares three strategies:
1. SPY-only (VIX≤15)
2. ISF.L-only (VIX≤20)
3. Multi-tier (pick best: SPY≤15, ISF.L≤20, else SHV)

Output: yearly breakdown + comparison table.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..broker.ibkr_data import IBKRDataClient
from .backtest import run_allocation_backtest
from .multi_tier_allocator import MultiTierAllocator

_log = logging.getLogger(__name__)


def run_phase_b_comparison(
    spy_df: pd.DataFrame,
    isfl_df: pd.DataFrame,
    shv_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    start_year: int = 2015,
    end_year: int = 2024,
    initial_cash: float = 100_000.0,
) -> dict:
    """Run all three strategies and compare.

    Args:
        spy_df, isfl_df, shv_df, vix_df: Daily OHLCV DataFrames
        start_year, end_year: Backtest window
        initial_cash: Starting capital per strategy

    Returns:
        Dict with results for each strategy + comparison summary
    """
    # Strategy 1: SPY-only (VIX=15)
    _log.info("Running SPY-only backtest (VIX≤15)...")
    spy_result = run_allocation_backtest(
        market_df=spy_df,
        defensive_df=shv_df,
        vix_df=vix_df,
        vix_threshold=15.0,
        pbull_threshold=0.5,
        initial_cash=initial_cash,
        market_ticker="SPY",
        mode="binary",
    )
    spy_summary = spy_result["summary"]

    # Strategy 2: ISF.L-only (VIX=20)
    _log.info("Running ISF.L-only backtest (VIX≤20)...")
    isfl_result = run_allocation_backtest(
        market_df=isfl_df,
        defensive_df=shv_df,
        vix_df=vix_df,
        vix_threshold=20.0,
        pbull_threshold=0.5,
        initial_cash=initial_cash,
        market_ticker="ISF.L",
        mode="binary",
    )
    isfl_summary = isfl_result["summary"]

    # Strategy 3: Multi-tier (test both threshold configs)
    _log.info("Running multi-tier allocation backtest (VIX thresholds: 15/20)...")
    allocator = MultiTierAllocator(vix_tier1=15.0, vix_tier2=20.0)
    multi_result = allocator.backtest(
        spy_df=spy_df,
        isfl_df=isfl_df,
        shv_df=shv_df,
        vix_df=vix_df,
        initial_cash=initial_cash,
    )
    multi_summary = multi_result["summary"]

    return {
        "spy": {"result": spy_result, "summary": spy_summary},
        "isfl": {"result": isfl_result, "summary": isfl_summary},
        "multi": {"result": multi_result, "summary": multi_summary},
    }


def main():
    parser = argparse.ArgumentParser(description="Phase B: Multi-tier allocation comparison")
    parser.add_argument("--start-date", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--initial-cash", type=float, default=100_000.0, help="Starting capital per strategy")
    parser.add_argument("--output", default="allocation_phase_b_comparison.csv", help="Output CSV path")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Fetch data
    _log.info("Fetching data from IBKR...")
    client = IBKRDataClient()
    spy_df = client.fetch_daily("SPY", period="max", use_cache=True, historical_only=True)
    isfl_df = client.fetch_daily("ISF.L", period="max", use_cache=True, historical_only=True)
    shv_df = client.fetch_daily("SHV", period="max", use_cache=True, historical_only=True)
    vix_df = client.fetch_index_daily("VIX", "CBOE", currency="USD", historical_only=True)

    if not all([spy_df is not None and not spy_df.empty,
                isfl_df is not None and not isfl_df.empty,
                shv_df is not None and not shv_df.empty,
                vix_df is not None and not vix_df.empty]):
        _log.error("Missing required data")
        return

    # Filter to backtest window
    start_date = args.start_date
    end_date = args.end_date
    spy_df = spy_df.loc[start_date:end_date]
    isfl_df = isfl_df.loc[start_date:end_date]
    shv_df = shv_df.loc[start_date:end_date]
    vix_df = vix_df.loc[start_date:end_date]

    # Run comparison
    _log.info(f"Running Phase B comparison ({start_date} to {end_date})...")
    results = run_phase_b_comparison(
        spy_df=spy_df,
        isfl_df=isfl_df,
        shv_df=shv_df,
        vix_df=vix_df,
        initial_cash=args.initial_cash,
    )

    # Print summary
    print("\n" + "="*100)
    print("PHASE B: MULTI-TIER ALLOCATION COMPARISON")
    print("="*100)
    print(f"\nPeriod: {start_date} to {end_date}")
    print(f"Initial capital: ${args.initial_cash:,.0f} per strategy\n")

    print("| Strategy | Sharpe | Sortino | Return % | Max DD % | Final Value | Tier Breakdown |")
    print("|---|---|---|---|---|---|---|")

    for name, data in results.items():
        summary = data["summary"]
        final_value = args.initial_cash * (1 + summary["total_return_pct"] / 100)

        if name == "multi":
            tier_info = f"Tier1: {summary.get('pct_tier1', 0):.0f}%, Tier2: {summary.get('pct_tier2', 0):.0f}%, Tier3: {summary.get('pct_tier3', 0):.0f}%"
        else:
            tier_info = "100% market"

        print(f"| {name.upper():12s} | {summary['sharpe']:6.2f} | {summary['sortino']:7.2f} | {summary['total_return_pct']:7.2f} | {summary['max_drawdown_pct']:7.2f} | ${final_value:11,.0f} | {tier_info} |")

    print("="*100)

    # Detailed output
    print("\nDetailed tier breakdown (multi-tier strategy):")
    multi_daily = results["multi"]["result"]["daily_nav"]
    print(f"  Days in Tier 1 (SPY): {results['multi']['summary'].get('days_tier1_spy', 0)} ({results['multi']['summary'].get('pct_tier1', 0):.1f}%)")
    print(f"  Days in Tier 2 (ISF.L): {results['multi']['summary'].get('days_tier2_isfl', 0)} ({results['multi']['summary'].get('pct_tier2', 0):.1f}%)")
    print(f"  Days in Tier 3 (SHV): {results['multi']['summary'].get('days_tier3_shv', 0)} ({results['multi']['summary'].get('pct_tier3', 0):.1f}%)")


if __name__ == "__main__":
    main()
