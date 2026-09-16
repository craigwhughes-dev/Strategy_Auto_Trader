"""Phase B: Multi-tier threshold sweep.

Test different ISF.L VIX thresholds (15, 17.5, 20) in multi-tier allocation.
Keep SPY at VIX=15, vary ISF.L threshold to find optimal DD/Sharpe trade-off.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from ..broker.ibkr_data import IBKRDataClient
from .multi_tier_allocator import MultiTierAllocator

_log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase B: Multi-tier threshold sweep")
    parser.add_argument("--start-date", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--initial-cash", type=float, default=100_000.0, help="Starting capital")

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

    # Test different ISF.L thresholds
    thresholds_to_test = [15.0, 17.5, 20.0]

    print("\n" + "="*120)
    print("PHASE B: MULTI-TIER THRESHOLD SWEEP (SPY=15, ISF.L varies)")
    print("="*120)
    print(f"\nPeriod: {start_date} to {end_date}")
    print(f"Initial capital: ${args.initial_cash:,.0f}\n")

    print("| ISF.L VIX | Sharpe | Sortino | Return % | Max DD % | Final Value | Tier Breakdown |")
    print("|---|---|---|---|---|---|---|")

    results = {}
    for isfl_threshold in thresholds_to_test:
        _log.info(f"Testing ISF.L threshold {isfl_threshold}...")
        allocator = MultiTierAllocator(vix_tier1=15.0, vix_tier2=isfl_threshold)
        result = allocator.backtest(
            spy_df=spy_df,
            isfl_df=isfl_df,
            shv_df=shv_df,
            vix_df=vix_df,
            initial_cash=args.initial_cash,
        )
        summary = result["summary"]
        final_value = args.initial_cash * (1 + summary["total_return_pct"] / 100)

        tier_info = f"T1: {summary.get('pct_tier1', 0):.0f}%, T2: {summary.get('pct_tier2', 0):.0f}%, T3: {summary.get('pct_tier3', 0):.0f}%"

        print(f"| <={isfl_threshold:5.1f}    | {summary['sharpe']:6.2f} | {summary['sortino']:7.2f} | {summary['total_return_pct']:7.2f} | {summary['max_drawdown_pct']:7.2f} | ${final_value:11,.0f} | {tier_info} |")

        results[isfl_threshold] = {
            "summary": summary,
            "final_value": final_value,
            "result": result,
        }

    print("="*120)

    # Analysis
    print("\nAnalysis:")
    for threshold in thresholds_to_test:
        summary = results[threshold]["summary"]
        print(f"\nISF.L threshold {threshold}:")
        print(f"  Days in Tier 1 (SPY): {summary.get('days_tier1_spy', 0)} ({summary.get('pct_tier1', 0):.1f}%)")
        print(f"  Days in Tier 2 (ISF.L): {summary.get('days_tier2_isfl', 0)} ({summary.get('pct_tier2', 0):.1f}%)")
        print(f"  Days in Tier 3 (SHV): {summary.get('days_tier3_shv', 0)} ({summary.get('pct_tier3', 0):.1f}%)")


if __name__ == "__main__":
    main()
