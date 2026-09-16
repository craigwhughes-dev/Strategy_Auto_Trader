"""Yearly allocation strategy breakdown vs S&P/FTSE benchmarks.

Runs backtest and aggregates results by year, showing:
- Allocation strategy yearly return/max_dd
- Days in market vs defensive
- Trade rebalances
- S&P/FTSE benchmark returns
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..broker.ibkr_data import IBKRDataClient
from .rotator import AllocationRotator
from .backtest import fetch_data, run_allocation_backtest

_log = logging.getLogger(__name__)


def calculate_index_returns(index_df: pd.DataFrame, year: int) -> float:
    """Calculate index return for a given year (price-only, no dividends)."""
    year_data = index_df.loc[f"{year}-01-01":f"{year}-12-31"]
    if len(year_data) < 2:
        return 0.0
    return (year_data["Close"].iloc[-1] - year_data["Close"].iloc[0]) / year_data["Close"].iloc[0] * 100


def analyze_yearly(
    market_df: pd.DataFrame,
    defensive_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    market_ticker: str = "SPY",
    defensive_ticker: str = "SHV",
    vix_threshold: float = 15.0,
    initial_cash: float = 100_000.0,
) -> pd.DataFrame:
    """Run backtest and analyze yearly breakdown.

    Args:
        market_df: Daily OHLCV for market asset
        defensive_df: Daily OHLCV for defensive asset
        vix_df: Daily VIX closes
        market_ticker: Name of market asset
        defensive_ticker: Name of defensive asset
        vix_threshold: VIX threshold for allocation
        initial_cash: Starting capital

    Returns:
        DataFrame with yearly breakdown (strategy return, benchmark return, allocation %)
    """
    # Run backtest
    rotator = AllocationRotator(
        market_ticker=market_ticker,
        vix_threshold=vix_threshold,
        pbull_threshold=0.5,  # Not used in binary mode, but required
        mode="binary",
    )

    result = run_allocation_backtest(
        market_df=market_df,
        defensive_df=defensive_df,
        vix_df=vix_df,
        vix_threshold=vix_threshold,
        pbull_threshold=0.5,
        initial_cash=initial_cash,
        market_ticker=market_ticker,
        mode="binary",
    )

    daily_nav = result["daily_nav"]
    daily_nav["date"] = pd.to_datetime(daily_nav["date"])
    daily_nav["year"] = daily_nav["date"].dt.year

    # Calculate yearly stats
    years = sorted(daily_nav["year"].unique())
    yearly_stats = []

    for year in years:
        year_data = daily_nav[daily_nav["year"] == year].copy()
        if len(year_data) == 0:
            continue

        # Strategy return for year
        start_nav = year_data["nav"].iloc[0]
        end_nav = year_data["nav"].iloc[-1]
        strategy_return = (end_nav - start_nav) / start_nav * 100

        # Max drawdown for year
        running_max = year_data["nav"].cummax()
        drawdown = (year_data["nav"] - running_max) / running_max * 100
        max_dd = drawdown.min()

        # Time in market
        days_in_market = (year_data["allocation_pct"] > 50).sum()
        total_days = len(year_data)
        pct_in_market = days_in_market / total_days * 100 if total_days > 0 else 0

        # Count rebalances (allocation changes)
        allocation_shifts = (year_data["allocation_pct"].diff().abs() > 1).sum()

        # Benchmark returns
        market_year_data = market_df.loc[f"{year}-01-01":f"{year}-12-31"]
        market_return = 0.0
        if len(market_year_data) >= 2:
            # Get close column (case-insensitive)
            close_col = next((c for c in market_year_data.columns if c.upper() == "CLOSE"), None)
            if close_col:
                market_return = (
                    (market_year_data[close_col].iloc[-1] - market_year_data[close_col].iloc[0]) /
                    market_year_data[close_col].iloc[0] * 100
                )

        yearly_stats.append({
            "year": year,
            "strategy_return_pct": strategy_return,
            "strategy_max_dd_pct": max_dd,
            "days_in_market": days_in_market,
            "total_days": total_days,
            "pct_in_market": pct_in_market,
            "rebalances": allocation_shifts,
            "market_return_pct": market_return,
            "strategy_nav_start": start_nav,
            "strategy_nav_end": end_nav,
        })

    return pd.DataFrame(yearly_stats)


def main():
    parser = argparse.ArgumentParser(description="Yearly allocation strategy breakdown")
    parser.add_argument("--start-date", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--market-ticker", default="SPY", help="Market asset")
    parser.add_argument("--defensive-ticker", default="SHV", help="Defensive asset")
    parser.add_argument("--vix-threshold", type=float, default=15.0, help="VIX threshold")
    parser.add_argument("--initial-cash", type=float, default=100_000.0, help="Starting capital")
    parser.add_argument("--output", default="allocation_yearly.csv", help="Output CSV path")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Fetch data
    data = fetch_data(
        start_date=args.start_date,
        end_date=args.end_date,
        market_ticker=args.market_ticker,
        source="ibkr",
    )

    if not all(k in data for k in [args.market_ticker, args.defensive_ticker, "VIX"]):
        _log.error("Missing required data")
        return

    # Analyze yearly
    yearly_df = analyze_yearly(
        market_df=data[args.market_ticker],
        defensive_df=data[args.defensive_ticker],
        vix_df=data["VIX"],
        market_ticker=args.market_ticker,
        defensive_ticker=args.defensive_ticker,
        vix_threshold=args.vix_threshold,
        initial_cash=args.initial_cash,
    )

    # Output
    yearly_df.to_csv(args.output, index=False)
    _log.info(f"Yearly breakdown saved to {args.output}")

    # Print summary
    print("\n" + "="*120)
    print("YEARLY ALLOCATION STRATEGY BREAKDOWN")
    print("="*120)
    print(yearly_df.to_string(index=False))
    print("="*120)

    # Calculate cumulative comparison
    print(f"\nCumulative (full period):")
    print(f"  Strategy return: {(yearly_df['strategy_return_pct'] + 100).prod() - 100:+.2f}%")
    print(f"  Market return:   {(yearly_df['market_return_pct'] + 100).prod() - 100:+.2f}%")
    print(f"  Avg time in market: {yearly_df['pct_in_market'].mean():.1f}%")


if __name__ == "__main__":
    main()
