"""Detailed portfolio holdings breakdown — what was actually invested in SPY vs SHV each year.

Shows:
- Dollar amount in SPY (market)
- Dollar amount in SHV (defensive)
- Days holding each
- Portfolio value progression
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from ..broker.ibkr_data import IBKRDataClient
from .backtest import fetch_data, run_allocation_backtest

_log = logging.getLogger(__name__)


def analyze_portfolio_detail(
    market_df: pd.DataFrame,
    defensive_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    market_ticker: str = "SPY",
    defensive_ticker: str = "SHV",
    vix_threshold: float = 15.0,
    initial_cash: float = 100_000.0,
) -> pd.DataFrame:
    """Run backtest and analyze portfolio composition by year.

    Args:
        market_df: Daily OHLCV for market asset (SPY)
        defensive_df: Daily OHLCV for defensive asset (SHV)
        vix_df: Daily VIX closes
        market_ticker: Name of market asset
        defensive_ticker: Name of defensive asset
        vix_threshold: VIX threshold for allocation
        initial_cash: Starting capital

    Returns:
        DataFrame with yearly portfolio composition
    """
    # Run backtest
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

    daily_nav = result["daily_nav"].copy()
    daily_nav["date"] = pd.to_datetime(daily_nav["date"])
    daily_nav["year"] = daily_nav["date"].dt.year

    # Calculate actual holdings
    # In binary mode: if allocation_pct > 50, holding market; else holding defensive
    daily_nav["holding"] = daily_nav["allocation_pct"].apply(lambda x: market_ticker if x > 50 else defensive_ticker)

    # Calculate position values (allocate current nav to holdings)
    daily_nav["market_value"] = daily_nav["nav"] * (daily_nav["allocation_pct"] / 100)
    daily_nav["defensive_value"] = daily_nav["nav"] * ((100 - daily_nav["allocation_pct"]) / 100)

    # Aggregate by year
    years = sorted(daily_nav["year"].unique())
    yearly_detail = []

    for year in years:
        year_data = daily_nav[daily_nav["year"] == year].copy()
        if len(year_data) == 0:
            continue

        # Days holding each asset
        days_market = (year_data["holding"] == market_ticker).sum()
        days_defensive = (year_data["holding"] == defensive_ticker).sum()

        # Portfolio values
        start_nav = year_data["nav"].iloc[0]
        end_nav = year_data["nav"].iloc[-1]

        # Average holdings
        avg_market_value = year_data["market_value"].mean()
        avg_defensive_value = year_data["defensive_value"].mean()

        # Peak holdings
        peak_market_value = year_data["market_value"].max()
        peak_defensive_value = year_data["defensive_value"].max()

        # Min holdings
        min_market_value = year_data["market_value"].min()
        min_defensive_value = year_data["defensive_value"].min()

        # Calculate P&L by holding
        # Rough estimate: change in value attributable to each holding
        market_pnl = (year_data["market_value"].iloc[-1] - year_data["market_value"].iloc[0]) / start_nav * 100 if start_nav > 0 else 0
        defensive_pnl = (year_data["defensive_value"].iloc[-1] - year_data["defensive_value"].iloc[0]) / start_nav * 100 if start_nav > 0 else 0

        yearly_detail.append({
            "year": year,
            "start_portfolio_value": start_nav,
            "end_portfolio_value": end_nav,
            "total_return_pct": (end_nav - start_nav) / start_nav * 100,
            "days_holding_spy": days_market,
            "days_holding_shv": days_defensive,
            "avg_spy_value": avg_market_value,
            "avg_shv_value": avg_defensive_value,
            "peak_spy_value": peak_market_value,
            "peak_shv_value": peak_defensive_value,
            "min_spy_value": min_market_value,
            "min_shv_value": min_defensive_value,
            "total_days": len(year_data),
        })

    return pd.DataFrame(yearly_detail)


def main():
    parser = argparse.ArgumentParser(description="Portfolio holdings breakdown by year")
    parser.add_argument("--start-date", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="allocation_portfolio_detail.csv", help="Output CSV path")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Fetch data
    data = fetch_data(
        start_date=args.start_date,
        end_date=args.end_date,
        market_ticker="SPY",
        source="ibkr",
    )

    if not all(k in data for k in ["SPY", "SHV", "VIX"]):
        _log.error("Missing required data")
        return

    # Analyze portfolio detail
    detail_df = analyze_portfolio_detail(
        market_df=data["SPY"],
        defensive_df=data["SHV"],
        vix_df=data["VIX"],
        market_ticker="SPY",
        defensive_ticker="SHV",
        vix_threshold=15.0,
        initial_cash=100_000.0,
    )

    # Output
    detail_df.to_csv(args.output, index=False)
    _log.info(f"Portfolio detail saved to {args.output}")

    # Print summary
    print("\n" + "="*160)
    print("PORTFOLIO HOLDINGS BREAKDOWN BY YEAR")
    print("="*160)
    print(detail_df.to_string(index=False))
    print("="*160)


if __name__ == "__main__":
    main()
