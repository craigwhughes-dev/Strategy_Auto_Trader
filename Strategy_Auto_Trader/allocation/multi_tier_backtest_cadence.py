"""Cadence sweep for multi-tier allocation: test buy delays 1-10 days.

Goal: Defer buys/upgrades to every N days (reduce entry churn) while allowing
immediate defensive sells. Preserve crisis response, minimize whipsaw in choppy years.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .multi_tier_allocator_cadence import MultiTierAllocatorCadence

_log = logging.getLogger(__name__)


def load_synthetic_daily(csv_path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """Load a synthetic hourly CSV and resample to daily OHLCV."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.loc[start_date:end_date]
    daily = df.resample("1D").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    }).dropna(subset=["Close"])
    return daily


def main():
    parser = argparse.ArgumentParser(description="Multi-tier allocation cadence sweep")
    parser.add_argument("--synthetic-data-dir", default="data_synthetic/hourly", help="Dir with SPY/ISF.L/CSH2.L/VIX.csv")
    parser.add_argument("--start-date", default="1999-09-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2026-09-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--initial-cash", type=float, default=100_000.0, help="Starting capital")
    parser.add_argument("--vix-tier1", type=float, default=15.0, help="VIX threshold for SPY tier")
    parser.add_argument("--vix-tier2", type=float, default=17.5, help="VIX threshold for ISF.L tier")
    parser.add_argument("--cadence-days", type=int, nargs="+", default=[1, 2, 5, 10], help="Days to defer buys")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    data_dir = Path(args.synthetic_data_dir)

    _log.info(f"Loading synthetic daily data ({args.start_date} to {args.end_date})...")
    spy_df = load_synthetic_daily(data_dir / "SPY.csv", args.start_date, args.end_date)
    isfl_df = load_synthetic_daily(data_dir / "ISF.L.csv", args.start_date, args.end_date)
    csh2_df = load_synthetic_daily(data_dir / "CSH2.L.csv", args.start_date, args.end_date)
    vix_df = load_synthetic_daily(data_dir / "VIX.csv", args.start_date, args.end_date)

    _log.info(f"SPY: {len(spy_df)} days, ISF.L: {len(isfl_df)} days, CSH2.L: {len(csh2_df)} days, VIX: {len(vix_df)} days")

    dates = spy_df.index.intersection(isfl_df.index).intersection(csh2_df.index).intersection(vix_df.index)
    _log.info(f"Common dates: {len(dates)} ({dates.min()} to {dates.max()})")

    print("\n" + "=" * 130)
    print("MULTI-TIER ALLOCATION: BUY CADENCE SWEEP")
    print("=" * 130)
    print(f"\nData: {data_dir}, common range {dates.min().date()} to {dates.max().date()} ({len(dates)} daily bars)")
    print(f"VIX tiers: {args.vix_tier1} / {args.vix_tier2}, Initial capital: ${args.initial_cash:,.0f}")
    print("Strategy: Sell defensively immediately, defer buys/upgrades by N days\n")

    print("| Buy Cadence | Sharpe | Sortino | Return % | Max DD % | Final Value | Switches | Bad Years (2005/2015) |")
    print("|---|---|---|---|---|---|---|---|---|")

    for cadence_days in sorted(args.cadence_days):
        allocator = MultiTierAllocatorCadence(
            vix_tier1=args.vix_tier1,
            vix_tier2=args.vix_tier2,
            buy_cadence_days=cadence_days,
        )
        result = allocator.backtest(
            spy_df=spy_df,
            isfl_df=isfl_df,
            shv_df=csh2_df,
            vix_df=vix_df,
            initial_cash=args.initial_cash,
        )

        summary = result["summary"]
        final_value = args.initial_cash * (1 + summary["total_return_pct"] / 100)

        # Count switches per year and extract 2005/2015 returns
        daily_nav = result["daily_nav"].copy()
        daily_nav["date"] = pd.to_datetime(daily_nav["date"])
        daily_nav["year"] = daily_nav["date"].dt.year
        daily_nav["asset_changed"] = daily_nav["asset"] != daily_nav["asset"].shift(1)

        switches_per_year = daily_nav.groupby("year")["asset_changed"].sum()
        total_switches = switches_per_year.sum()

        # Extract 2005 and 2015 returns
        def year_return(year_val):
            yr_data = daily_nav[daily_nav["year"] == year_val]
            if len(yr_data) < 2:
                return 0.0
            start_nav = yr_data["nav"].iloc[0] / (1 + yr_data["daily_return_pct"].iloc[0] / 100)
            end_nav = yr_data["nav"].iloc[-1]
            return (end_nav - start_nav) / start_nav * 100

        ret_2005 = year_return(2005)
        ret_2015 = year_return(2015)

        bad_years_str = f"2005: {ret_2005:+.2f}%, 2015: {ret_2015:+.2f}%"

        print(f"|{cadence_days:11d} days | {summary['sharpe']:6.2f} | {summary['sortino']:7.2f} | {summary['total_return_pct']:9.2f} | {summary['max_drawdown_pct']:7.2f} | ${final_value:13,.0f} | {int(total_switches):8d} | {bad_years_str:>20s} |")

    print("=" * 130)


if __name__ == "__main__":
    main()
