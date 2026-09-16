"""Multi-tier allocation backtest on synthetic hourly data (26yr stress test).

Mirrors phase_b_threshold_sweep.py but sources SPY/ISF.L/SHV/VIX from
data_synthetic/hourly/*.csv (resampled to daily) instead of IBKR real data.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .multi_tier_allocator import MultiTierAllocator

_log = logging.getLogger(__name__)


def load_synthetic_daily(csv_path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """Load a synthetic hourly CSV and resample to daily OHLCV."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.loc[start_date:end_date]
    daily = df.resample("1D").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    }).dropna(subset=["Close"])
    return daily


def yearly_breakdown(
    result: dict,
    spy_df: pd.DataFrame,
    isfl_df: pd.DataFrame,
    initial_cash: float,
) -> pd.DataFrame:
    """Per-year P&L by tier + combined, plus standalone SPY/ISF.L buy-and-hold % gain.

    Tier P&L is the sum of daily NAV deltas on days the strategy held that tier's asset
    (approximation: NAV delta on a given day is attributed to whichever asset was held
    that day, which is exact since the strategy is always 100% in one asset).
    """
    daily_nav = result["daily_nav"].copy()
    daily_nav["date"] = pd.to_datetime(daily_nav["date"])
    daily_nav["year"] = daily_nav["date"].dt.year
    daily_nav["nav_prev"] = daily_nav["nav"].shift(1).fillna(initial_cash)
    daily_nav["nav_delta"] = daily_nav["nav"] - daily_nav["nav_prev"]

    rows = []
    for year, grp in daily_nav.groupby("year"):
        start_nav = grp["nav_prev"].iloc[0]
        end_nav = grp["nav"].iloc[-1]
        combined_return_pct = (end_nav - start_nav) / start_nav * 100
        combined_pnl = end_nav - start_nav

        pnl_by_tier = grp.groupby("asset")["nav_delta"].sum()
        pnl_spy = pnl_by_tier.get("SPY", 0.0)
        pnl_isfl = pnl_by_tier.get("ISF.L", 0.0)
        pnl_csh2 = pnl_by_tier.get("CSH2.L", 0.0)

        def bh_pct(df: pd.DataFrame) -> float:
            yr = df.loc[f"{year}-01-01":f"{year}-12-31"]
            if len(yr) < 2:
                return 0.0
            return (yr["Close"].iloc[-1] - yr["Close"].iloc[0]) / yr["Close"].iloc[0] * 100

        rows.append({
            "year": year,
            "combined_return_pct": combined_return_pct,
            "combined_pnl": combined_pnl,
            "spy_tier_pnl": pnl_spy,
            "isfl_tier_pnl": pnl_isfl,
            "csh2_tier_pnl": pnl_csh2,
            "spy_bh_pct": bh_pct(spy_df),
            "isfl_bh_pct": bh_pct(isfl_df),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Multi-tier allocation: 26yr synthetic sweep")
    parser.add_argument("--synthetic-data-dir", default="data_synthetic/hourly", help="Dir with SPY/ISF.L/SHV/VIX.csv")
    parser.add_argument("--start-date", default="1999-09-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2026-09-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--initial-cash", type=float, default=100_000.0, help="Starting capital")
    parser.add_argument("--vix-tier1", type=float, default=15.0, help="VIX threshold for SPY tier")
    parser.add_argument("--isfl-thresholds", type=float, nargs="+", default=[15.0, 17.5, 20.0], help="ISF.L VIX thresholds to sweep")
    parser.add_argument("--yearly-threshold", type=float, default=None, help="If set, print per-year tier P&L breakdown for this ISF.L threshold")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    data_dir = Path(args.synthetic_data_dir)

    _log.info(f"Loading synthetic daily data ({args.start_date} to {args.end_date})...")
    spy_df = load_synthetic_daily(data_dir / "SPY.csv", args.start_date, args.end_date)
    isfl_df = load_synthetic_daily(data_dir / "ISF.L.csv", args.start_date, args.end_date)
    shv_df = load_synthetic_daily(data_dir / "CSH2.L.csv", args.start_date, args.end_date)
    vix_df = load_synthetic_daily(data_dir / "VIX.csv", args.start_date, args.end_date)

    _log.info(f"SPY: {len(spy_df)} days, ISF.L: {len(isfl_df)} days, CSH2.L: {len(shv_df)} days, VIX: {len(vix_df)} days")

    dates = spy_df.index.intersection(isfl_df.index).intersection(shv_df.index).intersection(vix_df.index)
    _log.info(f"Common dates: {len(dates)} ({dates.min()} to {dates.max()})")

    print("\n" + "=" * 120)
    print("MULTI-TIER ALLOCATION: 26-YEAR SYNTHETIC SWEEP (SPY / ISF.L / CSH2.L)")
    print("=" * 120)
    print(f"\nData: {data_dir}, common range {dates.min().date()} to {dates.max().date()} ({len(dates)} daily bars)")
    print(f"Initial capital: ${args.initial_cash:,.0f}\n")

    print("| ISF.L VIX | Sharpe | Sortino | Return % | Max DD % | Final Value | Tier Breakdown |")
    print("|---|---|---|---|---|---|---|")

    for isfl_threshold in args.isfl_thresholds:
        allocator = MultiTierAllocator(vix_tier1=args.vix_tier1, vix_tier2=isfl_threshold)
        result = allocator.backtest(
            spy_df=spy_df, isfl_df=isfl_df, shv_df=shv_df, vix_df=vix_df,
            initial_cash=args.initial_cash,
        )
        summary = result["summary"]
        final_value = args.initial_cash * (1 + summary["total_return_pct"] / 100)
        tier_info = f"T1: {summary.get('pct_tier1', 0):.0f}%, T2: {summary.get('pct_tier2', 0):.0f}%, T3: {summary.get('pct_tier3', 0):.0f}%"
        print(f"| <={isfl_threshold:5.1f}    | {summary['sharpe']:6.2f} | {summary['sortino']:7.2f} | {summary['total_return_pct']:9.2f} | {summary['max_drawdown_pct']:7.2f} | ${final_value:13,.0f} | {tier_info} |")

        if args.yearly_threshold is not None and isfl_threshold == args.yearly_threshold:
            yearly_df = yearly_breakdown(result, spy_df, isfl_df, args.initial_cash)

            print("\n" + "=" * 140)
            print(f"YEARLY BREAKDOWN (ISF.L threshold <= {isfl_threshold})")
            print("=" * 140)
            print(f"{'Year':<6}{'Combined %':>11}{'Combined P&L':>15}{'SPY tier P&L':>15}{'ISF.L tier P&L':>16}{'CSH2 tier P&L':>15}{'SPY B&H %':>12}{'FTSE B&H %':>12}")
            for _, r in yearly_df.iterrows():
                print(f"{int(r['year']):<6}{r['combined_return_pct']:>10.2f}%{r['combined_pnl']:>14,.0f} {r['spy_tier_pnl']:>14,.0f} {r['isfl_tier_pnl']:>15,.0f} {r['csh2_tier_pnl']:>14,.0f} {r['spy_bh_pct']:>11.2f}% {r['isfl_bh_pct']:>11.2f}%")
            print("=" * 140)
            print(f"Totals: combined P&L ${yearly_df['combined_pnl'].sum():,.0f}, SPY-tier P&L ${yearly_df['spy_tier_pnl'].sum():,.0f}, ISF.L-tier P&L ${yearly_df['isfl_tier_pnl'].sum():,.0f}, CSH2-tier P&L ${yearly_df['csh2_tier_pnl'].sum():,.0f}")
            yearly_out = Path(f"data/allocation_backtest/multi_tier_yearly_synthetic_isfl{isfl_threshold}.csv")
            yearly_out.parent.mkdir(parents=True, exist_ok=True)
            # Format percentages as strings to avoid 100x multiplication on re-read
            yearly_df_out = yearly_df.copy()
            yearly_df_out['combined_return_pct'] = yearly_df_out['combined_return_pct'].apply(lambda x: f"{x:.2f}%")
            yearly_df_out['spy_bh_pct'] = yearly_df_out['spy_bh_pct'].apply(lambda x: f"{x:.2f}%")
            yearly_df_out['isfl_bh_pct'] = yearly_df_out['isfl_bh_pct'].apply(lambda x: f"{x:.2f}%")
            yearly_df_out.to_csv(yearly_out, index=False)
            print(f"Saved: {yearly_out}")

    print("=" * 120)


if __name__ == "__main__":
    main()
