"""Multi-tier allocation backtest: CSH2.L (tier3) vs SHV baseline, 26yr synthetic.

Compare SPY/ISF.L/CSH2.L against SPY/ISF.L/SHV using same allocation logic.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .multi_tier_allocator import MultiTierAllocator

_log = logging.getLogger(__name__)


def yearly_breakdown(result: dict, initial_cash: float) -> pd.DataFrame:
    """Per-year P&L by tier + combined."""
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
        pnl_tier3 = pnl_by_tier.get("CSH2.L", 0.0)

        rows.append({
            "year": year,
            "combined_return_pct": combined_return_pct,
            "combined_pnl": combined_pnl,
            "spy_tier_pnl": pnl_spy,
            "isfl_tier_pnl": pnl_isfl,
            "tier3_pnl": pnl_tier3,
        })

    return pd.DataFrame(rows)


def load_synthetic_daily(csv_path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """Load a synthetic hourly CSV and resample to daily OHLCV."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.loc[start_date:end_date]
    daily = df.resample("1D").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    }).dropna(subset=["Close"])
    return daily


def main():
    parser = argparse.ArgumentParser(description="Multi-tier CSH2.L vs SHV: 26yr synthetic comparison")
    parser.add_argument("--synthetic-data-dir", default="data_synthetic/hourly", help="Dir with SPY/ISF.L/SHV/CSH2.L/VIX.csv")
    parser.add_argument("--start-date", default="1999-09-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2026-09-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--initial-cash", type=float, default=100_000.0, help="Starting capital")
    parser.add_argument("--vix-tier1", type=float, default=15.0, help="VIX threshold for SPY tier")
    parser.add_argument("--isfl-threshold", type=float, default=17.5, help="ISF.L VIX threshold")
    parser.add_argument("--yearly", action="store_true", help="Show yearly breakdown")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    data_dir = Path(args.synthetic_data_dir)

    _log.info(f"Loading synthetic daily data ({args.start_date} to {args.end_date})...")
    spy_df = load_synthetic_daily(data_dir / "SPY.csv", args.start_date, args.end_date)
    isfl_df = load_synthetic_daily(data_dir / "ISF.L.csv", args.start_date, args.end_date)
    shv_df = load_synthetic_daily(data_dir / "SHV.csv", args.start_date, args.end_date)
    csh2_df = load_synthetic_daily(data_dir / "CSH2.L.csv", args.start_date, args.end_date)
    vix_df = load_synthetic_daily(data_dir / "VIX.csv", args.start_date, args.end_date)

    _log.info(f"SPY: {len(spy_df)}, ISF.L: {len(isfl_df)}, SHV: {len(shv_df)}, CSH2.L: {len(csh2_df)}, VIX: {len(vix_df)}")

    # SHV baseline
    dates_shv = spy_df.index.intersection(isfl_df.index).intersection(shv_df.index).intersection(vix_df.index)
    _log.info(f"SHV common dates: {len(dates_shv)} ({dates_shv.min()} to {dates_shv.max()})")

    # CSH2 variant
    dates_csh2 = spy_df.index.intersection(isfl_df.index).intersection(csh2_df.index).intersection(vix_df.index)
    _log.info(f"CSH2 common dates: {len(dates_csh2)} ({dates_csh2.min()} to {dates_csh2.max()})")

    print("\n" + "=" * 140)
    print("MULTI-TIER ALLOCATION: CSH2.L vs SHV (26-YEAR SYNTHETIC COMPARISON)")
    print("=" * 140)
    print(f"\nData: {data_dir}")
    print(f"Initial capital: ${args.initial_cash:,.0f}")
    print(f"SPY VIX tier: {args.vix_tier1}, ISF.L VIX tier: {args.isfl_threshold}\n")

    # SHV backtest
    print("SHV (TIER 3) BASELINE")
    print("-" * 140)
    allocator_shv = MultiTierAllocator(vix_tier1=args.vix_tier1, vix_tier2=args.isfl_threshold)
    result_shv = allocator_shv.backtest(
        spy_df=spy_df.loc[dates_shv], isfl_df=isfl_df.loc[dates_shv], shv_df=shv_df.loc[dates_shv], vix_df=vix_df.loc[dates_shv],
        initial_cash=args.initial_cash,
    )
    summary_shv = result_shv["summary"]
    final_value_shv = args.initial_cash * (1 + summary_shv["total_return_pct"] / 100)
    tier_info_shv = f"T1: {summary_shv.get('pct_tier1', 0):.0f}%, T2: {summary_shv.get('pct_tier2', 0):.0f}%, T3: {summary_shv.get('pct_tier3', 0):.0f}%"
    print(f"Sharpe: {summary_shv['sharpe']:6.2f}, Sortino: {summary_shv['sortino']:7.2f}, Return: {summary_shv['total_return_pct']:9.2f}%, Max DD: {summary_shv['max_drawdown_pct']:7.2f}%")
    print(f"Final value: ${final_value_shv:,.0f}")
    print(f"Tier allocation: {tier_info_shv}\n")

    # CSH2 backtest
    print("CSH2.L (TIER 3) CANDIDATE")
    print("-" * 140)
    allocator_csh2 = MultiTierAllocator(vix_tier1=args.vix_tier1, vix_tier2=args.isfl_threshold)
    result_csh2 = allocator_csh2.backtest(
        spy_df=spy_df.loc[dates_csh2], isfl_df=isfl_df.loc[dates_csh2], shv_df=csh2_df.loc[dates_csh2], vix_df=vix_df.loc[dates_csh2],
        initial_cash=args.initial_cash,
    )
    summary_csh2 = result_csh2["summary"]
    final_value_csh2 = args.initial_cash * (1 + summary_csh2["total_return_pct"] / 100)
    tier_info_csh2 = f"T1: {summary_csh2.get('pct_tier1', 0):.0f}%, T2: {summary_csh2.get('pct_tier2', 0):.0f}%, T3: {summary_csh2.get('pct_tier3', 0):.0f}%"
    print(f"Sharpe: {summary_csh2['sharpe']:6.2f}, Sortino: {summary_csh2['sortino']:7.2f}, Return: {summary_csh2['total_return_pct']:9.2f}%, Max DD: {summary_csh2['max_drawdown_pct']:7.2f}%")
    print(f"Final value: ${final_value_csh2:,.0f}")
    print(f"Tier allocation: {tier_info_csh2}\n")

    # Comparison
    print("=" * 140)
    print("COMPARISON (CSH2 vs SHV)")
    print("=" * 140)
    sharpe_delta = summary_csh2['sharpe'] - summary_shv['sharpe']
    sortino_delta = summary_csh2['sortino'] - summary_shv['sortino']
    return_delta = summary_csh2['total_return_pct'] - summary_shv['total_return_pct']
    dd_delta = summary_csh2['max_drawdown_pct'] - summary_shv['max_drawdown_pct']
    value_delta = final_value_csh2 - final_value_shv

    print(f"Sharpe:    CSH2 {summary_csh2['sharpe']:6.2f} vs SHV {summary_shv['sharpe']:6.2f}  ({sharpe_delta:+.2f}) {'WIN CSH2' if sharpe_delta > 0 else 'WIN SHV'}")
    print(f"Sortino:   CSH2 {summary_csh2['sortino']:6.2f} vs SHV {summary_shv['sortino']:6.2f}  ({sortino_delta:+.2f}) {'WIN CSH2' if sortino_delta > 0 else 'WIN SHV'}")
    print(f"Return:    CSH2 {summary_csh2['total_return_pct']:6.2f}% vs SHV {summary_shv['total_return_pct']:6.2f}%  ({return_delta:+.2f}%) {'WIN CSH2' if return_delta > 0 else 'WIN SHV'}")
    print(f"Max DD:    CSH2 {summary_csh2['max_drawdown_pct']:6.2f}% vs SHV {summary_shv['max_drawdown_pct']:6.2f}%  ({dd_delta:+.2f}%) {'WIN CSH2' if dd_delta < 0 else 'WIN SHV'}")
    print(f"Final $:   CSH2 ${final_value_csh2:,.0f} vs SHV ${final_value_shv:,.0f}  ({value_delta:+,.0f})")
    print("=" * 140)

    if args.yearly:
        yearly_csh2 = yearly_breakdown(result_csh2, args.initial_cash)
        yearly_shv = yearly_breakdown(result_shv, args.initial_cash)

        print("\nYEARLY BREAKDOWN (CSH2 vs SHV)")
        print("=" * 140)
        print(f"{'Year':<6} {'CSH2 %':>10} {'CSH2 PnL':>14} {'SHV %':>10} {'SHV PnL':>14} {'Diff %':>10} {'Diff $':>14}")
        for _, r_csh2 in yearly_csh2.iterrows():
            year = int(r_csh2['year'])
            r_shv = yearly_shv[yearly_shv['year'] == year]
            if len(r_shv) > 0:
                r_shv = r_shv.iloc[0]
                diff_pct = r_csh2['combined_return_pct'] - r_shv['combined_return_pct']
                diff_pnl = r_csh2['combined_pnl'] - r_shv['combined_pnl']
                print(f"{year:<6} {r_csh2['combined_return_pct']:>9.2f}% {r_csh2['combined_pnl']:>13,.0f} {r_shv['combined_return_pct']:>9.2f}% {r_shv['combined_pnl']:>13,.0f} {diff_pct:>9.2f}% {diff_pnl:>13,.0f}")
        print("=" * 140)


if __name__ == "__main__":
    main()
