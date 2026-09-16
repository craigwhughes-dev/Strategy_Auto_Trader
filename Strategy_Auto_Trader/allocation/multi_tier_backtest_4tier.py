"""4-tier allocation backtest: Compare 3-tier baseline vs 4-tier (Nasdaq added).

Backtest both configurations to decide: is 4-tier worth the complexity?
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .multi_tier_allocator import MultiTierAllocator
from .multi_tier_allocator_4tier import MultiTierAllocator4Tier

_log = logging.getLogger(__name__)


def load_synthetic_daily(csv_path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """Load a synthetic hourly CSV and resample to daily OHLCV."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.loc[start_date:end_date]

    # Handle EQGB (Close-only) vs other files (OHLCV)
    if "Open" not in df.columns:
        # EQGB case: only Close column
        daily = df.resample("1D").agg({
            "Close": "last"
        }).dropna(subset=["Close"])
    else:
        daily = df.resample("1D").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
        }).dropna(subset=["Close"])
    return daily


def main():
    parser = argparse.ArgumentParser(description="3-tier vs 4-tier allocation comparison")
    parser.add_argument("--synthetic-data-dir", default="data_synthetic/hourly", help="Dir with synthetic data")
    parser.add_argument("--start-date", default="1999-09-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2026-09-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--initial-cash", type=float, default=100_000.0, help="Starting capital")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    data_dir = Path(args.synthetic_data_dir)

    _log.info(f"Loading synthetic daily data ({args.start_date} to {args.end_date})...")
    spy_df = load_synthetic_daily(data_dir / "SPY.csv", args.start_date, args.end_date)
    isfl_df = load_synthetic_daily(data_dir / "ISF.L.csv", args.start_date, args.end_date)
    csh2_df = load_synthetic_daily(data_dir / "CSH2.L.csv", args.start_date, args.end_date)
    eqgb_df = load_synthetic_daily(data_dir / "EQGB_COMPLETE.csv", args.start_date, args.end_date)
    vix_df = load_synthetic_daily(data_dir / "VIX.csv", args.start_date, args.end_date)
    vxn_df = load_synthetic_daily(data_dir / "VXN.csv", args.start_date, args.end_date)

    _log.info(f"SPY: {len(spy_df)}, ISF.L: {len(isfl_df)}, CSH2: {len(csh2_df)}, EQGB: {len(eqgb_df)}, VIX: {len(vix_df)}, VXN: {len(vxn_df)}")

    dates_3tier = spy_df.index.intersection(isfl_df.index).intersection(csh2_df.index).intersection(vix_df.index)
    dates_4tier = spy_df.index.intersection(isfl_df.index).intersection(csh2_df.index).intersection(eqgb_df.index).intersection(vix_df.index).intersection(vxn_df.index)

    print("\n" + "=" * 140)
    print("3-TIER vs 4-TIER ALLOCATION: DECISION DATA")
    print("=" * 140)
    print(f"\nCONSTRAINT: VXN only available from 2007-11-20 (no 1999-2007 history)")
    print(f"3-tier baseline data: {len(dates_3tier)} days (1999-2026)")
    print(f"4-tier data: {len(dates_4tier)} days (overlap only)")
    print(f"  -> Fair comparison requires both tested on same date range")
    if len(dates_4tier) > 0:
        print(f"\nComparing on common period {dates_4tier.min().date()} to {dates_4tier.max().date()} ({len(dates_4tier)} days):\n")
    else:
        print(f"\n[ERROR] No common dates across all 4-tier inputs. Data alignment failed.")

    # 3-tier baseline (on same date range as 4-tier for fair comparison)
    print("3-TIER BASELINE (SPY/ISF.L/CSH2.L with VIX gates, on common 4-tier date range):")
    print("-" * 140)

    # Subset data to common dates
    spy_df_common = spy_df.loc[dates_4tier]
    isfl_df_common = isfl_df.loc[dates_4tier]
    csh2_df_common = csh2_df.loc[dates_4tier]
    vix_df_common = vix_df.loc[dates_4tier]

    allocator_3tier = MultiTierAllocator(vix_tier1=15.0, vix_tier2=17.5)
    result_3tier = allocator_3tier.backtest(
        spy_df=spy_df_common,
        isfl_df=isfl_df_common,
        shv_df=csh2_df_common,
        vix_df=vix_df_common,
        initial_cash=args.initial_cash,
    )
    summary_3tier = result_3tier["summary"]
    final_3tier = args.initial_cash * (1 + summary_3tier["total_return_pct"] / 100)

    print(f"Sharpe: {summary_3tier['sharpe']:7.2f}")
    print(f"Sortino: {summary_3tier['sortino']:7.2f}")
    print(f"Return: {summary_3tier['total_return_pct']:9.2f}%")
    print(f"Max DD: {summary_3tier['max_drawdown_pct']:7.2f}%")
    print(f"Final Value: ${final_3tier:,.0f}")
    print(f"Tier allocation: T1={summary_3tier.get('pct_tier1', 0):.0f}%, T2={summary_3tier.get('pct_tier2', 0):.0f}%, T3={summary_3tier.get('pct_tier3', 0):.0f}%")

    # 4-tier with different VXN thresholds
    print("\n4-TIER VARIANTS (Nasdaq added with VXN gate):")
    print("-" * 140)
    print("| VXN Threshold | Sharpe | Sortino | Return % | Max DD % | Final Value | Nasdaq % | SPY % | ISF % | Cash % |")
    print("|---|---|---|---|---|---|---|---|---|---|")

    # Subset to common dates
    eqgb_df_common = eqgb_df.loc[dates_4tier]
    vxn_df_common = vxn_df.loc[dates_4tier]

    for vxn_thresh in [12.0, 15.0, 18.0, 20.0, 23.0, 25.0]:
        allocator_4tier = MultiTierAllocator4Tier(vxn_threshold=vxn_thresh, vix_tier1=15.0, vix_tier2=17.5)
        result_4tier = allocator_4tier.backtest(
            nasdaq_df=eqgb_df_common,
            spy_df=spy_df_common,
            isfl_df=isfl_df_common,
            csh2_df=csh2_df_common,
            vxn_df=vxn_df_common,
            vix_df=vix_df_common,
            initial_cash=args.initial_cash,
        )
        summary_4tier = result_4tier["summary"]
        final_4tier = args.initial_cash * (1 + summary_4tier["total_return_pct"] / 100)

        nasdaq_pct = summary_4tier.get("pct_tier1", 0)
        spy_pct = summary_4tier.get("pct_tier2", 0)
        isf_pct = summary_4tier.get("pct_tier3", 0)
        cash_pct = summary_4tier.get("pct_tier4", 0)

        print(f"|    <{vxn_thresh:5.1f}     | {summary_4tier['sharpe']:6.2f} | {summary_4tier['sortino']:7.2f} | {summary_4tier['total_return_pct']:9.2f} | {summary_4tier['max_drawdown_pct']:7.2f} | ${final_4tier:13,.0f} | {nasdaq_pct:7.0f}% | {spy_pct:4.0f}% | {isf_pct:4.0f}% | {cash_pct:4.0f}% |")

    print("=" * 140)
    print("\nDECISION SUMMARY:")
    print(f"3-tier baseline: Sharpe {summary_3tier['sharpe']:.2f}, return +{summary_3tier['total_return_pct']:.0f}%")
    print("Question: Does any 4-tier variant beat 3-tier Sharpe by >0.1 (material improvement)?")
    print("If yes → worth restructuring to 4-tier. If no → stick with 3-tier + parallel Nasdaq candidate approach.")

if __name__ == "__main__":
    main()
