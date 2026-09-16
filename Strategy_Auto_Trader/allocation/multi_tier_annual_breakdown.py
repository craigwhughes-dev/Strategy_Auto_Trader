"""Annual breakdown: 3-tier vs 4-tier (VXN<18), properly scaled for Excel % format."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .multi_tier_allocator import MultiTierAllocator
from .multi_tier_allocator_4tier import MultiTierAllocator4Tier

_log = logging.getLogger(__name__)


def load_synthetic_daily(csv_path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """Load synthetic hourly CSV and resample to daily OHLCV."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.loc[start_date:end_date]

    if "Open" not in df.columns:
        daily = df.resample("1D").agg({"Close": "last"}).dropna(subset=["Close"])
    else:
        daily = df.resample("1D").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
        }).dropna(subset=["Close"])
    return daily


def annual_breakdown(
    result: dict,
    spy_df: pd.DataFrame,
    isfl_df: pd.DataFrame,
    eqgb_df: pd.DataFrame,
    initial_cash: float,
) -> pd.DataFrame:
    """Per-year P&L by tier + combined, plus standalone SPY/ISF.L buy-and-hold.

    Returns DataFrame with unformatted numbers (decimals for % columns).
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
        combined_return = (end_nav - start_nav) / start_nav  # Decimal, not %
        combined_pnl = end_nav - start_nav

        pnl_by_tier = grp.groupby("asset")["nav_delta"].sum()

        # Get tier names dynamically
        tier_names = grp["asset"].unique()
        tier_pnls = {name: pnl_by_tier.get(name, 0.0) for name in tier_names}

        # Underlying market returns (what market actually did that year)
        def market_return(df: pd.DataFrame) -> float:
            yr = df.loc[f"{year}-01-01":f"{year}-12-31"]
            if len(yr) < 2:
                return 0.0
            return (yr["Close"].iloc[-1] - yr["Close"].iloc[0]) / yr["Close"].iloc[0]

        spy_mkt = market_return(spy_df)
        isfl_mkt = market_return(isfl_df)
        eqgb_mkt = market_return(eqgb_df)

        rows.append({
            "year": year,
            "combined_return": combined_return,  # Decimal (0.05 = 5%)
            "combined_pnl": combined_pnl,
            **{f"{name}_pnl": tier_pnls[name] for name in tier_names},
            "spy_mkt_pct": spy_mkt,
            "isfl_ftse_mkt_pct": isfl_mkt,
            "eqgb_nasdaq_mkt_pct": eqgb_mkt,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Annual breakdown: 3-tier vs 4-tier")
    parser.add_argument("--synthetic-data-dir", default="data_synthetic/hourly")
    parser.add_argument("--vxn-file", default="data_synthetic/hourly/VXN_EXTENDED_1999_2026.csv")
    parser.add_argument("--start-date", default="1999-09-01")
    parser.add_argument("--end-date", default="2026-09-15")
    parser.add_argument("--out-dir", default="data/allocation_backtest")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    data_dir = Path(args.synthetic_data_dir)

    _log.info(f"Loading data ({args.start_date} to {args.end_date})...")
    spy_df = load_synthetic_daily(data_dir / "SPY.csv", args.start_date, args.end_date)
    isfl_df = load_synthetic_daily(data_dir / "ISF.L.csv", args.start_date, args.end_date)
    csh2_df = load_synthetic_daily(data_dir / "CSH2.L.csv", args.start_date, args.end_date)
    eqgb_df = load_synthetic_daily(data_dir / "EQGB_COMPLETE.csv", args.start_date, args.end_date)
    vix_df = load_synthetic_daily(data_dir / "VIX.csv", args.start_date, args.end_date)
    vxn_df = load_synthetic_daily(Path(args.vxn_file), args.start_date, args.end_date)

    # Normalize indices to date (not datetime)
    for df in [spy_df, isfl_df, csh2_df, eqgb_df, vix_df, vxn_df]:
        df.index = pd.to_datetime(df.index.date)

    dates = (
        spy_df.index.intersection(isfl_df.index)
        .intersection(csh2_df.index)
        .intersection(eqgb_df.index)
        .intersection(vix_df.index)
        .intersection(vxn_df.index)
    )

    # Subset to common dates
    spy_df = spy_df.loc[dates]
    isfl_df = isfl_df.loc[dates]
    csh2_df = csh2_df.loc[dates]
    eqgb_df = eqgb_df.loc[dates]
    vix_df = vix_df.loc[dates]
    vxn_df = vxn_df.loc[dates]

    _log.info(f"Common dates: {len(dates)} ({dates.min()} to {dates.max()})")

    # Run 3-tier
    _log.info("Running 3-tier baseline...")
    allocator_3tier = MultiTierAllocator(vix_tier1=15.0, vix_tier2=17.5)
    result_3tier = allocator_3tier.backtest(
        spy_df=spy_df,
        isfl_df=isfl_df,
        shv_df=csh2_df,
        vix_df=vix_df,
        initial_cash=100_000.0,
    )

    # Run 4-tier
    _log.info("Running 4-tier (VXN<18)...")
    allocator_4tier = MultiTierAllocator4Tier(vxn_threshold=18.0, vix_tier1=15.0, vix_tier2=17.5)
    result_4tier = allocator_4tier.backtest(
        nasdaq_df=eqgb_df,
        spy_df=spy_df,
        isfl_df=isfl_df,
        csh2_df=csh2_df,
        vxn_df=vxn_df,
        vix_df=vix_df,
        initial_cash=100_000.0,
    )

    # Annual breakdowns
    df_3tier = annual_breakdown(result_3tier, spy_df, isfl_df, eqgb_df, 100_000.0)
    df_4tier = annual_breakdown(result_4tier, spy_df, isfl_df, eqgb_df, 100_000.0)

    # Merge on year
    df_merged = df_3tier.merge(
        df_4tier,
        on="year",
        suffixes=("_3tier", "_4tier"),
        how="outer"
    ).sort_values("year")

    # Output CSV with proper scaling (decimals, not %)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "4tier_vs_3tier_annual.csv"

    df_merged.to_csv(out_file, index=False)
    _log.info(f"Saved: {out_file}")

    # Display summary
    print("\n" + "=" * 160)
    print("ANNUAL BREAKDOWN: 3-TIER vs 4-TIER (VXN<18)")
    print("=" * 160)
    print(f"\nNote: % columns are scaled as decimals (0.05 = 5% when formatted as % in Excel)\n")

    # Columns: year, combined_return_3tier, combined_pnl_3tier, SPY_pnl_3tier, ISF_pnl_3tier, CSH2_pnl_3tier,
    #          combined_return_4tier, combined_pnl_4tier, Nasdaq_pnl_4tier, SPY_pnl_4tier, ISF_pnl_4tier, CSH2_pnl_4tier,
    #          spy_bh, isfl_bh

    print(df_merged.to_string(index=False))
    print("\n" + "=" * 160)
    print(f"All % columns stored as decimals. Open in Excel and format as % to see percentages.")
    print(f"\n3-tier final value: ${(100000 * (1 + df_3tier['combined_return'].sum())):,.0f}")
    print(f"4-tier final value: ${(100000 * (1 + df_4tier['combined_return'].sum())):,.0f}")


if __name__ == "__main__":
    main()
