#!/usr/bin/env python3
"""Multi-tier allocation backtest on synthetic daily data (2003-2026)."""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

from Strategy_Auto_Trader.allocation.multi_tier_allocator import MultiTierAllocator
from Strategy_Auto_Trader.broker.ibkr_data import IBKRDataClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYNTHETIC_HOURLY_DIR = Path("data_synthetic/hourly")

def load_synthetic_daily(ticker: str) -> pd.DataFrame | None:
    """Load synthetic hourly, resample to daily OHLC."""
    hourly_path = SYNTHETIC_HOURLY_DIR / f"{ticker}.csv"
    if not hourly_path.exists():
        logger.error(f"Synthetic hourly file not found: {hourly_path}")
        return None

    df = pd.read_csv(hourly_path, index_col=0, parse_dates=True)
    if df.empty:
        return None

    # Resample hourly to daily OHLC
    daily = pd.DataFrame({
        'Open': df['Open'].resample('D').first(),
        'High': df['High'].resample('D').max(),
        'Low': df['Low'].resample('D').min(),
        'Close': df['Close'].resample('D').last(),
        'Volume': df['Volume'].resample('D').sum(),
    })

    return daily.dropna()

def main():
    logger.info("Loading synthetic daily data (SPY, ISF.L, SHV)...")
    spy_df = load_synthetic_daily("SPY")
    isfl_df = load_synthetic_daily("ISF.L")
    shv_df = load_synthetic_daily("SHV")

    if spy_df is None or isfl_df is None or shv_df is None:
        logger.error("Failed to load synthetic data for all tickers")
        return

    logger.info(f"SPY: {len(spy_df)} days ({spy_df.index.min().date()} to {spy_df.index.max().date()})")
    logger.info(f"ISF.L: {len(isfl_df)} days ({isfl_df.index.min().date()} to {isfl_df.index.max().date()})")
    logger.info(f"SHV: {len(shv_df)} days ({shv_df.index.min().date()} to {shv_df.index.max().date()})")

    # Fetch real VIX data
    logger.info("Fetching real VIX data...")
    client = IBKRDataClient()
    vix_df = client.fetch_index_daily("VIX", "CBOE", currency="USD", historical_only=True)
    if vix_df is None or vix_df.empty:
        logger.error("Failed to fetch VIX data")
        return

    # Run multi-tier allocation backtest with validated thresholds
    logger.info("Running multi-tier allocation backtest (VIX 15.0 / 17.5)...")
    allocator = MultiTierAllocator(vix_tier1=15.0, vix_tier2=17.5)

    result = allocator.backtest(
        spy_df=spy_df,
        isfl_df=isfl_df,
        shv_df=shv_df,
        vix_df=vix_df,
        initial_cash=100_000.0,
    )

    summary = result["summary"]

    print("\n" + "="*80)
    print("MULTI-TIER ALLOCATION SYNTHETIC BACKTEST (2003-2026)")
    print("="*80)
    print(f"Thresholds: VIX tier1=15.0, tier2=17.5")
    print(f"Initial capital: £100,000")
    print()
    print(f"Sharpe:           {summary['sharpe']:.2f}")
    print(f"Sortino:          {summary['sortino']:.2f}")
    print(f"Total Return:     {summary['total_return_pct']:+.2f}%")
    print(f"Max Drawdown:     {summary['max_drawdown_pct']:.2f}%")
    print(f"Final Value:      £{100_000 * (1 + summary['total_return_pct']/100):,.0f}")
    print()
    print(f"Tier 1 (SPY):     {summary['pct_tier1']:.1f}% of days ({summary['days_tier1_spy']} days)")
    print(f"Tier 2 (ISF.L):   {summary['pct_tier2']:.1f}% of days ({summary['days_tier2_isfl']} days)")
    print(f"Tier 3 (SHV):     {summary['pct_tier3']:.1f}% of days ({summary['days_tier3_shv']} days)")
    print("="*80)

    # Save results
    output_path = Path("data_synthetic/journals/multi_tier_synthetic_summary.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("MULTI-TIER ALLOCATION SYNTHETIC BACKTEST (2003-2026)\n")
        f.write("="*80 + "\n")
        f.write(f"Sharpe:           {summary['sharpe']:.2f}\n")
        f.write(f"Sortino:          {summary['sortino']:.2f}\n")
        f.write(f"Total Return:     {summary['total_return_pct']:+.2f}%\n")
        f.write(f"Max Drawdown:     {summary['max_drawdown_pct']:.2f}%\n")
        f.write(f"Final Value:      £{100_000 * (1 + summary['total_return_pct']/100):,.0f}\n")
        f.write(f"\nTier breakdown:\n")
        f.write(f"Tier 1 (SPY):     {summary['pct_tier1']:.1f}%\n")
        f.write(f"Tier 2 (ISF.L):   {summary['pct_tier2']:.1f}%\n")
        f.write(f"Tier 3 (SHV):     {summary['pct_tier3']:.1f}%\n")

    logger.info(f"Summary saved to {output_path}")

if __name__ == "__main__":
    main()
