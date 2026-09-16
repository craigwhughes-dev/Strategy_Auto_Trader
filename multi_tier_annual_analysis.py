#!/usr/bin/env python3
"""Annual breakdown of multi-tier allocation synthetic backtest."""

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

def analyze_annual(daily_nav_df: pd.DataFrame, initial_cash: float = 100_000) -> pd.DataFrame:
    """Break down performance by year."""
    daily_nav = daily_nav_df.copy()
    daily_nav['date'] = pd.to_datetime(daily_nav['date'])
    daily_nav['year'] = daily_nav['date'].dt.year

    results = []
    for year in sorted(daily_nav['year'].unique()):
        year_data = daily_nav[daily_nav['year'] == year]
        if len(year_data) < 2:
            continue

        year_navs = year_data['nav'].values
        year_rets = np.diff(year_navs) / year_navs[:-1]

        # Annual return
        annual_ret_pct = (year_navs[-1] - year_navs[0]) / year_navs[0] * 100

        # Sharpe (annualized)
        daily_mean = np.mean(year_rets)
        daily_std = np.std(year_rets, ddof=1) if len(year_rets) > 1 else 0
        sharpe = (daily_mean * 252 / daily_std) if daily_std > 0 else 0

        # Sortino (downside vol only)
        down_rets = year_rets[year_rets < 0]
        down_std = np.std(down_rets, ddof=1) if len(down_rets) > 1 else 0
        sortino = (daily_mean * 252 / down_std) if down_std > 0 else 0

        # Max drawdown
        running_max = np.maximum.accumulate(year_navs)
        dd = (year_navs - running_max) / running_max
        max_dd = np.min(dd) if len(dd) > 0 else 0

        # Asset allocation %
        tier_pcts = year_data.groupby('tier').size() / len(year_data) * 100

        results.append({
            'Year': year,
            'Days': len(year_data),
            'Return %': annual_ret_pct,
            'Sharpe': sharpe,
            'Sortino': sortino,
            'Max DD %': max_dd * 100,
            'End NAV': year_navs[-1],
            'Tier1(SPY)%': tier_pcts.get(1, 0),
            'Tier2(ISF)%': tier_pcts.get(2, 0),
            'Tier3(SHV)%': tier_pcts.get(3, 0),
        })

    return pd.DataFrame(results)

def main():
    logger.info("Loading synthetic daily data...")
    spy_df = load_synthetic_daily("SPY")
    isfl_df = load_synthetic_daily("ISF.L")
    shv_df = load_synthetic_daily("SHV")

    if spy_df is None or isfl_df is None or shv_df is None:
        logger.error("Failed to load synthetic data")
        return

    logger.info("Fetching VIX...")
    client = IBKRDataClient()
    vix_df = client.fetch_index_daily("VIX", "CBOE", currency="USD", historical_only=True)

    logger.info("Running backtest...")
    allocator = MultiTierAllocator(vix_tier1=15.0, vix_tier2=17.5)
    result = allocator.backtest(spy_df, isfl_df, shv_df, vix_df, initial_cash=100_000.0)

    daily_nav = result['daily_nav']

    logger.info("Computing annual breakdown...")
    annual_df = analyze_annual(daily_nav, initial_cash=100_000)

    print("\n" + "="*120)
    print("MULTI-TIER ALLOCATION: ANNUAL BREAKDOWN (2003-2026)")
    print("="*120)
    print(annual_df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print("="*120)

    # Summary totals
    total_ret = (daily_nav['nav'].iloc[-1] - 100_000) / 100_000 * 100
    print(f"\nCumulative Return: {total_ret:+.2f}%")
    print(f"Final NAV: £{daily_nav['nav'].iloc[-1]:,.0f}")

    # Save to file
    output_path = Path("data_synthetic/journals/multi_tier_annual_breakdown.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annual_df.to_csv(output_path, index=False)
    logger.info(f"Annual breakdown saved to {output_path}")

if __name__ == "__main__":
    main()
