#!/usr/bin/env python3
"""Multi-tier annual breakdown with benchmarks + asset PnL contribution."""

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

    daily = pd.DataFrame({
        'Open': df['Open'].resample('D').first(),
        'High': df['High'].resample('D').max(),
        'Low': df['Low'].resample('D').min(),
        'Close': df['Close'].resample('D').last(),
        'Volume': df['Volume'].resample('D').sum(),
    })

    return daily.dropna()

def get_annual_returns(df: pd.DataFrame) -> dict:
    """Compute annual returns from daily close prices."""
    if df.empty or len(df) < 2:
        return {}

    df = df.copy()
    df['year'] = df.index.year

    annual_ret = {}
    for year in df['year'].unique():
        year_data = df[df['year'] == year]
        if len(year_data) < 2:
            continue
        start_price = year_data['Close'].iloc[0]
        end_price = year_data['Close'].iloc[-1]
        annual_ret[year] = (end_price - start_price) / start_price * 100

    return annual_ret

def compute_asset_pnl(daily_nav_df: pd.DataFrame,
                     spy_df: pd.DataFrame,
                     isfl_df: pd.DataFrame,
                     shv_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate PnL contribution from each asset."""
    daily_nav = daily_nav_df.copy()
    daily_nav['date'] = pd.to_datetime(daily_nav['date'])
    daily_nav['year'] = daily_nav['date'].dt.year

    spy_rets = get_annual_returns(spy_df)
    isfl_rets = get_annual_returns(isfl_df)
    shv_rets = get_annual_returns(shv_df)

    results = []
    for year in sorted(daily_nav['year'].unique()):
        year_data = daily_nav[daily_nav['year'] == year]
        if len(year_data) < 2:
            continue

        nav_start = year_data['nav'].iloc[0]
        nav_end = year_data['nav'].iloc[-1]
        nav_ret_pct = (nav_end - nav_start) / nav_start * 100

        # Tier allocation %
        tier_pcts = year_data.groupby('tier').size() / len(year_data) * 100
        spy_pct = tier_pcts.get(1, 0)
        isfl_pct = tier_pcts.get(2, 0)
        shv_pct = tier_pcts.get(3, 0)

        # Benchmark returns
        spy_annual = spy_rets.get(year, np.nan)
        isfl_annual = isfl_rets.get(year, np.nan)
        shv_annual = shv_rets.get(year, np.nan)

        # Estimate PnL contribution (allocation % × asset return)
        # This is a simplification; real attribution is more complex
        spy_contrib = spy_pct / 100 * spy_annual if not np.isnan(spy_annual) else 0
        isfl_contrib = isfl_pct / 100 * isfl_annual if not np.isnan(isfl_annual) else 0
        shv_contrib = shv_pct / 100 * shv_annual if not np.isnan(shv_annual) else 0

        # Average cash rate (CSH2 or SHV yield proxy)
        cash_rate = shv_annual * 0.2 if not np.isnan(shv_annual) else 0  # Rough proxy

        results.append({
            'Year': year,
            'Days': len(year_data),
            'Multi-Tier Return %': nav_ret_pct,
            'SPY Annual %': spy_annual,
            'ISF.L Annual %': isfl_annual,
            'SHV Annual %': shv_annual,
            'SPY Alloc %': spy_pct,
            'ISF Alloc %': isfl_pct,
            'SHV Alloc %': shv_pct,
            'SPY PnL Contrib (bps)': spy_contrib * 100,
            'ISF PnL Contrib (bps)': isfl_contrib * 100,
            'SHV PnL Contrib (bps)': shv_contrib * 100,
            'Cash Rate (bps)': cash_rate * 100,
        })

    return pd.DataFrame(results)

def main():
    logger.info("Loading synthetic data...")
    spy_df = load_synthetic_daily("SPY")
    isfl_df = load_synthetic_daily("ISF.L")
    shv_df = load_synthetic_daily("SHV")

    if spy_df is None or isfl_df is None or shv_df is None:
        logger.error("Failed to load synthetic data")
        return

    logger.info("Loading VIX...")
    # Use synthetic VIX when running synthetic assets (1990-2026 coverage)
    vix_path = Path("data_synthetic/daily/VIX_1990_2026.csv")
    if vix_path.exists():
        logger.info("Using synthetic VIX 1990-2026")
        vix_df = pd.read_csv(vix_path, index_col=0, parse_dates=True)
    else:
        logger.info("Fetching real VIX from IBKR (2005-2026 only)")
        client = IBKRDataClient()
        vix_df = client.fetch_index_daily("VIX", "CBOE", currency="USD", historical_only=True)

    logger.info("Running backtest...")
    allocator = MultiTierAllocator(vix_tier1=15.0, vix_tier2=17.5)
    result = allocator.backtest(spy_df, isfl_df, shv_df, vix_df, initial_cash=100_000.0)

    daily_nav = result['daily_nav']

    logger.info("Computing detailed annual breakdown...")
    detail_df = compute_asset_pnl(daily_nav, spy_df, isfl_df, shv_df)

    print("\n" + "="*180)
    print("MULTI-TIER ALLOCATION: DETAILED ANNUAL BREAKDOWN (2007-2026)")
    print("="*180)

    # Format for display
    display_cols = [
        'Year', 'Days',
        'Multi-Tier Return %', 'SPY Annual %', 'ISF.L Annual %', 'SHV Annual %',
        'SPY Alloc %', 'ISF Alloc %', 'SHV Alloc %',
        'SPY PnL Contrib (bps)', 'ISF PnL Contrib (bps)', 'SHV PnL Contrib (bps)',
        'Cash Rate (bps)'
    ]

    # Format: percentages (%) vs ratios (decimals)
    for col in detail_df.columns:
        if 'Alloc %' in col:
            detail_df[col] = detail_df[col].apply(lambda x: f"{x:.1f}%")
        elif any(substr in col for substr in ['Return %', 'Annual %']):
            detail_df[col] = detail_df[col].apply(lambda x: f"{x:+.2f}%")
        elif any(substr in col for substr in ['Contrib %', 'Cash Rate %']):
            detail_df[col] = detail_df[col].apply(lambda x: f"{x:+.3f}")  # ratio, not %

    print(detail_df[display_cols].to_string(index=False))
    print("="*180)

    # Save
    output_path = Path("data_synthetic/journals/multi_tier_annual_detail.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
