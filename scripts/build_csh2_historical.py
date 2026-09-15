"""Build CSH2.L historical daily returns: BoE Bank Rate 2002-2024 + IBKR real 2024-09 onward.

BoE Bank Rate is a proxy for money-market fund yields (CSH2.L tracks overnight returns).
Fills backtest window 2002-2026 by patching BoE-constructed returns with real IBKR data.

Output: data/cache/csh2_daily_returns.csv (date, close, daily_return)
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Strategy_Auto_Trader.quant_hmm.quant_engine import fetch_hourly


def parse_boe_rate_history(csv_path: str) -> pd.Series:
    """Parse BoE Bank Rate CSV, return daily interpolated rates (%).

    CSV format: "Date Changed","Rate"
    Returns: Series indexed by date, values = annual rate (%).
    """
    df = pd.read_csv(csv_path)
    df['Date Changed'] = pd.to_datetime(df['Date Changed'], format='%d %b %y')
    df = df.sort_values('Date Changed').reset_index(drop=True)

    # Extend backwards to 2002-01-01 (use first known rate)
    first_date = df['Date Changed'].iloc[0]
    first_rate = df['Rate'].iloc[0]
    start = pd.Timestamp('2002-01-01')
    if first_date > start:
        df = pd.concat([
            pd.DataFrame({'Date Changed': [start], 'Rate': [first_rate]}),
            df
        ], ignore_index=True)

    # Create daily series: hold each rate until next change
    all_dates = pd.date_range(start='2002-01-01', end=pd.Timestamp.now(), freq='D')
    daily_rates = []
    rate_idx = 0

    for date in all_dates:
        # Find applicable rate (last rate change on or before this date)
        while rate_idx + 1 < len(df) and df['Date Changed'].iloc[rate_idx + 1] <= date:
            rate_idx += 1
        daily_rates.append(df['Rate'].iloc[rate_idx])

    return pd.Series(daily_rates, index=all_dates, name='boe_rate')


def rate_to_daily_return(annual_rate_pct: float) -> float:
    """Convert annual rate (%) to daily return (decimal).

    Assumes compounding: (1 + annual_rate/100) ^ (1/365) - 1
    """
    annual_rate_decimal = annual_rate_pct / 100.0
    return (1.0 + annual_rate_decimal) ** (1.0 / 365.0) - 1.0


def build_csh2_returns(boe_csv: str, output_dir: str = "data/cache") -> pd.DataFrame:
    """Build CSH2.L daily returns: BoE 2002-2024 + real IBKR 2024-09-onward.

    Steps:
    1. Parse BoE Bank Rate CSV → daily rates
    2. Convert to daily returns
    3. Fetch real IBKR CSH2.L hourly → resample to daily close
    4. Patch: use BoE-constructed up to IBKR start, then real IBKR onward
    5. Save to output_dir/csh2_daily_returns.csv

    Returns: DataFrame with columns [date, close, daily_return]
    """
    print("1. Parsing BoE Bank Rate history...")
    boe_rates = parse_boe_rate_history(boe_csv)
    print(f"   BoE rates: {boe_rates.index[0].date()} to {boe_rates.index[-1].date()}")

    # Convert to daily returns (construct prices from rates)
    print("2. Converting BoE rates to daily returns...")
    boe_daily_ret = boe_rates.apply(rate_to_daily_return)
    # Build price series from returns (starting at 100)
    boe_price = 100.0 * (1.0 + boe_daily_ret).cumprod()
    boe_df = pd.DataFrame({
        'close': boe_price,
        'daily_return': boe_daily_ret,
    }, index=boe_rates.index)
    print(f"   BoE-constructed: {len(boe_df)} days, price range {boe_df['close'].min():.2f}-{boe_df['close'].max():.2f}")

    # Fetch real IBKR CSH2.L
    print("3. Fetching IBKR CSH2.L hourly...")
    ibkr_hourly = fetch_hourly("CSH2.L", source="ibkr", historical_only=True)
    if ibkr_hourly is None or ibkr_hourly.empty:
        print("   WARNING: CSH2.L IBKR cache empty, using BoE-only")
        result = boe_df.copy()
    else:
        print(f"   IBKR: {len(ibkr_hourly)} bars, {ibkr_hourly.index[0]} to {ibkr_hourly.index[-1]}")

        # Resample to daily close
        ibkr_daily = ibkr_hourly['Close'].resample('D').last().dropna()
        ibkr_daily_ret = ibkr_daily.pct_change()
        ibkr_df = pd.DataFrame({
            'close': ibkr_daily,
            'daily_return': ibkr_daily_ret,
        })
        ibkr_df.index = ibkr_df.index.tz_localize(None)
        print(f"   IBKR daily: {len(ibkr_df)} days, {ibkr_df.index[0].date()} to {ibkr_df.index[-1].date()}")

        # Patch: BoE up to IBKR start, then real IBKR
        ibkr_start = ibkr_df.index[0]
        result = pd.concat([
            boe_df.loc[:'2024-09-15'],  # BoE up to gap
            ibkr_df,  # Real IBKR onward
        ]).sort_index().drop_duplicates(keep='last')
        print(f"   Patched: {len(result)} days, {result.index[0].date()} to {result.index[-1].date()}")

    # Save
    output_path = Path(output_dir) / "csh2_daily_returns.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path)
    print(f"4. Saved: {output_path}")

    return result


if __name__ == "__main__":
    boe_csv = "Bank Rate history and data Bank of England Database.csv"
    df = build_csh2_returns(boe_csv)
    print(f"\nFinal: {len(df)} rows")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Daily return stats: mean={df['daily_return'].mean()*100:.4f}%, std={df['daily_return'].std()*100:.4f}%")
