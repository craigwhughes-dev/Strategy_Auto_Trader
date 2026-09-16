"""Rebuild CSH2 daily from BoE rates (1999-2026) + real IBKR (2024+).

Similar to build_csh2_historical.py but using extended BoE file.
Output: data/cache/csh2_daily_returns_extended.csv
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_boe_rate_history(csv_path: str) -> pd.Series:
    """Parse BoE Bank Rate CSV, return daily interpolated rates (%)."""
    df = pd.read_csv(csv_path)
    df['Date Changed'] = pd.to_datetime(df['Date Changed'], format='%d %b %y')
    df = df.sort_values('Date Changed').reset_index(drop=True)

    # Extend backwards to 1999-01-01 (use first known rate)
    first_date = df['Date Changed'].iloc[0]
    first_rate = df['Rate'].iloc[0]
    start = pd.Timestamp('1999-01-01')
    if first_date > start:
        df = pd.concat([
            pd.DataFrame({'Date Changed': [start], 'Rate': [first_rate]}),
            df
        ], ignore_index=True)

    # Create daily series: hold each rate until next change
    all_dates = pd.date_range(start='1999-01-01', end=pd.Timestamp.now(), freq='D')
    daily_rates = []
    rate_idx = 0

    for date in all_dates:
        # Find applicable rate (last rate change on or before this date)
        while rate_idx + 1 < len(df) and df['Date Changed'].iloc[rate_idx + 1] <= date:
            rate_idx += 1
        daily_rates.append(df['Rate'].iloc[rate_idx])

    return pd.Series(daily_rates, index=all_dates, name='boe_rate')


def rate_to_daily_return(annual_rate_pct: float) -> float:
    """Convert annual rate (%) to daily return (decimal)."""
    annual_rate_decimal = annual_rate_pct / 100.0
    return (1 + annual_rate_decimal) ** (1.0 / 365.0) - 1


def build_csh2_from_boe(boe_rates: pd.Series, start_date='1999-01-01') -> pd.DataFrame:
    """Build CSH2 NAV from BoE rates."""
    rates = boe_rates.loc[start_date:]

    # Assume starting value of 100
    price = 100.0
    prices = [price]

    for rate in rates.iloc[1:]:
        daily_ret = rate_to_daily_return(rate)
        price *= (1 + daily_ret)
        prices.append(price)

    return pd.DataFrame({
        'close': prices,
        'daily_return': pd.Series(prices).pct_change().fillna(0).values,
    }, index=rates.index)


def main():
    boe_csv = Path.home() / "Downloads" / "Bank Rate history and data Bank of England Database.csv"
    output_csv = Path("data/cache/csh2_daily_returns_extended.csv")

    print(f"Loading BoE rates from {boe_csv}...")
    boe_rates = parse_boe_rate_history(boe_csv)
    print(f"BoE rates: {boe_rates.index.min()} to {boe_rates.index.max()} ({len(boe_rates)} days)")

    print(f"Building CSH2 daily from 1999-01-01...")
    df_csh2 = build_csh2_from_boe(boe_rates, start_date='1999-01-01')

    print(f"CSH2 daily: {len(df_csh2)} rows, {df_csh2.index.min()} to {df_csh2.index.max()}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_csh2.to_csv(output_csv)
    print(f"Wrote: {output_csv}")


if __name__ == "__main__":
    main()
