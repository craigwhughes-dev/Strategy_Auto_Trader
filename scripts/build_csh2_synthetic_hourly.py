"""Bridge CSH2 daily (1999-2026) to hourly OHLCV via Brownian bridge.

Input: data/cache/csh2_daily_returns_extended.csv (BoE 1999-2026)
Output: data_synthetic/hourly/CSH2.L.csv (hourly, 7 bars/day, 1999-2026)

Usage:
    python scripts/build_csh2_synthetic_hourly.py
"""

import sys
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Strategy_Auto_Trader.synthetic_backtest_data import bridge
from Strategy_Auto_Trader.core.atomic_io import atomic_write_csv

_BARS_PER_DAY = 7
_VOL_WINDOW = 21


def build_csh2_synthetic_hourly():
    """Load CSH2 daily (1999-2026), bridge to hourly."""
    daily_csv = Path("data/cache/csh2_daily_returns_extended.csv")
    output_dir = Path("data_synthetic/hourly")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "CSH2.L.csv"

    # Load CSH2 extended (1999-2026)
    df_daily = pd.read_csv(daily_csv, index_col=0, parse_dates=True)
    df_daily.index = pd.to_datetime(df_daily.index)

    print(f"CSH2 daily (extended): {len(df_daily)} rows, {df_daily.index.min()} to {df_daily.index.max()}")

    # Compute rolling volatility (annualized daily vol)
    daily_prices = df_daily["close"].values
    daily_returns = np.diff(daily_prices) / daily_prices[:-1]
    rolling_vol = np.std(daily_returns[-_VOL_WINDOW:]) * np.sqrt(252)
    if rolling_vol < 0.001:
        rolling_vol = 0.001  # floor for money market

    print(f"CSH2 rolling daily vol: {rolling_vol:.6f} (annualized)")

    # Bridge each day to hourly
    rng = np.random.default_rng(42)
    hourly_rows = []
    for i in range(len(df_daily) - 1):
        date = df_daily.index[i]
        price_open = df_daily["close"].iloc[i]
        price_close = df_daily["close"].iloc[i + 1]

        # Bridge this day
        day_path = bridge.generate_bridge_path(
            prev_close=price_open,
            next_close=price_close,
            sigma=rolling_vol,
            n_steps=_BARS_PER_DAY,
            rng=rng,
            sigma_scale=1.0 / np.sqrt(_BARS_PER_DAY),  # correct Brownian scaling
        )

        # Generate OHLCV for each bar
        for j in range(_BARS_PER_DAY):
            bar_time = date + timedelta(hours=j + 1)  # bars at 1h, 2h, ..., 7h
            bar_open = day_path[j]
            bar_close = day_path[j + 1] if j + 1 < len(day_path) else price_close
            bar_high = max(bar_open, bar_close)
            bar_low = min(bar_open, bar_close)
            bar_volume = 1.0  # money market, volume is not meaningful

            hourly_rows.append({
                "Datetime": bar_time,
                "Open": bar_open,
                "High": bar_high,
                "Low": bar_low,
                "Close": bar_close,
                "Volume": bar_volume,
            })

        if (i + 1) % 500 == 0:
            print(f"  bridged {i + 1}/{len(df_daily) - 1} days")

    df_hourly = pd.DataFrame(hourly_rows)
    df_hourly["Datetime"] = pd.to_datetime(df_hourly["Datetime"], utc=True)
    df_hourly.set_index("Datetime", inplace=True)

    print(f"CSH2 hourly: {len(df_hourly)} rows, {df_hourly.index.min()} to {df_hourly.index.max()}")

    # Write atomically
    atomic_write_csv(output_csv, df_hourly)
    print(f"Wrote: {output_csv}")


if __name__ == "__main__":
    build_csh2_synthetic_hourly()
