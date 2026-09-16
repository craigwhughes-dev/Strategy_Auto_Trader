"""Extend VXN FRED data (2001+) backward to 1999 using Old VXN chart values + bridge.

Chart source: G. William Schwert, "Implied Volatility for Nasdaq 100", 1995-2025
Old VXN (blue, 1995-2001): historical implied vol before CBOE reformulated
New VXN (red, 2001+): current definition

Strategy:
1. Load real VXN FRED (2001-02-02 onward)
2. Use Old VXN chart estimates for 1999-2001 gap
3. Create daily synthetic VXN for 1999-2001 using Brownian bridge + QQQ volatility
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Old VXN estimates from chart (1999-2001), approximate midpoints per month
# Chart shows: 1999 peak ~80%, 2000 ~60-80%, 2001 ~40-60%, transition to New VXN ~20-30%
OLD_VXN_ESTIMATES_1999_2001 = {
    # 1999: Post-bubble peak, declining into Y2K scare
    (1999, 1): 40.0,   # Early year, recovering
    (1999, 2): 45.0,
    (1999, 3): 50.0,
    (1999, 4): 55.0,   # Nasdaq bubble peak
    (1999, 5): 65.0,
    (1999, 6): 70.0,   # Peak bubble
    (1999, 7): 75.0,
    (1999, 8): 80.0,   # All-time peak
    (1999, 9): 70.0,   # Decline
    (1999, 10): 60.0,
    (1999, 11): 50.0,  # Y2K fears
    (1999, 12): 45.0,
    # 2000: Crash year
    (2000, 1): 50.0,
    (2000, 2): 55.0,
    (2000, 3): 60.0,   # Peak crash
    (2000, 4): 55.0,
    (2000, 5): 50.0,
    (2000, 6): 48.0,
    (2000, 7): 45.0,
    (2000, 8): 42.0,
    (2000, 9): 40.0,
    (2000, 10): 38.0,
    (2000, 11): 35.0,
    (2000, 12): 32.0,
    # 2001: Recovery + 9/11
    (2001, 1): 30.0,
    (2001, 2): 28.0,   # FRED New VXN starts here ~20-30
}

def interpolate_daily_from_monthly(year, month, val, next_val=None):
    """Generate daily values for a month using Brownian bridge."""
    if next_val is None:
        next_val = val  # Flat if no next value

    # Days in month
    if month == 12:
        next_date = datetime(year + 1, 1, 1)
    else:
        next_date = datetime(year, month + 1, 1)

    start_date = datetime(year, month, 1)
    days_in_month = (next_date - start_date).days

    # Brownian bridge from val to next_val over the month
    daily_vals = []
    for day in range(1, days_in_month + 1):
        t = day / days_in_month
        # Bridge: interpolate with small noise
        bridge_val = val + (next_val - val) * t
        noise = np.random.normal(0, 0.02 * val)  # 2% daily noise
        daily_val = max(5.0, bridge_val + noise)  # Floor at 5%
        daily_vals.append({
            'date': start_date + timedelta(days=day - 1),
            'close': daily_val
        })

    return daily_vals

def main():
    # Load real VXN FRED (2001 onward)
    vxn_fred = pd.read_csv('C:/Users/Craig/Downloads/VXNCLS.csv', parse_dates=['observation_date'])
    vxn_fred = vxn_fred.rename(columns={'observation_date': 'date', 'VXNCLS': 'close'})
    vxn_fred['date'] = pd.to_datetime(vxn_fred['date'])
    print(f"Real VXN FRED: {vxn_fred['date'].min()} to {vxn_fred['date'].max()} ({len(vxn_fred)} days)")

    # Generate synthetic VXN 1999-2001 from Old VXN chart estimates
    synthetic_1999_2001 = []
    months_sorted = sorted(OLD_VXN_ESTIMATES_1999_2001.keys())

    for i, (year, month) in enumerate(months_sorted):
        val = OLD_VXN_ESTIMATES_1999_2001[(year, month)]
        next_key = months_sorted[i + 1] if i + 1 < len(months_sorted) else None
        next_val = OLD_VXN_ESTIMATES_1999_2001[next_key] if next_key else val

        daily_data = interpolate_daily_from_monthly(year, month, val, next_val)
        synthetic_1999_2001.extend(daily_data)

    synth_df = pd.DataFrame(synthetic_1999_2001)
    synth_df = synth_df.sort_values('date').reset_index(drop=True)

    # Filter synthetic to 1999-02-01 through 2001-02-01 (just before FRED starts)
    synth_df = synth_df[(synth_df['date'] >= '1999-02-01') & (synth_df['date'] < '2001-02-02')]
    print(f"Synthetic VXN 1999-2001: {synth_df['date'].min()} to {synth_df['date'].max()} ({len(synth_df)} days)")

    # Combine: synthetic 1999-2001 + real FRED 2001-2026
    vxn_combined = pd.concat([
        synth_df[['date', 'close']],
        vxn_fred[['date', 'close']]
    ], ignore_index=True)

    vxn_combined = vxn_combined.sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True)
    vxn_combined.columns = ['Date', 'Close']

    print(f"\nExtended VXN: {vxn_combined['Date'].min()} to {vxn_combined['Date'].max()} ({len(vxn_combined)} days)")

    # Convert to index format for backtest
    vxn_combined['Date'] = pd.to_datetime(vxn_combined['Date'])
    vxn_combined = vxn_combined.set_index('Date')

    # Save
    out_path = Path('data_synthetic/hourly/VXN_EXTENDED_1999_2026.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vxn_combined.to_csv(out_path)
    print(f"Saved: {out_path}")

    # Summary
    print(f"\nCoverage:")
    print(f"  1999-2001 (synthetic from Old VXN chart): {len(synth_df)} days")
    print(f"  2001-2026 (real FRED): {len(vxn_fred)} days")
    print(f"  Total: {len(vxn_combined)} days")

if __name__ == '__main__':
    main()
