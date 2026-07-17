#!/usr/bin/env python3
"""Estimate max concurrent positions for all strategies."""

import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path("reports/full_scan/summary.csv")
df = pd.read_csv(csv_path)
df_ok = df[df['status'] == 'ok'].copy()

# Get actual max concurrent for default (from journals)
default_max_concurrent = 406
default_trades = df_ok[df_ok['strategy'] == 'default']['n_trades_journaled'].sum()
default_hours = df_ok[df_ok['strategy'] == 'default']['n_bars'].sum()

# Back-calculate avg hold time from actual max concurrent
# max_concurrent = (trades * hold_hours) / total_hours
# => hold_hours = (max_concurrent * total_hours) / trades
avg_hold_hours_default = (default_max_concurrent * default_hours) / default_trades if default_trades > 0 else 1

print("=" * 80)
print("MAX CONCURRENT POSITIONS (Estimated)")
print("=" * 80)
print(f"\nDefault strategy calibration:")
print(f"  Total trades: {default_trades:,}")
print(f"  Total bars: {default_hours:,}")
print(f"  Max concurrent (actual): {default_max_concurrent}")
print(f"  Implied avg hold: ~{avg_hold_hours_default:.2f} hours per trade\n")

# Estimate for all strategies
results = []
for strat in sorted(df_ok['strategy'].unique()):
    strat_df = df_ok[df_ok['strategy'] == strat]

    n_trades = strat_df['n_trades_journaled'].sum()
    n_bars = strat_df['n_bars'].sum()

    # Estimate max concurrent: trades × hold_hours / total_hours
    est_max_concurrent = int(round((n_trades * avg_hold_hours_default) / n_bars)) if n_bars > 0 else 0

    results.append((strat, n_trades, est_max_concurrent))

# Sort by max concurrent descending
results = sorted(results, key=lambda x: x[2], reverse=True)

print(f"{'Strategy':<25} {'Total Trades':<15} {'Max Concurrent':<15}")
print("-" * 80)
for strat, trades, max_concurrent in results:
    print(f"{strat:<25} {int(trades):<15,d} {max_concurrent:<15d}")

print("=" * 80)
print("\nNote: Estimates based on trade count + avg hold time (5h). Default strategy")
print("calibrated against actual journal analysis (406 concurrent).")
