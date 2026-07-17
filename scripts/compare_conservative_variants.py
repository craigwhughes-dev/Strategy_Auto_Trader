#!/usr/bin/env python3
"""Compare Conservative vs Conservative_Optimised using original journal data."""

import pandas as pd
import numpy as np
from pathlib import Path

original_dir = Path("data/journals/full_scan/conservative")

all_trades = []
for csv_file in sorted(original_dir.glob("*.csv")):  # All tickers
    try:
        df = pd.read_csv(csv_file)
        ticker = csv_file.stem
        df['ticker'] = ticker
        all_trades.append(df)
        print(f"Loaded {ticker}: {len(df)} trades")
    except Exception as e:
        print(f"Error {csv_file.name}: {e}")

if all_trades:
    full_df = pd.concat(all_trades, ignore_index=True)

    print("\n" + "=" * 120)
    print("CONSERVATIVE vs CONSERVATIVE_OPTIMISED (SMA200 health filter)")
    print("=" * 120)

    # All conservative trades
    conservative_all = full_df.copy()

    # Only trades where price > SMA200 at entry
    conservative_optimised = full_df[full_df['entry_above_sma200'] == True].copy()

    print(f"\nConservative (all):               {len(conservative_all):,} trades")
    print(f"Conservative_Optimised (>SMA200): {len(conservative_optimised):,} trades")
    print(f"Filter reduction:                 {(1 - len(conservative_optimised)/len(conservative_all))*100:.1f}% fewer trades")

    print("\n" + "=" * 120)
    print(f"{'Metric':<35} {'Conservative':<18} {'Optimised':<18} {'Change':<15}")
    print("-" * 120)

    # Compare metrics
    c_pnl = conservative_all['pnl_usd'].sum()
    o_pnl = conservative_optimised['pnl_usd'].sum()
    print(f"{'Total P&L (USD)':<35} ${c_pnl:>16,.2f} ${o_pnl:>16,.2f} ${o_pnl-c_pnl:>+13,.2f}")

    c_avg = conservative_all['pnl_usd'].mean()
    o_avg = conservative_optimised['pnl_usd'].mean()
    print(f"{'Avg P&L per trade (USD)':<35} ${c_avg:>16,.2f} ${o_avg:>16,.2f} ${o_avg-c_avg:>+13,.2f}")

    c_win = (conservative_all['return_pct'] > 0).sum() / len(conservative_all) * 100
    o_win = (conservative_optimised['return_pct'] > 0).sum() / len(conservative_optimised) * 100
    print(f"{'Win rate (%)':<35} {c_win:>16.1f}% {o_win:>16.1f}% {o_win-c_win:>+13.1f}%")

    c_mean = conservative_all['return_pct'].mean() * 100
    o_mean = conservative_optimised['return_pct'].mean() * 100
    print(f"{'Mean return per trade (%)':<35} {c_mean:>16.3f}% {o_mean:>16.3f}% {o_mean-c_mean:>+13.3f}%")

    c_median = conservative_all['return_pct'].median() * 100
    o_median = conservative_optimised['return_pct'].median() * 100
    print(f"{'Median return per trade (%)':<35} {c_median:>16.3f}% {o_median:>16.3f}% {o_median-c_median:>+13.3f}%")

    c_days = conservative_all['days_held'].mean()
    o_days = conservative_optimised['days_held'].mean()
    print(f"{'Avg hold days':<35} {c_days:>16.1f} {o_days:>16.1f} {o_days-c_days:>+13.1f}")

    print("\n" + "=" * 120)
    print("ANALYSIS")
    print("=" * 120)
    losing_c = (conservative_all['pnl_usd'] < 0).sum()
    losing_o = (conservative_optimised['pnl_usd'] < 0).sum()
    print(f"Losing trades Conservative:           {losing_c:,} ({losing_c/len(conservative_all)*100:.1f}%)")
    print(f"Losing trades Conservative_Optimised: {losing_o:,} ({losing_o/len(conservative_optimised)*100:.1f}%)")
    print(f"Trades eliminated by filter:          {len(conservative_all) - len(conservative_optimised):,}")
    print(f"P&L impact:                           {o_pnl - c_pnl:+,.2f} ({(o_pnl-c_pnl)/c_pnl*100 if c_pnl != 0 else 0:+.1f}%)")
else:
    print("No data loaded")
