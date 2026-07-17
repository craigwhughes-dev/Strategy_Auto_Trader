#!/usr/bin/env python3
"""Summarize conservative strategy trades from all journals."""

import pandas as pd
import numpy as np
from pathlib import Path

journal_dir = Path("data/journals/full_scan/conservative")
journal_files = list(journal_dir.glob("*.csv"))

print(f"Loading {len(journal_files)} conservative journal files...")

# Combine all trades
all_trades = []
for jf in journal_files:
    try:
        df = pd.read_csv(jf)
        all_trades.append(df)
    except:
        pass

trades = pd.concat(all_trades, ignore_index=True)
print(f"Total trades: {len(trades)}")

# Basic stats
print("\n" + "=" * 100)
print("CONSERVATIVE STRATEGY - TRADE SUMMARY")
print("=" * 100)

print(f"\nTotal trades: {len(trades):,}")
print(f"Tickers traded: {trades['ticker'].nunique()}")
print(f"Period: {trades['date_opened'].min()} to {trades['date_closed'].max()}")

# Win/loss breakdown
winning = (trades['return_pct'] > 0).sum()
losing = (trades['return_pct'] < 0).sum()
breakeven = (trades['return_pct'] == 0).sum()

print(f"\nWins: {winning:,} ({winning/len(trades)*100:.1f}%)")
print(f"Losses: {losing:,} ({losing/len(trades)*100:.1f}%)")
print(f"Breakeven: {breakeven:,} ({breakeven/len(trades)*100:.1f}%)")

# Return stats
avg_return = trades['return_pct'].mean()
med_return = trades['return_pct'].median()
std_return = trades['return_pct'].std()
min_return = trades['return_pct'].min()
max_return = trades['return_pct'].max()

print(f"\nReturn % (per trade):")
print(f"  Mean: {avg_return:+.3f}%")
print(f"  Median: {med_return:+.3f}%")
print(f"  Std dev: {std_return:.3f}%")
print(f"  Min: {min_return:+.3f}%")
print(f"  Max: {max_return:+.3f}%")

# Hold time stats
if 'days_held' in trades.columns:
    avg_hold = trades['days_held'].mean()
    med_hold = trades['days_held'].median()
    print(f"\nHold time (days):")
    print(f"  Mean: {avg_hold:.1f}")
    print(f"  Median: {med_hold:.1f}")

# Exit reasons
if 'exit_reason' in trades.columns:
    print(f"\nExit reasons:")
    exit_counts = trades['exit_reason'].value_counts()
    for reason, count in exit_counts.head(5).items():
        print(f"  {reason}: {count:,} ({count/len(trades)*100:.1f}%)")

# Entry signals
if 'entry_signal' in trades.columns:
    print(f"\nEntry signals:")
    entry_counts = trades['entry_signal'].value_counts()
    for signal, count in entry_counts.head(5).items():
        print(f"  {signal}: {count:,} ({count/len(trades)*100:.1f}%)")

# Top/worst tickers
print(f"\nBest performing tickers (avg return %):")
ticker_perf = trades.groupby('ticker')['return_pct'].agg(['mean', 'count']).sort_values('mean', ascending=False)
for ticker, row in ticker_perf.head(5).iterrows():
    print(f"  {ticker}: {row['mean']:+.2f}% ({int(row['count'])} trades)")

print(f"\nWorst performing tickers (avg return %):")
for ticker, row in ticker_perf.tail(5).iterrows():
    print(f"  {ticker}: {row['mean']:+.2f}% ({int(row['count'])} trades)")

# P&L summary
total_pnl = trades['pnl_usd'].sum()
avg_pnl = trades['pnl_usd'].mean()
print(f"\nP&L (USD):")
print(f"  Total: ${total_pnl:,.2f}")
print(f"  Avg per trade: ${avg_pnl:,.2f}")

print("\n" + "=" * 100)

# Save combined trades to file
output_file = Path("reports/conservative_all_trades.csv")
trades.to_csv(output_file, index=False)
print(f"\nFull trade data saved to: {output_file}")
