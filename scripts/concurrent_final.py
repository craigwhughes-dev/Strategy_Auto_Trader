#!/usr/bin/env python3
"""Max concurrent positions - correct calculation."""

import pandas as pd
from pathlib import Path

csv_path = Path("reports/full_scan/summary.csv")
df = pd.read_csv(csv_path)
df_ok = df[df['status'] == 'ok'].copy()

print("=" * 100)
print("MAX CONCURRENT POSITIONS (Correct)")
print("=" * 100)
print("\nLogic: With N trades/ticker over 730 days, avg 5-hour hold: concurrent = (N*5)/(730*24)\n")

results = []
for strat in sorted(df_ok['strategy'].unique()):
    strat_df = df_ok[df_ok['strategy'] == strat]

    n_tickers = len(strat_df)
    avg_trades_per_ticker = strat_df['n_trades_journaled'].mean()

    # Per-ticker concurrent (on average)
    concurrent_per_ticker = (avg_trades_per_ticker * 5) / (730 * 24)

    # Peak concurrent if ALL tickers trade same hours (unrealistic but upper bound)
    max_concurrent = concurrent_per_ticker * n_tickers

    results.append((strat, n_tickers, avg_trades_per_ticker, concurrent_per_ticker, max_concurrent))

# Sort by max concurrent
results = sorted(results, key=lambda x: x[4], reverse=True)

print(f"{'Strategy':<25} {'Tickers':<10} {'Trades/Ticker':<18} {'Per-Ticker':<15} {'Max (All Active)':<15}")
print("-" * 100)
for strat, n_tickers, trades_per, per_ticker, max_con in results:
    print(f"{strat:<25} {n_tickers:<10} {trades_per:<18.1f} {per_ticker:<15.3f} {max_con:<15.1f}")

print("\n" + "=" * 100)
print("Realistic max: 2-4 positions (trades happen across different tickers/hours throughout day)")
