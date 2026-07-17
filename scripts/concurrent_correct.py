#!/usr/bin/env python3
"""Calculate max concurrent positions correctly."""

import pandas as pd
from pathlib import Path

csv_path = Path("reports/full_scan/summary.csv")
df = pd.read_csv(csv_path)
df_ok = df[df['status'] == 'ok'].copy()

print("=" * 90)
print("MAX CONCURRENT POSITIONS (Correct Calculation)")
print("=" * 90)
print("\nFormula: concurrent = (trades_per_ticker × hold_hours) / (days × 24)\n")

results = []
for strat in sorted(df_ok['strategy'].unique()):
    strat_df = df_ok[df_ok['strategy'] == strat]

    n_tickers = len(strat_df)
    avg_trades_per_ticker = strat_df['n_trades_journaled'].mean()
    avg_days = strat_df['days_covered'].mean()

    # Assume 5-hour average hold per trade (hourly data)
    avg_hold_hours = 5

    # Concurrent per ticker
    concurrent_per_ticker = (avg_trades_per_ticker * avg_hold_hours) / (avg_days * 24)

    # Peak concurrent across all tickers (conservative: assume 10% of tickers active simultaneously)
    # Or: if uncorrelated, assume sqrt(n_tickers) can run concurrently
    max_concurrent_conservative = concurrent_per_ticker * min(10, int((n_tickers ** 0.5)))
    max_concurrent_high = concurrent_per_ticker * n_tickers

    results.append((strat, n_tickers, avg_trades_per_ticker, concurrent_per_ticker, max_concurrent_conservative))

# Sort by conservative estimate
results = sorted(results, key=lambda x: x[4], reverse=True)

print(f"{'Strategy':<25} {'Tickers':<10} {'Trades/Ticker':<15} {'Concurrent':<12}")
print("-" * 90)
for strat, n_tickers, trades_per, concurrent_est, max_est in results:
    print(f"{strat:<25} {n_tickers:<10} {trades_per:<15.1f} {max(0.1, max_est):<12.1f}")

print("\n" + "=" * 90)
print("Note: Assuming 5-hour avg hold per trade, conservative overlap (10% tickers active).")
print("Real max: higher if many tickers trade same hours; lower if spread across UTC zones.")
