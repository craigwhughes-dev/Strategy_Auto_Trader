#!/usr/bin/env python3
"""Per-ticker P&L summary across all strategies."""

import pandas as pd
from pathlib import Path

journal_dir = Path("reports/simplified_journals")
output_file = Path("reports/ticker_pnl_summary.csv")

results = []

# Load each strategy's simplified journal
for csv_file in sorted(journal_dir.glob("*_simplified.csv")):
    strategy = csv_file.stem.replace("_simplified", "")

    try:
        df = pd.read_csv(csv_file)

        # Group by ticker
        ticker_stats = df.groupby('ticker').agg({
            'pnl_usd': ['sum', 'count', 'mean'],
            'return_pct': ['mean', 'count'],
        }).round(4)

        ticker_stats.columns = ['pnl_total', 'trades', 'pnl_avg', 'return_mean', 'trades_count']
        ticker_stats['strategy'] = strategy

        results.append(ticker_stats)
        print(f"{strategy}: {len(ticker_stats)} tickers")

    except Exception as e:
        print(f"Error {csv_file.name}: {e}")

# Combine all results
combined = pd.concat(results, ignore_index=False)
combined = combined.reset_index()
combined = combined.rename(columns={'ticker': 'ticker'})

# Create pivot: rows = tickers, columns = strategy P&L
pivot = combined.pivot_table(
    index='ticker',
    columns='strategy',
    values='pnl_total',
    aggfunc='sum'
)

# Add totals
pivot['total_pnl'] = pivot.sum(axis=1)
pivot['num_strategies'] = pivot.notna().sum(axis=1) - 1  # Exclude total column

# Sort by total P&L descending
pivot = pivot.sort_values('total_pnl', ascending=False)

# Export
pivot.to_csv(output_file)

print(f"\nExported {len(pivot)} tickers to {output_file}")
print(f"\nTop 10 tickers by total P&L:")
print(pivot[['ai', 'breakout_momentum', 'conservative', 'default', 'optimised', 'trend', 'total_pnl']].head(10).to_string())

print(f"\nBottom 10 tickers:")
print(pivot[['ai', 'breakout_momentum', 'conservative', 'default', 'optimised', 'trend', 'total_pnl']].tail(10).to_string())

print(f"\nTotal P&L all tickers/strategies: ${pivot['total_pnl'].sum():,.2f}")
print(f"Mean P&L per ticker: ${pivot['total_pnl'].mean():,.2f}")
