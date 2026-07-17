#!/usr/bin/env python3
"""Analyze what differentiates conservative strategy winners vs losers."""

import pandas as pd
import numpy as np
from pathlib import Path

# Load ticker P&L summary
pnl_df = pd.read_csv("reports/ticker_pnl_summary.csv")
conservative_pnl = pnl_df[['ticker', 'conservative', 'total_pnl']].copy()
conservative_pnl = conservative_pnl[conservative_pnl['conservative'].notna()]

# Load full scan summary metrics
summary_df = pd.read_csv("reports/full_scan/summary.csv")
conservative_df = summary_df[(summary_df['strategy'] == 'conservative') & (summary_df['status'] == 'ok')].copy()

# Merge
merged = conservative_pnl.merge(conservative_df, on='ticker', how='inner')

print(f"Conservative strategy: {len(merged)} tickers with data")
print("\n" + "=" * 120)

# Split winners vs losers
winners = merged[merged['conservative'] > 0].copy()
losers = merged[merged['conservative'] < 0].copy()

print(f"Winners: {len(winners)} tickers, total P&L: ${winners['conservative'].sum():,.2f}")
print(f"Losers:  {len(losers)} tickers, total P&L: ${losers['conservative'].sum():,.2f}")

# Key metrics to compare
metrics = [
    'sharpe_strategy', 'sortino_strategy', 'calmar_strategy',
    'total_return_strategy', 'total_return_bh',
    'max_drawdown_strategy', 'max_drawdown_bh',
    'n_trades_journaled', 'final_kelly',
    'ann_vol', 'downside_vol',
    'efficiency_ratio', 'choppiness_idx',
    'trend_quality', 'autocorr', 'sign_change_freq',
    'information_ratio', 'up_capture', 'down_capture',
]

print("\n" + "=" * 120)
print("COMPARISON: Winners vs Losers")
print("=" * 120)
print(f"{'Metric':<30} {'Winners (mean)':<20} {'Losers (mean)':<20} {'Difference':<15}")
print("-" * 120)

comparison = []
for metric in metrics:
    if metric not in merged.columns:
        continue

    w_val = winners[metric].replace([np.inf, -np.inf], np.nan).mean()
    l_val = losers[metric].replace([np.inf, -np.inf], np.nan).mean()

    if pd.isna(w_val) or pd.isna(l_val):
        continue

    diff = w_val - l_val
    pct_diff = (diff / abs(l_val) * 100) if l_val != 0 else 0

    comparison.append({
        'metric': metric,
        'winners': w_val,
        'losers': l_val,
        'diff': diff,
        'pct_diff': pct_diff,
    })

    print(f"{metric:<30} {w_val:<20.4f} {l_val:<20.4f} {diff:+.4f} ({pct_diff:+.1f}%)")

print("\n" + "=" * 120)
print("TOP 5 WINNERS:")
print("=" * 120)
top_winners = winners.nlargest(5, 'conservative')[['ticker', 'conservative', 'sharpe_strategy', 'total_return_strategy', 'max_drawdown_strategy', 'n_trades_journaled', 'final_kelly']]
print(top_winners.to_string(index=False))

print("\n" + "=" * 120)
print("TOP 5 LOSERS:")
print("=" * 120)
top_losers = losers.nsmallest(5, 'conservative')[['ticker', 'conservative', 'sharpe_strategy', 'total_return_strategy', 'max_drawdown_strategy', 'n_trades_journaled', 'final_kelly']]
print(top_losers.to_string(index=False))

# Rank metrics by their correlation with P&L
print("\n" + "=" * 120)
print("METRICS RANKED BY CORRELATION WITH P&L (differentiator strength)")
print("=" * 120)

correlations = []
for metric in metrics:
    if metric not in merged.columns:
        continue

    # Clean data (remove inf, nan)
    clean_data = merged[[metric, 'conservative']].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean_data) < 2:
        continue

    corr = clean_data[metric].corr(clean_data['conservative'])
    if pd.notna(corr):
        correlations.append({'metric': metric, 'correlation': corr})

correlations = sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)

for i, item in enumerate(correlations[:15], 1):
    print(f"{i:2}. {item['metric']:<30} r={item['correlation']:+.4f}")

print("\n" + "=" * 120)
