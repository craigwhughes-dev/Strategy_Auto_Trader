#!/usr/bin/env python3
"""Analyze winners vs losers for ALL strategies. Identify actionable filters.

For each strategy:
1. Split tickers into winners (P&L > 0) vs losers (P&L < 0)
2. Compare all available metrics (Sharpe, Sortino, volatility, Kelly, etc)
3. Rank metrics by correlation with P&L (strongest differentiators first)
4. Identify if a simple filter could improve the strategy
5. Recommend optimised version if filter is strong enough
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load data
pnl_df = pd.read_csv("reports/ticker_pnl_summary.csv")
summary_df = pd.read_csv("reports/full_scan/summary.csv")
summary_df = summary_df[summary_df['status'] == 'ok'].copy()

strategies = [s for s in summary_df['strategy'].unique() if s != 'default']
strategies = sorted(strategies)

print("=" * 150)
print("STRATEGY OPTIMIZATION ANALYSIS: Winners vs Losers")
print("=" * 150)

for strategy in strategies:
    print(f"\n\n{'=' * 150}")
    print(f"STRATEGY: {strategy.upper()}")
    print(f"{'=' * 150}")

    # Get P&L for this strategy
    if strategy not in pnl_df.columns:
        print(f"  No P&L data for {strategy}")
        continue

    strategy_pnl = pnl_df[['ticker', strategy]].copy()
    strategy_pnl = strategy_pnl[strategy_pnl[strategy].notna()].copy()
    strategy_pnl.columns = ['ticker', 'pnl']

    # Get metrics for this strategy
    strat_df = summary_df[summary_df['strategy'] == strategy].copy()
    merged = strategy_pnl.merge(strat_df, on='ticker', how='inner')

    winners = merged[merged['pnl'] > 0]
    losers = merged[merged['pnl'] < 0]

    print(f"\nSummary:")
    print(f"  Winners: {len(winners):>3} tickers, P&L: ${winners['pnl'].sum():>12,.0f}")
    print(f"  Losers:  {len(losers):>3} tickers, P&L: ${losers['pnl'].sum():>12,.0f}")
    print(f"  Total:   {len(merged):>3} tickers, P&L: ${merged['pnl'].sum():>12,.0f}")

    # Key metrics to compare
    metrics = [
        'sharpe_strategy', 'sortino_strategy', 'calmar_strategy',
        'total_return_strategy', 'total_return_bh',
        'max_drawdown_strategy', 'max_drawdown_bh',
        'n_trades_journaled', 'final_kelly',
        'ann_vol', 'downside_vol',
        'efficiency_ratio', 'choppiness_idx', 'trend_quality',
        'autocorr', 'sign_change_freq',
        'information_ratio', 'up_capture', 'down_capture',
    ]

    # Calculate correlations
    correlations = []
    for metric in metrics:
        if metric not in merged.columns:
            continue

        clean_data = merged[[metric, 'pnl']].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean_data) < 2:
            continue

        corr = clean_data[metric].corr(clean_data['pnl'])
        if pd.notna(corr):
            correlations.append({
                'metric': metric,
                'correlation': corr,
                'w_mean': winners[metric].replace([np.inf, -np.inf], np.nan).mean(),
                'l_mean': losers[metric].replace([np.inf, -np.inf], np.nan).mean(),
            })

    correlations = sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)

    # Show top 10 differentiators
    print(f"\nTop 10 Differentiators (by correlation with P&L):")
    print(f"  {'Rank':<5} {'Metric':<30} {'Correlation':<12} {'Winners':<15} {'Losers':<15}")
    print(f"  {'-'*5} {'-'*30} {'-'*12} {'-'*15} {'-'*15}")

    for i, item in enumerate(correlations[:10], 1):
        w = item['w_mean']
        l = item['l_mean']
        corr = item['correlation']
        metric = item['metric']

        w_str = f"{w:>13.2f}" if not np.isnan(w) else "        NaN"
        l_str = f"{l:>13.2f}" if not np.isnan(l) else "        NaN"

        print(f"  {i:<5} {metric:<30} {corr:>+10.3f}  {w_str}  {l_str}")

    # Recommend filter if top correlator is strong
    top_metric = correlations[0] if correlations else None
    if top_metric and abs(top_metric['correlation']) > 0.15:
        metric_name = top_metric['metric']
        print(f"\n*** ACTIONABLE FILTER FOUND ***")
        print(f"  Metric: {metric_name} (r={top_metric['correlation']:+.3f})")
        print(f"  Winners mean:  {top_metric['w_mean']:>10.3f}")
        print(f"  Losers mean:   {top_metric['l_mean']:>10.3f}")
        print(f"  Difference:    {top_metric['w_mean'] - top_metric['l_mean']:>10.3f}")
        print(f"  Recommendation: Create {strategy}_optimised with gate on {metric_name}")
    else:
        print(f"\n  No strong differentiator found (top r={top_metric['correlation']:+.3f})")

print("\n\n" + "=" * 150)
print("SUMMARY: Strategies with Actionable Filters")
print("=" * 150)
print("\nReview recommendations above and create optimised versions for strategies with r > 0.15")
