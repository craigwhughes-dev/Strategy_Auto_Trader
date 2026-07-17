#!/usr/bin/env python3
"""Analyze full_scan summary.csv — aggregate stats per strategy."""

import pandas as pd
import numpy as np
from pathlib import Path

# Load data
csv_path = Path(__file__).parent.parent / "reports" / "full_scan" / "summary.csv"
df = pd.read_csv(csv_path)

# Filter to 'ok' status only
df_ok = df[df['status'] == 'ok'].copy()

print("=" * 80)
print(f"Full Scan Analysis: {len(df_ok)} tickers × strategies, 9 strategies")
print("=" * 80)
print()

# Group by strategy
strategies = df_ok['strategy'].unique()
results = {}

for strat in sorted(strategies):
    strat_df = df_ok[df_ok['strategy'] == strat]

    # Filter out NaN/inf for numeric operations
    sharpe = strat_df['sharpe_strategy'].replace([np.inf, -np.inf], np.nan)
    sortino = strat_df['sortino_strategy'].replace([np.inf, -np.inf], np.nan)
    calmar = strat_df['calmar_strategy'].replace([np.inf, -np.inf], np.nan)
    ret_strat = strat_df['total_return_strategy']
    ret_bh = strat_df['total_return_bh']
    pf = strat_df['information_ratio']
    drawdown = strat_df['max_drawdown_strategy'].replace([np.inf, -np.inf], np.nan)

    # Win rate: strategy return > b&h return
    win_rate = (ret_strat > ret_bh).sum() / len(strat_df) * 100

    # Beat b&h absolute return
    beat_bh_count = (ret_strat > ret_bh).sum()

    results[strat] = {
        'n_tickers': len(strat_df),
        'avg_sharpe_strat': sharpe.mean(),
        'avg_sharpe_bh': strat_df['sharpe_bh'].mean(),
        'avg_sortino_strat': sortino.mean(),
        'avg_sortino_bh': strat_df['sortino_bh'].mean(),
        'avg_calmar_strat': calmar.mean(),
        'avg_calmar_bh': strat_df['calmar_bh'].mean(),
        'avg_return_strat': ret_strat.mean(),
        'avg_return_bh': ret_bh.mean(),
        'win_rate_pct': win_rate,
        'beat_bh': beat_bh_count,
        'avg_drawdown': drawdown.mean(),
        'avg_n_trades': strat_df['n_trades_journaled'].mean(),
        'avg_time_in_market': (strat_df['n_trades_journaled'] * 5 / strat_df['n_bars'] * 100).mean(),  # rough: avg 5 bars per trade
        'median_kelly': strat_df['final_kelly'].median(),
    }

# Display results ranked by Sharpe
print("RANKING BY AVG SHARPE (strategy):")
print("-" * 80)
ranked = sorted(results.items(), key=lambda x: x[1]['avg_sharpe_strat'], reverse=True)
for i, (strat, stats) in enumerate(ranked, 1):
    print(f"{i}. {strat:20s} Sharpe {stats['avg_sharpe_strat']:6.2f} (BH {stats['avg_sharpe_bh']:5.2f}) | "
          f"Sortino {stats['avg_sortino_strat']:6.2f} | Return {stats['avg_return_strat']:+7.1f}% "
          f"(BH {stats['avg_return_bh']:+7.1f}%) | Win {stats['win_rate_pct']:5.1f}%")

print()
print("DETAILED STATS BY STRATEGY:")
print("-" * 80)
for strat, stats in ranked:
    print()
    print(f"{strat.upper()}")
    print(f"  Tickers: {stats['n_tickers']}")
    print(f"  Sharpe (strategy/BH): {stats['avg_sharpe_strat']:.2f} / {stats['avg_sharpe_bh']:.2f}")
    print(f"  Sortino (strategy/BH): {stats['avg_sortino_strat']:.2f} / {stats['avg_sortino_bh']:.2f}")
    print(f"  Calmar (strategy/BH): {stats['avg_calmar_strat']:.2f} / {stats['avg_calmar_bh']:.2f}")
    print(f"  Return %: {stats['avg_return_strat']:+.1f}% vs B&H {stats['avg_return_bh']:+.1f}%")
    print(f"  Win rate (beat B&H): {stats['win_rate_pct']:.1f}% ({stats['beat_bh']}/{stats['n_tickers']} tickers)")
    print(f"  Avg max drawdown: {stats['avg_drawdown']:.2f}%")
    print(f"  Avg trades per ticker: {stats['avg_n_trades']:.0f}")
    print(f"  Median Kelly fraction: {stats['median_kelly']:.3f}")

print()
print("=" * 80)
print("KEY OBSERVATIONS:")
print("=" * 80)

# Sharpe ranking
sharpe_ranked = sorted(results.items(), key=lambda x: x[1]['avg_sharpe_strat'], reverse=True)
print(f"[BEST] Highest Sharpe: {sharpe_ranked[0][0]} ({sharpe_ranked[0][1]['avg_sharpe_strat']:.2f})")
print(f"[WORST] Lowest Sharpe: {sharpe_ranked[-1][0]} ({sharpe_ranked[-1][1]['avg_sharpe_strat']:.2f})")

# Return ranking
return_ranked = sorted(results.items(), key=lambda x: x[1]['avg_return_strat'], reverse=True)
print(f"[BEST] Highest return: {return_ranked[0][0]} ({return_ranked[0][1]['avg_return_strat']:+.1f}%)")
print(f"[WORST] Lowest return: {return_ranked[-1][0]} ({return_ranked[-1][1]['avg_return_strat']:+.1f}%)")

# Best risk-adjusted (Sortino)
sortino_ranked = sorted(results.items(), key=lambda x: x[1]['avg_sortino_strat'], reverse=True)
print(f"[BEST] Sortino (downside): {sortino_ranked[0][0]} ({sortino_ranked[0][1]['avg_sortino_strat']:.2f})")

# Win rate
win_ranked = sorted(results.items(), key=lambda x: x[1]['win_rate_pct'], reverse=True)
print(f"[BEST] Highest win rate: {win_ranked[0][0]} ({win_ranked[0][1]['win_rate_pct']:.1f}%)")
print(f"[WORST] Lowest win rate: {win_ranked[-1][0]} ({win_ranked[-1][1]['win_rate_pct']:.1f}%)")

# Activity level
trades_ranked = sorted(results.items(), key=lambda x: x[1]['avg_n_trades'], reverse=True)
print(f"[ACTIVE] Most trades: {trades_ranked[0][0]} ({trades_ranked[0][1]['avg_n_trades']:.0f} avg/ticker)")
print(f"[QUIET] Least trades: {trades_ranked[-1][0]} ({trades_ranked[-1][1]['avg_n_trades']:.0f} avg/ticker)")

print()
print("=" * 80)
