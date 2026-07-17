#!/usr/bin/env python3
"""Enhanced analysis: max invested (Kelly × 10k) + time in market from summary.csv."""

import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path(__file__).parent.parent / "reports" / "full_scan" / "summary.csv"
df = pd.read_csv(csv_path)
df_ok = df[df['status'] == 'ok'].copy()

print("=" * 110)
print("RANKING BY SHARPE (with 10k pot metrics)")
print("=" * 110)

# Group by strategy
strategies = sorted(df_ok['strategy'].unique())
results = []

for strat in strategies:
    strat_df = df_ok[df_ok['strategy'] == strat]

    sharpe = strat_df['sharpe_strategy'].replace([np.inf, -np.inf], np.nan)
    sortino = strat_df['sortino_strategy'].replace([np.inf, -np.inf], np.nan)
    ret_strat = strat_df['total_return_strategy']
    ret_bh = strat_df['total_return_bh']
    kelly = strat_df['final_kelly']
    n_trades = strat_df['n_trades_journaled']
    n_bars = strat_df['n_bars']
    days_covered = strat_df['days_covered']

    # Win rate
    win_rate = (ret_strat > ret_bh).sum() / len(strat_df) * 100

    # Time in market: estimate from trades
    # Assume avg 5-hour hold per trade
    est_hours_invested = (n_trades * 5).sum()
    est_hours_total = n_bars.sum()
    time_in_market_pct = (est_hours_invested / est_hours_total * 100) if est_hours_total > 0 else 0

    # Max invested: median kelly × 10k (conservative single-position estimate)
    # Or: could be multiple positions at once. Use median kelly as typical per-position size.
    kelly_median = kelly.median()
    max_invested_single = kelly_median * 10000

    results.append({
        'strategy': strat,
        'n_tickers': len(strat_df),
        'sharpe': sharpe.mean(),
        'sortino': sortino.mean(),
        'return': ret_strat.mean(),
        'return_bh': ret_bh.mean(),
        'win_rate': win_rate,
        'kelly_median': kelly_median,
        'time_in_market_pct': time_in_market_pct,
        'max_invested': max_invested_single,
    })

# Sort by sharpe
results = sorted(results, key=lambda x: x['sharpe'], reverse=True)

print(f"\n{'Rank':<5} {'Strategy':<20} {'Sharpe':<8} {'Sortino':<8} {'Return':<10} {'Win %':<8} {'Time IM %':<10} {'Max $':<10}")
print("-" * 110)

for i, r in enumerate(results, 1):
    print(f"{i:<5} {r['strategy']:<20} {r['sharpe']:>7.2f} {r['sortino']:>7.2f} "
          f"{r['return']:>8.1f}% {r['win_rate']:>6.1f}% "
          f"{r['time_in_market_pct']:>8.1f}% ${r['max_invested']:>8.0f}")

print("\n" + "=" * 110)
print("\nKELLY FRACTIONS (median per strategy):")
print("-" * 110)
for r in results:
    print(f"{r['strategy']:<20} Kelly {r['kelly_median']:.4f}  =>  ${r['max_invested']:>8.0f} per concurrent trade on 10k pot")

print("\n" + "=" * 110)
