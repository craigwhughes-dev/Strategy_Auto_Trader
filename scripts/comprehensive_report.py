#!/usr/bin/env python3
"""Comprehensive strategy comparison report."""

import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path("reports/full_scan/summary.csv")
df = pd.read_csv(csv_path)
df_ok = df[df['status'] == 'ok'].copy()

print("=" * 150)
print("COMPREHENSIVE STRATEGY COMPARISON")
print("=" * 150)

results = []
for strat in sorted(df_ok['strategy'].unique()):
    strat_df = df_ok[df_ok['strategy'] == strat]

    # Sharpe/Sortino
    sharpe = strat_df['sharpe_strategy'].replace([np.inf, -np.inf], np.nan).mean()
    sharpe_bh = strat_df['sharpe_bh'].mean()
    sortino = strat_df['sortino_strategy'].replace([np.inf, -np.inf], np.nan).mean()

    # Return
    ret_strat = strat_df['total_return_strategy'].mean()
    ret_bh = strat_df['total_return_bh'].mean()

    # Win rate
    win_rate = (strat_df['total_return_strategy'] > strat_df['total_return_bh']).sum() / len(strat_df) * 100

    # Trades per ticker
    trades_per_ticker = strat_df['n_trades_journaled'].mean()

    # Kelly (avg investment size on 10k pot)
    kelly_median = strat_df['final_kelly'].median()
    avg_investment = kelly_median * 10000

    # Max concurrent
    n_tickers = len(strat_df)
    concurrent_per_ticker = (trades_per_ticker * 5) / (730 * 24)
    max_concurrent = concurrent_per_ticker * n_tickers

    # P&L on 10K portfolio (at max concurrent, annualized)
    # Each position: kelly_median * 10k, earning avg_return% per ~730d period
    pnl_10k = max_concurrent * avg_investment * (ret_strat / 100)

    # Max invested (kelly × 10k, single position)
    max_invested_single = kelly_median * 10000

    results.append({
        'strategy': strat,
        'sharpe': sharpe,
        'sharpe_bh': sharpe_bh,
        'sortino': sortino,
        'return': ret_strat,
        'return_bh': ret_bh,
        'win_rate': win_rate,
        'trades_per_ticker': trades_per_ticker,
        'kelly': kelly_median,
        'avg_investment': avg_investment,
        'max_concurrent': max_concurrent,
        'pnl_10k': pnl_10k,
        'max_invested': max_invested_single,
    })

# Sort by Sharpe
results = sorted(results, key=lambda x: x['sharpe'], reverse=True)

# Print header
print(f"{'Rank':<5} {'Strategy':<20} {'Sharpe':<12} {'Sortino':<10} {'Return':<10} {'Win %':<8} {'Trades/Tk':<10} {'Avg Inv':<12} {'Max Conc':<10} {'P&L 10K':<12} {'Max Inv':<10}")
print("-" * 150)

for i, r in enumerate(results, 1):
    sharpe_str = f"{r['sharpe']:.2f}" if not np.isnan(r['sharpe']) else "nan"
    sortino_str = f"{r['sortino']:.2f}" if not np.isnan(r['sortino']) else "nan"
    pnl_str = f"£{r['pnl_10k']:,.0f}" if r['pnl_10k'] != 0 else "£0"

    print(f"{i:<5} {r['strategy']:<20} {sharpe_str:<12} {sortino_str:<10} {r['return']:>8.1f}% {r['win_rate']:>6.1f}% "
          f"{r['trades_per_ticker']:>8.1f} £{r['avg_investment']:>9,.0f} {r['max_concurrent']:>8.1f} {pnl_str:>11} £{r['max_invested']:>8,.0f}")

print("\n" + "=" * 150)
print("\nColumn Definitions:")
print("  Sharpe        : Risk-adjusted return (strategy / B&H benchmark)")
print("  Sortino       : Downside risk-adjusted return")
print("  Return        : Average total return % per ticker")
print("  Win %         : Pct of tickers beating B&H")
print("  Trades/Tk     : Avg trades per ticker over 730d")
print("  Avg Inv       : Kelly fraction * £10k (typical position size)")
print("  Max Conc      : Peak concurrent positions if all tickers trade together")
print("  P&L 10K       : Expected P&L on £10k pot at max concurrent (rough estimate)")
print("  Max Inv       : Maximum capital deployed per concurrent trade (= Avg Inv)")
print("=" * 150)
