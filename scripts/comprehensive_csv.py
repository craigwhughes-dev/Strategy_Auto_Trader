#!/usr/bin/env python3
"""Export comprehensive comparison as CSV."""

import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path("reports/full_scan/summary.csv")
df = pd.read_csv(csv_path)
df_ok = df[df['status'] == 'ok'].copy()

results = []
for strat in sorted(df_ok['strategy'].unique()):
    strat_df = df_ok[df_ok['strategy'] == strat]

    sharpe = strat_df['sharpe_strategy'].replace([np.inf, -np.inf], np.nan).mean()
    sharpe_bh = strat_df['sharpe_bh'].mean()
    sortino = strat_df['sortino_strategy'].replace([np.inf, -np.inf], np.nan).mean()
    ret_strat = strat_df['total_return_strategy'].mean()
    ret_bh = strat_df['total_return_bh'].mean()
    win_rate = (strat_df['total_return_strategy'] > strat_df['total_return_bh']).sum() / len(strat_df) * 100
    trades_per_ticker = strat_df['n_trades_journaled'].mean()
    kelly_median = strat_df['final_kelly'].median()
    avg_investment = kelly_median * 10000
    n_tickers = len(strat_df)
    concurrent_per_ticker = (trades_per_ticker * 5) / (730 * 24)
    max_concurrent = concurrent_per_ticker * n_tickers
    pnl_10k = max_concurrent * avg_investment * (ret_strat / 100)

    max_concurrent_gbp = max_concurrent * avg_investment

    results.append({
        'Rank': len(results) + 1,
        'Strategy': strat,
        'Sharpe (Strat)': round(sharpe, 2),
        'Sharpe (B&H)': round(sharpe_bh, 2),
        'Sortino': round(sortino, 2),
        'Return (%)': round(ret_strat, 1),
        'Win Rate (%)': round(win_rate, 1),
        'Trades/Ticker': round(trades_per_ticker, 1),
        'Avg Investment (GBP)': int(round(avg_investment)),
        'Max Concurrent': round(max_concurrent, 1),
        'Max Concurrent (GBP)': int(round(max_concurrent_gbp)),
        'P&L on 10K (GBP)': round(pnl_10k, 2),
        'Kelly Fraction': round(kelly_median, 4),
    })

# Sort by sharpe
results = sorted(results, key=lambda x: x['Sharpe (Strat)'], reverse=True)
for i, r in enumerate(results, 1):
    r['Rank'] = i

out_df = pd.DataFrame(results)
out_df.to_csv('reports/comprehensive_strategy_report.csv', index=False)

print("Saved to: reports/comprehensive_strategy_report.csv")
print(out_df.to_string(index=False))
