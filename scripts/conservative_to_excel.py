#!/usr/bin/env python3
"""Combine all conservative trades into single Excel file."""

import pandas as pd
from pathlib import Path

journal_dir = Path("data/journals/full_scan/conservative")
journal_files = list(journal_dir.glob("*.csv"))

print(f"Loading {len(journal_files)} conservative journal files...")

all_trades = []
for jf in journal_files:
    try:
        df = pd.read_csv(jf)
        # Ensure ticker column exists
        if 'ticker' not in df.columns:
            df['ticker'] = jf.stem
        all_trades.append(df)
    except Exception as e:
        print(f"Error reading {jf}: {e}")

trades = pd.concat(all_trades, ignore_index=True)
print(f"Combined {len(trades):,} trades from {len(journal_files)} tickers")

# Reorder columns - put ticker first
cols = trades.columns.tolist()
if 'ticker' in cols:
    cols.remove('ticker')
    cols = ['ticker'] + cols
    trades = trades[cols]

# Export to Excel
output_file = Path("reports/conservative_all_trades.xlsx")

# Summary data
summary_data = {
    'Metric': ['Total Trades', 'Unique Tickers', 'Winning Trades', 'Losing Trades', 'Win Rate %',
               'Mean Return %', 'Median Return %', 'Std Dev %', 'Avg Hold Days', 'Total P&L USD', 'Avg P&L USD'],
    'Value': [
        len(trades),
        trades['ticker'].nunique(),
        (trades['return_pct'] > 0).sum(),
        (trades['return_pct'] < 0).sum(),
        round((trades['return_pct'] > 0).sum() / len(trades) * 100, 2),
        round(trades['return_pct'].mean(), 4),
        round(trades['return_pct'].median(), 4),
        round(trades['return_pct'].std(), 4),
        round(trades['days_held'].mean(), 1) if 'days_held' in trades.columns else 'N/A',
        round(trades['pnl_usd'].sum(), 2) if 'pnl_usd' in trades.columns else 'N/A',
        round(trades['pnl_usd'].mean(), 2) if 'pnl_usd' in trades.columns else 'N/A',
    ]
}
summary_df = pd.DataFrame(summary_data)

# Write both sheets
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    summary_df.to_excel(writer, sheet_name='Summary', index=False)
    trades.to_excel(writer, sheet_name='All Trades', index=False)

print(f"\nExcel file created: {output_file}")
print(f"Sheets: Summary, All Trades")
print(f"Total rows: {len(trades):,}")
print(f"Total columns: {len(trades.columns)}")
print(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
