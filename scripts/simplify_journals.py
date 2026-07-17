#!/usr/bin/env python3
"""Simplify journal exports - keep only useful columns."""

import pandas as pd
from pathlib import Path

CAPITAL = 10000  # From backtest config

journal_dir = Path("data/journals/full_scan")
output_dir = Path("reports/simplified_journals")
output_dir.mkdir(exist_ok=True)

# Process each strategy
for strat_dir in journal_dir.iterdir():
    if not strat_dir.is_dir():
        continue

    strategy = strat_dir.name
    print(f"\nProcessing {strategy}...")

    all_trades = []
    for jf in strat_dir.glob("*.csv"):
        try:
            df = pd.read_csv(jf)
            ticker = jf.stem

            # Extract key columns
            trade = pd.DataFrame({
                'ticker': ticker,
                'date_opened': df['date_opened'],
                'date_closed': df['date_closed'],
                'entry_price': df['entry_price'],
                'exit_price': df['exit_price'],
                'return_pct': df['return_pct'],
                'pnl_usd': df['pnl_usd'],
                'days_held': df['days_held'],
                'exit_reason': df['exit_reason'],
                'kelly_fraction': df['kelly_fraction'],
                'entry_volatility': df['entry_rolling_vol_20'] if 'entry_rolling_vol_20' in df.columns else None,
                'peak_gain': df['peak_gain'],
                'peak_loss': df['peak_loss'],
                'entry_signal': df['entry_signal'],
            })

            # Calculate derived fields
            trade['amount_invested'] = trade['kelly_fraction'] * CAPITAL
            trade['shares_held'] = trade['amount_invested'] / trade['entry_price']
            trade['volume_ratio_at_entry'] = df['volume_ratio'] if 'volume_ratio' in df.columns else None

            all_trades.append(trade)
        except Exception as e:
            print(f"  Error {jf.name}: {e}")
            continue

    if not all_trades:
        print(f"  No trades found")
        continue

    combined = pd.concat(all_trades, ignore_index=True)

    # Export simplified CSV
    output_file = output_dir / f"{strategy}_simplified.csv"
    combined.to_csv(output_file, index=False)

    print(f"  {len(combined):,} trades -> {output_file.name}")
    print(f"  Avg invested: £{combined['amount_invested'].mean():,.2f}")
    print(f"  Max invested: £{combined['amount_invested'].max():,.2f}")
    print(f"  Avg P&L: ${combined['pnl_usd'].mean():,.2f}")
    print(f"  Total P&L: ${combined['pnl_usd'].sum():,.2f}")

print(f"\nDone. Output: {output_dir}")
