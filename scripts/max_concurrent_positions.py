#!/usr/bin/env python3
"""Find max concurrent positions per strategy from journals."""

import pandas as pd
from pathlib import Path
from collections import defaultdict

# Load all journals, extract open/close dates per strategy
journal_dir = Path("data/journals/full_scan")
strategy_trades = defaultdict(list)

for jf in journal_dir.glob("*.csv"):
    try:
        df = pd.read_csv(jf)
        for _, row in df.iterrows():
            if pd.notna(row.get('strategy')) and pd.notna(row.get('date_opened')) and pd.notna(row.get('date_closed')):
                strategy_trades[row['strategy']].append({
                    'open': pd.to_datetime(row['date_opened'], utc=True),
                    'close': pd.to_datetime(row['date_closed'], utc=True),
                })
    except:
        pass

# Find max concurrent for each strategy
print("=" * 70)
print("MAX CONCURRENT POSITIONS PER STRATEGY")
print("=" * 70)

results = []
for strat in sorted(strategy_trades.keys()):
    trades = strategy_trades[strat]
    if not trades:
        continue

    # Get all event timestamps
    events = []
    for t in trades:
        events.append((t['open'], 'open'))
        events.append((t['close'], 'close'))

    events.sort()

    # Sweep line to find max concurrent
    current = 0
    max_concurrent = 0
    for ts, event_type in events:
        if event_type == 'open':
            current += 1
            max_concurrent = max(max_concurrent, current)
        else:
            current -= 1

    results.append((strat, len(trades), max_concurrent))
    print(f"{strat:<20} Total trades: {len(trades):>4}   Max concurrent: {max_concurrent:>3}")

print("=" * 70)
