"""Step 5: Signal-level comparison — does the same ticker produce different entry signals on IBKR vs yfinance data?"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports/ibkr_yf_analysis/signal_diff.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

from Strategy_Auto_Trader.quant_hmm.quant_engine import fetch_hourly
from Strategy_Auto_Trader.strategy.base.registry import resolve_strategy
from Strategy_Auto_Trader.quant_hmm.consolidated_engine import consolidated_backtest

# 3 representative tickers: high-div US, LSE, growth/no-div US
TICKERS = [
    ("XOM",    "High-div US"),
    ("SHEL.L", "LSE"),
    ("NVDA",   "Growth/no-div US"),
]
STRATEGY = "optimised_new"

rows = []
for ticker, label in TICKERS:
    print(f"\n{'='*60}")
    print(f"{ticker} ({label})")
    entry_s, exit_s = resolve_strategy(STRATEGY, ticker=ticker)
    for source in ["yfinance", "ibkr"]:
        try:
            df = fetch_hourly(ticker, source=source)
        except Exception as e:
            print(f"  {source}: fetch failed — {e}")
            continue
        if df is None or df.empty:
            print(f"  {source}: empty data")
            continue

        try:
            result = consolidated_backtest(
                df,
                regime_model=None,
                entry_strategy=entry_s,
                exit_strategy=exit_s,
            )
        except Exception as e:
            print(f"  {source}: backtest failed — {e}")
            continue

        trades = result.get("trades", [])

        n_trades  = len(trades)
        total_ret = sum(t.get("return_pct", 0) for t in trades)
        win_rate  = sum(1 for t in trades if t.get("return_pct", 0) > 0) / n_trades if n_trades else 0

        print(f"  {source:10s}: trades={n_trades}  win={win_rate:.1%}  "
              f"total_return={total_ret:.1%}")

        rows.append({
            "ticker": ticker,
            "label": label,
            "source": source,
            "n_trades": n_trades,
            "win_rate": round(win_rate, 3),
            "total_return_pct": round(total_ret, 4),
            "mean_return_pct": round(total_ret / n_trades, 4) if n_trades else 0,
        })

    # Compare
    if len(rows) >= 2 and rows[-1]["ticker"] == rows[-2]["ticker"]:
        yf_row = rows[-2]
        ib_row = rows[-1]
        if yf_row["source"] == "yfinance" and ib_row["source"] == "ibkr":
            print(f"  d_trades={ib_row['n_trades']-yf_row['n_trades']:+d}  "
                  f"d_win_rate={ib_row['win_rate']-yf_row['win_rate']:+.1%}  "
                  f"d_total_return={ib_row['total_return_pct']-yf_row['total_return_pct']:+.1%}")

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

print(f"\n{'='*60}")
print("Summary:")
print(df.to_string(index=False))
print(f"\nSaved: {OUT}")
