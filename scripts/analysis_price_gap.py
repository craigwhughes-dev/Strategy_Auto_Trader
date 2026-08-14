"""Step 1: Price adjustment gap audit — yfinance adjusted vs IBKR TRADES bar-by-bar."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IBKR_CACHE = ROOT / "data/cache/ibkr_hourly"
OUT = ROOT / "reports/ibkr_yf_analysis/price_gap.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Sample: tickers from the trade_diff matched set (both sources have data)
TRADE_DIFF = ROOT / "reports/ibkr_yf_analysis/trade_diff.csv"

import yfinance as yf

def load_ibkr_csv(ticker: str) -> pd.DataFrame | None:
    fname = ticker.replace("/", "_").replace(".", "_") + ".csv"
    path = IBKR_CACHE / fname
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df

def load_yf_hourly(ticker: str) -> pd.DataFrame | None:
    # yfinance 1h capped at last 730 days — always fetch max window
    try:
        df = yf.download(ticker, period="730d", interval="1h",
                         auto_adjust=True, progress=False)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    except Exception:
        return None

# Pick sample: tickers from trade_diff (already traded by both sources)
td = pd.read_csv(TRADE_DIFF)
sample_tickers = td["ticker"].value_counts().head(20).index.tolist()

# Supplement with some known holdings if needed
extra_us  = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]
extra_lse = ["SHEL.L", "HSBA.L", "AZN.L", "BP.L", "ULVR.L"]
sample_tickers = list(dict.fromkeys(sample_tickers + extra_us + extra_lse))[:30]

rows = []
for ticker in sample_tickers:
    ib = load_ibkr_csv(ticker)
    if ib is None:
        continue

    yf_df = load_yf_hourly(ticker)
    if yf_df is None or yf_df.empty:
        continue

    # Align on common timestamps
    ib_close  = ib["Close"].rename("ib")
    # yf.download returns MultiIndex cols — squeeze to Series first
    yf_raw = yf_df["Close"]
    if isinstance(yf_raw, pd.DataFrame):
        yf_raw = yf_raw.iloc[:, 0]
    yf_close = yf_raw.rename("yf")

    both = pd.concat([ib_close, yf_close], axis=1, sort=False).dropna()
    if len(both) < 50:
        continue

    ratio = both["ib"] / both["yf"]
    exchange = "LSE" if ticker.endswith(".L") else "US"
    rows.append({
        "ticker": ticker,
        "exchange": exchange,
        "n_bars": len(both),
        "ratio_mean": ratio.mean(),
        "ratio_std": ratio.std(),
        "ratio_min": ratio.min(),
        "ratio_max": ratio.max(),
        "ratio_first": ratio.iloc[0],
        "ratio_last": ratio.iloc[-1],
        "ratio_drift": ratio.iloc[-1] - ratio.iloc[0],
        "overlap_start": str(both.index.min().date()),
        "overlap_end":   str(both.index.max().date()),
    })
    print(f"  {ticker:12s} n={len(both):5d}  ratio mean={ratio.mean():.4f}  "
          f"std={ratio.std():.4f}  drift={ratio.iloc[-1]-ratio.iloc[0]:.4f}")

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

print(f"\n--- Price ratio (IBKR/yfinance) summary ---")
print(f"Tickers processed: {len(df)}")
if df.empty:
    print("No tickers processed — check IBKR cache and yfinance availability.")
    sys.exit(0)
for exch in ["US", "LSE"]:
    sub = df[df["exchange"] == exch]
    if sub.empty:
        continue
    print(f"\n{exch} (n={len(sub)}):")
    print(f"  ratio mean={sub['ratio_mean'].mean():.4f}  "
          f"std_mean={sub['ratio_std'].mean():.4f}  "
          f"drift mean={sub['ratio_drift'].mean():.4f}")
    print(f"  Pence/pound suspects (ratio near 100): {(sub['ratio_mean'] > 50).sum()}")
    print(f"  Significant drift (>0.01): {(sub['ratio_drift'].abs() > 0.01).sum()}")

print(f"\nAll:")
print(f"  ratio mean={df['ratio_mean'].mean():.4f}  "
      f"std_mean={df['ratio_std'].mean():.4f}  "
      f"drift mean={df['ratio_drift'].mean():.4f}")

print(f"\nSaved: {OUT}")
