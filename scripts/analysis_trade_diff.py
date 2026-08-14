"""Step 3: Trade-level diff — match trades by ticker+date, compare return_pct and P&L."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
YF_J  = ROOT / "data/journals/live_sim_seasonal_seasonal.csv"
IB_J  = ROOT / "data/journals/live_sim_ibkr_matched.csv"
OUT   = ROOT / "reports/ibkr_yf_analysis/trade_diff.csv"

yf = pd.read_csv(YF_J)
ib = pd.read_csv(IB_J)
yf["date_opened"] = pd.to_datetime(yf["date_opened"], utc=True)
ib["date_opened"] = pd.to_datetime(ib["date_opened"], utc=True)

yf["source"] = "yfinance"
ib["source"] = "ibkr"

# Match by ticker + date_opened within ±3 days
matches = []
for _, yrow in yf.iterrows():
    t = yrow["ticker"]
    d = yrow["date_opened"]
    candidates = ib[
        (ib["ticker"] == t) &
        (abs((ib["date_opened"] - d).dt.total_seconds()) <= 3 * 86400)
    ]
    if len(candidates) == 0:
        continue
    # Closest match
    irow = candidates.iloc[(abs(candidates["date_opened"] - d)).argmin()]
    matches.append({
        "ticker": t,
        "exchange": "LSE" if t.endswith(".L") else "US",
        "yf_date_opened": yrow["date_opened"],
        "ib_date_opened": irow["date_opened"],
        "date_delta_days": (irow["date_opened"] - yrow["date_opened"]).total_seconds() / 86400,
        "yf_entry_price": yrow["entry_price"],
        "ib_entry_price": irow["entry_price"],
        "price_ratio": irow["entry_price"] / yrow["entry_price"] if yrow["entry_price"] != 0 else np.nan,
        "yf_return_pct": yrow["return_pct"],
        "ib_return_pct": irow["return_pct"],
        "return_delta": irow["return_pct"] - yrow["return_pct"],
        "yf_pnl": yrow["pnl_usd"],
        "ib_pnl": irow["pnl_usd"],
        "pnl_delta": irow["pnl_usd"] - yrow["pnl_usd"],
        "yf_exit_reason": yrow["exit_reason"],
        "ib_exit_reason": irow["exit_reason"],
    })

df = pd.DataFrame(matches)
df.to_csv(OUT, index=False)

n = len(df)
print(f"Matched trades: {n}  (yf={len(yf)}, ib={len(ib)})")
print(f"yf-unmatched: {len(yf) - n}   ib-unmatched: {len(ib) - n}")

print(f"\n--- Price ratio (IBKR/yfinance entry price) ---")
print(f"All:  mean={df['price_ratio'].mean():.4f}  std={df['price_ratio'].std():.4f}  "
      f"min={df['price_ratio'].min():.4f}  max={df['price_ratio'].max():.4f}")
for exch in ["US", "LSE"]:
    sub = df[df["exchange"] == exch]
    print(f"{exch}: mean={sub['price_ratio'].mean():.4f}  std={sub['price_ratio'].std():.4f}  n={len(sub)}")

print(f"\n--- Return delta (IBKR - yfinance) ---")
print(f"All:  mean={df['return_delta'].mean():.4f}  median={df['return_delta'].median():.4f}  "
      f"std={df['return_delta'].std():.4f}")
for exch in ["US", "LSE"]:
    sub = df[df["exchange"] == exch]
    print(f"{exch}: mean={sub['return_delta'].mean():.4f}  median={sub['return_delta'].median():.4f}  n={len(sub)}")

print(f"\n--- P&L delta (IBKR - yfinance) ---")
print(f"All:  total={df['pnl_delta'].sum():.0f}  mean={df['pnl_delta'].mean():.0f}")
for exch in ["US", "LSE"]:
    sub = df[df["exchange"] == exch]
    print(f"{exch}: total={sub['pnl_delta'].sum():.0f}  mean={sub['pnl_delta'].mean():.0f}  n={len(sub)}")

print(f"\nWorst 10 trades by P&L delta:")
print(df.nsmallest(10, "pnl_delta")[["ticker","exchange","yf_date_opened","price_ratio","yf_return_pct","ib_return_pct","return_delta","pnl_delta"]].to_string(index=False))

print(f"\nSaved: {OUT}")
