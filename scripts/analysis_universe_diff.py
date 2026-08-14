"""Step 2: Universe divergence — which tickers traded in each source, win-rate diff."""

from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
YF_J  = ROOT / "data/journals/live_sim_seasonal_seasonal.csv"
IB_J  = ROOT / "data/journals/live_sim_ibkr_matched.csv"
OUT   = ROOT / "reports/ibkr_yf_analysis/universe_diff.csv"

yf = pd.read_csv(YF_J, parse_dates=["date_opened", "date_closed"])
ib = pd.read_csv(IB_J, parse_dates=["date_opened", "date_closed"])

yf_tickers = set(yf["ticker"].unique())
ib_tickers = set(ib["ticker"].unique())

both   = yf_tickers & ib_tickers
yf_only = yf_tickers - ib_tickers
ib_only = ib_tickers - yf_tickers

print(f"yfinance tickers: {len(yf_tickers)}   IBKR tickers: {len(ib_tickers)}")
print(f"In both: {len(both)}   yf-only: {len(yf_only)}   ibkr-only: {len(ib_only)}")
print(f"Jaccard overlap: {len(both) / len(yf_tickers | ib_tickers):.2%}")
print(f"\nyf-only:   {sorted(yf_only)}")
print(f"\nibkr-only: {sorted(ib_only)}")

# Per-ticker stats for shared tickers
rows = []
for t in sorted(both):
    yf_t = yf[yf["ticker"] == t]
    ib_t = ib[ib["ticker"] == t]
    yf_win = (yf_t["return_pct"] > 0).mean()
    ib_win = (ib_t["return_pct"] > 0).mean()
    yf_pnl = yf_t["pnl_usd"].sum()
    ib_pnl = ib_t["pnl_usd"].sum()
    rows.append({
        "ticker": t,
        "yf_trades": len(yf_t), "ib_trades": len(ib_t),
        "yf_win_rate": round(yf_win, 3), "ib_win_rate": round(ib_win, 3),
        "win_rate_delta": round(ib_win - yf_win, 3),
        "yf_pnl": round(yf_pnl, 2), "ib_pnl": round(ib_pnl, 2),
        "pnl_delta": round(ib_pnl - yf_pnl, 2),
    })

df_out = pd.DataFrame(rows).sort_values("pnl_delta")
df_out.to_csv(OUT, index=False)
print(f"\nBottom 10 by P&L delta (IBKR underperforms most):")
print(df_out.head(10).to_string(index=False))
print(f"\nTop 10 by P&L delta (IBKR outperforms):")
print(df_out.tail(10).to_string(index=False))
print(f"\nSaved: {OUT}")
