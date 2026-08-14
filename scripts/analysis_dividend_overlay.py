"""Step 4: Dividend event overlay — do trades near dividend dates show larger IBKR/yf divergence?"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
TRADE_DIFF = ROOT / "reports/ibkr_yf_analysis/trade_diff.csv"
OUT = ROOT / "reports/ibkr_yf_analysis/dividend_overlay.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

td = pd.read_csv(TRADE_DIFF)
td["yf_date_opened"] = pd.to_datetime(td["yf_date_opened"], utc=True)

# Top tickers by absolute P&L delta (most divergent)
top_tickers = (td.groupby("ticker")["pnl_delta"]
               .apply(lambda x: x.abs().sum())
               .nlargest(15).index.tolist())

rows = []
for ticker in top_tickers:
    try:
        divs = yf.Ticker(ticker).dividends
    except Exception:
        continue
    if divs is None or divs.empty:
        print(f"  {ticker}: no dividends")
        rows.append({"ticker": ticker, "n_trades": 0, "n_div_trades": 0,
                     "div_pnl_delta_mean": np.nan, "clean_pnl_delta_mean": np.nan})
        continue

    divs.index = pd.to_datetime(divs.index, utc=True)
    trades_t = td[td["ticker"] == ticker].copy()

    WINDOW_DAYS = 5
    def near_dividend(dt):
        diffs = abs((divs.index - dt).total_seconds()) / 86400
        return diffs.min() <= WINDOW_DAYS if len(diffs) else False

    trades_t["near_div"] = trades_t["yf_date_opened"].apply(near_dividend)
    n_div = trades_t["near_div"].sum()
    n_clean = (~trades_t["near_div"]).sum()

    div_delta   = trades_t.loc[trades_t["near_div"],  "pnl_delta"].mean() if n_div  else np.nan
    clean_delta = trades_t.loc[~trades_t["near_div"], "pnl_delta"].mean() if n_clean else np.nan

    print(f"  {ticker:12s}  n_trades={len(trades_t):3d}  near_div={n_div}  "
          f"div_delta={div_delta:.1f}  clean_delta={clean_delta:.1f}")

    rows.append({
        "ticker": ticker,
        "n_trades": len(trades_t),
        "n_div_trades": int(n_div),
        "n_clean_trades": int(n_clean),
        "div_pnl_delta_mean": round(div_delta, 2) if not np.isnan(div_delta) else np.nan,
        "clean_pnl_delta_mean": round(clean_delta, 2) if not np.isnan(clean_delta) else np.nan,
    })

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

# Overall: do dividend-proximate trades show more divergence?
td_all = pd.read_csv(TRADE_DIFF)
td_all["yf_date_opened"] = pd.to_datetime(td_all["yf_date_opened"], utc=True)

all_near = []
for ticker in td_all["ticker"].unique():
    try:
        divs = yf.Ticker(ticker).dividends
    except Exception:
        continue
    if divs is None or divs.empty:
        continue
    divs.index = pd.to_datetime(divs.index, utc=True)
    trades_t = td_all[td_all["ticker"] == ticker]
    for idx, row in trades_t.iterrows():
        diffs = abs((divs.index - row["yf_date_opened"]).total_seconds()) / 86400
        all_near.append(diffs.min() <= 5 if len(diffs) else False)

if all_near:
    td_all = td_all.iloc[:len(all_near)].copy()
    td_all["near_div"] = all_near
    print(f"\n--- All {len(td_all)} matched trades ---")
    print(f"Near dividend (±5d): {td_all['near_div'].sum()}  |  Clean: {(~td_all['near_div']).sum()}")
    print(f"Div trade pnl_delta:   mean={td_all.loc[td_all['near_div'],'pnl_delta'].mean():.1f}  "
          f"std={td_all.loc[td_all['near_div'],'pnl_delta'].std():.1f}")
    print(f"Clean trade pnl_delta: mean={td_all.loc[~td_all['near_div'],'pnl_delta'].mean():.1f}  "
          f"std={td_all.loc[~td_all['near_div'],'pnl_delta'].std():.1f}")

print(f"\nSaved: {OUT}")
