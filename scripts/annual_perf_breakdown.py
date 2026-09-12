import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

JOURNAL = "data_synthetic/journals/synth_26yr.csv"
POS_SUM = "data_synthetic/journals/live_sim_synthetic_position_summary_20260912T092749.csv"
INITIAL = 100_000.0

# --- Load journal (closed trades) ---
trades = pd.read_csv(JOURNAL)
trades["date_closed"] = pd.to_datetime(trades["date_closed"], utc=True, errors="coerce")
trades["date_opened"] = pd.to_datetime(trades["date_opened"], utc=True, errors="coerce")
trades["close_year"] = trades["date_closed"].dt.year
trades["group"] = trades["ticker"].apply(lambda t: "FTSE" if str(t).endswith(".L") else "SP500")

# Realized P&L per group per close-year
grp = trades.groupby(["close_year", "group"])["pnl_usd"].sum().unstack(fill_value=0)
grp.columns.name = None
for c in ["FTSE", "SP500"]:
    if c not in grp.columns:
        grp[c] = 0.0

# --- Portfolio annual return from position summary ---
ps = pd.read_csv(POS_SUM)
ps["date"] = pd.to_datetime(ps["date"], errors="coerce")
ps = ps.sort_values("date")
ps["year"] = ps["date"].dt.year
annual_port = ps.groupby("year").last()[["portfolio_value"]].copy()
annual_port["prev"] = annual_port["portfolio_value"].shift(1)
annual_port.loc[annual_port.index[0], "prev"] = INITIAL
annual_port["port_ret"] = (annual_port["portfolio_value"] / annual_port["prev"]) - 1

# --- Fetch index annual returns ---
def fetch_annual(ticker, start="1999-01-01", end="2026-09-30"):
    try:
        raw = yf.download(ticker, start=start, end=end, interval="1mo",
                          auto_adjust=True, progress=False)
        if raw.empty:
            return pd.Series(dtype=float)
        close = raw["Close"].squeeze()
        close.index = pd.to_datetime(close.index)
        yr = close.groupby(close.index.year).last()
        ret = yr.pct_change()
        return ret
    except Exception as e:
        print(f"  fetch failed {ticker}: {e}")
        return pd.Series(dtype=float)

print("Fetching ^GSPC and ^FTSE...")
sp_ret  = fetch_annual("^GSPC")
ftse_ret = fetch_annual("^FTSE")

# --- Build combined table ---
years = sorted(annual_port.index.astype(int))
print("\n")
print(f"{'Year':<6} {'Port':>8} {'FTSE£PnL':>10} {'SP$PnL':>10} | {'Port%':>7} {'S&P%':>7} {'FTSE%':>7}")
print("-" * 66)
for yr in years:
    pv   = annual_port.loc[yr, "portfolio_value"] if yr in annual_port.index else float("nan")
    pr   = annual_port.loc[yr, "port_ret"]        if yr in annual_port.index else float("nan")
    fp   = grp.loc[yr, "FTSE"]  if yr in grp.index else 0.0
    sp   = grp.loc[yr, "SP500"] if yr in grp.index else 0.0
    spx  = sp_ret.get(yr,  float("nan"))
    ftx  = ftse_ret.get(yr, float("nan"))

    fp_s  = f"{fp:>+10,.0f}" if not np.isnan(fp)  else f"{'—':>10}"
    sp_s  = f"{sp:>+10,.0f}" if not np.isnan(sp)  else f"{'—':>10}"
    pr_s  = f"{pr:>+7.1%}"   if not np.isnan(pr)  else f"{'—':>7}"
    spx_s = f"{spx:>+7.1%}"  if not np.isnan(spx) else f"{'n/a':>7}"
    ftx_s = f"{ftx:>+7.1%}"  if not np.isnan(ftx) else f"{'n/a':>7}"
    print(f"{yr:<6} {pv:>8,.0f} {fp_s} {sp_s} | {pr_s} {spx_s} {ftx_s}")

# Sharpe/Sortino
r = annual_port["port_ret"].values
rfr = 0.04
excess = r - rfr
sharpe = excess.mean() / excess.std(ddof=1)
downside = r[r < rfr] - rfr
sortino = excess.mean() / np.sqrt((downside**2).mean()) if len(downside) else float("nan")
cagr = (annual_port["portfolio_value"].iloc[-1] / INITIAL) ** (1.0 / len(years)) - 1

print(f"\nCAGR ({len(years)}yr): {cagr:.1%}   Sharpe: {sharpe:.2f}   Sortino: {sortino:.2f}")
print("\nNote: FTSE£PnL and SP$PnL are realized P&L in trade currency (mixed GBp/USD) for trades closing that year.")
print("      S&P% and FTSE% are price-only index returns (^GSPC, ^FTSE), no dividends.")
