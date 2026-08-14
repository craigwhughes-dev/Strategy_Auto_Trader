"""
Controlled IBKR comparison: same 70 tickers, same date window as yfinance Sharpe-2.032 run.

Fetches IBKR data for each ticker, clips to >= YF_START (2023-09-18, the exact
date yfinance hourly data begins), then runs generate_candidates + arbitrate
with the same params as the seasonal Sharpe-2.032 run. Only variable vs that
run: data source (IBKR TRADES vs yfinance adjusted).
"""

from __future__ import annotations
from pathlib import Path
import sys
import math
import pandas as pd
import numpy as np
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Strategy_Auto_Trader.quant_hmm.quant_engine import fetch_hourly
from Strategy_Auto_Trader.quant_hmm.ticker_ranking import generate_candidates
from Strategy_Auto_Trader.markov_cli.live_sim import arbitrate
from Strategy_Auto_Trader.output.journal import append_trades

YF_START = "2023-09-18"   # exact start of yfinance hourly window (verified)
ARBITRATION_START = "2023-12-14"
INITIAL_CASH = 100_000.0
TRADE_COST = 1.0
COST_MODEL = "ibkr_tiered_spread"
STRATEGY = "optimised_new"

# Exact 70 tickers from the Sharpe 2.032 seasonal run journal
TICKERS = [
    "ABT", "ADM.L", "ATO", "AUTO.L", "AXP", "BRK-B", "BT-A.L", "CDNS",
    "CF", "CHRW", "CINF", "CMCSA", "CNC", "COR", "CSCO", "CVS", "DELL",
    "DRI", "DUK", "DVA", "DVN", "EDV.L", "ELV", "EXPN.L", "FE", "FTNT",
    "GOOG", "GOOGL", "HAS", "HCA", "HIG", "HPE", "HRL", "HUM", "IAG.L",
    "IBKR", "IBM", "ICE", "ITRK.L", "J", "KMB", "KR", "L", "MA", "MCK",
    "MDT", "MKC", "MNG.L", "MSI", "NDAQ", "NEE", "NVDA", "O", "ORLY",
    "PCT.L", "PHM", "PM", "PWR", "ROP", "SHEL.L", "SMCI", "SNPS", "SYK",
    "TER", "TJX", "TPL", "TRGP", "VEEV", "VICI", "VLO",
]

OUT_JOURNAL = ROOT / "data/journals/diag_ibkr_resampled_yf70.csv"
OUT_POS     = ROOT / "data/journals/diag_ibkr_resampled_yf70_pos.csv"

print(f"Loading IBKR data for {len(TICKERS)} tickers, clipping to >= {YF_START}")
cutoff_dt = pd.Timestamp(YF_START, tz="UTC")

df_by_ticker: dict[str, pd.DataFrame] = {}
failed = []
for ticker in TICKERS:
    try:
        df = fetch_hourly(ticker, source="ibkr")
        if df is None or df.empty:
            print(f"  {ticker}: no IBKR data — skip")
            failed.append(ticker)
            continue
        # Normalise index to UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        # Resample to :30-aligned 60-min bars matching yfinance convention (analysis only)
        df = df.resample("1h", offset="30min").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        ).dropna(subset=["Close"])
        df_clipped = df[df.index >= cutoff_dt]
        if len(df_clipped) < 200:
            print(f"  {ticker}: only {len(df_clipped)} bars after clip — skip")
            failed.append(ticker)
            continue
        df_by_ticker[ticker] = df_clipped
        print(f"  {ticker}: {df.index.min().date()} -> clipped to {df_clipped.index.min().date()}  ({len(df_clipped)} bars)")
    except Exception as e:
        print(f"  {ticker}: fetch error {e}")
        failed.append(ticker)

if failed:
    print(f"\nMissing/failed ({len(failed)}): {failed}")

usable = [t for t in TICKERS if t in df_by_ticker]
print(f"\n{len(usable)} tickers with clipped IBKR data; generating candidates...")

candidates, price_by_ticker, trend_quality_by_ticker = generate_candidates(
    tickers=usable,
    strategy_name=STRATEGY,
    vol_filter_ok=True,
    workers=1,                     # must be 1 when df_by_ticker provided
    use_seasonal_volume=True,
    source="ibkr",
    df_by_ticker=df_by_ticker,
    use_persistent_cache=False,    # don't pollute real HMM cache
)

# Apply same start-date filter as live_sim main()
arb_cutoff = pd.Timestamp(ARBITRATION_START)
candidates_filtered = [c for c in candidates if c.date_opened.tz_localize(None) >= arb_cutoff]
print(f"Candidates: {len(candidates)} total, {len(candidates_filtered)} after {ARBITRATION_START} cutoff")

result = arbitrate(
    candidates=candidates_filtered,
    initial_cash=INITIAL_CASH,
    trade_cost=TRADE_COST,
    cost_model_name=COST_MODEL,
    price_by_ticker=price_by_ticker,
)

executed = result["executed"]
equity   = result["equity_curve"]

print(f"\n{'='*60}")
print(f"Admitted: {result['n_admitted']}/{result['n_candidates']}  "
      f"(cash-rejected: {result['n_rejected_cash']}, kelly<=0: {result['n_rejected_kelly']})")
print(f"Final cash: £{result['final_cash']:,.2f}")

if equity:
    df_eq = pd.DataFrame(equity)
    df_eq["date"] = pd.to_datetime(df_eq["date"])
    final_val = df_eq["portfolio_value"].iloc[-1]
    pnl = final_val - INITIAL_CASH
    print(f"Final portfolio: £{final_val:,.2f}  (P&L £{pnl:+,.2f})")
    print(f"Max drawdown: {df_eq['portfolio_value'].pct_change().min()*100:.2f}%")

    # Sharpe: same methodology as BACKTEST_LOG (inter-event returns × sqrt(252))
    pv = df_eq["portfolio_value"].values
    rets = np.diff(pv) / pv[:-1]
    if rets.std() > 0:
        sharpe = (rets.mean() / rets.std()) * math.sqrt(252)
        sortino_neg = rets[rets < 0]
        sortino = (rets.mean() / sortino_neg.std()) * math.sqrt(252) if len(sortino_neg) > 1 else np.nan
        print(f"Sharpe (inter-event x sqrt252): {sharpe:.3f}")
        print(f"Sortino (inter-event x sqrt252): {sortino:.3f}")
    else:
        print("Sharpe: undefined (zero variance)")

    # Write journals
    append_trades(OUT_JOURNAL, executed)
    df_eq.to_csv(OUT_POS, index=False)
    print(f"\nJournal: {OUT_JOURNAL}")
    print(f"Position summary: {OUT_POS}")
else:
    print("No equity curve produced.")
