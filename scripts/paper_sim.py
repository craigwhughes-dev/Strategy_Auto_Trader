"""Paper-trade simulation: run the consolidated engine for a small ticker set,
filter results to a start date, aggregate hourly bars to daily closing state,
and write per-ticker CSVs plus a variable-legend table.

Usage:
    uv run python scripts/paper_sim.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from markov_hedge_fund_method.quant_hmm.quant_engine import fetch_hourly
from markov_hedge_fund_method.quant_hmm.consolidated_engine import consolidated_backtest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "SPY", "NVDA", "META"]
START_DATE = "2026-01-12"          # filter output to on/after this date
OUT_DIR = ROOT / "reports" / "paper_sim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Engine defaults (matches run.py / watchlist defaults)
ENGINE_KWARGS = dict(
    entry_prob=0.65,
    exit_prob=0.40,
    stop_loss_pct=0.05,
    take_profit_pct=0.15,
    volume_min_ratio=0.8,
    initial_cash=20_000.0,
    trade_cost=10.0,
    use_kelly=True,
    regime_smooth=24,
    min_hold_bars=48,
    buy_threshold=3.0,
    sell_threshold=-3.0,
    vol_stop_mult=0.0,
    vol_stop_window=20,
    profit_stop_scale=0.0,
    min_stop_pct=0.05,
    trailing_stop=0.0,
    max_hold_days=0,
    exit_on_rsi_reversal=False,
    exit_on_macd_cross=False,
    exit_on_consolidation=False,
    use_sar_stop=False,
)

# ---------------------------------------------------------------------------
# Variable legend — Y = directly used in signal/gate/exit/sizing decisions
#                   N = context / performance tracking only
# ---------------------------------------------------------------------------

VARIABLE_LEGEND = {
    # Market data
    "close":              ("Market data",   "N", "Closing price of the hourly bar"),
    "volume_ratio":       ("Market data",   "Y", "Volume / 100-bar avg; drives volume vote (+1/>1.2, -1/<0.7) and quality gate entry veto (<1.0) and adverse-exit veto (<0.8)"),
    # HMM regime
    "p_bull":             ("HMM regime",    "N", "Raw P(Bull) from HMM forward filter — not used directly; smoothed before use"),
    "p_bear":             ("HMM regime",    "N", "Raw P(Bear) = 1 - p_bull — informational"),
    "p_bull_smooth":      ("HMM regime",    "Y", "Rolling mean of p_bull over last 24 bars; used to discretize hmm_vote and compute regime_signal"),
    "regime_signal":      ("HMM regime",    "Y", "p_bull_smooth - p_bear; used in quality gate entry veto (<0.25) and adverse-exit veto (<-0.20)"),
    "hmm_vote":           ("HMM regime",    "Y", "Discretised regime: -1=Bear (p_bull_smooth<0.40), 0=Sideways, +1=Bull (>=0.65); weight 1.5 in composite vote"),
    # Momentum indicators
    "rsi":                ("Momentum",      "Y", "RSI(14); vote +1 if >=50 or recently crossed above 50, -1 if <40 or recently crossed below 40; weight 1.5"),
    "rsi_x_above_50":     ("Momentum",      "Y", "RSI crossed above 50 in last 4 bars; used in RSI vote (+1) and quality gate entry veto"),
    "rsi_x_below_40":     ("Momentum",      "Y", "RSI crossed below 40 in last 4 bars; used in RSI vote (-1) and quality gate adverse-exit veto"),
    "above_sma20":        ("Momentum",      "Y", "Price > 20-bar SMA; combined with above_sma50 for trend vote (weight 1.0) and quality gate checks"),
    "above_sma50":        ("Momentum",      "Y", "Price > 50-bar SMA; combined with above_sma20 for trend vote and quality gate checks"),
    "above_sma200":       ("Momentum",      "Y", "Price > 200-bar SMA; weight 2.0 in vote (strongest filter); also in quality gate entry/exit veto"),
    # Signal outputs
    "signal_flag":        ("Signal",        "Y", "Pre-gate composite vote output: BUY / HOLD / SELL"),
    "signal_score":       ("Signal",        "Y", "Raw weighted vote score (max ~7.5); BUY if >=3.0, SELL if <=-3.0"),
    "gate_flag":          ("Signal",        "Y", "Post-quality-gate flag; may veto BUY -> HOLD or force SELL"),
    "gate_reason":        ("Signal",        "N", "Reason string when gate overrides; empty if gate passes unchanged"),
    # Trade state
    "position":           ("Trade state",   "Y", "Current position fraction (Kelly-sized; 0 = flat)"),
    "trade_event":        ("Trade state",   "Y", "BUY / SELL / empty — action taken this bar"),
    "sell_reason":        ("Trade state",   "Y", "Why the exit fired (stop_loss, take_profit, signal, etc.)"),
    "entry_price":        ("Trade state",   "Y", "Price at which position was opened"),
    "stop_level":         ("Trade state",   "Y", "Hard stop price (entry × (1 - stop_loss_pct=5%))"),
    "target_level":       ("Trade state",   "Y", "Hard take-profit price (entry × (1 + take_profit_pct=15%))"),
    "days_in_trade":      ("Trade state",   "Y", "Hourly bars held; gate allows SELL only after min_hold_bars=48"),
    "peak_since_entry":   ("Trade state",   "Y", "Highest close since entry; used to compute trailing-stop drawdown"),
    # Sizing
    "kelly_fraction":     ("Sizing",        "Y", "Current Kelly estimate; position = max(2%, kelly_fraction)"),
    # Returns
    "bar_return":         ("Performance",   "N", "Raw bar log-return (buy-and-hold baseline)"),
    "strategy_return":    ("Performance",   "N", "Position × bar_return — strategy P&L this bar"),
    "strategy_equity":    ("Performance",   "N", "Cumulative strategy equity index"),
    "bh_equity":          ("Performance",   "N", "Cumulative buy-and-hold equity index"),
    "portfolio_value":    ("Performance",   "N", "Simulated portfolio value in GBP/USD"),
}


def _hourly_to_daily(detail: pd.DataFrame) -> pd.DataFrame:
    """Collapse hourly detail to one row per calendar day (last bar of each day)."""
    detail = detail.copy()
    detail.index = pd.to_datetime(detail.index)
    daily = detail.groupby(detail.index.date).last()
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    return daily


def _interleave_strat_columns(daily: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *daily* with a 'Strat' (Y/N) column after each data column.

    The Y/N value is constant for the whole CSV — it comes from VARIABLE_LEGEND
    and indicates whether that variable is part of the current strategy.
    """
    col_data: dict[str, object] = {}
    ordered: list[str] = []
    for col in daily.columns:
        col_data[col] = daily[col]
        ordered.append(col)
        strat_val = VARIABLE_LEGEND.get(col, ("", "N", ""))[1]
        strat_key = f"__s_{col}"
        col_data[strat_key] = strat_val
        ordered.append(strat_key)
    result = pd.DataFrame(col_data, index=daily.index)[ordered]
    result.columns = [
        "Strat" if c.startswith("__s_") else c
        for c in result.columns
    ]
    return result


def _write_legend(path: Path) -> None:
    rows = []
    for col, (group, in_strategy, description) in VARIABLE_LEGEND.items():
        rows.append({
            "variable": col,
            "group": group,
            "in_strategy": in_strategy,
            "description": description,
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  legend -> {path.name}")


def run_ticker(ticker: str) -> pd.DataFrame | None:
    print(f"\n[{ticker}] fetching hourly data...")
    df = fetch_hourly(ticker, period="730d")
    if df is None or df.empty:
        print(f"  ERROR: no data for {ticker}")
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    print(f"  {len(df)} bars  {df.index.min()} to {df.index.max()}")

    print(f"  running consolidated backtest...")
    bt = consolidated_backtest(df, **ENGINE_KWARGS)
    if bt["n_bars"] == 0:
        print(f"  insufficient data for {ticker}")
        return None

    detail: pd.DataFrame = bt["detail"]

    # Filter to simulation start date (index is tz-aware from yfinance)
    detail.index = pd.to_datetime(detail.index)
    start_ts = pd.Timestamp(START_DATE)
    if detail.index.tz is not None:
        start_ts = start_ts.tz_localize(detail.index.tz)
    detail = detail[detail.index >= start_ts]
    if detail.empty:
        print(f"  no bars after {START_DATE} for {ticker}")
        return None

    # Add portfolio_value, strategy_equity, bh_equity if present in detail
    for col in ("strategy_equity", "bh_equity", "portfolio_value"):
        if col not in detail.columns:
            detail[col] = float("nan")

    daily = _hourly_to_daily(detail)

    # Summary
    trades = daily["trade_event"].isin(["BUY", "SELL"]).sum()
    buys   = (daily["trade_event"] == "BUY").sum()
    sells  = (daily["trade_event"] == "SELL").sum()
    final_pv = daily["portfolio_value"].dropna().iloc[-1] if "portfolio_value" in daily.columns else float("nan")
    print(f"  {len(daily)} days | {buys} buys, {sells} sells | final portfolio: £{final_pv:,.0f}")

    out_path = OUT_DIR / f"{ticker}_daily_state.csv"
    _interleave_strat_columns(daily).to_csv(out_path)
    print(f"  written -> {out_path.name}")

    return daily


def main() -> None:
    print(f"Paper-trade simulation: {START_DATE} to present")
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Output: {OUT_DIR}")
    print("=" * 60)

    summaries = []
    for ticker in TICKERS:
        daily = run_ticker(ticker)
        if daily is None:
            continue
        last = daily.iloc[-1]
        summaries.append({
            "ticker": ticker,
            "days": len(daily),
            "buys": int((daily["trade_event"] == "BUY").sum()),
            "sells": int((daily["trade_event"] == "SELL").sum()),
            "last_gate_flag": str(last.get("gate_flag", "")),
            "last_position": float(last.get("position", 0.0)),
            "last_kelly": float(last.get("kelly_fraction", 0.1)),
            "last_p_bull_smooth": float(last.get("p_bull_smooth", 0.0)),
            "final_portfolio": float(last.get("portfolio_value", 0.0)),
        })

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_path = OUT_DIR / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary -> {summary_path.name}")
        print(summary_df.to_string(index=False))

    _write_legend(OUT_DIR / "variable_legend.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
