"""Consolidated engine regression check: run on a basket of tickers and report P&L metrics.

Previously compared daily vs consolidated engine side-by-side.  Now that the daily
engine has been retired, this script runs the consolidated engine on live data and
reports the key stats as a snapshot, so regressions across code changes can be detected
by diffing the JSON output over time.

Usage
-----
    uv run python scripts/regression_compare.py [--tickers AAPL MSFT] [--out results.json]

Defaults to a small representative basket drawn from config/watchlist.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _watchlist_sample(n: int = 5) -> list[str]:
    wl_path = REPO_ROOT / "config" / "watchlist.json"
    if not wl_path.exists():
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
    with open(wl_path) as f:
        data = json.load(f)
    tickers = data.get("tickers", [])[:n]
    return tickers if tickers else ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]


def _fetch_hourly(ticker: str, period: str = "730d") -> pd.DataFrame | None:
    from Strategy_Auto_Trader.quant_hmm.quant_engine import fetch_hourly
    return fetch_hourly(ticker, period=period)


def _run_consolidated_engine(df: pd.DataFrame) -> dict:
    from Strategy_Auto_Trader.quant_hmm.consolidated_engine import consolidated_backtest
    result = consolidated_backtest(
        df,
        entry_prob=0.65, exit_prob=0.40,
        stop_loss_pct=0.05, take_profit_pct=0.15,
        volume_min_ratio=0.8,
        min_train_bars=500, hmm_refit_bars=500,
        initial_cash=20_000.0, trade_cost=10.0,
        use_kelly=True, kelly_lookback=20,
        regime_smooth=24, min_hold_bars=48,
        buy_threshold=3.0, sell_threshold=-3.0,
        vol_stop_mult=0.5, vol_stop_window=20,
        exit_on_rsi_reversal=True,
    )
    return {
        "n_trades": result["n_buys"],
        "total_return": result["total_return_strategy"],
        "sharpe": result["sharpe_strategy"],
        "max_dd": result["max_drawdown_strategy"],
        "final_portfolio": result["final_portfolio"],
    }


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def run_comparison(tickers: list[str], out_path: Path | None) -> None:
    results = {}
    header = f"{'Ticker':<8}  {'n_trades':>8}  {'return':>8}  {'sharpe':>8}  {'max_dd':>8}  {'portfolio':>10}"
    sep = "-" * len(header)

    print(header)
    print(sep)

    for ticker in tickers:
        print(f"\n=== {ticker} ===")
        row: dict = {"ticker": ticker}

        try:
            df_hourly = _fetch_hourly(ticker)
            if df_hourly is None or len(df_hourly) < 550:
                print(f"  SKIP (insufficient hourly data: "
                      f"{len(df_hourly) if df_hourly is not None else 0} bars)")
                row["consolidated"] = None
            else:
                cs = _run_consolidated_engine(df_hourly)
                row["consolidated"] = cs
                print(f"  {_fmt(cs['n_trades']):>8}  {_fmt(cs['total_return']):>8}  "
                      f"{_fmt(cs['sharpe']):>8}  {_fmt(cs['max_dd']):>8}  {_fmt(cs['final_portfolio']):>10}")
        except Exception as e:
            print(f"  ERROR — {e}")
            row["consolidated"] = {"error": str(e)}

        results[ticker] = row

    print(f"\n{sep}")
    if out_path:
        def _serial(o):
            if isinstance(o, float):
                return None if not np.isfinite(o) else o
            return o
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=_serial)
        print(f"\nResults written to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run consolidated engine on a basket of tickers")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Tickers to run (default: 5-ticker watchlist sample)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional path to write JSON results")
    args = parser.parse_args()

    tickers = args.tickers or _watchlist_sample(5)
    print(f"Consolidated engine snapshot: {len(tickers)} tickers: {', '.join(tickers)}\n")
    run_comparison(tickers, args.out)


if __name__ == "__main__":
    main()
