"""Screen a large universe of tickers quickly using the consolidated engine.

Usage:
    uv run python -m Strategy_Auto_Trader.markov_cli.screen [--input scan_tickers.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent


def _screen_one(ticker: str, years: int = 2) -> dict | None:
    """Run a fast consolidated backtest on one ticker and return P&L metrics.

    Uses aggressive settings (small train window, single HMM refit) to keep
    the screen cheap enough to run across a large universe.
    """
    from ..quant_hmm.quant_engine import fetch_hourly
    from ..quant_hmm.consolidated_engine import consolidated_backtest

    df = fetch_hourly(ticker, period="730d")
    if df is None or len(df) < 200:
        return None

    try:
        bt = consolidated_backtest(
            df,
            min_train_bars=150, hmm_refit_bars=50_000,  # fit once, never refit
            regime_smooth=12, min_hold_bars=24,
            buy_threshold=2.0, sell_threshold=-2.0,
            stop_loss_pct=0.05, take_profit_pct=0.15,
            vol_stop_mult=0.5, vol_stop_window=20,
            profit_stop_scale=0.5, min_stop_pct=0.05,
            initial_cash=20_000.0, trade_cost=10.0,
            use_kelly=True,
        )
    except Exception:
        return None

    if bt["n_bars"] == 0:
        return None

    return {
        "ticker": ticker,
        "strategy_return": bt["total_return_strategy"],
        "bh_return": bt["total_return_bh"],
        "sharpe": bt["sharpe_strategy"],
        "sortino": bt.get("sortino_strategy", float("nan")),
        "calmar": bt.get("calmar_strategy", float("nan")),
        "max_dd": bt["max_drawdown_strategy"],
        "final_portfolio": bt["final_portfolio"],
        "pl": bt["total_pl"],
        "n_trades": bt["n_buys"],
        "profitable": bt["total_pl"] > 0,
        "beats_bh": bt["total_return_strategy"] > bt["total_return_bh"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="screen")
    parser.add_argument("--input", type=Path, default=ROOT / "config" / "scan_tickers.json")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    all_tickers = data.get("sp500", []) + data.get("ftse100", [])
    all_tickers = sorted(set(all_tickers))
    total = len(all_tickers)
    print(f"Screening {total} tickers (consolidated engine, fast settings)")

    results = []
    winners = []
    t0 = time.time()

    for i, ticker in enumerate(all_tickers, 1):
        if i % 25 == 0 or i == 1:
            elapsed = time.time() - t0
            rate = elapsed / i if i > 1 else 0
            eta = rate * (total - i)
            print(f"  [{i}/{total}]  {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining...")

        r = _screen_one(ticker)
        if r is None:
            continue
        results.append(r)

        if r["profitable"] or r["beats_bh"]:
            winners.append(r)
            tag = ""
            if r["profitable"] and r["beats_bh"]:
                tag = "PROFIT + BEATS B&H"
            elif r["profitable"]:
                tag = "PROFIT"
            else:
                tag = "BEATS B&H"
            print(f"  + {ticker:<10s}  strat={r['strategy_return']*100:+.1f}%  "
                  f"bh={r['bh_return']*100:+.1f}%  "
                  f"sharpe={r['sharpe']:.2f}  [{tag}]")

    elapsed = time.time() - t0
    print(f"\nDone: {len(results)} analysed, {total - len(results)} skipped, "
          f"{len(winners)} winners in {elapsed:.0f}s")

    out_path = ROOT / "reports" / "screen_winners.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(winners, f, indent=2)
    print(f"Winners saved to {out_path}")

    winners_sorted = sorted(winners, key=lambda r: r["strategy_return"], reverse=True)
    print(f"\n{'Ticker':<10s} {'Strat':>8s} {'B&H':>8s} {'Sharpe':>7s} {'Sortino':>8s} {'Calmar':>7s} {'MaxDD':>7s} {'P&L':>10s} {'Trades':>6s}")
    print("-" * 77)
    for r in winners_sorted:
        def _f2(v):
            return f"{v:.2f}" if np.isfinite(v) else "NaN"
        print(f"{r['ticker']:<10s} {r['strategy_return']*100:>+7.1f}% "
              f"{r['bh_return']*100:>+7.1f}% {_f2(r['sharpe']):>7s} "
              f"{_f2(r.get('sortino', float('nan'))):>8s} {_f2(r.get('calmar', float('nan'))):>7s} "
              f"{r['max_dd']*100:>6.1f}% {r['pl']:>+9,.0f} {r['n_trades']:>6d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
