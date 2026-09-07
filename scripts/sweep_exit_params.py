"""Sweep exit parameters on a sample of synthetic tickers to find directional signal.

Each parameter is swept independently (others at current baseline) using the
warmed synthetic HMM cache — much faster than a full live_sim run.

Usage:
    uv run python scripts/sweep_exit_params.py
    uv run python scripts/sweep_exit_params.py --tickers AAPL MSFT NFLX --n-tickers 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from Strategy_Auto_Trader.quant_hmm.consolidated_engine import consolidated_backtest
from Strategy_Auto_Trader.quant_hmm.ticker_ranking import resolve_strategy
from Strategy_Auto_Trader.strategy.optimised_new import OptimisedNewEntry, OptimisedNewExit
from Strategy_Auto_Trader.plugins.persistent_hmm import PersistentHMMRegimeModel
from Strategy_Auto_Trader.synthetic_backtest_data.generate import (
    load_synthetic_hourly,
    SYNTHETIC_HMM_CACHE_DIR,
)
from Strategy_Auto_Trader.plugins.costs import make_cost_model

_HOURLY_DIR = Path("data_synthetic/hourly")
_STRATEGY = "optimised_new"
_START = "2000-01-01"

# Top tickers from the no-vol-filter 26yr run (selected set, alphabetical sample)
_DEFAULT_TICKERS = [
    "ANET", "AWK", "BG", "BKR", "BLK", "CF", "CI", "CME", "COO", "CPT",
    "CRL", "CSX", "DVA", "EFX", "ELV", "ES", "ETN", "FAST", "FIX", "FRT",
]

# Current baseline (optimised_new as of 2026-09-07)
_BASELINE = dict(
    stop_loss_pct=0.08,
    vol_stop_mult=0.5,
    vol_stop_window=20,
    profit_stop_scale=0.30,
    min_stop_pct=0.03,
    min_hold_bars=48,
    take_profit_pct=999.0,
    trailing_stop=0.0,
    max_hold_days=0,
    breakeven_trailing=False,  # Plan A NEGATIVE (swept 2026-09-07) — keep off
    min_entry_score=7.0,       # adopted 2026-09-07: real IBKR sweep +0.75% vs +0.12%
)

_SWEEPS: dict[str, list] = {
    "profit_stop_scale":  [0.0, 0.10, 0.20, 0.30, 0.50],
    "vol_stop_mult":      [0.5, 1.0, 1.5, 2.0, 3.0],
    "stop_loss_pct":      [0.04, 0.06, 0.08, 0.10],
    "min_hold_bars":      [24, 48, 96, 168],
    "breakeven_trailing": [False, True],
    "min_entry_score":    [6.0, 7.0, 8.0],  # 6.0 = effectively off (= buy_threshold)
}

# Plan B: regime-forced exit minimum hold bars before bypass kicks in
_REGIME_FORCED_SWEEP: list[int | None] = [None, 6, 12, 24, 48]

_COST_MODEL = make_cost_model("ibkr_tiered_spread", "SPY", 1.0)


def _run_backtest(
    df: pd.DataFrame,
    hmm_path: Path,
    min_hold_bars_regime_exit: int | None = None,
    **overrides,
) -> dict | None:
    params = {**_BASELINE, **overrides}
    regime_model = PersistentHMMRegimeModel(
        hmm_path, dates=df.index, closes=df["Close"].values,
    )
    entry_s = OptimisedNewEntry(
        vol_filter_ok=True,
        min_entry_score=params.get("min_entry_score"),
    )
    exit_s = OptimisedNewExit(
        stop_loss_pct=params["stop_loss_pct"],
        vol_stop_mult=params["vol_stop_mult"],
        vol_stop_window=params["vol_stop_window"],
        profit_stop_scale=params["profit_stop_scale"],
        min_stop_pct=params["min_stop_pct"],
        max_hold_days=params["max_hold_days"],
        breakeven_trailing=params.get("breakeven_trailing", False),
    )
    if min_hold_bars_regime_exit is not None:
        exit_s.min_hold_bars_regime_exit = min_hold_bars_regime_exit
    try:
        bt = consolidated_backtest(
            df,
            regime_model=regime_model,
            entry_strategy=entry_s,
            exit_strategy=exit_s,
            min_hold_bars=params["min_hold_bars"],
            trailing_stop=params["trailing_stop"],
            take_profit_pct=params["take_profit_pct"],
            cost_model=_COST_MODEL,
        )
    except Exception:
        return None
    detail = bt.get("detail", pd.DataFrame())
    if detail.empty:
        return None

    from Strategy_Auto_Trader.output.journal import extract_trades_from_detail
    trades = extract_trades_from_detail("X", detail.reset_index(), strategy=_STRATEGY)
    if not trades:
        return None

    rets = np.array([t.return_pct for t in trades])
    peaks = np.array([t.peak_gain for t in trades])
    win_mask = rets > 0
    n = len(rets)
    win_rate = win_mask.mean()
    mean_ret = rets.mean()
    peak_capture = (rets[win_mask] / peaks[win_mask]).mean() if win_mask.any() else 0.0
    sharpe = (mean_ret / rets.std() * np.sqrt(252 * 7)) if rets.std() > 0 else 0.0
    return dict(n=n, win_rate=win_rate, mean_ret=mean_ret,
                peak_capture=peak_capture, sharpe=sharpe)


def _aggregate(results: list[dict]) -> dict:
    if not results:
        return {}
    keys = [k for k in results[0] if k != "n"]
    agg = {k: np.mean([r[k] for r in results]) for k in keys}
    agg["n_total"] = sum(r["n"] for r in results)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--n-tickers", type=int, default=20)
    args = parser.parse_args()

    tickers = args.tickers or _DEFAULT_TICKERS[:args.n_tickers]

    # Pre-load synthetic data — skip tickers without files
    ticker_data: dict[str, tuple[pd.DataFrame, Path]] = {}
    for t in tickers:
        df = load_synthetic_hourly(t, hourly_dir=_HOURLY_DIR)
        if df is None:
            continue
        df = df.loc[_START:]
        hmm_path = SYNTHETIC_HMM_CACHE_DIR / f"{t}.pkl"
        if not hmm_path.exists():
            continue
        ticker_data[t] = (df, hmm_path)

    print(f"\nExit parameter sweep — {len(ticker_data)} tickers, strategy={_STRATEGY}")
    print(f"Baseline: stop={_BASELINE['stop_loss_pct']:.0%}  "
          f"vol_mult={_BASELINE['vol_stop_mult']}  "
          f"profit_scale={_BASELINE['profit_stop_scale']}  "
          f"min_hold={_BASELINE['min_hold_bars']}bars\n")

    for param, values in _SWEEPS.items():
        print(f"{'-'*72}")
        print(f"Sweeping: {param}")
        print(f"  {'Value':<12} {'Trades':>7} {'WinRate':>8} {'MeanRet':>9} {'PeakCapt':>10} {'Sharpe':>8}")
        for val in values:
            ticker_results = []
            for t, (df, hmm_path) in ticker_data.items():
                r = _run_backtest(df, hmm_path, **{param: val})
                if r:
                    ticker_results.append(r)
            if not ticker_results:
                print(f"  {val!s:<12} {'no data':>7}")
                continue
            agg = _aggregate(ticker_results)
            marker = " << baseline" if val == _BASELINE.get(param) else ""
            print(f"  {val!s:<12} {agg['n_total']:>7} "
                  f"{agg['win_rate']:>8.1%} "
                  f"{agg['mean_ret']:>+9.2%} "
                  f"{agg['peak_capture']:>10.1%} "
                  f"{agg['sharpe']:>+8.3f}{marker}")
        print()

    print(f"{'-'*72}")
    print("Plan B: regime-forced exit (min_hold_bars_regime_exit sweep)")
    print("  None = feature off (baseline behavior)")
    print(f"  {'Value':<12} {'Trades':>7} {'WinRate':>8} {'MeanRet':>9} {'PeakCapt':>10} {'Sharpe':>8}")
    for val in _REGIME_FORCED_SWEEP:
        ticker_results = []
        for t, (df, hmm_path) in ticker_data.items():
            r = _run_backtest(df, hmm_path, min_hold_bars_regime_exit=val)
            if r:
                ticker_results.append(r)
        if not ticker_results:
            print(f"  {val!s:<12} {'no data':>7}")
            continue
        agg = _aggregate(ticker_results)
        label = "None (off)" if val is None else str(val)
        print(f"  {label:<12} {agg['n_total']:>7} "
              f"{agg['win_rate']:>8.1%} "
              f"{agg['mean_ret']:>+9.2%} "
              f"{agg['peak_capture']:>10.1%} "
              f"{agg['sharpe']:>+8.3f}")
    print()


if __name__ == "__main__":
    main()
