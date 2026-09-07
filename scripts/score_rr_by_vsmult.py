"""Compare R:R per entry_score at vol_stop_mult=1.0 (baseline) vs 0.5.

Runs consolidated_backtest directly on synthetic data for a sample of tickers,
then breaks trade results down by entry_score bucket. Goal: confirm vol_stop_mult=0.5
shrinks avg_loss for score-6 entries toward the 4.63% break-even target.

Usage:
    uv run python scripts/score_rr_by_vsmult.py
    uv run python scripts/score_rr_by_vsmult.py --n-tickers 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from Strategy_Auto_Trader.quant_hmm.consolidated_engine import consolidated_backtest
from Strategy_Auto_Trader.strategy.optimised_new import OptimisedNewEntry, OptimisedNewExit
from Strategy_Auto_Trader.plugins.persistent_hmm import PersistentHMMRegimeModel
from Strategy_Auto_Trader.synthetic_backtest_data.generate import (
    load_synthetic_hourly,
    SYNTHETIC_HMM_CACHE_DIR,
)
from Strategy_Auto_Trader.plugins.costs import make_cost_model
from Strategy_Auto_Trader.output.journal import extract_trades_from_detail

_HOURLY_DIR = Path("data_synthetic/hourly")
_STRATEGY = "optimised_new"
_START = "2000-01-01"

_DEFAULT_TICKERS = [
    "ANET", "AWK", "BG", "BKR", "BLK", "CF", "CI", "CME", "COO", "CPT",
    "CRL", "CSX", "DVA", "EFX", "ELV", "ES", "ETN", "FAST", "FIX", "FRT",
]

_COST_MODEL = make_cost_model("ibkr_tiered_spread", "SPY", 1.0)

_BASELINE = dict(
    stop_loss_pct=0.08,
    vol_stop_window=20,
    profit_stop_scale=0.30,
    min_stop_pct=0.03,
    min_hold_bars=48,
    take_profit_pct=999.0,
    trailing_stop=0.0,
    max_hold_days=0,
)

_VARIANTS = [
    ("vol_stop_mult=1.0 (baseline)", 1.0),
    ("vol_stop_mult=0.5 (new)",      0.5),
]


def _run_backtest(df: pd.DataFrame, hmm_path: Path, vol_stop_mult: float) -> list | None:
    regime_model = PersistentHMMRegimeModel(
        hmm_path, dates=df.index, closes=df["Close"].values,
    )
    entry_s = OptimisedNewEntry(vol_filter_ok=True)
    exit_s = OptimisedNewExit(
        stop_loss_pct=_BASELINE["stop_loss_pct"],
        vol_stop_mult=vol_stop_mult,
        vol_stop_window=_BASELINE["vol_stop_window"],
        profit_stop_scale=_BASELINE["profit_stop_scale"],
        min_stop_pct=_BASELINE["min_stop_pct"],
        max_hold_days=_BASELINE["max_hold_days"],
    )
    try:
        bt = consolidated_backtest(
            df,
            regime_model=regime_model,
            entry_strategy=entry_s,
            exit_strategy=exit_s,
            min_hold_bars=_BASELINE["min_hold_bars"],
            trailing_stop=_BASELINE["trailing_stop"],
            take_profit_pct=_BASELINE["take_profit_pct"],
            cost_model=_COST_MODEL,
        )
    except Exception:
        return None
    detail = bt.get("detail", pd.DataFrame())
    if detail.empty:
        return None
    return extract_trades_from_detail("X", detail.reset_index(), strategy=_STRATEGY)


def _score_table(trades: list) -> None:
    if not trades:
        print("  (no trades)")
        return

    df = pd.DataFrame({
        "score":    [t.entry_score for t in trades],
        "ret":      [t.return_pct  for t in trades],
        "peak":     [t.peak_gain   for t in trades],
        "exit":     [t.exit_reason for t in trades],
    })

    print(f"  {'Score':<8} {'Trades':>7} {'Win%':>7} {'MeanRet':>9} {'AvgWin':>8} {'AvgLoss':>9} "
          f"{'BreakEven%':>11} {'PeakCapt':>10} {'HardStop%':>10}")
    for score, grp in df.groupby("score"):
        rets = grp["ret"].values
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        peaks = grp["peak"].values
        avg_win = wins.mean() if len(wins) else 0.0
        avg_loss = abs(losses.mean()) if len(losses) else 0.0
        be_wr = avg_loss / (avg_win + avg_loss) if (avg_win + avg_loss) > 0 else float("nan")
        peak_cap = (rets[rets > 0] / grp["peak"].values[rets > 0]).mean() if len(wins) else 0.0
        hard_stop_pct = (grp["exit"].str.contains("rr_stop_loss", na=False)).mean()
        print(f"  {score:<8.0f} {len(rets):>7} {(rets>0).mean():>7.1%} {rets.mean():>+9.2%} "
              f"{avg_win:>+8.2%} {-avg_loss:>+9.2%} "
              f"{be_wr:>11.1%} {peak_cap:>10.1%} {hard_stop_pct:>10.1%}")
    print(f"  {'ALL':<8} {len(df):>7} {(df['ret']>0).mean():>7.1%} {df['ret'].mean():>+9.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--n-tickers", type=int, default=20)
    args = parser.parse_args()

    tickers = args.tickers or _DEFAULT_TICKERS[:args.n_tickers]

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

    print(f"\nR:R by entry_score -- {len(ticker_data)} tickers, strategy={_STRATEGY}")
    print(f"Break-even% = required win rate to break even at current avg_win/avg_loss\n")

    for label, vol_stop_mult in _VARIANTS:
        print(f"{'='*80}")
        print(f"{label}")
        print(f"{'='*80}")
        all_trades = []
        for t, (df, hmm_path) in ticker_data.items():
            trades = _run_backtest(df, hmm_path, vol_stop_mult)
            if trades:
                all_trades.extend(trades)
        _score_table(all_trades)
        print()


if __name__ == "__main__":
    main()
