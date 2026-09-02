"""Warm the synthetic-data HMM cache for a fixed date window.

Loads each ticker's already-generated synthetic hourly CSV (see generate.py),
slices it to [start_date, end_date], and runs it through
quant_hmm.ticker_ranking.run_ticker_backtest with use_persistent_cache=True
and hmm_cache_dir=SYNTHETIC_HMM_CACHE_DIR — the isolated cache dir so this
can never touch/corrupt a real ticker's on-disk HMM state (see generate.py's
docstring).

strategy_name only needs to be *some* strategy whose weights give "hmm" a
nonzero weight (default.py's does) — consolidated_engine's
skip_unused_indicators=True default means the HMM step is skipped entirely
for every bar when the chosen strategy doesn't use the hmm signal, which
would silently warm nothing. The regime_model's own output does not depend
on which such strategy is chosen; "default" is used purely to satisfy that
gate.

Cache-window ordering (see PersistentHMMRegimeModel's docstring): this
window becomes the cache's permanent start-of-history anchor. A future run
starting *earlier* than start_date invalidates and fully recomputes this
window — only extending *later* than end_date is free. Pick start_date with
enough lookback before the period you actually care about that
min_train_bars (default 500) of HMM warmup completes first — otherwise the
start of your window of interest has no regime signal yet (NaN/skipped).
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ..quant_hmm.ticker_ranking import run_ticker_backtest
from . import generate
from .generate import SYNTHETIC_HMM_CACHE_DIR

logger = logging.getLogger(__name__)

_SYNTHETIC_HOURLY_DIR = Path(__file__).resolve().parent.parent.parent / "data_synthetic" / "hourly"
_UNIVERSE_FILE = Path(__file__).resolve().parent.parent.parent / "config" / "universe_sp_ftse.json"

_DEFAULT_STRATEGY = "default"


def _load_synthetic_hourly(ticker: str) -> pd.DataFrame | None:
    """Thin wrapper over generate.load_synthetic_hourly, reading
    _SYNTHETIC_HOURLY_DIR at call time so tests can monkeypatch it."""
    return generate.load_synthetic_hourly(ticker, hourly_dir=_SYNTHETIC_HOURLY_DIR)


def warm_hmm_cache_for_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    strategy_name: str = _DEFAULT_STRATEGY,
) -> dict:
    """Step one ticker's regime model through [start_date, end_date] of its
    synthetic hourly data and persist the result to SYNTHETIC_HMM_CACHE_DIR.
    Returns a status dict rather than raising."""
    try:
        df = _load_synthetic_hourly(ticker)
        if df is None:
            return {"ticker": ticker, "status": "no_data",
                     "note": "no synthetic hourly CSV — run generate.py first"}

        window = df.loc[start_date:end_date]
        if window.empty:
            return {"ticker": ticker, "status": "no_data",
                     "note": f"no bars in [{start_date}, {end_date}]"}

        detail, _df = run_ticker_backtest(
            ticker, strategy_name, df=window,
            use_persistent_cache=True, hmm_cache_dir=SYNTHETIC_HMM_CACHE_DIR,
        )
        if detail is None:
            return {"ticker": ticker, "status": "no_data",
                     "note": "insufficient bars for min_train_bars"}
        return {"ticker": ticker, "status": "ok", "n_bars": len(window)}
    except Exception as exc:
        return {"ticker": ticker, "status": "error", "note": f"{type(exc).__name__}: {exc}"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=None,
                         help="Ticker list (default: config/universe_sp_ftse.json's full universe)")
    parser.add_argument("--start-date", required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--strategy", default=_DEFAULT_STRATEGY)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    tickers = args.tickers
    if tickers is None:
        data = json.loads(_UNIVERSE_FILE.read_text(encoding="utf-8"))
        tickers = data["tickers"]

    def _report(result: dict) -> None:
        if result["status"] == "ok":
            print(f"{result['ticker']}: warmed {result['n_bars']} bars")
        else:
            logger.warning("%s: %s (%s)", result["ticker"], result["status"], result.get("note", ""))

    if args.workers == 1:
        for ticker in tickers:
            _report(warm_hmm_cache_for_ticker(ticker, args.start_date, args.end_date, args.strategy))
        return

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(warm_hmm_cache_for_ticker, t, args.start_date, args.end_date, args.strategy): t
            for t in tickers
        }
        for future in as_completed(futures):
            _report(future.result())


if __name__ == "__main__":
    main()
