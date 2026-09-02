"""Generate synthetic hourly OHLCV data from real daily closes.

Method: fetch a ticker's real daily closes — from the local Stooq dump
(stooq_daily.py) if present, else IBKR (broker.ibkr_data; no yfinance
anywhere in this module) — compute a rolling daily volatility, then for
each consecutive pair of real daily closes synthesize
`bars_per_day` hourly bars via a Brownian bridge (bridge.py) — a random
intraday path that is forced to land exactly on the next real daily close.
Only the shape of each day is synthetic; the day-to-day closes themselves
are always real.

`bars_per_day=7` default matches quant_engine._HOURS_PER_YEAR=1700
(1700/252 trading days per year ~= 6.75, rounds to 7) so a synthetic hourly
series has the same bars-per-year density the rest of the codebase assumes
(Sharpe/Sortino annualization, HMM bar-count expectations, etc).

Volume: the real daily total (from the same daily source as the closes) is
distributed across each day's synthetic bars proportional to bar-level
price movement — see bridge.py's docstring for the method and its
limitations.

Output is written to a directory entirely separate from the real IBKR
hourly/daily caches (data/cache/ibkr_hourly/, data/cache/ibkr_daily/) and
the local Stooq daily dump (data/cache/stooq_daily/) so synthetic and real
data can never get mixed on disk. Default output dir: data_synthetic/hourly/
(a top-level sibling of data/, not nested inside it — deliberately outside
the data/ tree so a wholesale "wipe generated data" on data/ can never
delete synthetic output).

Deferred: wiring this into live_sim.py automatically. For now, the output
DataFrame is shaped exactly like broker.ibkr_data.IBKRDataClient.fetch_hourly's
return contract (tz-aware index, Open/High/Low/Close/Volume columns), so it
can already be passed as
    generate_candidates(df_by_ticker={ticker: df}, use_persistent_cache=False)
— the same df_by_ticker override monte_carlo_live_sim.py already uses for
synthetic paths — once that wiring is deliberately added.

HMM-cache warning for that future wiring: the persistent HMM cache
(data/cache/hmm_cache/<ticker>.pkl, built by quant_hmm.ticker_ranking) is
keyed by ticker name alone, real or synthetic. Any backtest/live_sim run
against this module's output must either pass use_persistent_cache=False
(same invariant Monte Carlo already enforces, see monte_carlo.py's
docstring / .claude/rules/cli.md) or, to keep the caching speedup,
hmm_cache_dir=SYNTHETIC_HMM_CACHE_DIR (below) — generate_candidates() takes
both params. Using the real cache path for a synthetic run would silently
corrupt the real ticker's on-disk HMM state.
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from ..broker.ibkr_data import IBKRDataClient
from ..core.atomic_io import atomic_write_csv
from . import bridge, stooq_daily, vol

logger = logging.getLogger(__name__)

_DEFAULT_VOL_WINDOW = 21
_DEFAULT_BARS_PER_DAY = 7

# Pass as hmm_cache_dir to live_sim.py's --synthetic-data-dir /
# generate_candidates(hmm_cache_dir=...) to keep HMM caching enabled for
# synthetic-data runs without touching the real per-ticker cache at
# data/cache/hmm_cache/ — see this module's docstring.
SYNTHETIC_HMM_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data_synthetic" / "hmm_cache"

DEFAULT_HOURLY_DIR = Path(__file__).resolve().parent.parent.parent / "data_synthetic" / "hourly"


def load_synthetic_hourly(ticker: str, hourly_dir: Path | None = None) -> pd.DataFrame | None:
    """Load a ticker's already-generated synthetic hourly CSV (written by
    generate_synthetic_hourly / main()). None if no file exists for this
    ticker. Same tz-localize-if-naive contract as broker.ibkr_data's cache
    loader."""
    hourly_dir = DEFAULT_HOURLY_DIR if hourly_dir is None else Path(hourly_dir)
    path = hourly_dir / f"{ticker.replace('/', '-')}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.empty:
        return None
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def generate_synthetic_hourly(
    ticker: str,
    vol_window: int = _DEFAULT_VOL_WINDOW,
    bars_per_day: int = _DEFAULT_BARS_PER_DAY,
    seed: int | None = None,
    client_id: int = 3,
) -> pd.DataFrame | None:
    """Fetch real daily closes for `ticker` — local Stooq dump first (fast,
    deeper UK history, see stooq_daily.py), IBKR fetch_daily as fallback for
    tickers outside the Stooq snapshot — and synthesize an hourly OHLCV
    series spanning the same history. Returns None if both sources fail."""
    daily = stooq_daily.load_stooq_daily(ticker)
    if daily is None or daily.empty:
        daily = IBKRDataClient(client_id=client_id).fetch_daily(ticker, period="max")
    if daily is None or daily.empty:
        return None

    sigma = vol.rolling_daily_vol(daily["Close"], window=vol_window)
    rng = np.random.default_rng(seed)

    frames: list[pd.DataFrame] = []
    closes = daily["Close"].to_numpy()
    volumes = daily["Volume"].to_numpy() if "Volume" in daily.columns else None
    timestamps = daily.index
    for i in range(1, len(daily)):
        day_sigma = sigma.iloc[i]
        if not np.isfinite(day_sigma) or day_sigma <= 0:
            continue
        day_df = bridge.build_hourly_ohlcv_for_day(
            prev_close=closes[i - 1],
            next_close=closes[i],
            sigma=day_sigma,
            n_bars=bars_per_day,
            rng=rng,
            daily_volume=volumes[i] if volumes is not None else None,
        )
        end = timestamps[i]
        day_df.index = pd.date_range(end=end, periods=bars_per_day, freq="1h")
        frames.append(day_df)

    if not frames:
        return None
    return pd.concat(frames).sort_index()


def _generate_and_write_worker(
    ticker: str, vol_window: int, bars_per_day: int, seed: int | None, output_dir: Path,
) -> dict:
    """Top-level module worker function for ProcessPoolExecutor — generate
    one ticker's synthetic hourly series and write it, returning a status
    dict rather than raising, so one ticker's failure doesn't kill the pool
    (mirrors full_scan.py's _scan_ticker_worker).

    Derives a per-process IBKR client_id (os.getpid()-based) for the
    IBKR-fallback path in generate_synthetic_hourly — under --workers > 1,
    every worker process hitting the fallback with the same hardcoded
    client_id collided on the same TWS connection slot and timed out
    (observed live: 5/6 Stooq-missing FTSE tickers failed this way in one
    run). One process = one PID = one client_id avoids the collision."""
    client_id = 1000 + (os.getpid() % 9000)
    try:
        df = generate_synthetic_hourly(
            ticker, vol_window=vol_window, bars_per_day=bars_per_day, seed=seed,
            client_id=client_id,
        )
        if df is None:
            return {"ticker": ticker, "status": "no_data"}
        path = output_dir / f"{ticker.replace('/', '-')}.csv"
        atomic_write_csv(path, df)
        return {"ticker": ticker, "status": "ok", "n_bars": len(df), "path": str(path)}
    except Exception as exc:
        return {"ticker": ticker, "status": "error", "note": f"{type(exc).__name__}: {exc}"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--vol-window", type=int, default=_DEFAULT_VOL_WINDOW)
    parser.add_argument("--bars-per-day", type=int, default=_DEFAULT_BARS_PER_DAY)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--workers", type=int, default=1,
                         help="Parallel worker processes, one ticker per task (1 = sequential, default: 1).")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(__file__).resolve().parent.parent.parent / "data_synthetic" / "hourly"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    def _report(result: dict) -> None:
        if result["status"] == "ok":
            print(f"{result['ticker']}: {result['n_bars']} synthetic hourly bars -> {result['path']}")
        elif result["status"] == "no_data":
            logger.warning("generate_synthetic_hourly(%s) failed, skipping", result["ticker"])
        else:
            logger.warning("generate_synthetic_hourly(%s) errored: %s", result["ticker"], result["note"])

    if args.workers == 1:
        for ticker in args.tickers:
            _report(_generate_and_write_worker(
                ticker, args.vol_window, args.bars_per_day, args.seed, output_dir))
        return

    executor = ProcessPoolExecutor(max_workers=args.workers)
    try:
        futures = {
            executor.submit(
                _generate_and_write_worker,
                ticker, args.vol_window, args.bars_per_day, args.seed, output_dir,
            ): ticker
            for ticker in args.tickers
        }
        for future in as_completed(futures):
            _report(future.result())
    except KeyboardInterrupt:
        logger.warning("Interrupted; canceling remaining tickers...")
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown()


if __name__ == "__main__":
    main()
