"""In-memory, process-scoped cache for the two yfinance fetches every
(ticker, strategy) pass repeats: fetch_hourly() and volatility_profile().

A multi-strategy sweep (full_scan_all_strategies.py, live_sim.py) runs every
strategy in-process against the same ticker list, re-issuing both fetches
once per strategy with zero reuse. This cache makes the 2nd..Nth strategy's
fetch a dict lookup instead of a network round-trip.

Deliberately in-memory only, not disk-persisted: full_scan_all_strategies.py
already runs every strategy in one process (see full_scan_all_strategies.py),
so there's no cross-process reuse to gain, and skipping disk persistence
avoids a staleness/corruption window (e.g. a stock split mid-sweep caching
pre-split prices for some strategies and post-split for others).

fetch_hourly()/volatility_profile() themselves are untouched — callers that
want current behavior (single-ticker CLI runs, live trading paths) keep
calling them directly.
"""

from __future__ import annotations

import os
import pandas as pd

from ..quant_hmm.quant_engine import fetch_hourly
from ..quant_hmm.vol_screen import volatility_profile

_cache: dict[tuple, object] = {}


def clear_cache() -> None:
    """Drop all cached entries — call between test cases."""
    _cache.clear()


def fetch_hourly_cached(ticker: str, period: str = "730d"):
    """fetch_hourly(), memoized per (ticker, period) for this process's lifetime.

    A None/empty result (fetch failure) is never cached — a transient network
    error must not permanently poison every later strategy's fetch for this
    ticker, so those calls retry instead of reading a stale failure.
    """
    key = ("hourly", ticker, period)
    cached = _cache.get(key)
    if cached is None:
        cached = fetch_hourly(ticker, period=period)
        if cached is not None and not cached.empty:
            _cache[key] = cached
    return cached


def volatility_profile_cached(ticker: str, period: str = "2y"):
    """volatility_profile(), memoized per (ticker, period) for this process's lifetime.

    A None result (fetch/compute failure) is never cached — see
    fetch_hourly_cached's docstring for why.
    """
    key = ("daily", ticker, period)
    cached = _cache.get(key)
    if cached is None:
        cached = volatility_profile(ticker, period=period)
        if cached is not None:
            _cache[key] = cached
    return cached


def fetch_hourly_stooq(ticker: str, period: str = "730d") -> pd.DataFrame | None:
    """Fetch hourly OHLCV data from local Stooq CSV files.

    Reads from data/stooq_raw/ directory structure:
      data/stooq_raw/data/hourly/us/{ticker}.txt
      data/stooq_raw/data/hourly/gb/{ticker}.txt (for .L suffix)

    Stooq CSV format: Date,Open,High,Low,Close,Volume
    Returns DataFrame with datetime index, same schema as yfinance.

    Hard-fails if ticker not found (no fallback to yfinance).
    """
    stooq_base = os.path.join(os.path.dirname(__file__), "..", "..", "data", "stooq_raw")

    # Normalize ticker: .L suffix maps to gb/ directory, others to us/
    if ticker.endswith(".L"):
        clean_ticker = ticker[:-2].lower()  # Remove .L, lowercase for filename
        market = "gb"
    else:
        clean_ticker = ticker.lower()  # Lowercase for filename
        market = "us"

    csv_path = os.path.join(stooq_base, "data", "hourly", market, f"{clean_ticker}.txt")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Stooq file not found: {csv_path} (ticker: {ticker})")

    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
        if df.empty:
            raise ValueError(f"Stooq file empty: {csv_path}")

        # Ensure columns match yfinance schema [Open, High, Low, Close, Volume]
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing columns in {csv_path}. Expected {required_cols}, got {list(df.columns)}")

        # Keep only required columns in yfinance order
        df = df[required_cols]

        # Sort by date ascending (ensure chronological order)
        df = df.sort_index()

        return df
    except Exception as e:
        raise RuntimeError(f"Failed to parse Stooq file {csv_path}: {e}") from e
