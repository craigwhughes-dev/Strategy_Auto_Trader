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
