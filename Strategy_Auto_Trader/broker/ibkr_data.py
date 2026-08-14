"""IBKR historical-bar client via ib_async — an opt-in alternative to
yfinance for `quant_engine.fetch_hourly` (see quant_engine.py's `source`
param). Kept separate from ibkr_adapter.py: that module is for live order
execution, this one is read-only historical data and is safe to exercise
without touching the trading connection.

Uses client_id=2 by default, reserved for data-fetch, so it never collides
with the live daemon's execution connection (client_id=1) on the same TWS
instance — both can run concurrently against one TWS/Gateway.

The on-disk cache (data/cache/ibkr_hourly/) is a growing, incrementally
updated store, not a point-in-time snapshot: fetch_hourly() only ever pages
for the *gap* between the last cached bar and now, merges, and re-saves. A
brand-new ticker with no cache yet still does the old-style bootstrap page
(walking back until either the requested period is covered or IBKR returns
an empty page, i.e. the start of available history) — but a multi-day
daemon outage is just a bigger gap through that same incremental path, no
separate "backfill after downtime" logic needed.

IBKR's reqHistoricalData is duration- and pacing-limited per request, so
years of hourly history requires paging backwards in chunks with throttling
between requests, concatenating until either the requested period/gap is
covered or a page returns no further/older bars.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..core.atomic_io import atomic_write_csv
from .symbols import ibkr_contract_params

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "ibkr_hourly"

# 862-1123 bars/request measured live for SPY/HSBA.L at "6 M" — comfortably
# under IBKR's ~2000-bar/request cap, cuts request count ~6x vs the old 30 D
# chunking (see broker/ibkr_data.py migration notes / project memory).
_PAGE_DURATION = "6 M"
_MAX_PAGES = 200          # safety valve against runaway paging
_PAGE_SLEEP_S = 2.0       # throttle between reqHistoricalData calls

_DURATION_UNIT_DAYS = {"D": 1, "W": 7, "M": 30, "Y": 365}


def _duration_str_to_days(duration: str) -> int:
    """Parse an IBKR durationStr ("6 M", "30 D") into an approximate day count."""
    n, unit = duration.strip().split()
    return int(n) * _DURATION_UNIT_DAYS[unit.upper()[0]]


def _period_to_days(period: str) -> int:
    """Parse a yfinance-style period string ("730d", "2y") into a day count."""
    period = period.strip().lower()
    if period.endswith("d"):
        return int(period[:-1])
    if period.endswith("y"):
        return int(period[:-1]) * 365
    if period.endswith("mo"):
        return int(period[:-2]) * 30
    raise ValueError(f"Unrecognized period format: {period!r}")


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.replace('/', '-')}.csv"


def _load_cache(ticker: str) -> pd.DataFrame | None:
    path = _cache_path(ticker)
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.empty:
        return None
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def _save_cache(ticker: str, df: pd.DataFrame) -> None:
    atomic_write_csv(_cache_path(ticker), df)


def _resample_30min_aligned(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Resample raw IBKR bars to :30-aligned 60-min windows matching yfinance convention.

    IBKR TRADES bars start at HH:30 for the first bar then shift to HH:00-aligned,
    causing RSI/SMA200/volume_ratio to be computed on different price slices vs yfinance.
    Resampling to offset="30min" makes bar boundaries identical to yfinance hourly bars,
    recovering the Sharpe gap (verified: Sharpe 1.42 raw → 2.75 resampled on matched set).
    Cache is left in raw form so incremental stop_at logic stays correct.
    """
    if df is None or df.empty:
        return df
    return (
        df.resample("1h", offset="30min")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna(subset=["Close"])
    )


class IBKRDataClient:
    """Wraps ib_async for historical-bar requests.

    ib_async is imported lazily so the rest of the package works even if
    it is not installed (mirrors IBKRAdapter's lazy-import convention).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 2,
        connect_timeout: float = 30.0,
    ) -> None:
        self._host = host
        self._port = port
        self._client_id = client_id
        self._connect_timeout = connect_timeout
        self._ib = None

    def connect(self) -> bool:
        """Connect to TWS / IB Gateway. Returns False (never raises) on any
        failure — TWS not running, wrong port, handshake timeout, etc. —
        so callers can fall back the same way a yfinance failure would."""
        try:
            from ib_async import IB
        except ImportError:
            return False
        try:
            self._ib = IB()
            self._ib.connect(self._host, self._port, clientId=self._client_id,
                             timeout=self._connect_timeout)
            return True
        except Exception:
            self._ib = None
            return False

    def disconnect(self) -> None:
        if self._ib is not None:
            self._ib.disconnect()
            self._ib = None

    def _fetch_pages(self, contract, what_to_show: str = "TRADES",
                      stop_at: pd.Timestamp | None = None,
                      min_days: int | None = None) -> pd.DataFrame:
        """Page backward from now through reqHistoricalData.

        stop_at (incremental gap-fill): page only until a page's oldest bar
        reaches stop_at, then return just the bars strictly newer than it —
        a multi-day gap (e.g. daemon downtime) is simply more pages through
        this same loop, no special-casing.

        min_days (bootstrap, stop_at=None): page until at least min_days is
        covered, or a page comes back empty — IBKR has reached the start of
        its available history for this contract.
        """
        from ib_async import util

        frames: list[pd.DataFrame] = []
        end_dt = ""
        days_covered = 0
        page_days = _duration_str_to_days(_PAGE_DURATION)
        for _ in range(_MAX_PAGES):
            bars = self._ib.reqHistoricalData(
                contract, endDateTime=end_dt, durationStr=_PAGE_DURATION,
                barSizeSetting="1 hour", whatToShow=what_to_show, useRTH=True,
            )
            if not bars:
                break
            page = util.df(bars)
            frames.append(page)
            days_covered += page_days
            end_dt = bars[0].date

            oldest = pd.Timestamp(bars[0].date)
            oldest = oldest.tz_localize("UTC") if oldest.tzinfo is None else oldest.tz_convert("UTC")

            if stop_at is not None and oldest <= stop_at:
                break
            if stop_at is None and min_days is not None and days_covered >= min_days:
                break
            self._ib.sleep(_PAGE_SLEEP_S)

        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames[::-1], ignore_index=True)
        out = out.drop_duplicates(subset="date").sort_values("date")
        out = out.set_index("date")
        out.index = pd.to_datetime(out.index, utc=True)
        out.index.name = None
        out = out.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })[["Open", "High", "Low", "Close", "Volume"]]
        if stop_at is not None:
            out = out[out.index > stop_at]
        return out

    def fetch_hourly(self, ticker: str, period: str = "730d", use_cache: bool = True,
                      what_to_show: str = "TRADES") -> pd.DataFrame | None:
        """Fetch hourly OHLCV, same return contract as quant_engine's
        yfinance path: pd.DataFrame | None, tz-aware index, OHLCV columns.

        Incremental: an existing cache is extended by paging only the gap
        since its last bar (see _fetch_pages), not re-pulled from scratch.
        A brand-new ticker gets a one-time bootstrap pull covering `period`.
        Results are merged into data/cache/ibkr_hourly/ so the cache only
        ever grows.

        what_to_show="ADJUSTED_LAST" is exposed for scripts/ibkr_data_pilot.py's
        split/dividend-adjustment validation against yfinance; the incremental
        cache itself always uses the default "TRADES" — see the migration
        plan's validation notes for why ADJUSTED_LAST was not adopted."""
        cached = _load_cache(ticker) if use_cache else None

        owns_connection = self._ib is None
        if owns_connection and not self.connect():
            return _resample_30min_aligned(cached)

        try:
            from ib_async import Stock
            contract = Stock(*ibkr_contract_params(ticker))
            self._ib.qualifyContracts(contract)
            if cached is not None:
                new_df = self._fetch_pages(contract, what_to_show=what_to_show,
                                            stop_at=cached.index[-1])
            else:
                new_df = self._fetch_pages(contract, what_to_show=what_to_show,
                                            min_days=_period_to_days(period))
        except Exception:
            return _resample_30min_aligned(cached)
        finally:
            if owns_connection:
                self.disconnect()

        if cached is not None:
            merged = pd.concat([cached, new_df]) if not new_df.empty else cached
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        else:
            merged = new_df

        if merged.empty:
            return _resample_30min_aligned(cached)
        if use_cache and not new_df.empty:
            _save_cache(ticker, merged)
        return _resample_30min_aligned(merged)
