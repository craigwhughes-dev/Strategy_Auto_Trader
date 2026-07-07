"""IBKR historical-bar client via ib_insync — an opt-in alternative to
yfinance for `quant_engine.fetch_hourly` (see quant_engine.py's `source`
param). Kept separate from ibkr_adapter.py: that module is for live order
execution, this one is read-only historical data and is safe to exercise
without touching the trading connection.

Uses client_id=2 by default, reserved for data-fetch, so it never collides
with the live daemon's execution connection (client_id=1) on the same TWS
instance — both can run concurrently against one TWS/Gateway.

IBKR's reqHistoricalData is duration- and pacing-limited per request, so
years of hourly history requires paging backwards in chunks with throttling
between requests, concatenating until either the requested period is
covered or a page returns no further/older bars.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .symbols import ibkr_contract_params

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "ibkr_hourly"

_PAGE_DURATION = "30 D"
_MAX_PAGES = 200          # safety valve against runaway paging
_PAGE_SLEEP_S = 2.0       # throttle between reqHistoricalData calls


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


def _load_cache(ticker: str) -> pd.DataFrame | None:
    path = CACHE_DIR / f"{ticker.replace('/', '-')}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df if not df.empty else None


def _save_cache(ticker: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{ticker.replace('/', '-')}.csv"
    df.to_csv(path)


class IBKRDataClient:
    """Wraps ib_insync for historical-bar requests.

    ib_insync is imported lazily so the rest of the package works even if
    it is not installed (mirrors IBKRAdapter's lazy-import convention).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
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
            from ib_insync import IB
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

    def _fetch_pages(self, contract, days_needed: int) -> pd.DataFrame:
        from ib_insync import util

        frames: list[pd.DataFrame] = []
        end_dt = ""
        days_covered = 0
        for _ in range(_MAX_PAGES):
            bars = self._ib.reqHistoricalData(
                contract, endDateTime=end_dt, durationStr=_PAGE_DURATION,
                barSizeSetting="1 hour", whatToShow="TRADES", useRTH=True,
            )
            if not bars:
                break
            page = util.df(bars)
            frames.append(page)
            days_covered += 30
            end_dt = bars[0].date
            if days_covered >= days_needed:
                break
            self._ib.sleep(_PAGE_SLEEP_S)

        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames[::-1], ignore_index=True)
        out = out.drop_duplicates(subset="date").sort_values("date")
        out = out.set_index("date")
        out.index = pd.to_datetime(out.index, utc=True)
        out.index.name = None
        return out.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })[["Open", "High", "Low", "Close", "Volume"]]

    def fetch_hourly(self, ticker: str, period: str = "730d", use_cache: bool = True) -> pd.DataFrame | None:
        """Fetch hourly OHLCV, same return contract as quant_engine's
        yfinance path: pd.DataFrame | None, tz-aware index, OHLCV columns.
        Pages through IBKR's pacing-limited reqHistoricalData and caches
        results under data/cache/ibkr_hourly/ so repeat backtests don't
        re-page every run."""
        days_needed = _period_to_days(period)

        cached = _load_cache(ticker) if use_cache else None
        if cached is not None:
            span = (cached.index[-1] - cached.index[0]).days
            if span >= days_needed:
                return cached

        owns_connection = self._ib is None
        if owns_connection and not self.connect():
            return cached

        try:
            from ib_insync import Stock
            contract = Stock(*ibkr_contract_params(ticker))
            self._ib.qualifyContracts(contract)
            df = self._fetch_pages(contract, days_needed)
        except Exception:
            return cached
        finally:
            if owns_connection:
                self.disconnect()

        if df.empty:
            return cached
        if use_cache:
            _save_cache(ticker, df)
        return df
