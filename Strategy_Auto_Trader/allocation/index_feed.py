"""Latest COMPLETED hourly bar of a CBOE volatility index, refreshed on a timer.

The tier allocator used to fetch VIX/VXN once per calendar day and reuse that reading for every
60-second cycle, so the tier was effectively set at the first cycle (~08:00 London, before the
VIX Global-Trading-Hours print begins at 08:15) and never reacted to the rest of the session.

Reading rule (kept identical to the intraday backtest in `intraday_engine.py`): a bar's Close
counts only once the bar has ended (`core.trading_sessions.bar_end`). The still-forming bar is
ignored, so live decisions land on the same hourly boundaries the backtest models. VXN has no
overnight print, so outside US hours both feeds simply keep returning the last completed bar.

Failure handling: a failed refresh keeps the last good value for `max_outage_seconds`, then
returns None (the allocator treats a missing VIX as cash and a missing VXN as "no Nasdaq").
Retries are also spaced by `refresh_seconds` so an IBKR outage is not hammered every cycle.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

from ..core.trading_sessions import US, bar_end

_log = logging.getLogger(__name__)

REFRESH_SECONDS = 300
MAX_OUTAGE_SECONDS = 3 * 3600


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def latest_completed_close(df: pd.DataFrame, now: pd.Timestamp) -> tuple[float, pd.Timestamp] | None:
    """(Close, bar_end) of the newest bar that has ended by `now`, or None if there is none."""
    index = df.index if df.index.tz is not None else df.index.tz_localize("UTC")
    done = bar_end(index, US) <= now
    if not done.any():
        return None
    position = int(done.nonzero()[0][-1])
    return float(df["Close"].iloc[position]), bar_end(index[[position]], US)[0]


class IndexFeed:
    """Cached, TTL-refreshed reader for one index. `current()` is safe to call every cycle."""

    def __init__(
        self,
        name: str,
        refresh_seconds: float = REFRESH_SECONDS,
        max_outage_seconds: float = MAX_OUTAGE_SECONDS,
        clock: Callable[[], pd.Timestamp] = _utc_now,
    ):
        self.name = name
        self._refresh_seconds = refresh_seconds
        self._max_outage_seconds = max_outage_seconds
        self._clock = clock
        self._df: pd.DataFrame | None = None
        self._last_attempt: pd.Timestamp | None = None
        self._last_success: pd.Timestamp | None = None

    def current(self, fetcher: Callable[[], pd.DataFrame | None]) -> float | None:
        now = self._clock()
        if self._last_attempt is None or (now - self._last_attempt).total_seconds() >= self._refresh_seconds:
            self._refresh(fetcher, now)
        if self._df is None:
            return None
        if (now - self._last_success).total_seconds() > self._max_outage_seconds:
            _log.warning(f"{self.name}: no successful fetch for over {self._max_outage_seconds / 3600:.0f}h — treating as unavailable")
            return None
        try:
            found = latest_completed_close(self._df, now)
        except Exception as e:
            _log.warning(f"Failed to extract {self.name} from cache: {e}")
            return None
        if found is None:
            return None
        close, ended = found
        _log.debug(f"{self.name} = {close:.2f} (bar ended {ended:%Y-%m-%d %H:%M} UTC)")
        return close

    def _refresh(self, fetcher: Callable[[], pd.DataFrame | None], now: pd.Timestamp) -> None:
        self._last_attempt = now
        try:
            df = fetcher()
        except Exception as e:
            _log.error(f"Failed to fetch {self.name}: {e}")
            return
        if df is None or df.empty:
            _log.warning(f"{self.name}: fetch returned no data — keeping previous reading")
            return
        self._df = df
        self._last_success = now
        _log.debug(f"{self.name} refreshed: {len(df)} rows")
