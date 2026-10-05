"""Hourly VXN/VVIX bars from CBOE's intraday minute feed, on the same grid as the tier allocator.

Source: CBOE's delayed-quotes intraday JSON (1-minute OHLC, one session). Undocumented endpoint,
so every failure returns None and IndexFeed keeps serving its last good reading for its outage limit.

Row labels are the instant a minute ends: the row stamped 11:30 holds the close at 11:30 ET. The
session's 16:00 print is absent from the minute feed. The 15:00 bar (which ends at 16:00) closes on
the 15:59 row, unless the official settlement close is published from the quote feed, in which case
that close replaces it. Settlement is only taken after 16:00 ET and only when the quote feed dates it
to the same session. A bar is dropped unless its close row falls within `max_row_lag` of the bar end,
which stops a lagging feed from passing a partial bar off as complete.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from collections.abc import Callable

import pandas as pd

from ..core.trading_sessions import US, session_bars

_log = logging.getLogger(__name__)

CBOE_INTRADAY_URL = "https://cdn.cboe.com/api/global/delayed_quotes/charts/intraday/_{symbol}.json"
CBOE_QUOTE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/quotes/_{symbol}.json"
# CBOE's CDN redirects clients without a browser User-Agent
USER_AGENT = "Mozilla/5.0"
MAX_ROW_LAG = pd.Timedelta(minutes=5)
SETTLEMENT_TIME = "16:00"
_OHLC = ("Open", "High", "Low", "Close")

GetJson = Callable[[str, float], dict]


def _get_json(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def parse_minute_rows(payload: dict) -> pd.DataFrame:
    """Minute OHLC from one intraday payload, UTC-indexed by the minute's end instant.

    Pre-print rows carry zeros; a row with no positive close is not a trade and is dropped.
    """
    rows = [
        {"datetime": r["datetime"], **{k.capitalize(): float(r["price"][k]) for k in ("open", "high", "low", "close")}}
        for r in payload["data"]
    ]
    frame = pd.DataFrame(rows)
    frame = frame[frame["Close"] > 0]
    if frame.empty:
        return pd.DataFrame(columns=list(_OHLC), index=pd.DatetimeIndex([], tz="UTC"))
    local = pd.to_datetime(frame["datetime"]).dt.tz_localize(US.tz)
    out = frame[list(_OHLC)].set_axis(pd.DatetimeIndex(local.dt.tz_convert("UTC")))
    return out.sort_index()


def fetch_official_close(symbol: str, timeout: float = 30.0, get_json: GetJson = _get_json) -> tuple[str, float] | None:
    """(session date 'YYYY-MM-DD', settlement close) from the quote feed, or None if it has no usable close."""
    data = get_json(CBOE_QUOTE_URL.format(symbol=symbol), timeout).get("data") or {}
    close = float(data.get("close") or 0)
    trade = data.get("last_trade_time") or ""
    if close <= 0 or len(trade) < 10:
        return None
    return trade[:10], close


def _is_settlement_bar(end: pd.Timestamp, session_date: str) -> bool:
    return end.tz_convert(US.tz).strftime("%Y-%m-%d %H:%M") == f"{session_date} {SETTLEMENT_TIME}"


def hourly_bars(
    minutes: pd.DataFrame,
    official: tuple[str, float] | None = None,
    max_row_lag: pd.Timedelta = MAX_ROW_LAG,
) -> pd.DataFrame:
    """Bars on the US session grid, indexed by bar START (UTC), columns Open/High/Low/Close.

    A bar covers minutes labelled (start, end]. Its Close is the last such minute's close, kept only
    if that minute ends within `max_row_lag` of the bar end. The bar ending at 16:00 ET takes the
    `official` (session date, close) when given for that session date. Empty and incomplete bars
    are dropped.
    """
    records = []
    days = sorted(set(minutes.index.tz_convert(US.tz).normalize().tz_localize(None)))
    for day in days:
        for _, bar in session_bars(day, US).iterrows():
            start, end = bar["start"], bar["end"]
            window = minutes[(minutes.index > start) & (minutes.index <= end)]
            if window.empty or window.index[-1] < end - max_row_lag:
                continue
            close = window["Close"].iloc[-1]
            if official is not None and official[0] == day.strftime("%Y-%m-%d") and _is_settlement_bar(end, official[0]):
                close = official[1]
            records.append({
                "start": start,
                "Open": window["Open"].iloc[0],
                "High": window["High"].max(),
                "Low": window["Low"].min(),
                "Close": close,
            })
    if not records:
        return pd.DataFrame(columns=list(_OHLC), index=pd.DatetimeIndex([], tz="UTC"))
    frame = pd.DataFrame(records).set_index("start")
    frame.index = pd.DatetimeIndex(frame.index, tz="UTC")
    frame.index.name = None
    return frame[list(_OHLC)]


def _settlement_for_session(
    symbol: str, minutes: pd.DataFrame, timeout: float, get_json: GetJson, now: Callable[[], pd.Timestamp],
) -> tuple[str, float] | None:
    """Official close for the session in `minutes`, only once 16:00 ET has passed. Failure means None."""
    session_date = minutes.index[-1].tz_convert(US.tz).strftime("%Y-%m-%d")
    if now().tz_convert(US.tz) < pd.Timestamp(f"{session_date} {SETTLEMENT_TIME}", tz=US.tz):
        return None
    try:
        official = fetch_official_close(symbol, timeout, get_json)
    except Exception as e:
        _log.warning(f"CBOE quote {symbol}: settlement fetch failed ({e}); using 15:59 close")
        return None
    if official is None or official[0] != session_date:
        return None
    return official


def fetch_cboe_hourly(
    symbol: str,
    timeout: float = 30.0,
    get_json: GetJson = _get_json,
    now: Callable[[], pd.Timestamp] = _utc_now,
) -> pd.DataFrame | None:
    """Hourly bars for one CBOE index, or None on any failure (IndexFeed then serves its last good value)."""
    try:
        minutes = parse_minute_rows(get_json(CBOE_INTRADAY_URL.format(symbol=symbol), timeout))
        if minutes.empty:
            _log.warning(f"CBOE intraday {symbol}: no trades in feed")
            return None
        official = _settlement_for_session(symbol, minutes, timeout, get_json, now)
        bars = hourly_bars(minutes, official)
    except Exception as e:
        _log.warning(f"CBOE intraday {symbol}: fetch failed ({e})")
        return None
    if bars.empty:
        _log.warning(f"CBOE intraday {symbol}: no complete bars in feed")
        return None
    return bars


def fetch_prior_close(symbol: str, timeout: float = 30.0, get_json: GetJson = _get_json) -> float | None:
    """Previous session's close from the quote feed. Its `close` field is the live last trade during
    the session, so it is not used for "last completed" readings; `prev_day_close` is."""
    data = get_json(CBOE_QUOTE_URL.format(symbol=symbol), timeout).get("data") or {}
    prev = float(data.get("prev_day_close") or 0)
    return prev if prev > 0 else None


def _prior_weekday(now: pd.Timestamp) -> pd.Timestamp:
    """Most recent weekday before `now`'s US calendar date. Holidays are not modelled: the date only
    stamps the reading, and the reading is still the last close."""
    day = now.tz_convert(US.tz).normalize().tz_localize(None) - pd.Timedelta(days=1)
    while day.weekday() >= 5:
        day -= pd.Timedelta(days=1)
    return day


def _single_bar(day: pd.Timestamp, close: float) -> pd.DataFrame:
    """One completed bar for `day`'s last US hour (15:00 ET, ends 16:00 ET), so IndexFeed treats it as complete."""
    start = pd.Timestamp(f"{day:%Y-%m-%d} 15:00", tz=US.tz).tz_convert("UTC")
    return pd.DataFrame({"Open": [close], "High": [close], "Low": [close], "Close": [close]},
                        index=pd.DatetimeIndex([start]))


def fetch_cboe_index_hourly(
    symbol: str,
    timeout: float = 30.0,
    get_json: GetJson = _get_json,
    now: Callable[[], pd.Timestamp] = _utc_now,
) -> pd.DataFrame | None:
    """Completed hourly bars for a CBOE index, falling back to the prior session's close before the feed has a complete bar.

    The minute feed holds one session only and has no complete bar until 10:00 ET, so the pre-open
    reading is the previous session's close, as the allocator expects.
    """
    bars = fetch_cboe_hourly(symbol, timeout, get_json, now)
    if bars is not None:
        return bars
    try:
        prior = fetch_prior_close(symbol, timeout, get_json)
    except Exception as e:
        _log.warning(f"CBOE quote {symbol}: prior close fetch failed ({e})")
        return None
    if prior is None:
        return None
    return _single_bar(_prior_weekday(now()), prior)


def fetch_cboe_daily_close(
    symbol: str,
    timeout: float = 30.0,
    get_json: GetJson = _get_json,
    now: Callable[[], pd.Timestamp] = _utc_now,
) -> pd.DataFrame | None:
    """One-row daily frame (Close, indexed by the prior session date), or None. Used once a day, pre-open."""
    try:
        prior = fetch_prior_close(symbol, timeout, get_json)
    except Exception as e:
        _log.warning(f"CBOE quote {symbol}: daily close fetch failed ({e})")
        return None
    if prior is None:
        _log.warning(f"CBOE quote {symbol}: no previous close")
        return None
    return pd.DataFrame({"Close": [prior]}, index=pd.DatetimeIndex([_prior_weekday(now())]))
