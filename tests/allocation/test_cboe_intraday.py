"""CBOE intraday minute feed -> US-session hourly bars for the tier allocator."""

from __future__ import annotations

import pandas as pd
import pytest

from Strategy_Auto_Trader.allocation.cboe_intraday import (
    fetch_cboe_daily_close,
    fetch_cboe_hourly,
    fetch_cboe_index_hourly,
    fetch_official_close,
    hourly_bars,
    parse_minute_rows,
)
from Strategy_Auto_Trader.allocation.index_feed import IndexFeed

# 2 Oct 2026 is EDT (UTC-4): ET 09:30 is 13:30 UTC.


def _row(hhmm: str, close: float, *, day: str = "2026-10-02") -> dict:
    return {"datetime": f"{day}T{hhmm}:00", "price": {"open": close, "high": close, "low": close, "close": close}}


def _payload(rows: list[dict]) -> dict:
    return {"timestamp": "2026-10-02 20:14:34", "data": rows}


def _session_rows(last_hhmm: str = "15:59") -> list[dict]:
    """One row per minute from 09:31 to last_hhmm; close = minutes since 09:00, so each bar's close is identifiable."""
    stamps = pd.date_range("2026-10-02 09:31", f"2026-10-02 {last_hhmm}", freq="min")
    return [
        _row(ts.strftime("%H:%M"), 100.0 + (ts.hour * 60 + ts.minute - 540) / 1000)
        for ts in stamps
    ]


def _minutes(rows: list[dict]) -> pd.DataFrame:
    return parse_minute_rows(_payload(rows))


def test_parse_drops_zero_rows_and_converts_et_to_utc():
    rows = [_row("09:31", 0.0), _row("09:32", 21.29)]
    frame = _minutes(rows)
    assert list(frame.index) == [pd.Timestamp("2026-10-02 13:32", tz="UTC")]
    assert frame["Close"].iloc[0] == pytest.approx(21.29)


def test_parse_empty_when_no_trades():
    frame = _minutes([_row("09:31", 0.0)])
    assert frame.empty


def test_bar_closes_on_row_at_its_end_minute():
    bars = hourly_bars(_minutes(_session_rows()))
    # 09:30 bar ends 10:00 -> close is the 10:00 row, not 09:59
    first = bars.loc[pd.Timestamp("2026-10-02 13:30", tz="UTC")]
    assert first["Close"] == pytest.approx(100.0 + (600 - 540) / 1000)
    # 10:00 bar ends 11:00 -> 11:00 row
    second = bars.loc[pd.Timestamp("2026-10-02 14:00", tz="UTC")]
    assert second["Close"] == pytest.approx(100.0 + (660 - 540) / 1000)


def test_bar_high_low_open_come_from_minutes_inside_the_bar():
    # 10:00 row closes the 09:30 bar, so it must not leak into the 10:00-11:00 bar
    rows = [_row("10:00", 20.0), _row("10:20", 25.0), _row("10:40", 18.0), _row("11:00", 21.0)]
    bars = hourly_bars(_minutes(rows))
    bar = bars.loc[pd.Timestamp("2026-10-02 14:00", tz="UTC")]
    assert bar["Open"] == 25.0 and bar["High"] == 25.0 and bar["Low"] == 18.0 and bar["Close"] == 21.0


def test_final_hour_closes_on_last_available_minute_when_16_00_absent():
    bars = hourly_bars(_minutes(_session_rows()))
    # 15:00 bar ends at 16:00, but the feed has no 16:00 row; close falls back to 15:59
    last = bars.loc[pd.Timestamp("2026-10-02 19:00", tz="UTC")]
    assert last["Close"] == pytest.approx(100.0 + (959 - 540) / 1000)
    # 16:00 bar (16:00-16:15) has no minutes, so it is absent
    assert pd.Timestamp("2026-10-02 20:00", tz="UTC") not in bars.index


def test_lagging_feed_drops_bar_whose_close_row_is_stale():
    # feed stops at 09:50: the 09:30 bar (ends 10:00) lacks a row within 5 min of its end
    bars = hourly_bars(_minutes(_session_rows(last_hhmm="09:50")))
    assert bars.empty


def test_bars_index_is_utc_bar_start_with_close_column():
    bars = hourly_bars(_minutes(_session_rows()))
    assert bars.index.tz is not None
    assert str(bars.index.tz) == "UTC"
    assert "Close" in bars.columns


def test_index_feed_serves_latest_completed_bar_from_cboe_bars():
    bars = hourly_bars(_minutes(_session_rows()))
    now = pd.Timestamp("2026-10-02 14:30", tz="UTC")  # 10:30 ET: 09:30 bar done, 10:00 bar not yet
    feed = IndexFeed("VXN", clock=lambda: now)
    close = feed.current(lambda: bars)
    assert close == pytest.approx(100.0 + (600 - 540) / 1000)


def test_fetch_returns_none_when_request_fails():
    def boom(url, timeout):
        raise OSError("network down")

    assert fetch_cboe_hourly("VXN", get_json=boom) is None


def test_fetch_returns_none_when_feed_has_no_complete_bars():
    assert fetch_cboe_hourly("VXN", get_json=lambda url, timeout: _payload([_row("09:31", 0.0)])) is None


def test_fetch_builds_bars_from_payload_and_requests_symbol_url():
    seen = []

    def record(url, timeout):
        seen.append(url)
        return _payload(_session_rows())

    bars = fetch_cboe_hourly("VVIX", get_json=record, now=BEFORE_SETTLEMENT)
    assert seen == ["https://cdn.cboe.com/api/global/delayed_quotes/charts/intraday/_VVIX.json"]
    assert bars is not None and not bars.empty


def _quote(close: float, trade: str = "2026-10-02T00:00:00-05:00") -> dict:
    return {"timestamp": "2026-10-02 20:14:34", "data": {"close": close, "last_trade_time": trade}}


def _router(intraday: dict, quote, seen: list[str] | None = None):
    """get_json stub: intraday payload for the minute URL, quote payload (or exception) for the quote URL."""
    def get(url, timeout):
        if seen is not None:
            seen.append(url)
        if "/quotes/" in url:
            if isinstance(quote, Exception):
                raise quote
            return quote
        return intraday
    return get


AFTER_SETTLEMENT = lambda: pd.Timestamp("2026-10-02 20:30", tz="UTC")  # 16:30 EDT
BEFORE_SETTLEMENT = lambda: pd.Timestamp("2026-10-02 19:30", tz="UTC")  # 15:30 EDT
FINAL_BAR = pd.Timestamp("2026-10-02 19:00", tz="UTC")  # 15:00 ET bar start


def test_official_close_replaces_final_bar_close_for_matching_session():
    bars = hourly_bars(_minutes(_session_rows()), official=("2026-10-02", 21.20))
    assert bars.loc[FINAL_BAR]["Close"] == pytest.approx(21.20)
    # earlier bars keep minute closes
    assert bars.loc[pd.Timestamp("2026-10-02 14:00", tz="UTC")]["Close"] == pytest.approx(100.0 + (660 - 540) / 1000)


def test_official_close_ignored_for_other_session_date():
    bars = hourly_bars(_minutes(_session_rows()), official=("2026-10-01", 21.20))
    assert bars.loc[FINAL_BAR]["Close"] == pytest.approx(100.0 + (959 - 540) / 1000)


def test_fetch_uses_official_close_after_settlement():
    bars = fetch_cboe_hourly("VXN", get_json=_router(_payload(_session_rows()), _quote(21.20)), now=AFTER_SETTLEMENT)
    assert bars.loc[FINAL_BAR]["Close"] == pytest.approx(21.20)


def test_fetch_does_not_request_quote_before_settlement():
    seen: list[str] = []
    bars = fetch_cboe_hourly("VXN", get_json=_router(_payload(_session_rows()), _quote(21.20), seen), now=BEFORE_SETTLEMENT)
    assert not any("/quotes/" in url for url in seen)
    assert bars.loc[FINAL_BAR]["Close"] == pytest.approx(100.0 + (959 - 540) / 1000)


def test_fetch_falls_back_when_quote_dated_to_other_session():
    bars = fetch_cboe_hourly("VXN", get_json=_router(_payload(_session_rows()), _quote(21.20, "2026-10-01T00:00:00-05:00")), now=AFTER_SETTLEMENT)
    assert bars.loc[FINAL_BAR]["Close"] == pytest.approx(100.0 + (959 - 540) / 1000)


def test_fetch_falls_back_when_quote_request_fails():
    bars = fetch_cboe_hourly("VXN", get_json=_router(_payload(_session_rows()), OSError("quote down")), now=AFTER_SETTLEMENT)
    assert bars is not None
    assert bars.loc[FINAL_BAR]["Close"] == pytest.approx(100.0 + (959 - 540) / 1000)


def test_fetch_official_close_rejects_zero_or_undated_quote():
    assert fetch_official_close("VXN", get_json=lambda u, t: _quote(0.0)) is None
    assert fetch_official_close("VXN", get_json=lambda u, t: _quote(21.2, "")) is None
    assert fetch_official_close("VXN", get_json=lambda u, t: _quote(21.2)) == ("2026-10-02", 21.2)


MONDAY_PREOPEN = lambda: pd.Timestamp("2026-10-05 12:45", tz="UTC")  # Mon 08:45 EDT, before the US open


def _prior(prev: float) -> dict:
    """Quote payload with a live `close` that differs from `prev_day_close`, as the feed reports mid-session."""
    return {"data": {"close": 88.9, "prev_day_close": prev, "last_trade_time": "2026-10-05T08:45:00"}}


def test_daily_close_is_previous_session_close_not_live_price():
    frame = fetch_cboe_daily_close("VVIX", get_json=lambda u, t: _prior(87.02), now=MONDAY_PREOPEN)
    assert frame["Close"].iloc[0] == pytest.approx(87.02)
    assert list(frame.index) == [pd.Timestamp("2026-10-02")]  # Friday, the prior weekday


def test_daily_close_requests_quote_url_for_symbol():
    seen: list[str] = []
    fetch_cboe_daily_close("VVIX", get_json=lambda u, t: (seen.append(u), _prior(87.02))[1], now=MONDAY_PREOPEN)
    assert seen == ["https://cdn.cboe.com/api/global/delayed_quotes/quotes/_VVIX.json"]


def test_daily_close_none_when_no_prior_close_or_request_fails():
    assert fetch_cboe_daily_close("VVIX", get_json=lambda u, t: _prior(0.0), now=MONDAY_PREOPEN) is None

    def boom(url, timeout):
        raise OSError("network down")

    assert fetch_cboe_daily_close("VVIX", get_json=boom, now=MONDAY_PREOPEN) is None


def test_vxn_preopen_uses_prior_session_close_as_last_completed_bar():
    # Monday pre-open: minute feed has only a zero row, so no complete bar exists yet
    empty = _payload([_row("09:31", 0.0)])
    bars = fetch_cboe_index_hourly("VXN", get_json=_router(empty, _prior(21.20)), now=MONDAY_PREOPEN)
    assert bars is not None
    feed = IndexFeed("VXN", clock=MONDAY_PREOPEN)
    assert feed.current(lambda: bars) == pytest.approx(21.20)


def test_vxn_uses_intraday_bars_once_a_complete_bar_exists():
    bars = fetch_cboe_index_hourly("VXN", get_json=_router(_payload(_session_rows()), _prior(21.20)), now=BEFORE_SETTLEMENT)
    assert bars.loc[pd.Timestamp("2026-10-02 13:30", tz="UTC")]["Close"] == pytest.approx(100.0 + (600 - 540) / 1000)


def test_vxn_none_when_both_intraday_and_quote_fail():
    def boom(url, timeout):
        raise OSError("network down")

    assert fetch_cboe_index_hourly("VXN", get_json=boom, now=MONDAY_PREOPEN) is None
