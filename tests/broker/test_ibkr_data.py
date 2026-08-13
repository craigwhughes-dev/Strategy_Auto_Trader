"""Tests for broker.ibkr_data — IBKR historical-bar client (paging, cache,
and failure-falls-back-to-None/cache contract), mirroring the mocking
pattern used for IBKRAdapter in test_broker.py."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from Strategy_Auto_Trader.broker import ibkr_data
from Strategy_Auto_Trader.broker.ibkr_data import IBKRDataClient


def _bar(date, o=100.0, h=101.0, l=99.0, c=100.5, v=1000):
    from ib_async.objects import BarData
    return BarData(date=date, open=o, high=h, low=l, close=c, volume=v)


def _make_page(start: datetime, n_hours: int) -> list:
    return [_bar(start + timedelta(hours=i)) for i in range(n_hours)]


class TestConnect:
    def test_connect_success(self):
        pytest.importorskip("ib_async")
        from unittest.mock import patch
        client = IBKRDataClient(port=4002, client_id=9)
        with patch("ib_async.IB") as MockIB:
            assert client.connect() is True
        MockIB.return_value.connect.assert_called_once_with(
            "127.0.0.1", 4002, clientId=9, timeout=30.0)

    def test_connect_failure_returns_false_not_raise(self):
        pytest.importorskip("ib_async")
        from unittest.mock import patch
        client = IBKRDataClient()
        with patch("ib_async.IB") as MockIB:
            MockIB.return_value.connect.side_effect = ConnectionRefusedError("no TWS")
            assert client.connect() is False
        assert client._ib is None

    def test_connect_missing_ib_async_returns_false(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "ib_async":
                raise ImportError("no module")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        client = IBKRDataClient()
        assert client.connect() is False


class TestFetchHourly:
    def test_bootstrap_pages_until_min_days_covered(self, tmp_path, monkeypatch):
        """Brand-new ticker, no cache: pages until min_days (from `period`)
        is covered — each "6 M" page counts as 180 days regardless of the
        bar count returned, so a 200-day bootstrap needs two pages."""
        pytest.importorskip("ib_async")
        from unittest.mock import MagicMock
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)

        client = IBKRDataClient()
        client._ib = MagicMock()
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        page1 = _make_page(start + timedelta(days=180), 5)
        page2 = _make_page(start, 5)
        client._ib.reqHistoricalData.side_effect = [page1, page2]

        out = client.fetch_hourly("AAPL", period="200d", use_cache=False)

        assert client._ib.reqHistoricalData.call_count == 2
        assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(out) == 10
        assert out.index.is_monotonic_increasing

    def test_incremental_fetch_only_pages_the_gap(self, tmp_path, monkeypatch):
        """A cache that already covers the requested period must still be
        extended with new bars, not returned as-is forever — this is the
        core fix over the old span-check design (a stale-forever cache)."""
        pytest.importorskip("ib_async")
        from unittest.mock import MagicMock
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)

        cached_idx = pd.date_range("2026-01-01 00:00", periods=5, freq="h", tz="UTC")
        cached = pd.DataFrame(
            {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 100}, index=cached_idx)
        ibkr_data._save_cache("AAPL", cached)

        client = IBKRDataClient()
        client._ib = MagicMock()
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        # One page spanning 00:00-09:00; its oldest bar (00:00) is <= the
        # cached last bar (04:00), so paging stops after a single request.
        client._ib.reqHistoricalData.side_effect = [_make_page(start, 10)]

        out = client.fetch_hourly("AAPL", period="730d", use_cache=True)

        assert client._ib.reqHistoricalData.call_count == 1
        assert len(out) == 10  # 5 cached + 5 genuinely new bars (05:00-09:00)
        assert out.index.is_monotonic_increasing

        reloaded = ibkr_data._load_cache("AAPL")
        assert len(reloaded) == 10

    def test_gap_with_no_bars_in_between_is_still_bridged(self, tmp_path, monkeypatch):
        """Gap-fill correctness doesn't assume bar density between the last
        cached bar and now — IBKR simply omits non-trading hours (weekends,
        holidays), and the stop_at comparison works on whatever timestamps
        come back, so a sparse gap needs no special-case calendar logic."""
        pytest.importorskip("ib_async")
        from unittest.mock import MagicMock
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)

        last_cached = pd.Timestamp("2026-01-02 16:00", tz="UTC")
        cached = pd.DataFrame(
            {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 100},
            index=pd.DatetimeIndex([last_cached]))
        ibkr_data._save_cache("AAPL", cached)

        client = IBKRDataClient()
        client._ib = MagicMock()
        next_session_open = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)
        # A real "6 M" page reaches back far past a few-day gap, so its
        # oldest bar is last_cached itself — the weekend in between simply
        # has no bars, same as IBKR would return.
        page = [_bar(last_cached.to_pydatetime())] + _make_page(next_session_open, 3)
        client._ib.reqHistoricalData.side_effect = [page]

        out = client.fetch_hourly("AAPL", period="730d", use_cache=True)

        assert client._ib.reqHistoricalData.call_count == 1
        assert len(out) == 4  # 1 cached + 3 new, nothing fabricated for the gap
        assert out.index[0] == last_cached

    def test_stops_when_page_returns_no_bars(self, tmp_path, monkeypatch):
        pytest.importorskip("ib_async")
        from unittest.mock import MagicMock
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)

        client = IBKRDataClient()
        client._ib = MagicMock()
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        client._ib.reqHistoricalData.side_effect = [_make_page(start, 5), []]

        out = client.fetch_hourly("AAPL", period="730d", use_cache=False)

        assert client._ib.reqHistoricalData.call_count == 2
        assert len(out) == 5

    def test_connection_failure_returns_none_when_no_cache(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)
        client = IBKRDataClient()
        client.connect = lambda: False
        assert client.fetch_hourly("AAPL", use_cache=True) is None

    def test_connection_failure_falls_back_to_cache(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)
        idx = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
        cached = pd.DataFrame(
            {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 100}, index=idx)
        ibkr_data._save_cache("AAPL", cached)

        client = IBKRDataClient()
        client.connect = lambda: False
        out = client.fetch_hourly("AAPL", period="1d", use_cache=True)
        assert out is not None
        assert len(out) == 5

    def test_no_new_bars_leaves_cache_file_untouched(self, monkeypatch, tmp_path):
        """If IBKR has nothing newer than the cache (e.g. called again within
        the same hour), the merge is a no-op and the cache file isn't
        rewritten — no point churning an atomic write for zero new rows."""
        pytest.importorskip("ib_async")
        from unittest.mock import MagicMock
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)
        idx = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
        cached = pd.DataFrame(
            {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 100}, index=idx)
        ibkr_data._save_cache("AAPL", cached)
        mtime_before = ibkr_data._cache_path("AAPL").stat().st_mtime_ns

        client = IBKRDataClient()
        client._ib = MagicMock()
        # Page's only bar is exactly the cached last bar — filtered out by
        # the stop_at > comparison, so nothing new survives.
        client._ib.reqHistoricalData.side_effect = [_make_page(idx[-1].to_pydatetime(), 1)]

        out = client.fetch_hourly("AAPL", period="730d", use_cache=True)

        assert len(out) == 5
        assert ibkr_data._cache_path("AAPL").stat().st_mtime_ns == mtime_before


class TestPeriodToDays:
    @pytest.mark.parametrize("period,expected", [
        ("730d", 730), ("2y", 730), ("1mo", 30),
    ])
    def test_parses_common_formats(self, period, expected):
        assert ibkr_data._period_to_days(period) == expected

    def test_unrecognized_format_raises(self):
        with pytest.raises(ValueError):
            ibkr_data._period_to_days("bogus")


class TestCacheRoundTrip:
    def test_save_then_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)
        idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
        df = pd.DataFrame(
            {"Open": [1, 2, 3], "High": [1, 2, 3], "Low": [1, 2, 3],
             "Close": [1, 2, 3], "Volume": [10, 20, 30]}, index=idx)
        ibkr_data._save_cache("HSBA.L", df)
        out = ibkr_data._load_cache("HSBA.L")
        assert out is not None
        assert len(out) == 3
        assert list(out["Close"]) == [1, 2, 3]

    def test_missing_cache_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)
        assert ibkr_data._load_cache("NOPE") is None
