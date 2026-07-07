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
    from ib_insync.objects import BarData
    return BarData(date=date, open=o, high=h, low=l, close=c, volume=v)


def _make_page(start: datetime, n_hours: int) -> list:
    return [_bar(start + timedelta(hours=i)) for i in range(n_hours)]


class TestConnect:
    def test_connect_success(self):
        pytest.importorskip("ib_insync")
        from unittest.mock import patch
        client = IBKRDataClient(port=4002, client_id=9)
        with patch("ib_insync.IB") as MockIB:
            assert client.connect() is True
        MockIB.return_value.connect.assert_called_once_with(
            "127.0.0.1", 4002, clientId=9, timeout=30.0)

    def test_connect_failure_returns_false_not_raise(self):
        pytest.importorskip("ib_insync")
        from unittest.mock import patch
        client = IBKRDataClient()
        with patch("ib_insync.IB") as MockIB:
            MockIB.return_value.connect.side_effect = ConnectionRefusedError("no TWS")
            assert client.connect() is False
        assert client._ib is None

    def test_connect_missing_ib_insync_returns_false(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "ib_insync":
                raise ImportError("no module")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        client = IBKRDataClient()
        assert client.connect() is False


class TestFetchHourly:
    def test_pages_until_period_covered(self, tmp_path, monkeypatch):
        pytest.importorskip("ib_insync")
        from unittest.mock import MagicMock
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)

        client = IBKRDataClient()
        client._ib = MagicMock()
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        # Two pages of 30 "days" each (represented here by bar count, since
        # the loop only cares about days_covered += 30 per page and the
        # returned bars' own date range for cursoring).
        page1 = _make_page(start + timedelta(days=30), 5)
        page2 = _make_page(start, 5)
        client._ib.reqHistoricalData.side_effect = [page1, page2]

        out = client.fetch_hourly("AAPL", period="45d", use_cache=False)

        assert client._ib.reqHistoricalData.call_count == 2
        assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(out) == 10
        assert out.index.is_monotonic_increasing

    def test_stops_when_page_returns_no_bars(self, tmp_path, monkeypatch):
        pytest.importorskip("ib_insync")
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

    def test_uses_cache_when_span_already_covers_period(self, monkeypatch, tmp_path):
        pytest.importorskip("ib_insync")
        from unittest.mock import MagicMock
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)
        idx = pd.date_range("2020-01-01", periods=3000, freq="h", tz="UTC")
        cached = pd.DataFrame(
            {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 100}, index=idx)
        ibkr_data._save_cache("AAPL", cached)

        client = IBKRDataClient()
        client._ib = MagicMock()   # would blow up if called
        out = client.fetch_hourly("AAPL", period="100d", use_cache=True)

        client._ib.reqHistoricalData.assert_not_called()
        assert len(out) == 3000


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
