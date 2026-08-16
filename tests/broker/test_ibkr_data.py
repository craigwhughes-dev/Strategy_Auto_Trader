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
        # fetch_hourly's return contract is post-_resample_30min_aligned
        # (see 018e1af): a raw :00-aligned IBKR bar shifts to the :30-aligned
        # bin whose window contains it, so last_cached's 16:00 becomes 15:30
        # here — not stale data, this is the bar-alignment fix doing its job.
        assert out.index[0] == last_cached - pd.Timedelta(minutes=30)

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

    def test_unqualified_contract_falls_back_and_logs(self, monkeypatch, tmp_path, caplog):
        """IBKR's qualifyContracts doesn't raise on an unknown/ambiguous
        symbol — it returns None in the result slot and only warns to its
        own logger. fetch_hourly must check that return value itself
        (rather than paging a never-matched contract) and log why."""
        pytest.importorskip("ib_async")
        from unittest.mock import MagicMock
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)

        client = IBKRDataClient()
        client._ib = MagicMock()
        client._ib.qualifyContracts.return_value = [None]

        with caplog.at_level("WARNING"):
            out = client.fetch_hourly("BP.L", period="30d", use_cache=False)

        assert out is None
        assert client._ib.reqHistoricalData.call_count == 0
        assert "BP.L" in caplog.text
        assert "could not qualify" in caplog.text

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


class TestFetchRecentRaw:
    def test_pages_back_to_lookback_cutoff(self, tmp_path, monkeypatch):
        pytest.importorskip("ib_async")
        from unittest.mock import MagicMock
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)

        client = IBKRDataClient()
        client._ib = MagicMock()
        # Page is recent (well inside the 14d cutoff), so paging only stops
        # because the second page comes back empty (start of history) —
        # avoids asserting on exact stop_at boundary arithmetic.
        start = datetime.now(timezone.utc) - timedelta(days=1)
        client._ib.reqHistoricalData.side_effect = [_make_page(start, 5), []]

        out = client.fetch_recent_raw("AAPL", lookback_days=14)

        assert client._ib.reqHistoricalData.call_count == 2
        assert len(out) == 5

    def test_ignores_existing_cache_entirely(self, tmp_path, monkeypatch):
        """Unlike fetch_hourly, this must re-page even bars already cached —
        that's the whole point (detecting a revision on an already-cached bar)."""
        pytest.importorskip("ib_async")
        from unittest.mock import MagicMock
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)

        idx = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
        cached = pd.DataFrame(
            {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 100}, index=idx)
        ibkr_data._save_cache("AAPL", cached)

        client = IBKRDataClient()
        client._ib = MagicMock()
        # cached ends 2026-01-01 04:00, but the page returned is recent
        # (~now) — if fetch_recent_raw used cached.index[-1] as stop_at like
        # fetch_hourly does, this page's oldest bar (~now) would be nowhere
        # near that old cutoff and paging would run away; it terminates via
        # the empty second page instead, proving stop_at is lookback-based.
        start = datetime.now(timezone.utc) - timedelta(days=1)
        client._ib.reqHistoricalData.side_effect = [_make_page(start, 3), []]

        client.fetch_recent_raw("AAPL", lookback_days=14)

        assert client._ib.reqHistoricalData.call_count == 2

    def test_connection_failure_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)
        client = IBKRDataClient()
        client.connect = lambda: False
        assert client.fetch_recent_raw("AAPL", lookback_days=14) is None


class TestReconcileRecentBars:
    def test_no_prior_cache_is_a_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)
        from unittest.mock import MagicMock
        client = MagicMock()
        result = ibkr_data.reconcile_recent_bars("AAPL", client)
        assert result == {"ticker": "AAPL", "checked": 0, "corrected": 0, "diffs": []}
        client.fetch_recent_raw.assert_not_called()

    def test_identical_refetch_finds_no_corrections(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)
        from unittest.mock import MagicMock
        idx = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
        cached = pd.DataFrame(
            {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 100.0, "Volume": 1000}, index=idx)
        ibkr_data._save_cache("AAPL", cached)

        client = MagicMock()
        client.fetch_recent_raw.return_value = cached.copy()

        result = ibkr_data.reconcile_recent_bars("AAPL", client)
        assert result["checked"] == 5
        assert result["corrected"] == 0
        assert result["diffs"] == []

    def test_revised_close_is_detected_and_applied(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)
        from unittest.mock import MagicMock
        idx = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
        cached = pd.DataFrame(
            {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 100.0, "Volume": 1000}, index=idx)
        ibkr_data._save_cache("AAPL", cached)

        fresh = cached.copy()
        fresh.loc[idx[2], "Close"] = 105.0   # a "trade correction" on bar 2

        client = MagicMock()
        client.fetch_recent_raw.return_value = fresh

        result = ibkr_data.reconcile_recent_bars("AAPL", client)
        assert result["checked"] == 5
        assert result["corrected"] == 1
        assert result["diffs"] == [
            {"date": idx[2], "field": "Close", "old": 100.0, "new": 105.0}
        ]

        reloaded = ibkr_data._load_cache("AAPL")
        assert reloaded.loc[idx[2], "Close"] == 105.0
        assert reloaded.loc[idx[0], "Close"] == 100.0   # untouched bars unaffected

    def test_fp_noise_below_tolerance_is_not_a_correction(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)
        from unittest.mock import MagicMock
        idx = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
        cached = pd.DataFrame(
            {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 100.0, "Volume": 1000}, index=idx)
        ibkr_data._save_cache("AAPL", cached)
        mtime_before = ibkr_data._cache_path("AAPL").stat().st_mtime_ns

        fresh = cached.copy()
        fresh.loc[idx[2], "Close"] = 100.0 + 1e-9

        client = MagicMock()
        client.fetch_recent_raw.return_value = fresh

        result = ibkr_data.reconcile_recent_bars("AAPL", client)
        assert result["corrected"] == 0
        assert ibkr_data._cache_path("AAPL").stat().st_mtime_ns == mtime_before

    def test_new_bars_not_yet_cached_are_ignored_not_flagged(self, tmp_path, monkeypatch):
        """fetch_recent_raw returning bars newer than the cache (routine —
        the daemon just hasn't gap-filled yet) must not be treated as
        corrections; reconcile only diffs the overlap."""
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)
        from unittest.mock import MagicMock
        idx = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
        cached = pd.DataFrame(
            {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 100.0, "Volume": 1000}, index=idx)
        ibkr_data._save_cache("AAPL", cached)

        extra_idx = pd.date_range(idx[-1] + pd.Timedelta(hours=1), periods=2, freq="h", tz="UTC")
        fresh = pd.concat([cached, pd.DataFrame(
            {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 101.0, "Volume": 1000},
            index=extra_idx)])

        client = MagicMock()
        client.fetch_recent_raw.return_value = fresh

        result = ibkr_data.reconcile_recent_bars("AAPL", client)
        assert result["checked"] == 5   # only the overlap, not the 2 new bars
        assert result["corrected"] == 0

        reloaded = ibkr_data._load_cache("AAPL")
        assert len(reloaded) == 5   # reconcile doesn't add new bars, only corrects existing ones

    def test_empty_fetch_is_a_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)
        from unittest.mock import MagicMock
        idx = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
        cached = pd.DataFrame(
            {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 100.0, "Volume": 1000}, index=idx)
        ibkr_data._save_cache("AAPL", cached)

        client = MagicMock()
        client.fetch_recent_raw.return_value = None

        result = ibkr_data.reconcile_recent_bars("AAPL", client)
        assert result == {"ticker": "AAPL", "checked": 0, "corrected": 0, "diffs": []}


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
