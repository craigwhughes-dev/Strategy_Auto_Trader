"""Tests for scripts/ibkr_backfill_universe.py — resumable throttled backfill."""

from __future__ import annotations

import json
from unittest import mock

import pandas as pd
import pytest


def _universe_file(tmp_path, tickers):
    path = tmp_path / "universe.json"
    path.write_text(json.dumps({"tickers": tickers}), encoding="utf-8")
    return path


class TestBackfillUniverse:
    def test_connect_failure_returns_1(self, tmp_path, monkeypatch):
        from scripts import ibkr_backfill_universe as backfill

        universe = _universe_file(tmp_path, ["AAPL"])
        with mock.patch.object(backfill.IBKRDataClient, "connect", return_value=False):
            rc = backfill.main(["--universe", str(universe)])
        assert rc == 1

    def test_skips_ticker_already_spanning_target_period(self, tmp_path, monkeypatch):
        """Resumability: a ticker whose cache already covers the target period
        must not trigger any network call — this is what makes re-running an
        interrupted backfill cheap."""
        from scripts import ibkr_backfill_universe as backfill
        from Strategy_Auto_Trader.broker import ibkr_data

        universe = _universe_file(tmp_path, ["AAPL"])
        # Date *range* is what the span check cares about, not bar density —
        # daily freq keeps this fast while still covering >1095 days.
        idx = pd.date_range("2020-01-01", periods=1200, freq="D", tz="UTC")
        cached = pd.DataFrame(
            {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 100}, index=idx)
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)
        ibkr_data._save_cache("AAPL", cached)

        with mock.patch.object(backfill.IBKRDataClient, "connect", return_value=True), \
             mock.patch.object(backfill.IBKRDataClient, "fetch_hourly") as mock_fetch, \
             mock.patch.object(backfill.IBKRDataClient, "disconnect"), \
             mock.patch("time.sleep"):
            rc = backfill.main(["--universe", str(universe), "--period", "1095d"])

        assert rc == 0
        mock_fetch.assert_not_called()

    def test_bootstraps_uncached_ticker(self, tmp_path, monkeypatch):
        from scripts import ibkr_backfill_universe as backfill
        from Strategy_Auto_Trader.broker import ibkr_data

        universe = _universe_file(tmp_path, ["AAPL"])
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)
        idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        fake_df = pd.DataFrame(
            {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 100}, index=idx)

        with mock.patch.object(backfill.IBKRDataClient, "connect", return_value=True), \
             mock.patch.object(backfill.IBKRDataClient, "fetch_hourly", return_value=fake_df) as mock_fetch, \
             mock.patch.object(backfill.IBKRDataClient, "disconnect"), \
             mock.patch("time.sleep"):
            rc = backfill.main(["--universe", str(universe), "--period", "1095d"])

        assert rc == 0
        mock_fetch.assert_called_once_with("AAPL", period="1095d")

    def test_limit_truncates_ticker_list(self, tmp_path, monkeypatch):
        from scripts import ibkr_backfill_universe as backfill
        from Strategy_Auto_Trader.broker import ibkr_data

        universe = _universe_file(tmp_path, ["AAPL", "MSFT", "GOOG"])
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)

        with mock.patch.object(backfill.IBKRDataClient, "connect", return_value=True), \
             mock.patch.object(backfill.IBKRDataClient, "fetch_hourly", return_value=None) as mock_fetch, \
             mock.patch.object(backfill.IBKRDataClient, "disconnect"), \
             mock.patch("time.sleep"):
            backfill.main(["--universe", str(universe), "--limit", "1"])

        assert mock_fetch.call_count == 1

    def test_sets_backfill_pacing_on_module(self, tmp_path, monkeypatch):
        """The wider inter-request pacing must actually be applied, not just
        computed — this is what keeps a full-universe run under IBKR's
        60-requests-per-10-minutes cap."""
        from scripts import ibkr_backfill_universe as backfill
        from Strategy_Auto_Trader.broker import ibkr_data

        universe = _universe_file(tmp_path, ["AAPL"])
        monkeypatch.setattr(ibkr_data, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(ibkr_data, "_PAGE_SLEEP_S", 2.0)

        with mock.patch.object(backfill.IBKRDataClient, "connect", return_value=True), \
             mock.patch.object(backfill.IBKRDataClient, "fetch_hourly", return_value=None), \
             mock.patch.object(backfill.IBKRDataClient, "disconnect"), \
             mock.patch("time.sleep"):
            backfill.main(["--universe", str(universe)])

        assert ibkr_data._PAGE_SLEEP_S == backfill._BACKFILL_PAGE_SLEEP_S
