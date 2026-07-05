"""Tests for live_daemon.py."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import pytest

from Strategy_Auto_Trader.markov_cli import live_daemon


@pytest.fixture
def config():
    """Sample overnight_strategy.json."""
    return {
        "markets": {
            "test_market": {
                "watchlist": "config/watchlist.json",
                "timezone": "Europe/London",
                "trading_start": "09:00",
                "trading_end": "17:00",
            }
        },
        "overnight_run_time": "02:00",
        "overnight_timezone": "Europe/London",
        "daytime": {
            "cycle_buffer_minutes": 5,
            "max_seconds_per_cycle": 1500,
            "poll_interval_seconds": 60,
        },
        "execution": {
            "capital_pot": 20000,
            "max_positions": 5,
            "daily_buy_limit": 2,
            "daily_sell_limit": None,
            "dry_run": True,
        },
    }


def test_is_trading_hours_during_trading():
    """Market is open during trading hours."""
    market_cfg = {
        "timezone": "Europe/London",
        "trading_start": "09:00",
        "trading_end": "17:00",
    }
    logger = mock.Mock()

    # Wednesday 10:30 in Europe/London
    with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.datetime") as mock_dt:
        mock_now = mock.Mock()
        mock_now.weekday.return_value = 2  # Wednesday
        mock_now.time.return_value = datetime(2026, 7, 1, 10, 30).time()
        mock_dt.now.return_value = mock_now
        mock_dt.strptime = datetime.strptime

        result = live_daemon.is_trading_hours(market_cfg, logger)
        assert result is True


def test_is_trading_hours_before_open():
    """Market is closed before opening time."""
    market_cfg = {
        "timezone": "Europe/London",
        "trading_start": "09:00",
        "trading_end": "17:00",
    }
    logger = mock.Mock()

    # Wednesday 08:30 (before 09:00)
    with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.datetime") as mock_dt:
        mock_now = mock.Mock()
        mock_now.weekday.return_value = 2  # Wednesday
        mock_now.time.return_value = datetime(2026, 7, 1, 8, 30).time()
        mock_dt.now.return_value = mock_now
        mock_dt.strptime = datetime.strptime

        result = live_daemon.is_trading_hours(market_cfg, logger)
        assert result is False


def test_is_trading_hours_weekend():
    """Market is closed on weekends."""
    market_cfg = {
        "timezone": "Europe/London",
        "trading_start": "09:00",
        "trading_end": "17:00",
    }
    logger = mock.Mock()

    # Saturday 10:30
    with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.datetime") as mock_dt:
        mock_now = mock.Mock()
        mock_now.weekday.return_value = 5  # Saturday
        mock_now.time.return_value = datetime(2026, 7, 4, 10, 30).time()
        mock_dt.now.return_value = mock_now
        mock_dt.strptime = datetime.strptime

        result = live_daemon.is_trading_hours(market_cfg, logger)
        assert result is False


def test_next_round_robin_slice_advances_cursor():
    """Round-robin cursor advances and wraps correctly."""
    in_scope = ["A", "B", "C", "D", "E"]
    daemon_state = {"cursors": {}}
    logger = mock.Mock()

    with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.datetime") as mock_dt:
        mock_dt.now.return_value = mock.Mock(date=mock.Mock(return_value=mock.Mock(isoformat=mock.Mock(return_value="2026-07-03"))))

        slice1 = live_daemon.next_round_robin_slice("test", in_scope, 2, daemon_state, logger)
        assert slice1 == ["A", "B"]

        slice2 = live_daemon.next_round_robin_slice("test", in_scope, 2, daemon_state, logger)
        assert slice2 == ["C", "D"]

        slice3 = live_daemon.next_round_robin_slice("test", in_scope, 2, daemon_state, logger)
        assert slice3 == ["E"]

        slice4 = live_daemon.next_round_robin_slice("test", in_scope, 2, daemon_state, logger)
        assert slice4 == ["A", "B"]
        logger.debug.assert_called_with("  test: round-robin wrapped")


def test_next_round_robin_empty_in_scope():
    """Round-robin handles empty in-scope list."""
    daemon_state = {"cursors": {}}
    logger = mock.Mock()

    slice_result = live_daemon.next_round_robin_slice("test", [], 5, daemon_state, logger)
    assert slice_result == []


def test_get_open_positions_filters_by_market_tickers():
    """Open positions are filtered to only those in the market's ticker list."""
    exec_state = {
        "positions": {
            "FTSE_TICKER": {"quantity": 10},
            "SP500_TICKER": {"quantity": 5},
        }
    }
    market_tickers = ["FTSE_TICKER", "OTHER_FTSE"]
    logger = mock.Mock()

    with mock.patch("pathlib.Path.exists", return_value=True):
        with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(exec_state))):
            result = live_daemon.get_open_positions("ftse", market_tickers, logger)

            assert "FTSE_TICKER" in result
            assert "SP500_TICKER" not in result


def test_check_overnight_screening_updates_state():
    """Overnight screening is called once per day at the right time."""
    config = {
        "overnight_run_time": "02:00",
        "overnight_timezone": "Europe/London",
        "markets": {},
    }
    daemon_state = {"last_overnight_date": None}
    logger = mock.Mock()

    with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.datetime") as mock_dt:
        # Exactly at 02:00
        mock_now = datetime(2026, 7, 3, 2, 0, tzinfo=ZoneInfo("Europe/London"))
        mock_dt.now.return_value = mock_now

        with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.main") as mock_overnight:
            with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.save_daemon_state"):
                live_daemon.check_overnight_screening(config, daemon_state, logger)

                assert mock_overnight.called
                assert daemon_state["last_overnight_date"] == "2026-07-03"


def test_check_overnight_screening_skips_if_already_run():
    """Overnight screening doesn't run twice in same day."""
    config = {
        "overnight_run_time": "02:00",
        "overnight_timezone": "Europe/London",
        "markets": {},
    }
    daemon_state = {"last_overnight_date": "2026-07-03"}
    logger = mock.Mock()

    with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.datetime") as mock_dt:
        mock_now = datetime(2026, 7, 3, 12, 0, tzinfo=ZoneInfo("Europe/London"))
        mock_dt.now.return_value = mock_now

        with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.main") as mock_overnight:
            live_daemon.check_overnight_screening(config, daemon_state, logger)
            assert not mock_overnight.called


def _run_process_cycle_capture_defaults(monkeypatch, market_cfg):
    """Run process_cycle with stubbed I/O, capturing what process_ticker receives."""
    from Strategy_Auto_Trader.markov_cli import batch

    captured = {}

    def fake_process_ticker(ticker_cfg, defaults, send_email):
        captured["ticker_cfg"] = ticker_cfg
        captured["defaults"] = defaults
        # FAIL status keeps process_cycle away from the execution stage
        return {"ticker": ticker_cfg["ticker"], "status": "FAIL: stub", "time": 0.0}

    monkeypatch.setattr(batch, "process_ticker", fake_process_ticker)
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["AAPL"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, a, l: [])

    config = {"daytime": {"max_seconds_per_cycle": 60, "cycle_buffer_minutes": 0}}
    live_daemon.process_cycle(
        "test_market", market_cfg, config, {"cursors": {}},
        portfolio=None, broker=None, logger=mock.Mock(),
    )
    return captured


def test_process_cycle_defaults_signal_reports_only_on(monkeypatch):
    """Daemon cycles default to rendering chart/report only on a signal event."""
    captured = _run_process_cycle_capture_defaults(monkeypatch, market_cfg={})
    assert captured["defaults"]["signal_reports_only"] is True


def test_process_cycle_feeds_closes_to_dry_run_broker(monkeypatch):
    """The NullBroker must be given this cycle's closes so dry fills are priced."""
    from Strategy_Auto_Trader.broker.null_adapter import NullBroker
    from Strategy_Auto_Trader.markov_cli import batch, execute

    def fake_process_ticker(ticker_cfg, defaults, send_email):
        return {"ticker": "AAPL", "status": "OK", "time": 0.0,
                "result": {"ticker": "AAPL", "close": 123.45}}

    monkeypatch.setattr(batch, "process_ticker", fake_process_ticker)
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["AAPL"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, a, l: [])
    monkeypatch.setattr(execute, "execute_signals",
                        lambda *a, **k: ([], [], []))

    broker = NullBroker(prices={})
    config = {"daytime": {"max_seconds_per_cycle": 60, "cycle_buffer_minutes": 0}}
    live_daemon.process_cycle(
        "test_market", {}, config, {"cursors": {}},
        portfolio=mock.Mock(), broker=broker, logger=mock.Mock(),
    )
    assert broker.get_last_price("AAPL") == pytest.approx(123.45)


def test_process_cycle_market_config_can_override_reports_default(monkeypatch):
    captured = _run_process_cycle_capture_defaults(
        monkeypatch, market_cfg={"defaults": {"signal_reports_only": False}})
    assert captured["defaults"]["signal_reports_only"] is False


class TestReconciliation:
    """run_reconciliation / check_nightly_reconciliation / halt enforcement."""

    def _portfolio(self, positions):
        p = mock.Mock()
        p.positions = positions
        return p

    def _broker(self, positions):
        b = mock.Mock()
        b.get_open_positions.return_value = positions
        return b

    def test_clean_pass_clears_halt(self, monkeypatch):
        monkeypatch.setattr(live_daemon, "save_daemon_state", lambda s: None)
        daemon_state = {"halt_new_entries": True}

        outcome = live_daemon.run_reconciliation(
            self._portfolio({"SPY": {"quantity": 10}}),
            self._broker({"SPY": 10}),
            daemon_state, mock.Mock(),
        )
        assert outcome == "clean"
        assert daemon_state["halt_new_entries"] is False
        assert daemon_state["reconciliation_discrepancies"] == []

    def test_mismatch_sets_halt_and_emails(self, monkeypatch):
        monkeypatch.setattr(live_daemon, "save_daemon_state", lambda s: None)
        sent = {}
        from Strategy_Auto_Trader.output import emailer
        monkeypatch.setattr(emailer, "send_reconciliation_alert",
                            lambda d: sent.update(discrepancies=d))
        daemon_state = {}

        outcome = live_daemon.run_reconciliation(
            self._portfolio({"SPY": {"quantity": 10}}),
            self._broker({"SPY": 7}),
            daemon_state, mock.Mock(),
        )
        assert outcome == "mismatch"
        assert daemon_state["halt_new_entries"] is True
        assert len(daemon_state["reconciliation_discrepancies"]) == 1
        assert len(sent["discrepancies"]) == 1

    def test_email_failure_does_not_mask_mismatch(self, monkeypatch):
        monkeypatch.setattr(live_daemon, "save_daemon_state", lambda s: None)
        from Strategy_Auto_Trader.output import emailer
        monkeypatch.setattr(emailer, "send_reconciliation_alert",
                            mock.Mock(side_effect=RuntimeError("smtp down")))
        daemon_state = {}

        outcome = live_daemon.run_reconciliation(
            self._portfolio({"SPY": {"quantity": 10}}),
            self._broker({}),
            daemon_state, mock.Mock(),
        )
        assert outcome == "mismatch"
        assert daemon_state["halt_new_entries"] is True

    def test_broker_fetch_error_returns_error_and_keeps_halt(self, monkeypatch):
        monkeypatch.setattr(live_daemon, "save_daemon_state", lambda s: None)
        broker = mock.Mock()
        broker.get_open_positions.side_effect = ConnectionError("TWS gone")
        daemon_state = {"halt_new_entries": True}

        outcome = live_daemon.run_reconciliation(
            self._portfolio({}), broker, daemon_state, mock.Mock(),
        )
        assert outcome == "error"
        assert daemon_state["halt_new_entries"] is True

    def _nightly(self, monkeypatch, daemon_state, outcome, at_hour=21, at_minute=30):
        config = {"overnight_timezone": "Europe/London",
                  "reconciliation_run_time": "21:30"}
        monkeypatch.setattr(live_daemon, "save_daemon_state", lambda s: None)
        run_mock = mock.Mock(return_value=outcome)
        monkeypatch.setattr(live_daemon, "run_reconciliation", run_mock)
        with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(
                2026, 7, 6, at_hour, at_minute, tzinfo=ZoneInfo("Europe/London"))
            live_daemon.check_nightly_reconciliation(
                config, daemon_state, mock.Mock(), mock.Mock(), mock.Mock())
        return run_mock

    def test_nightly_runs_at_configured_time(self, monkeypatch):
        daemon_state = {}
        run_mock = self._nightly(monkeypatch, daemon_state, "clean")
        assert run_mock.called
        assert daemon_state["last_reconcile_date"] == "2026-07-06"

    def test_nightly_skips_before_run_time(self, monkeypatch):
        daemon_state = {}
        run_mock = self._nightly(monkeypatch, daemon_state, "clean", at_hour=15)
        assert not run_mock.called
        assert "last_reconcile_date" not in daemon_state

    def test_nightly_skips_if_already_run_today(self, monkeypatch):
        daemon_state = {"last_reconcile_date": "2026-07-06"}
        run_mock = self._nightly(monkeypatch, daemon_state, "clean")
        assert not run_mock.called

    def test_nightly_mismatch_still_marks_date(self, monkeypatch):
        daemon_state = {}
        self._nightly(monkeypatch, daemon_state, "mismatch")
        assert daemon_state["last_reconcile_date"] == "2026-07-06"

    def test_nightly_fetch_error_retries(self, monkeypatch):
        daemon_state = {}
        self._nightly(monkeypatch, daemon_state, "error")
        assert "last_reconcile_date" not in daemon_state


def test_process_cycle_halt_flag_blocks_new_entries(monkeypatch):
    """halt_new_entries forces daily_buy_limit=0 into execute_signals."""
    from Strategy_Auto_Trader.markov_cli import batch, execute

    def fake_process_ticker(ticker_cfg, defaults, send_email):
        return {"ticker": "AAPL", "status": "OK", "time": 0.0,
                "result": {"ticker": "AAPL", "close": 100.0}}

    captured = {}

    def fake_execute_signals(tickers, data_dir, portfolio, limit_tracker,
                             broker, daily_buy_limit, daily_sell_limit):
        captured["daily_buy_limit"] = daily_buy_limit
        return [], [], []

    monkeypatch.setattr(batch, "process_ticker", fake_process_ticker)
    monkeypatch.setattr(execute, "execute_signals", fake_execute_signals)
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["AAPL"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, a, l: [])

    config = {
        "daytime": {"max_seconds_per_cycle": 60, "cycle_buffer_minutes": 0},
        "execution": {"daily_buy_limit": 5},
    }
    daemon_state = {"cursors": {}, "halt_new_entries": True}
    live_daemon.process_cycle(
        "test_market", {}, config, daemon_state,
        portfolio=mock.Mock(), broker=mock.Mock(spec=[]), logger=mock.Mock(),
    )
    assert captured["daily_buy_limit"] == 0


def test_main_refuses_to_start_when_self_check_fails(monkeypatch, config):
    """A failed startup self-check must abort the daemon with exit code 1
    before any broker/portfolio setup happens."""
    from Strategy_Auto_Trader.core import self_check

    monkeypatch.setattr(live_daemon, "setup_logging", lambda: mock.Mock())
    monkeypatch.setattr(live_daemon, "load_config", lambda: config)
    monkeypatch.setattr(
        self_check, "run_startup_checks",
        mock.Mock(side_effect=self_check.SelfCheckError("hmm broken")))

    assert live_daemon.main() == 1


def test_main_self_check_requires_broker_only_when_not_dry_run(monkeypatch, config):
    """dry_run=False must make the startup checks include the broker module."""
    from Strategy_Auto_Trader.core import self_check

    config["execution"]["dry_run"] = False
    monkeypatch.setattr(live_daemon, "setup_logging", lambda: mock.Mock())
    monkeypatch.setattr(live_daemon, "load_config", lambda: config)

    captured = {}

    def fake_checks(*, require_hmm=True, require_broker=False, logger=None):
        captured["require_broker"] = require_broker
        raise self_check.SelfCheckError("stop here")

    monkeypatch.setattr(self_check, "run_startup_checks", fake_checks)
    assert live_daemon.main() == 1
    assert captured["require_broker"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
