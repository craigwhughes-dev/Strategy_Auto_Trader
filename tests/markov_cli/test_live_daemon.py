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
                             broker, daily_buy_limit, daily_sell_limit, **kwargs):
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
    monkeypatch.setattr(live_daemon, "acquire_process_lock",
                        lambda logger, takeover=False: True)
    monkeypatch.setattr(live_daemon, "release_process_lock", lambda logger: None)
    monkeypatch.setattr(live_daemon, "kill_stray_daemons", lambda logger: 0)
    monkeypatch.setattr(
        self_check, "run_startup_checks",
        mock.Mock(side_effect=self_check.SelfCheckError("hmm broken")))

    assert live_daemon.main([]) == 1


def test_main_self_check_skips_broker_even_when_live(monkeypatch, config):
    """Startup checks never require the broker — the connection is deferred
    so the daemon can start (and generate signals) while TWS is down."""
    from Strategy_Auto_Trader.core import self_check

    config["execution"]["dry_run"] = False
    monkeypatch.setattr(live_daemon, "setup_logging", lambda: mock.Mock())
    monkeypatch.setattr(live_daemon, "load_config", lambda: config)
    monkeypatch.setattr(live_daemon, "validate_startup_environment", lambda logger: True)
    monkeypatch.setattr(live_daemon, "acquire_process_lock",
                        lambda logger, takeover=False: True)
    monkeypatch.setattr(live_daemon, "release_process_lock", lambda logger: None)
    monkeypatch.setattr(live_daemon, "kill_stray_daemons", lambda logger: 0)

    captured = {}

    def fake_checks(*, require_hmm=True, require_broker=False, logger=None):
        captured["require_broker"] = require_broker
        raise self_check.SelfCheckError("stop here")

    monkeypatch.setattr(self_check, "run_startup_checks", fake_checks)
    assert live_daemon.main([]) == 1
    assert captured["require_broker"] is False


def test_main_keyboard_interrupt_skips_sleep(monkeypatch, config, tmp_path):
    """Ctrl+C shutdown must exit promptly — no poll-interval sleep in finally."""
    from Strategy_Auto_Trader.core import self_check

    monkeypatch.setattr(live_daemon, "setup_logging", lambda: mock.Mock())
    monkeypatch.setattr(live_daemon, "load_config", lambda: config)
    monkeypatch.setattr(live_daemon, "validate_startup_environment", lambda logger: True)
    monkeypatch.setattr(live_daemon, "acquire_process_lock",
                        lambda logger, takeover=False: True)
    monkeypatch.setattr(live_daemon, "release_process_lock", lambda logger: None)
    monkeypatch.setattr(live_daemon, "kill_stray_daemons", lambda logger: 0)
    monkeypatch.setattr(live_daemon, "cleanup_incomplete_runs", lambda d, l: 0)
    monkeypatch.setattr(self_check, "run_startup_checks", lambda **kwargs: None)
    monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)

    snapshot_written = []
    monkeypatch.setattr(live_daemon, "write_app_status_snapshot",
                        lambda *a, **k: snapshot_written.append(True))
    monkeypatch.setattr(live_daemon, "check_overnight_screening",
                        mock.Mock(side_effect=KeyboardInterrupt))

    sleeps = []
    monkeypatch.setattr(live_daemon.time, "sleep", lambda s: sleeps.append(s))

    assert live_daemon.main([]) == 0
    assert sleeps == []  # No sleep on shutdown
    assert snapshot_written  # Snapshot still written on the final iteration


class TestExecuteSignalsWithRetry:
    """Auto-reconnect and retry on socket errors."""

    def test_success_first_attempt(self, monkeypatch):
        """Successful execution on first attempt returns immediately."""
        from Strategy_Auto_Trader.markov_cli import execute as execute_mod

        monkeypatch.setattr(execute_mod, "execute_signals",
                            lambda *a, **k: (["BUY"], ["SELL"], []))

        result = live_daemon.execute_signals_with_retry(
            "ftse", ["AAPL"], None, None, None, mock.Mock(),
            2, None, mock.Mock()
        )
        assert result == (["BUY"], ["SELL"], [])

    def test_socket_error_triggers_reconnect(self, monkeypatch):
        """Socket error triggers broker disconnect/reconnect and retry."""
        from Strategy_Auto_Trader.markov_cli import execute as execute_mod

        call_count = [0]
        broker = mock.Mock()
        logger = mock.Mock()

        def fake_execute(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("Socket disconnect: not connected to 127.0.0.1:7497")
            return (["BUY"], ["SELL"], [])

        monkeypatch.setattr(execute_mod, "execute_signals", fake_execute)

        result = live_daemon.execute_signals_with_retry(
            "ftse", ["AAPL"], None, None, None, broker,
            2, None, logger, max_retries=3
        )

        assert result == (["BUY"], ["SELL"], [])
        assert broker.disconnect.called
        assert broker.connect.called

    def test_socket_error_with_reconnect_failure(self, monkeypatch):
        """Socket error retries even if reconnect fails."""
        from Strategy_Auto_Trader.markov_cli import execute as execute_mod

        call_count = [0]
        broker = mock.Mock()
        broker.connect.side_effect = RuntimeError("TWS not responding")
        logger = mock.Mock()

        def fake_execute(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ConnectionError("Socket error")
            return (["BUY"], [], [])

        monkeypatch.setattr(execute_mod, "execute_signals", fake_execute)

        result = live_daemon.execute_signals_with_retry(
            "ftse", ["AAPL"], None, None, None, broker,
            2, None, logger, max_retries=3
        )

        assert result == (["BUY"], [], [])
        assert call_count[0] == 3  # Called on attempts 1, 2, 3

    def test_socket_error_max_retries_exhausted(self, monkeypatch):
        """Socket error after max retries returns empty results, doesn't raise."""
        from Strategy_Auto_Trader.markov_cli import execute as execute_mod

        broker = mock.Mock()
        logger = mock.Mock()

        def fake_execute(*args, **kwargs):
            raise ConnectionError("Socket disconnect")

        monkeypatch.setattr(execute_mod, "execute_signals", fake_execute)

        result = live_daemon.execute_signals_with_retry(
            "ftse", ["AAPL", "MSFT"], None, None, None, broker,
            2, None, logger, max_retries=2
        )

        assert result == ([], [], ["AAPL", "MSFT"])  # Tickers listed as skipped
        assert logger.error.called

    def test_timeout_error_triggers_retry(self, monkeypatch):
        """TimeoutError (socket-level) triggers reconnect."""
        from Strategy_Auto_Trader.markov_cli import execute as execute_mod

        call_count = [0]
        broker = mock.Mock()
        logger = mock.Mock()

        def fake_execute(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise TimeoutError("Socket timeout")
            return ([], ["SELL"], [])

        monkeypatch.setattr(execute_mod, "execute_signals", fake_execute)

        result = live_daemon.execute_signals_with_retry(
            "ftse", ["AAPL"], None, None, None, broker,
            2, None, logger
        )

        assert result == ([], ["SELL"], [])
        assert broker.disconnect.called

    def test_os_error_triggers_retry(self, monkeypatch):
        """OSError (socket-level) triggers reconnect."""
        from Strategy_Auto_Trader.markov_cli import execute as execute_mod

        call_count = [0]
        broker = mock.Mock()
        logger = mock.Mock()

        def fake_execute(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise OSError("[Errno 10054] Connection reset by peer")
            return (["BUY"], [], [])

        monkeypatch.setattr(execute_mod, "execute_signals", fake_execute)

        result = live_daemon.execute_signals_with_retry(
            "ftse", ["AAPL"], None, None, None, broker,
            2, None, logger
        )

        assert result == (["BUY"], [], [])

    def test_non_socket_error_raises_immediately(self, monkeypatch):
        """Non-connection errors are raised immediately without retry."""
        from Strategy_Auto_Trader.markov_cli import execute as execute_mod

        broker = mock.Mock()
        logger = mock.Mock()

        def fake_execute(*args, **kwargs):
            raise ValueError("Invalid ticker symbol")

        monkeypatch.setattr(execute_mod, "execute_signals", fake_execute)

        with pytest.raises(ValueError, match="Invalid ticker symbol"):
            live_daemon.execute_signals_with_retry(
                "ftse", ["AAPL"], None, None, None, broker,
                2, None, logger
            )

        assert not broker.disconnect.called

    def test_string_pattern_socket_error_triggers_retry(self, monkeypatch):
        """Exception with 'socket' or 'disconnect' in message triggers retry."""
        from Strategy_Auto_Trader.markov_cli import execute as execute_mod

        call_count = [0]
        broker = mock.Mock()
        logger = mock.Mock()

        def fake_execute(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Socket error: connection lost")
            return (["BUY"], ["SELL"], [])

        monkeypatch.setattr(execute_mod, "execute_signals", fake_execute)

        result = live_daemon.execute_signals_with_retry(
            "ftse", ["AAPL"], None, None, None, broker,
            2, None, logger
        )

        assert result == (["BUY"], ["SELL"], [])
        assert broker.disconnect.called

    def test_exponential_backoff_sleep_times(self, monkeypatch):
        """Retries use exponential backoff: 1s, 2s, 4s."""
        from Strategy_Auto_Trader.markov_cli import execute as execute_mod
        import time

        call_count = [0]
        sleep_times = []
        broker = mock.Mock()
        logger = mock.Mock()

        def fake_execute(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 4:
                raise ConnectionError("Socket error")
            return (["BUY"], [], [])

        def fake_sleep(secs):
            sleep_times.append(secs)

        monkeypatch.setattr(execute_mod, "execute_signals", fake_execute)
        monkeypatch.setattr(time, "sleep", fake_sleep)

        result = live_daemon.execute_signals_with_retry(
            "ftse", ["AAPL"], None, None, None, broker,
            2, None, logger, max_retries=4
        )

        assert result == (["BUY"], [], [])
        # Expected: exponential backoff [1, 2, 4] + intermediate 0.5s pauses between disconnect/reconnect
        assert sleep_times == [1, 0.5, 2, 0.5, 4, 0.5]


class TestAppStatusSnapshot:
    """Phase 0: app_status.json snapshot generation."""

    def test_write_app_status_snapshot_creates_file(self, tmp_path, monkeypatch):
        """write_app_status_snapshot writes app_status.json atomically."""
        from Strategy_Auto_Trader.markov_cli import live_daemon
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult

        monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)
        monkeypatch.setattr("Strategy_Auto_Trader.markov_cli.live_daemon.os.getpid", lambda: 12345)

        # Create a portfolio with one position
        pm = PortfolioManager(20_000, 5, tmp_path / "execution_state.json")
        fill = FillResult("AAPL", "BUY", 195.0, 10, "2026-07-01T00:00:00+00:00")
        pm.record_entry("AAPL", fill, 0.15, 185.0, 224.0, market="sp500", currency="USD")

        daemon_state = {
            "halt_new_entries": False,
            "reconciliation_discrepancies": [],
            "last_reconcile_date": "2026-07-01",
        }
        config = {
            "execution": {"dry_run": True},
            "markets": {
                "sp500": {"timezone": "America/New_York", "trading_start": "09:30", "trading_end": "16:00"},
            },
        }
        last_cycle_hour = {"sp500": 14}
        logger = mock.Mock()

        live_daemon.write_app_status_snapshot(pm, daemon_state, config, last_cycle_hour, logger)

        status_path = tmp_path / "app_status.json"
        assert status_path.exists()
        snapshot = json.loads(status_path.read_text(encoding="utf-8"))
        assert snapshot["schema_version"] == 1
        assert snapshot["daemon_pid"] == 12345
        assert snapshot["dry_run"] is True
        assert snapshot["halt_new_entries"] is False
        assert "AAPL" in snapshot["positions"]
        assert snapshot["positions"]["AAPL"]["market"] == "sp500"
        assert snapshot["positions"]["AAPL"]["currency"] == "USD"
        assert snapshot["positions"]["AAPL"]["quantity"] == 10

    def test_write_app_status_snapshot_includes_heartbeat(self, tmp_path, monkeypatch):
        """Snapshot heartbeat_utc is set to current time (liveness)."""
        from Strategy_Auto_Trader.markov_cli import live_daemon
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager

        monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)
        pm = PortfolioManager(20_000, 5, tmp_path / "execution_state.json")
        daemon_state = {}
        config = {"execution": {"dry_run": True}, "markets": {}}

        live_daemon.write_app_status_snapshot(pm, daemon_state, config, {}, mock.Mock())

        snapshot = json.loads((tmp_path / "app_status.json").read_text(encoding="utf-8"))
        assert "heartbeat_utc" in snapshot
        # Verify it's a valid ISO timestamp
        heartbeat = snapshot["heartbeat_utc"]
        assert "T" in heartbeat
        assert "+" in heartbeat or "Z" in heartbeat

    def test_write_app_status_snapshot_includes_halt_flag(self, tmp_path, monkeypatch):
        """Snapshot includes halt_new_entries flag for reconciliation mismatch."""
        from Strategy_Auto_Trader.markov_cli import live_daemon
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager

        monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)
        pm = PortfolioManager(20_000, 5, tmp_path / "execution_state.json")
        daemon_state = {
            "halt_new_entries": True,
            "reconciliation_discrepancies": ["SPY: broker=10, internal=5"],
        }
        config = {"execution": {"dry_run": False}, "markets": {}}

        live_daemon.write_app_status_snapshot(pm, daemon_state, config, {}, mock.Mock())

        snapshot = json.loads((tmp_path / "app_status.json").read_text(encoding="utf-8"))
        assert snapshot["halt_new_entries"] is True
        assert len(snapshot["reconciliation_discrepancies"]) == 1
        assert snapshot["reconciliation_discrepancies"][0] == "SPY: broker=10, internal=5"

    def test_get_market_currency_ftse(self):
        """get_market_currency maps FTSE markets to GBP."""
        from Strategy_Auto_Trader.markov_cli import live_daemon
        assert live_daemon.get_market_currency("ftse", {}) == "GBP"
        assert live_daemon.get_market_currency("FTSE100", {}) == "GBP"

    def test_get_market_currency_us(self):
        """get_market_currency maps US markets to USD."""
        from Strategy_Auto_Trader.markov_cli import live_daemon
        assert live_daemon.get_market_currency("sp500", {}) == "USD"
        assert live_daemon.get_market_currency("USA", {}) == "USD"


class TestProcessLock:
    """Held OS-lock single-instance enforcement."""

    @pytest.fixture(autouse=True)
    def isolated_state_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)
        yield tmp_path
        live_daemon.release_process_lock(mock.Mock())

    def test_acquire_records_own_pid(self, isolated_state_dir):
        import os
        assert live_daemon.acquire_process_lock(mock.Mock()) is True
        pid_file = isolated_state_dir / "daemon.pid"
        assert pid_file.exists()
        assert int(pid_file.read_text().split("|")[0]) == os.getpid()

    def test_second_acquire_fails_while_lock_held(self):
        logger = mock.Mock()
        assert live_daemon.acquire_process_lock(logger) is True
        assert live_daemon.acquire_process_lock(logger) is False
        assert logger.error.called

    def test_reacquire_succeeds_after_release(self):
        logger = mock.Mock()
        assert live_daemon.acquire_process_lock(logger) is True
        live_daemon.release_process_lock(logger)
        assert live_daemon.acquire_process_lock(logger) is True

    def test_release_removes_pid_file(self, isolated_state_dir):
        logger = mock.Mock()
        assert live_daemon.acquire_process_lock(logger) is True
        live_daemon.release_process_lock(logger)
        assert not (isolated_state_dir / "daemon.pid").exists()

    def test_takeover_refuses_non_daemon_holder(self):
        """PID-reuse guard: takeover must not kill a process whose cmdline
        is not a live_daemon (here: the pytest process holding the lock)."""
        logger = mock.Mock()
        assert live_daemon.acquire_process_lock(logger) is True
        assert live_daemon.acquire_process_lock(logger, takeover=True) is False
        assert logger.critical.called


class TestDaemonCmdlineMatch:
    def test_matches_module_invocation(self):
        assert live_daemon._is_daemon_cmdline(
            ["python", "-m", "Strategy_Auto_Trader.markov_cli.live_daemon"])

    def test_matches_script_path(self):
        assert live_daemon._is_daemon_cmdline(
            [r"C:\x\python.exe", r"Strategy_Auto_Trader\markov_cli\live_daemon.py"])

    def test_does_not_match_test_file(self):
        assert not live_daemon._is_daemon_cmdline(
            ["python", "-m", "pytest", "tests/markov_cli/test_live_daemon.py"])

    def test_does_not_match_empty(self):
        assert not live_daemon._is_daemon_cmdline(None)
        assert not live_daemon._is_daemon_cmdline([])


class TestKillStrayDaemons:
    """Orphan sweep kills only true stray daemon processes."""

    def _fake_psutil(self, monkeypatch, procs, me_pid=100, ancestor_pids=(50,)):
        import types

        me = mock.Mock(pid=me_pid)
        me.parents.return_value = [mock.Mock(pid=p) for p in ancestor_pids]
        me.children.return_value = []

        class NoSuchProcess(Exception):
            pass

        class AccessDenied(Exception):
            pass

        fake = types.SimpleNamespace(
            Process=lambda pid=None: me,
            process_iter=lambda attrs: procs,
            NoSuchProcess=NoSuchProcess,
            AccessDenied=AccessDenied,
        )
        monkeypatch.setattr(live_daemon, "psutil", fake)
        return fake

    @staticmethod
    def _proc(pid, name, cmdline):
        p = mock.Mock(pid=pid)
        p.info = {"pid": pid, "name": name, "cmdline": cmdline}
        return p

    def test_kills_stray_and_spares_self_ancestors_and_others(self, monkeypatch):
        daemon_cmd = ["python", "-m", "Strategy_Auto_Trader.markov_cli.live_daemon"]
        procs = [
            self._proc(200, "python.exe", daemon_cmd),          # stray -> kill
            self._proc(100, "python.exe", daemon_cmd),          # self
            self._proc(50, "python.exe", daemon_cmd),           # ancestor shim
            self._proc(300, "python.exe",
                       ["python", "-m", "pytest", "tests/x.py"]),  # unrelated python
            self._proc(400, "notepad.exe", ["notepad"]),        # not python
        ]
        self._fake_psutil(monkeypatch, procs)

        killed = []
        monkeypatch.setattr(live_daemon, "_kill_daemon_process",
                            lambda pid, logger: killed.append(pid) or True)

        assert live_daemon.kill_stray_daemons(mock.Mock()) == 1
        assert killed == [200]

    def test_no_strays_kills_nothing(self, monkeypatch):
        self._fake_psutil(monkeypatch, [
            self._proc(100, "python.exe",
                       ["python", "-m", "Strategy_Auto_Trader.markov_cli.live_daemon"]),
        ])
        killed = []
        monkeypatch.setattr(live_daemon, "_kill_daemon_process",
                            lambda pid, logger: killed.append(pid) or True)
        assert live_daemon.kill_stray_daemons(mock.Mock()) == 0
        assert killed == []

    def test_psutil_missing_skips_sweep(self, monkeypatch):
        monkeypatch.setattr(live_daemon, "psutil", None)
        logger = mock.Mock()
        assert live_daemon.kill_stray_daemons(logger) == 0
        assert logger.warning.called


def test_process_cycle_paused_by_user_blocks_new_entries(monkeypatch):
    """paused_by_user forces daily_buy_limit=0 into execute_signals."""
    from Strategy_Auto_Trader.markov_cli import batch, execute

    def fake_process_ticker(ticker_cfg, defaults, send_email):
        return {"ticker": "AAPL", "status": "OK", "time": 0.0,
                "result": {"ticker": "AAPL", "close": 100.0}}

    captured = {}

    def fake_execute_signals(tickers, data_dir, portfolio, limit_tracker,
                             broker, daily_buy_limit, daily_sell_limit, **kwargs):
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
    daemon_state = {"cursors": {}, "paused_by_user": True}
    live_daemon.process_cycle(
        "test_market", {}, config, daemon_state,
        portfolio=mock.Mock(), broker=mock.Mock(spec=[]), logger=mock.Mock(),
    )
    assert captured["daily_buy_limit"] == 0


def test_process_cycle_halt_and_paused_independent(monkeypatch):
    """halt_new_entries and paused_by_user flags are independent."""
    from Strategy_Auto_Trader.markov_cli import batch, execute

    def fake_process_ticker(ticker_cfg, defaults, send_email):
        return {"ticker": "AAPL", "status": "OK", "time": 0.0,
                "result": {"ticker": "AAPL", "close": 100.0}}

    captured = []

    def fake_execute_signals(tickers, data_dir, portfolio, limit_tracker,
                             broker, daily_buy_limit, daily_sell_limit, **kwargs):
        captured.append(daily_buy_limit)
        return [], [], []

    monkeypatch.setattr(batch, "process_ticker", fake_process_ticker)
    monkeypatch.setattr(execute, "execute_signals", fake_execute_signals)
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["AAPL"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, a, l: [])

    config = {
        "daytime": {"max_seconds_per_cycle": 60, "cycle_buffer_minutes": 0},
        "execution": {"daily_buy_limit": 5},
    }

    # Test 1: halt_new_entries alone
    daemon_state1 = {"cursors": {}, "halt_new_entries": True}
    live_daemon.process_cycle(
        "test_market", {}, config, daemon_state1,
        portfolio=mock.Mock(), broker=mock.Mock(spec=[]), logger=mock.Mock(),
    )

    # Test 2: paused_by_user alone
    daemon_state2 = {"cursors": {}, "paused_by_user": True}
    live_daemon.process_cycle(
        "test_market", {}, config, daemon_state2,
        portfolio=mock.Mock(), broker=mock.Mock(spec=[]), logger=mock.Mock(),
    )

    # Both should have resulted in daily_buy_limit=0
    assert captured == [0, 0]


def test_write_app_status_snapshot_includes_paused_by_user(tmp_path, monkeypatch):
    """Snapshot includes paused_by_user flag."""
    from Strategy_Auto_Trader.markov_cli import live_daemon
    from Strategy_Auto_Trader.broker.portfolio import PortfolioManager

    monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)
    pm = PortfolioManager(20_000, 5, tmp_path / "execution_state.json")
    daemon_state = {
        "paused_by_user": True,
    }
    config = {"execution": {"dry_run": True}, "markets": {}}

    live_daemon.write_app_status_snapshot(pm, daemon_state, config, {}, mock.Mock())

    snapshot = json.loads((tmp_path / "app_status.json").read_text(encoding="utf-8"))
    assert snapshot["paused_by_user"] is True


def test_write_app_status_snapshot_paused_by_user_defaults_false(tmp_path, monkeypatch):
    """Snapshot includes paused_by_user, defaults to False if not in daemon_state."""
    from Strategy_Auto_Trader.markov_cli import live_daemon
    from Strategy_Auto_Trader.broker.portfolio import PortfolioManager

    monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)
    pm = PortfolioManager(20_000, 5, tmp_path / "execution_state.json")
    daemon_state = {}  # No paused_by_user key
    config = {"execution": {"dry_run": True}, "markets": {}}

    live_daemon.write_app_status_snapshot(pm, daemon_state, config, {}, mock.Mock())

    snapshot = json.loads((tmp_path / "app_status.json").read_text(encoding="utf-8"))
    assert snapshot["paused_by_user"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
