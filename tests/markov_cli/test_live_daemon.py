"""Tests for live_daemon.py."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import pytest

from Strategy_Auto_Trader.markov_cli import live_daemon


@pytest.fixture(autouse=True)
def _stub_manual_commands_wrapper(monkeypatch):
    """process_cycle now calls process_manual_commands_wrapper() between tickers
    (item 8: interleaved manual-command polling). That wrapper defaults to the
    real repo's state/commands/ directory (no commands_dir override), so without
    this stub every process_cycle test would read/claim real pending command
    files on disk. Tests that specifically exercise the interleave behavior
    override this with their own monkeypatch.setattr call."""
    monkeypatch.setattr(live_daemon, "process_manual_commands_wrapper", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _stub_app_status_snapshot(monkeypatch):
    """process_cycle also calls _write_app_status_snapshot_safe() between tickers
    now (heartbeat staleness fix), which writes to the real repo's
    state/app_status.json (STATE_DIR has no override). Without this stub, every
    process_cycle test would overwrite the live daemon's real heartbeat file
    with test fixture data. Tests exercising the snapshot interleave itself
    override this with their own monkeypatch.setattr call."""
    monkeypatch.setattr(live_daemon, "_write_app_status_snapshot_safe", lambda *a, **k: None)


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
    now = datetime(2026, 7, 1, 10, 30, tzinfo=ZoneInfo("Europe/London"))
    result = live_daemon.is_trading_hours(market_cfg, logger, now=now)
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
    now = datetime(2026, 7, 1, 8, 30, tzinfo=ZoneInfo("Europe/London"))
    result = live_daemon.is_trading_hours(market_cfg, logger, now=now)
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
    now = datetime(2026, 7, 4, 10, 30, tzinfo=ZoneInfo("Europe/London"))
    result = live_daemon.is_trading_hours(market_cfg, logger, now=now)
    assert result is False


def test_next_round_robin_slice_resumes_from_actual_progress():
    """Round-robin resumes from however far the previous cycle actually got
    (not a fixed batch size), and wraps once the full list has been covered."""
    in_scope = ["A", "B", "C", "D", "E"]
    daemon_state = {"cursors": {}}
    logger = mock.Mock()

    with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.datetime") as mock_dt:
        mock_dt.now.return_value = mock.Mock(date=mock.Mock(return_value=mock.Mock(isoformat=mock.Mock(return_value="2026-07-03"))))

        # First cycle: full remaining list offered, but budget only allows 2.
        candidates1, cursor1 = live_daemon.next_round_robin_slice("test", in_scope, daemon_state, logger)
        assert candidates1 == ["A", "B", "C", "D", "E"]
        assert cursor1 == 0
        live_daemon.advance_round_robin_cursor("test", len(in_scope), cursor1, 2, daemon_state, logger)

        # Second cycle: resumes at C (not back at A), budget allows 2 more.
        candidates2, cursor2 = live_daemon.next_round_robin_slice("test", in_scope, daemon_state, logger)
        assert candidates2 == ["C", "D", "E"]
        assert cursor2 == 2
        live_daemon.advance_round_robin_cursor("test", len(in_scope), cursor2, 2, daemon_state, logger)

        # Third cycle: resumes at E, processes the last one, wraps.
        candidates3, cursor3 = live_daemon.next_round_robin_slice("test", in_scope, daemon_state, logger)
        assert candidates3 == ["E"]
        assert cursor3 == 4
        live_daemon.advance_round_robin_cursor("test", len(in_scope), cursor3, 1, daemon_state, logger)
        logger.debug.assert_called_with("  test: round-robin wrapped")

        # Fourth cycle: back at the top.
        candidates4, cursor4 = live_daemon.next_round_robin_slice("test", in_scope, daemon_state, logger)
        assert candidates4 == ["A", "B", "C", "D", "E"]
        assert cursor4 == 0


def test_next_round_robin_empty_in_scope():
    """Round-robin handles empty in-scope list."""
    daemon_state = {"cursors": {}}
    logger = mock.Mock()

    candidates, cursor = live_daemon.next_round_robin_slice("test", [], daemon_state, logger)
    assert candidates == []
    assert cursor == 0


def test_get_open_positions_filters_by_recorded_market_field():
    """Open positions are filtered by each position's own "market" field,
    not by watchlist/ticker-list membership."""
    exec_state = {
        "positions": {
            "FTSE_TICKER": {"quantity": 10, "market": "ftse"},
            "SP500_TICKER": {"quantity": 5, "market": "sp500"},
        }
    }
    logger = mock.Mock()

    with mock.patch("pathlib.Path.exists", return_value=True):
        with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(exec_state))):
            result = live_daemon.get_open_positions("ftse", logger)

            assert "FTSE_TICKER" in result
            assert "SP500_TICKER" not in result


def test_get_open_positions_includes_ticker_dropped_from_watchlist():
    """A position stays attributed to its market by the recorded "market"
    field even if the ticker has been removed from that market's watchlist
    entirely — this is the actual gap being fixed (previously used ticker-
    list membership as a proxy, which silently dropped orphaned positions)."""
    exec_state = {
        "positions": {
            "ORPHANED_TICKER": {"quantity": 10, "market": "ftse"},
        }
    }
    logger = mock.Mock()

    with mock.patch("pathlib.Path.exists", return_value=True):
        with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(exec_state))):
            result = live_daemon.get_open_positions("ftse", logger)

            assert "ORPHANED_TICKER" in result


def test_get_open_positions_missing_market_field_defaults_permissive():
    """Legacy position data without a "market" field defaults to matching
    the current market rather than being silently dropped."""
    exec_state = {
        "positions": {
            "LEGACY_TICKER": {"quantity": 10},
        }
    }
    logger = mock.Mock()

    with mock.patch("pathlib.Path.exists", return_value=True):
        with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(exec_state))):
            result = live_daemon.get_open_positions("ftse", logger)

            assert "LEGACY_TICKER" in result


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


def _ibkr_reconcile(daemon_state, at_hour=1, at_minute=0, day=6, cfg_overrides=None):
    config = {
        "overnight_timezone": "Europe/London",
        "ibkr_data_reconcile": {"enabled": True, "run_time": "01:00", **(cfg_overrides or {})},
    }
    run_mock = mock.Mock()
    logger = mock.Mock()
    with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(
            2026, 7, day, at_hour, at_minute, tzinfo=ZoneInfo("Europe/London"))
        live_daemon.check_ibkr_data_reconciliation(
            config, daemon_state, logger, run_reconcile=run_mock, save_state=lambda s: None)
    return run_mock, logger


def test_ibkr_data_reconciliation_runs_at_configured_time():
    daemon_state = {}
    run_mock, _ = _ibkr_reconcile(daemon_state)
    assert run_mock.called
    assert daemon_state["last_ibkr_data_reconcile_date"] == "2026-07-06"


def test_ibkr_data_reconciliation_skips_before_run_time():
    daemon_state = {}
    run_mock, _ = _ibkr_reconcile(daemon_state, at_hour=0, at_minute=30)
    assert not run_mock.called
    assert "last_ibkr_data_reconcile_date" not in daemon_state


def test_ibkr_data_reconciliation_skips_if_already_run_today():
    daemon_state = {"last_ibkr_data_reconcile_date": "2026-07-06"}
    run_mock, _ = _ibkr_reconcile(daemon_state)
    assert not run_mock.called


def test_ibkr_data_reconciliation_skips_when_disabled():
    daemon_state = {}
    run_mock, _ = _ibkr_reconcile(daemon_state, cfg_overrides={"enabled": False})
    assert not run_mock.called
    assert "last_ibkr_data_reconcile_date" not in daemon_state


def test_ibkr_data_reconciliation_error_does_not_mark_date_done():
    """A failed reconcile run (subprocess error, TWS unreachable) must retry
    on the next poll within the run window, not silently skip for the rest
    of the day — mirrors check_nightly_reconciliation's error-retry contract."""
    daemon_state = {}
    config = {
        "overnight_timezone": "Europe/London",
        "ibkr_data_reconcile": {"enabled": True, "run_time": "01:00"},
    }
    run_mock = mock.Mock(side_effect=RuntimeError("ibkr_reconcile exited 1"))
    logger = mock.Mock()
    with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 7, 6, 1, 0, tzinfo=ZoneInfo("Europe/London"))
        live_daemon.check_ibkr_data_reconciliation(
            config, daemon_state, logger, run_reconcile=run_mock, save_state=lambda s: None)
    assert "last_ibkr_data_reconcile_date" not in daemon_state
    assert "last_ibkr_reconcile_fail_ts" in daemon_state
    assert logger.error.called


def test_ibkr_data_reconciliation_backs_off_after_failure():
    """Cooldown gate suppresses retry within 10 minutes of a failure."""
    import time as _time
    daemon_state = {"last_ibkr_reconcile_fail_ts": _time.time() - 30}
    run_mock, _ = _ibkr_reconcile(daemon_state)
    assert not run_mock.called


def test_ibkr_data_reconciliation_retries_after_cooldown():
    """Once the 10-minute cooldown expires, reconciliation is attempted again."""
    import time as _time
    daemon_state = {"last_ibkr_reconcile_fail_ts": _time.time() - 700}
    run_mock, _ = _ibkr_reconcile(daemon_state)
    assert run_mock.called


def test_run_ibkr_data_reconcile_subprocess_builds_expected_command(monkeypatch):
    """Verifies the subprocess invocation shape (module path, host/port from
    broker config, client_id distinct from execution/data-fetch/backfill)."""
    captured = {}

    def fake_run(cmd, cwd, timeout, capture_output, text):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return mock.Mock(returncode=0, stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("socket.create_connection", mock.MagicMock())
    config = {"broker": {"host": "127.0.0.1", "port": 4002, "client_id": 1}}
    cfg = {"lookback_days": 14, "client_id": 4}

    live_daemon._run_ibkr_data_reconcile_subprocess(config, cfg)

    cmd = captured["cmd"]
    assert "Strategy_Auto_Trader.markov_cli.ibkr_reconcile" in cmd
    assert "--lookback-days" in cmd and "14" in cmd
    assert "--client-id" in cmd and "4" in cmd
    assert "--port" in cmd and "4002" in cmd


def test_run_ibkr_data_reconcile_subprocess_raises_on_nonzero_exit(monkeypatch):
    def fake_run(cmd, cwd, timeout, capture_output, text):
        return mock.Mock(returncode=1, stderr="Could not connect to TWS/Gateway")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("socket.create_connection", mock.MagicMock())
    with pytest.raises(RuntimeError, match="ibkr_reconcile exited 1"):
        live_daemon._run_ibkr_data_reconcile_subprocess(
            {"broker": {}}, {"lookback_days": 14, "client_id": 4})


def test_run_ibkr_data_reconcile_subprocess_pre_ping_failure(monkeypatch):
    """ConnectionRefusedError on TCP pre-ping raises RuntimeError before
    spawning the subprocess — clean log message, no process spawn overhead."""
    monkeypatch.setattr("socket.create_connection",
                        mock.Mock(side_effect=OSError("Connection refused")))
    subprocess_called = []
    monkeypatch.setattr("subprocess.run",
                        lambda *a, **k: subprocess_called.append(1))

    with pytest.raises(RuntimeError, match="TWS/Gateway not reachable"):
        live_daemon._run_ibkr_data_reconcile_subprocess(
            {"broker": {"host": "127.0.0.1", "port": 4002}},
            {"lookback_days": 14, "client_id": 4})

    assert not subprocess_called  # subprocess must NOT be spawned when ping fails


def test_top_k_screen_health_no_state_file_disabled_is_noop(monkeypatch, tmp_path):
    """Missing file is a noop when top_k_screen is disabled — no halt, no alert."""
    monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)
    state = {}
    cfg = {}  # top_k_screen.enabled defaults to False
    logger = mock.Mock()

    with mock.patch("Strategy_Auto_Trader.output.emailer.send_top_k_screen_alert") as mock_alert:
        live_daemon._check_top_k_screen_health(state, cfg, logger)

    mock_alert.assert_not_called()
    assert not state.get("halt_top_k_stale")


def test_top_k_screen_health_no_state_file_enabled_halts(monkeypatch, tmp_path):
    """Missing file when top_k_screen is enabled triggers halt + alert."""
    monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)
    state = {}
    cfg = {"top_k_screen": {"enabled": True}}
    logger = mock.Mock()

    with mock.patch("Strategy_Auto_Trader.output.emailer.send_top_k_screen_alert") as mock_alert:
        live_daemon._check_top_k_screen_health(state, cfg, logger)

    mock_alert.assert_called_once_with("missing", None)
    assert state["halt_top_k_stale"] is True


def test_top_k_screen_health_status_ok_no_alert(monkeypatch, tmp_path):
    monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)
    (tmp_path / "top_k_universe.json").write_text(
        json.dumps({"date": datetime.now(timezone.utc).date().isoformat(), "status": "ok"}),
        encoding="utf-8",
    )
    state = {}
    logger = mock.Mock()

    with mock.patch("Strategy_Auto_Trader.output.emailer.send_top_k_screen_alert") as mock_alert:
        live_daemon._check_top_k_screen_health(state, {}, logger)

    mock_alert.assert_not_called()
    assert state["halt_top_k_stale"] is False


def test_top_k_screen_health_fallback_status_sends_alert(monkeypatch, tmp_path):
    monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)
    today = datetime.now(timezone.utc).date().isoformat()
    (tmp_path / "top_k_universe.json").write_text(
        json.dumps({"date": today, "status": "fallback"}), encoding="utf-8",
    )
    state = {}
    logger = mock.Mock()

    with mock.patch("Strategy_Auto_Trader.output.emailer.send_top_k_screen_alert") as mock_alert:
        live_daemon._check_top_k_screen_health(state, {}, logger)

    mock_alert.assert_called_once_with("fallback", today)
    assert state["halt_top_k_stale"] is True


def test_top_k_screen_health_stale_state_sends_alert_even_if_status_ok(monkeypatch, tmp_path):
    """A state file more than a day old means the ranking has been failing
    for multiple nights — flag it as stale even if its own status says ok
    (that status reflects the run that PRODUCED it, not its current age)."""
    monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)
    old_date = (datetime.now(timezone.utc).date() - timedelta(days=3)).isoformat()
    (tmp_path / "top_k_universe.json").write_text(
        json.dumps({"date": old_date, "status": "ok"}), encoding="utf-8",
    )
    state = {}
    logger = mock.Mock()

    with mock.patch("Strategy_Auto_Trader.output.emailer.send_top_k_screen_alert") as mock_alert:
        live_daemon._check_top_k_screen_health(state, {}, logger)

    mock_alert.assert_called_once_with("stale", old_date)
    assert state["halt_top_k_stale"] is True


def test_top_k_screen_health_recovery_clears_halt(monkeypatch, tmp_path):
    """When ranking recovers to ok after a stale/fail, halt_top_k_stale is cleared."""
    monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)
    (tmp_path / "top_k_universe.json").write_text(
        json.dumps({"date": datetime.now(timezone.utc).date().isoformat(), "status": "ok"}),
        encoding="utf-8",
    )
    state = {"halt_top_k_stale": True}  # previously halted
    logger = mock.Mock()

    with mock.patch("Strategy_Auto_Trader.output.emailer.send_top_k_screen_alert") as mock_alert:
        live_daemon._check_top_k_screen_health(state, {}, logger)

    mock_alert.assert_not_called()
    assert state["halt_top_k_stale"] is False


def test_orphaned_positions_no_scope_file_is_noop(monkeypatch, tmp_path):
    monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)
    config = {"markets": {"ftse": {}}}
    logger = mock.Mock()

    with mock.patch("Strategy_Auto_Trader.output.emailer.send_orphaned_position_alert") as mock_alert:
        live_daemon._check_orphaned_positions(config, logger)

    mock_alert.assert_not_called()


def test_orphaned_positions_empty_list_no_alert(monkeypatch, tmp_path):
    monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)
    (tmp_path / "in_scope_ftse.json").write_text(
        json.dumps({"kept": [], "orphaned_positions": []}), encoding="utf-8",
    )
    config = {"markets": {"ftse": {}}}
    logger = mock.Mock()

    with mock.patch("Strategy_Auto_Trader.output.emailer.send_orphaned_position_alert") as mock_alert:
        live_daemon._check_orphaned_positions(config, logger)

    mock_alert.assert_not_called()


def test_orphaned_positions_present_sends_alert(monkeypatch, tmp_path):
    monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)
    (tmp_path / "in_scope_ftse.json").write_text(
        json.dumps({"kept": ["VOD.L"], "orphaned_positions": ["VOD.L"]}), encoding="utf-8",
    )
    config = {"markets": {"ftse": {}}}
    logger = mock.Mock()

    with mock.patch("Strategy_Auto_Trader.output.emailer.send_orphaned_position_alert") as mock_alert:
        live_daemon._check_orphaned_positions(config, logger)

    mock_alert.assert_called_once_with("ftse", ["VOD.L"])


def test_orphaned_positions_checks_every_configured_market(monkeypatch, tmp_path):
    monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)
    (tmp_path / "in_scope_ftse.json").write_text(
        json.dumps({"kept": [], "orphaned_positions": ["A"]}), encoding="utf-8",
    )
    (tmp_path / "in_scope_sp500.json").write_text(
        json.dumps({"kept": [], "orphaned_positions": ["B"]}), encoding="utf-8",
    )
    config = {"markets": {"ftse": {}, "sp500": {}}}
    logger = mock.Mock()

    with mock.patch("Strategy_Auto_Trader.output.emailer.send_orphaned_position_alert") as mock_alert:
        live_daemon._check_orphaned_positions(config, logger)

    assert mock_alert.call_count == 2


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
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, l: [])

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
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, l: [])
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


def test_process_cycle_defaults_source_ibkr(monkeypatch):
    """The live daemon defaults to the local IBKR-backed cache, not yfinance —
    this is what makes live and backtest share the same on-disk data
    (ibkr_backfill_universe.py completed 2026-08-14, see HANDOFF.md)."""
    captured = _run_process_cycle_capture_defaults(monkeypatch, market_cfg={})
    assert captured["defaults"]["source"] == "ibkr"


def test_process_cycle_market_config_can_override_source_default(monkeypatch):
    captured = _run_process_cycle_capture_defaults(
        monkeypatch, market_cfg={"defaults": {"source": "yfinance"}})
    assert captured["defaults"]["source"] == "yfinance"


# -- item 8: interleaved manual-command polling -----------------------------

def test_process_cycle_checks_manual_commands_between_must_run_tickers(monkeypatch):
    """A pending manual sell must be picked up between must-run tickers, not
    only once per full market pass (that pass can run ~20+ min)."""
    from Strategy_Auto_Trader.markov_cli import batch

    calls = []
    monkeypatch.setattr(live_daemon, "process_manual_commands_wrapper",
                         lambda *a, **k: calls.append(1))
    monkeypatch.setattr(batch, "process_ticker",
                         lambda ticker_cfg, defaults, send_email: {
                             "ticker": ticker_cfg["ticker"], "status": "FAIL: stub", "time": 0.0})
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["AAPL", "MSFT"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, l: ["AAPL", "MSFT"])

    config = {"daytime": {"max_seconds_per_cycle": 60, "cycle_buffer_minutes": 0}}
    live_daemon.process_cycle(
        "test_market", {}, config, {"cursors": {}},
        portfolio=None, broker=None, logger=mock.Mock(),
    )
    assert len(calls) == 2  # once before each must-run ticker


def test_process_cycle_checks_manual_commands_between_round_robin_tickers(monkeypatch):
    """Same interleave, for the round-robin candidate stage."""
    from Strategy_Auto_Trader.markov_cli import batch

    calls = []
    monkeypatch.setattr(live_daemon, "process_manual_commands_wrapper",
                         lambda *a, **k: calls.append(1))
    monkeypatch.setattr(batch, "process_ticker",
                         lambda ticker_cfg, defaults, send_email: {
                             "ticker": ticker_cfg["ticker"], "status": "FAIL: stub", "time": 0.0})
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["AAPL", "MSFT", "GOOGL"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, l: [])
    monkeypatch.setattr(live_daemon, "next_round_robin_slice",
                         lambda market_name, candidates, daemon_state, logger: (candidates, 0))

    config = {"daytime": {"max_seconds_per_cycle": 60, "cycle_buffer_minutes": 0}}
    live_daemon.process_cycle(
        "test_market", {}, config, {"cursors": {}},
        portfolio=None, broker=None, logger=mock.Mock(),
    )
    assert len(calls) == 3  # once before each round-robin candidate


def test_process_cycle_persists_partial_round_robin_progress_across_budget_cutoff(monkeypatch):
    """Regression test for the cursor-always-resets-to-0 bug: when the time
    budget cuts the round-robin loop off partway through, the next cycle must
    resume where it stopped, not restart from the top of the list (real
    next_round_robin_slice / advance_round_robin_cursor, not mocked)."""
    from Strategy_Auto_Trader.markov_cli import batch

    processed_tickers = []

    def fake_process_ticker(ticker_cfg, defaults, send_email):
        processed_tickers.append(ticker_cfg["ticker"])
        # FAIL status keeps process_cycle away from the execution stage
        return {"ticker": ticker_cfg["ticker"], "status": "FAIL: stub", "time": 0.0}

    monkeypatch.setattr(batch, "process_ticker", fake_process_ticker)
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["AAPL", "MSFT", "GOOGL"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, l: [])
    monkeypatch.setattr(live_daemon, "process_manual_commands_wrapper", lambda *a, **k: None)

    config = {"daytime": {"max_seconds_per_cycle": 10, "cycle_buffer_minutes": 0}}
    daemon_state = {"cursors": {}}
    today_key = f"test_market:{datetime.now().date().isoformat()}"

    # time.time() call order in process_cycle: cycle_start, pre-loop remaining
    # check, then one now_remaining check per ticker attempted, then the final
    # elapsed-time log line. Budget runs out right before GOOGL is reached.
    fake_times = iter([0, 1, 2, 3, 11, 12])
    monkeypatch.setattr(live_daemon.time, "time", lambda: next(fake_times))

    live_daemon.process_cycle(
        "test_market", {}, config, daemon_state,
        portfolio=None, broker=None, logger=mock.Mock(),
    )

    assert processed_tickers == ["AAPL", "MSFT"]
    assert daemon_state["cursors"][today_key] == 2  # stopped after 2, not reset to 0

    # Next cycle: plenty of budget, cursor should resume at GOOGL only.
    fake_times = iter([0, 1, 2, 3])
    monkeypatch.setattr(live_daemon.time, "time", lambda: next(fake_times))
    live_daemon.process_cycle(
        "test_market", {}, config, daemon_state,
        portfolio=None, broker=None, logger=mock.Mock(),
    )

    assert processed_tickers == ["AAPL", "MSFT", "GOOGL"]
    assert daemon_state["cursors"][today_key] == 0  # wrapped after covering the full list


# -- heartbeat fix: interleaved app_status.json snapshots -------------------

def test_process_cycle_writes_status_snapshot_between_must_run_tickers(monkeypatch):
    """app_status.json is the phone app's only heartbeat signal. It must be
    refreshed between tickers, not only after the full market pass — a
    must-run/round-robin scan can block for 20+ minutes, during which the old
    behavior left a stale/dead-PID heartbeat the whole time."""
    from Strategy_Auto_Trader.markov_cli import batch

    calls = []
    monkeypatch.setattr(live_daemon, "_write_app_status_snapshot_safe",
                         lambda *a, **k: calls.append(1))
    monkeypatch.setattr(batch, "process_ticker",
                         lambda ticker_cfg, defaults, send_email: {
                             "ticker": ticker_cfg["ticker"], "status": "FAIL: stub", "time": 0.0})
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["AAPL", "MSFT"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, l: ["AAPL", "MSFT"])

    config = {"daytime": {"max_seconds_per_cycle": 60, "cycle_buffer_minutes": 0}}
    live_daemon.process_cycle(
        "test_market", {}, config, {"cursors": {}},
        portfolio=None, broker=None, logger=mock.Mock(),
    )
    assert len(calls) == 2  # once before each must-run ticker


def test_process_cycle_writes_status_snapshot_between_round_robin_tickers(monkeypatch):
    """Same interleave, for the round-robin candidate stage."""
    from Strategy_Auto_Trader.markov_cli import batch

    calls = []
    monkeypatch.setattr(live_daemon, "_write_app_status_snapshot_safe",
                         lambda *a, **k: calls.append(1))
    monkeypatch.setattr(batch, "process_ticker",
                         lambda ticker_cfg, defaults, send_email: {
                             "ticker": ticker_cfg["ticker"], "status": "FAIL: stub", "time": 0.0})
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["AAPL", "MSFT", "GOOGL"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, l: [])
    monkeypatch.setattr(live_daemon, "next_round_robin_slice",
                         lambda market_name, candidates, daemon_state, logger: (candidates, 0))

    config = {"daytime": {"max_seconds_per_cycle": 60, "cycle_buffer_minutes": 0}}
    live_daemon.process_cycle(
        "test_market", {}, config, {"cursors": {}},
        portfolio=None, broker=None, logger=mock.Mock(),
    )
    assert len(calls) == 3  # once before each round-robin candidate


def test_process_cycle_passes_last_cycle_hour_through_to_snapshot(monkeypatch):
    """last_cycle_hour must thread from process_cycle's caller into the
    snapshot writer so markets_status reflects the real cycle state, not an
    always-empty dict."""
    from Strategy_Auto_Trader.markov_cli import batch

    captured = []
    monkeypatch.setattr(live_daemon, "_write_app_status_snapshot_safe",
                         lambda portfolio, daemon_state, config, last_cycle_hour, logger:
                             captured.append(last_cycle_hour))
    monkeypatch.setattr(batch, "process_ticker",
                         lambda ticker_cfg, defaults, send_email: {
                             "ticker": ticker_cfg["ticker"], "status": "FAIL: stub", "time": 0.0})
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["AAPL"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, l: ["AAPL"])

    config = {"daytime": {"max_seconds_per_cycle": 60, "cycle_buffer_minutes": 0}}
    sentinel = {"test_market": 13}
    live_daemon.process_cycle(
        "test_market", {}, config, {"cursors": {}},
        portfolio=None, broker=None, logger=mock.Mock(),
        last_cycle_hour=sentinel,
    )
    assert captured == [sentinel]


class TestReconciliation:
    """run_reconciliation / check_nightly_reconciliation / halt enforcement."""

    def _portfolio(self, positions):
        p = mock.Mock()
        p.positions = positions
        p.accrue_daily_interest.return_value = 0.0
        p.save = mock.Mock()
        return p

    def _broker(self, positions):
        b = mock.Mock()
        b.get_open_positions.return_value = positions
        return b

    def test_reconciliation_connects_broker_when_not_connected(self, monkeypatch):
        broker = self._broker({"SPY": 10})
        broker.is_connected.return_value = False
        daemon_state = {}

        outcome = live_daemon.run_reconciliation(
            self._portfolio({"SPY": {"quantity": 10}}), broker, daemon_state, mock.Mock(),
            save_state=lambda s: None,
        )
        assert outcome == "clean"
        broker.connect.assert_called_once()
        broker.get_open_positions.assert_called_once()

    def test_reconciliation_skips_connect_when_already_connected(self, monkeypatch):
        broker = self._broker({})
        daemon_state = {}

        live_daemon.run_reconciliation(
            self._portfolio({}), broker, daemon_state, mock.Mock(),
            save_state=lambda s: None,
        )
        # Bare Mock().is_connected() is truthy, matching an already-connected broker.
        broker.connect.assert_not_called()

    def test_reconciliation_connect_failure_returns_error(self, monkeypatch):
        broker = mock.Mock()
        broker.is_connected.return_value = False
        broker.connect.side_effect = ConnectionError("TWS not running")
        daemon_state = {}

        outcome = live_daemon.run_reconciliation(
            self._portfolio({}), broker, daemon_state, mock.Mock(),
            save_state=lambda s: None,
        )
        assert outcome == "error"
        broker.get_open_positions.assert_not_called()

    def test_clean_pass_clears_halt(self, monkeypatch):
        daemon_state = {"halt_new_entries": True}

        outcome = live_daemon.run_reconciliation(
            self._portfolio({"SPY": {"quantity": 10}}),
            self._broker({"SPY": 10}),
            daemon_state, mock.Mock(),
            save_state=lambda s: None,
        )
        assert outcome == "clean"
        assert daemon_state["halt_new_entries"] is False
        assert daemon_state["reconciliation_discrepancies"] == []

    def test_mismatch_sets_halt_and_emails(self, monkeypatch):
        sent = {}
        def fake_alert(d):
            sent.update(discrepancies=d)
        daemon_state = {}

        outcome = live_daemon.run_reconciliation(
            self._portfolio({"SPY": {"quantity": 10}}),
            self._broker({"SPY": 7}),
            daemon_state, mock.Mock(),
            save_state=lambda s: None,
            send_alert=fake_alert,
        )
        assert outcome == "mismatch"
        assert daemon_state["halt_new_entries"] is True
        assert len(daemon_state["reconciliation_discrepancies"]) == 1
        assert len(sent["discrepancies"]) == 1

    def test_email_failure_does_not_mask_mismatch(self, monkeypatch):
        def fake_alert_fail(d):
            raise RuntimeError("smtp down")
        daemon_state = {}

        outcome = live_daemon.run_reconciliation(
            self._portfolio({"SPY": {"quantity": 10}}),
            self._broker({}),
            daemon_state, mock.Mock(),
            save_state=lambda s: None,
            send_alert=fake_alert_fail,
        )
        assert outcome == "mismatch"
        assert daemon_state["halt_new_entries"] is True

    def test_broker_fetch_error_returns_error_and_keeps_halt(self, monkeypatch):
        broker = mock.Mock()
        broker.get_open_positions.side_effect = ConnectionError("TWS gone")
        daemon_state = {"halt_new_entries": True}

        outcome = live_daemon.run_reconciliation(
            self._portfolio({}), broker, daemon_state, mock.Mock(),
            save_state=lambda s: None,
        )
        assert outcome == "error"
        assert daemon_state["halt_new_entries"] is True

    def _nightly(self, monkeypatch, daemon_state, outcome, at_hour=21, at_minute=30, day=6):
        config = {"overnight_timezone": "Europe/London",
                  "reconciliation_run_time": "21:30"}
        run_mock = mock.Mock(return_value=outcome)
        # Set up portfolio mock with interest accrual
        portfolio_mock = mock.Mock()
        portfolio_mock.accrue_daily_interest.return_value = 0.0
        portfolio_mock.save = mock.Mock()
        with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(
                2026, 7, day, at_hour, at_minute, tzinfo=ZoneInfo("Europe/London"))
            live_daemon.check_nightly_reconciliation(
                config, daemon_state, portfolio_mock, mock.Mock(), mock.Mock(),
                run_recon=run_mock, save_state=lambda s: None)
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

    def test_nightly_error_retry_storm_same_day_counts_once(self, monkeypatch):
        """30 failed attempts in one evening's window must count as 1 error-day."""
        daemon_state = {}
        for _ in range(30):
            self._nightly(monkeypatch, daemon_state, "error", day=6)
        assert daemon_state["reconciliation_consecutive_error_days"] == 1

    def test_nightly_error_two_consecutive_days_sends_alert(self, monkeypatch):
        sent = []
        def fake_escalation_alert(d):
            sent.append(d)
        daemon_state = {}

        config = {"overnight_timezone": "Europe/London",
                  "reconciliation_run_time": "21:30"}
        run_mock = mock.Mock(return_value="error")

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(
                2026, 7, 6, 21, 30, tzinfo=ZoneInfo("Europe/London"))
            live_daemon.check_nightly_reconciliation(
                config, daemon_state, mock.Mock(), mock.Mock(), mock.Mock(),
                run_recon=run_mock, save_state=lambda s: None, send_alert=fake_escalation_alert)
        assert sent == []
        assert daemon_state["reconciliation_consecutive_error_days"] == 1

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(
                2026, 7, 7, 21, 30, tzinfo=ZoneInfo("Europe/London"))
            live_daemon.check_nightly_reconciliation(
                config, daemon_state, mock.Mock(), mock.Mock(), mock.Mock(),
                run_recon=run_mock, save_state=lambda s: None, send_alert=fake_escalation_alert)
        assert len(sent) == 1
        assert daemon_state["reconciliation_consecutive_error_days"] == 2
        assert daemon_state["reconciliation_alert_sent"] is True

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(
                2026, 7, 8, 21, 30, tzinfo=ZoneInfo("Europe/London"))
            live_daemon.check_nightly_reconciliation(
                config, daemon_state, mock.Mock(), mock.Mock(), mock.Mock(),
                run_recon=run_mock, save_state=lambda s: None, send_alert=fake_escalation_alert)
        # A third bad day doesn't resend the alert.
        assert len(sent) == 1

    def test_nightly_clean_pass_resets_error_streak(self, monkeypatch):
        daemon_state = {}

        config = {"overnight_timezone": "Europe/London",
                  "reconciliation_run_time": "21:30"}
        run_mock_error = mock.Mock(return_value="error")
        run_mock_clean = mock.Mock(return_value="clean")

        def _portfolio_mock():
            p = mock.Mock()
            p.accrue_daily_interest.return_value = 0.0
            p.save = mock.Mock()
            return p

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(
                2026, 7, 6, 21, 30, tzinfo=ZoneInfo("Europe/London"))
            live_daemon.check_nightly_reconciliation(
                config, daemon_state, _portfolio_mock(), mock.Mock(), mock.Mock(),
                run_recon=run_mock_error, save_state=lambda s: None, send_alert=lambda d: None)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(
                2026, 7, 7, 21, 30, tzinfo=ZoneInfo("Europe/London"))
            live_daemon.check_nightly_reconciliation(
                config, daemon_state, _portfolio_mock(), mock.Mock(), mock.Mock(),
                run_recon=run_mock_error, save_state=lambda s: None, send_alert=lambda d: None)
        assert daemon_state["reconciliation_consecutive_error_days"] == 2

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(
                2026, 7, 8, 21, 30, tzinfo=ZoneInfo("Europe/London"))
            live_daemon.check_nightly_reconciliation(
                config, daemon_state, _portfolio_mock(), mock.Mock(), mock.Mock(),
                run_recon=run_mock_clean, save_state=lambda s: None, send_alert=lambda d: None)
        assert daemon_state["reconciliation_consecutive_error_days"] == 0
        assert daemon_state["reconciliation_alert_sent"] is False

    def test_nightly_clean_pass_sends_roundup_email(self, monkeypatch):
        """Successful reconciliation triggers nightly roundup email."""
        daemon_state = {}
        sent_emails = []

        config = {
            "overnight_timezone": "Europe/London",
            "reconciliation_run_time": "21:30",
            "markets": {"ftse": {"defaults": {}}}
        }
        run_mock = mock.Mock(return_value="clean")

        def _portfolio_mock():
            p = mock.Mock()
            p.accrue_daily_interest.return_value = 0.0
            p.save = mock.Mock()
            return p

        # Mock the email sending
        def fake_roundup(results, failed):
            sent_emails.append({"results": len(results), "failed": len(failed)})

        monkeypatch.setattr(
            "Strategy_Auto_Trader.markov_cli.live_daemon._send_nightly_roundup",
            lambda cfg, logger: sent_emails.append({"called": True})
        )

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(
                2026, 7, 6, 21, 30, tzinfo=ZoneInfo("Europe/London"))
            live_daemon.check_nightly_reconciliation(
                config, daemon_state, _portfolio_mock(), mock.Mock(), mock.Mock(),
                run_recon=run_mock, save_state=lambda s: None)

        assert len(sent_emails) == 1
        assert sent_emails[0]["called"] is True

    def test_reconciliation_default_wiring_saves_state(self, monkeypatch, tmp_path):
        """run_reconciliation with no injected params uses module-level save_daemon_state."""
        monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)
        broker = self._broker({})
        broker.is_connected.return_value = True
        daemon_state = {}

        outcome = live_daemon.run_reconciliation(
            self._portfolio({}), broker, daemon_state, mock.Mock()
        )

        assert outcome == "clean"
        # Verify the daemon-state file was actually written via default save_state
        state_path = tmp_path / "daemon_state.json"
        assert state_path.exists()
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        assert saved["halt_new_entries"] is False

    def test_mismatch_dedup_suppresses_repeated_alerts(self):
        """Same mismatch + already alerted → skip email, log warning instead."""
        sent = {}
        def fake_alert(d):
            sent["alert_sent"] = True
            sent["discrepancies"] = d

        discrepancies_list = ["SPY: internal state shows 10 shares, broker shows 7"]
        daemon_state = {
            "reconciliation_discrepancies": discrepancies_list,
            "reconciliation_mismatch_alerted": True,
        }

        outcome = live_daemon.run_reconciliation(
            self._portfolio({"SPY": {"quantity": 10}}),
            self._broker({"SPY": 7}),
            daemon_state, mock.Mock(),
            save_state=lambda s: None,
            send_alert=fake_alert,
        )
        assert outcome == "mismatch"
        assert daemon_state["halt_new_entries"] is True
        assert "alert_sent" not in sent  # Email suppressed
        assert daemon_state["reconciliation_mismatch_alerted"] is True

    def test_mismatch_dedup_fires_on_different_discrepancies(self):
        """New discrepancy list + already alerted → alert fires (regression test for Fix 2)."""
        sent = []
        def fake_alert(d):
            sent.append(d)

        old_discrepancies = ["SPY: internal state shows 10 shares, broker shows 7"]
        daemon_state = {
            "reconciliation_discrepancies": old_discrepancies,
            "reconciliation_mismatch_alerted": True,
        }

        outcome = live_daemon.run_reconciliation(
            self._portfolio({"AAPL": {"quantity": 5}}),
            self._broker({}),
            daemon_state, mock.Mock(),
            save_state=lambda s: None,
            send_alert=fake_alert,
        )
        assert outcome == "mismatch"
        assert daemon_state["halt_new_entries"] is True
        assert len(sent) == 1  # New alert sent
        assert daemon_state["reconciliation_discrepancies"] == ["AAPL: internal state shows 5 shares, broker shows no position"]
        assert daemon_state["reconciliation_mismatch_alerted"] is True

    def test_mismatch_dedup_sends_on_first_occurrence(self):
        """New mismatch or first time → send alert."""
        sent = []
        def fake_alert(d):
            sent.append(d)

        daemon_state = {}

        outcome = live_daemon.run_reconciliation(
            self._portfolio({"SPY": {"quantity": 10}}),
            self._broker({"SPY": 7}),
            daemon_state, mock.Mock(),
            save_state=lambda s: None,
            send_alert=fake_alert,
        )
        assert outcome == "mismatch"
        assert len(sent) == 1
        assert daemon_state["reconciliation_mismatch_alerted"] is True

    def test_mismatch_dedup_flag_reset_on_clean(self):
        """Clean pass resets the mismatch-alerted flag."""
        daemon_state = {
            "reconciliation_mismatch_alerted": True,
            "halt_new_entries": True,
        }

        outcome = live_daemon.run_reconciliation(
            self._portfolio({"SPY": {"quantity": 10}}),
            self._broker({"SPY": 10}),
            daemon_state, mock.Mock(),
            save_state=lambda s: None,
        )
        assert outcome == "clean"
        assert daemon_state["halt_new_entries"] is False
        assert daemon_state["reconciliation_mismatch_alerted"] is False

    def test_run_startup_reconciliation_forces_halt_on_entry(self):
        """startup_reconciliation must force halt_new_entries=True immediately."""
        daemon_state = {"halt_new_entries": False}

        outcome = live_daemon.run_startup_reconciliation(
            daemon_state,
            self._portfolio({"SPY": {"quantity": 10}}),
            self._broker({"SPY": 10}),
            mock.Mock(),
            run_recon=lambda *a, **k: "clean",
            save_state=lambda s: None,
        )
        assert daemon_state["halt_new_entries"] is True
        assert outcome is True

    def test_run_startup_reconciliation_delegates_to_run_recon(self):
        """startup_reconciliation calls the provided run_recon function."""
        calls = []
        def fake_recon(*args, **kwargs):
            calls.append((args, kwargs))
            return "clean"

        daemon_state = {}
        outcome = live_daemon.run_startup_reconciliation(
            daemon_state,
            self._portfolio({"SPY": {"quantity": 10}}),
            self._broker({"SPY": 10}),
            mock.Mock(),
            run_recon=fake_recon,
            save_state=lambda s: None,
        )
        assert outcome is True
        assert len(calls) == 1

    def test_run_startup_reconciliation_returns_true_on_clean(self):
        """reconciliation clean → return True (reconciliation done)."""
        daemon_state = {}

        outcome = live_daemon.run_startup_reconciliation(
            daemon_state,
            self._portfolio({"SPY": {"quantity": 10}}),
            self._broker({"SPY": 10}),
            mock.Mock(),
            run_recon=lambda *a, **k: "clean",
            save_state=lambda s: None,
        )
        assert outcome is True

    def test_run_startup_reconciliation_returns_true_on_mismatch(self):
        """reconciliation mismatch → return True (definitive outcome)."""
        daemon_state = {}

        outcome = live_daemon.run_startup_reconciliation(
            daemon_state,
            self._portfolio({"SPY": {"quantity": 10}}),
            self._broker({"SPY": 7}),
            mock.Mock(),
            run_recon=lambda *a, **k: "mismatch",
            save_state=lambda s: None,
            send_interrupt_alert=lambda *a: None,
        )
        assert outcome is True

    def test_run_startup_reconciliation_returns_false_on_error(self, tmp_path):
        """reconciliation error → return False (caller retries)."""
        daemon_state = {}
        marker_path = tmp_path / "marker.json"

        outcome = live_daemon.run_startup_reconciliation(
            daemon_state,
            self._portfolio({"SPY": {"quantity": 10}}),
            self._broker({})  # Will fail on positions fetch
            ,
            mock.Mock(),
            run_recon=lambda *a, **k: "error",
            save_state=lambda s: None,
            send_interrupt_alert=lambda *a: None,
            marker_path=marker_path,
        )
        assert outcome is False

    def test_run_startup_reconciliation_escalates_if_marker_present_and_error(self, tmp_path):
        """error + marker → immediate escalation alert."""
        from Strategy_Auto_Trader.broker.in_flight_marker import write_marker

        alerted = []
        def fake_escalate(market_name, error, buys, sells, unresolved):
            alerted.append({
                "market_name": market_name,
                "error": error,
                "buys": buys,
                "sells": sells,
                "unresolved": unresolved,
            })

        daemon_state = {}
        marker_path = tmp_path / "marker.json"
        write_marker(marker_path, "SPY", "BUY", 10)

        outcome = live_daemon.run_startup_reconciliation(
            daemon_state,
            self._portfolio({}),
            mock.Mock(),
            mock.Mock(),
            run_recon=lambda *a, **k: "error",
            save_state=lambda s: None,
            send_interrupt_alert=fake_escalate,
            marker_path=marker_path,
        )
        assert outcome is False
        assert len(alerted) == 1
        assert alerted[0]["unresolved"] == ["SPY"]

    def test_run_startup_reconciliation_clears_marker_on_clean(self, tmp_path):
        """reconciliation clean + marker present + no stale broker order → marker cleared."""
        from Strategy_Auto_Trader.broker.in_flight_marker import write_marker, read_marker

        daemon_state = {}
        marker_path = tmp_path / "marker.json"
        write_marker(marker_path, "SPY", "BUY", 10)
        assert read_marker(marker_path) is not None

        broker = self._broker({"SPY": 10})
        broker.get_open_orders.return_value = []
        outcome = live_daemon.run_startup_reconciliation(
            daemon_state,
            self._portfolio({"SPY": {"quantity": 10}}),
            broker,
            mock.Mock(),
            run_recon=lambda *a, **k: "clean",
            save_state=lambda s: None,
            marker_path=marker_path,
        )
        assert outcome is True
        assert read_marker(marker_path) is None

    def test_run_startup_reconciliation_keeps_marker_if_order_still_live(self, tmp_path):
        """reconciliation clean but the marker's order is still working at the
        broker (the GSK.L-style gap: order accepted right before a disconnect,
        not filled yet so positions compare clean) → halt stays, marker kept."""
        from Strategy_Auto_Trader.broker.in_flight_marker import write_marker, read_marker

        alerted = []
        daemon_state = {}
        marker_path = tmp_path / "marker.json"
        write_marker(marker_path, "GSK.L", "BUY", 10)

        broker = self._broker({})
        broker.get_open_orders.return_value = [
            {"ticker": "GSK.L", "action": "BUY", "status": "PreSubmitted"}
        ]
        outcome = live_daemon.run_startup_reconciliation(
            daemon_state,
            self._portfolio({}),
            broker,
            mock.Mock(),
            run_recon=lambda *a, **k: "clean",
            save_state=lambda s: None,
            send_interrupt_alert=lambda *a: alerted.append(a),
            marker_path=marker_path,
        )
        assert outcome is True
        assert daemon_state["halt_new_entries"] is True
        assert read_marker(marker_path) is not None
        assert len(alerted) == 1

    def test_run_startup_reconciliation_clears_marker_ignores_other_ticker_orders(self, tmp_path):
        """A live order for an unrelated ticker must not block clearing the
        marker for the ticker that was actually interrupted."""
        from Strategy_Auto_Trader.broker.in_flight_marker import write_marker, read_marker

        daemon_state = {}
        marker_path = tmp_path / "marker.json"
        write_marker(marker_path, "SPY", "BUY", 10)

        broker = self._broker({"SPY": 10})
        broker.get_open_orders.return_value = [
            {"ticker": "AAPL", "action": "SELL", "status": "Submitted"}
        ]
        outcome = live_daemon.run_startup_reconciliation(
            daemon_state,
            self._portfolio({"SPY": {"quantity": 10}}),
            broker,
            mock.Mock(),
            run_recon=lambda *a, **k: "clean",
            save_state=lambda s: None,
            marker_path=marker_path,
        )
        assert outcome is True
        assert read_marker(marker_path) is None


def test_process_cycle_halt_flag_blocks_new_entries(monkeypatch):
    """halt_new_entries passes allow_new_entries=False into execute_signals."""
    from Strategy_Auto_Trader.markov_cli import batch, execute

    def fake_process_ticker(ticker_cfg, defaults, send_email):
        return {"ticker": "AAPL", "status": "OK", "time": 0.0,
                "result": {"ticker": "AAPL", "close": 100.0}}

    captured = {}

    def fake_execute_signals(tickers, data_dir, portfolio, limit_tracker,
                             broker, allow_new_entries=True, **kwargs):
        captured["allow_new_entries"] = allow_new_entries
        return [], [], []

    monkeypatch.setattr(batch, "process_ticker", fake_process_ticker)
    monkeypatch.setattr(execute, "execute_signals", fake_execute_signals)
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["AAPL"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, l: [])

    config = {
        "daytime": {"max_seconds_per_cycle": 60, "cycle_buffer_minutes": 0},
        "execution": {},
    }
    daemon_state = {"cursors": {}, "halt_new_entries": True}
    live_daemon.process_cycle(
        "test_market", {}, config, daemon_state,
        portfolio=mock.Mock(), broker=mock.Mock(spec=[]), logger=mock.Mock(),
    )
    assert captured["allow_new_entries"] is False


def test_process_cycle_halt_top_k_stale_blocks_new_entries(monkeypatch):
    """halt_top_k_stale passes allow_new_entries=False into execute_signals."""
    from Strategy_Auto_Trader.markov_cli import batch, execute

    def fake_process_ticker(ticker_cfg, defaults, send_email):
        return {"ticker": "AAPL", "status": "OK", "time": 0.0,
                "result": {"ticker": "AAPL", "close": 100.0}}

    captured = {}

    def fake_execute_signals(tickers, data_dir, portfolio, limit_tracker,
                             broker, allow_new_entries=True, **kwargs):
        captured["allow_new_entries"] = allow_new_entries
        return [], [], []

    monkeypatch.setattr(batch, "process_ticker", fake_process_ticker)
    monkeypatch.setattr(execute, "execute_signals", fake_execute_signals)
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["AAPL"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, l: [])

    config = {
        "daytime": {"max_seconds_per_cycle": 60, "cycle_buffer_minutes": 0},
        "execution": {},
    }
    daemon_state = {"cursors": {}, "halt_top_k_stale": True}
    live_daemon.process_cycle(
        "test_market", {}, config, daemon_state,
        portfolio=mock.Mock(), broker=mock.Mock(spec=[]), logger=mock.Mock(),
    )
    assert captured["allow_new_entries"] is False


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


def test_cleanup_incomplete_runs_skips_young_dirs(tmp_path):
    """A run_dir with only inputData.csv but written moments ago must survive —
    it may be an orphaned run_single child still mid-backtest (crashed parent
    daemon, not yet caught by kill_stray_daemons), and deleting it out from
    under that live writer is exactly the race that caused the ABBV OSError.
    """
    run_dir = tmp_path / "ABBV_20260811T160129Z"
    run_dir.mkdir()
    (run_dir / "inputData.csv").write_text("data")

    cleaned = live_daemon.cleanup_incomplete_runs(tmp_path, mock.Mock(), min_age_seconds=600)

    assert cleaned == 0
    assert run_dir.exists()


def test_cleanup_incomplete_runs_removes_stale_dirs(tmp_path):
    """A genuinely abandoned incomplete run (old enough that no live writer
    could still be producing it) is removed."""
    import os

    run_dir = tmp_path / "ABBV_20260811T160129Z"
    run_dir.mkdir()
    input_csv = run_dir / "inputData.csv"
    input_csv.write_text("data")
    old_time = live_daemon.time.time() - 700
    os.utime(input_csv, (old_time, old_time))

    cleaned = live_daemon.cleanup_incomplete_runs(tmp_path, mock.Mock(), min_age_seconds=600)

    assert cleaned == 1
    assert not run_dir.exists()


def test_cleanup_incomplete_runs_keeps_completed_runs(tmp_path):
    """A run_dir with output files present is never removed, regardless of age."""
    import os

    run_dir = tmp_path / "ABBV_20260811T160129Z"
    run_dir.mkdir()
    input_csv = run_dir / "inputData.csv"
    input_csv.write_text("data")
    (run_dir / "compositeBacktest.csv").write_text("data")
    old_time = live_daemon.time.time() - 700
    os.utime(input_csv, (old_time, old_time))

    cleaned = live_daemon.cleanup_incomplete_runs(tmp_path, mock.Mock(), min_age_seconds=600)

    assert cleaned == 0
    assert run_dir.exists()


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
    monkeypatch.setattr(live_daemon, "_write_app_status_snapshot_safe",
                        lambda *a, **k: snapshot_written.append(True))
    monkeypatch.setattr(live_daemon, "check_overnight_screening",
                        mock.Mock(side_effect=KeyboardInterrupt))

    sleeps = []
    monkeypatch.setattr(live_daemon.time, "sleep", lambda s: sleeps.append(s))

    assert live_daemon.main([]) == 0
    assert sleeps == []  # No sleep on shutdown
    assert snapshot_written  # Snapshot still written on the final iteration


def test_main_startup_reconciliation_retries_until_done(monkeypatch, config, tmp_path):
    """Startup reconciliation retries once per poll until resolved, then stops."""
    from Strategy_Auto_Trader.core import self_check

    startup_recon_calls = []

    def fake_startup_recon(*args, **kwargs):
        startup_recon_calls.append(1)
        return len(startup_recon_calls) >= 2  # Fail first time, succeed second time

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
    monkeypatch.setattr(live_daemon, "run_startup_reconciliation", fake_startup_recon)

    loop_count = [0]

    def fake_check_overnight(*args):
        loop_count[0] += 1
        if loop_count[0] >= 3:  # Exit after 2-3 loop iterations
            raise KeyboardInterrupt

    monkeypatch.setattr(live_daemon, "check_overnight_screening", fake_check_overnight)
    monkeypatch.setattr(live_daemon, "check_nightly_reconciliation", lambda *a, **k: None)
    monkeypatch.setattr(live_daemon, "process_manual_commands_wrapper", lambda *a, **k: None)
    monkeypatch.setattr(live_daemon, "_write_app_status_snapshot_safe", lambda *a, **k: None)
    monkeypatch.setattr(live_daemon, "is_trading_hours", lambda *a, **k: False)
    monkeypatch.setattr(live_daemon.time, "sleep", lambda s: None)

    config["execution"]["dry_run"] = False
    assert live_daemon.main([]) == 0

    assert len(startup_recon_calls) == 2  # Called twice: fails, retries, succeeds


def test_main_retries_pending_tickers_after_reconciliation_clears(monkeypatch, config, tmp_path):
    """Once startup reconciliation succeeds, retry_pending_tickers is invoked —
    this is what lets an interrupted ticker fire again within one poll interval
    instead of waiting for that market's next hourly cycle."""
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
    monkeypatch.setattr(live_daemon, "run_startup_reconciliation", lambda *a, **k: True)

    retry_calls = []
    monkeypatch.setattr(live_daemon, "retry_pending_tickers",
                        lambda *a, **k: retry_calls.append(1))

    loop_count = [0]

    def fake_check_overnight(*args):
        loop_count[0] += 1
        if loop_count[0] >= 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(live_daemon, "check_overnight_screening", fake_check_overnight)
    monkeypatch.setattr(live_daemon, "check_nightly_reconciliation", lambda *a, **k: None)
    monkeypatch.setattr(live_daemon, "process_manual_commands_wrapper", lambda *a, **k: None)
    monkeypatch.setattr(live_daemon, "_write_app_status_snapshot_safe", lambda *a, **k: None)
    monkeypatch.setattr(live_daemon, "is_trading_hours", lambda *a, **k: False)
    monkeypatch.setattr(live_daemon.time, "sleep", lambda s: None)

    config["execution"]["dry_run"] = False
    assert live_daemon.main([]) == 0

    assert retry_calls  # retry_pending_tickers was invoked after reconciliation succeeded


class TestNightlyRoundup:
    """Nightly roundup email collection and sending."""

    def test_send_nightly_roundup_collects_all_tickers(self, monkeypatch):
        """Collect results for all in-scope tickers from all markets."""
        sent = []

        def fake_roundup(results, failed):
            sent.append({"results": results, "failed": failed})

        def fake_collect(ticker):
            if ticker == "SPY":
                return {"ticker": ticker, "current_signal": "BUY", "score": 2.5, "close": 450.0,
                        "portfolio_value": 21000, "strategy_return": 0.05, "bh_return": 0.03}
            elif ticker == "QQQ":
                return {"ticker": ticker, "current_signal": "HOLD", "score": 0.5, "close": 350.0,
                        "portfolio_value": 20500, "strategy_return": 0.025, "bh_return": 0.04}
            return None

        config = {
            "markets": {
                "us": {"defaults": {}},
            }
        }

        monkeypatch.setattr("Strategy_Auto_Trader.markov_cli.live_daemon.load_in_scope_tickers",
                          lambda market, logger: ["SPY", "QQQ"] if market == "us" else [])
        monkeypatch.setattr("Strategy_Auto_Trader.markov_cli.batch._collect_results", fake_collect)
        monkeypatch.setattr("Strategy_Auto_Trader.output.emailer.send_daily_roundup", fake_roundup)

        live_daemon._send_nightly_roundup(config, mock.Mock())

        assert len(sent) == 1
        assert len(sent[0]["results"]) == 2
        assert sent[0]["results"][0]["ticker"] == "SPY"
        assert sent[0]["results"][1]["ticker"] == "QQQ"
        assert len(sent[0]["failed"]) == 0

    def test_send_nightly_roundup_handles_missing_results(self, monkeypatch):
        """Failed tickers are recorded in failed list."""
        sent = []

        def fake_roundup(results, failed):
            sent.append({"results": results, "failed": failed})

        def fake_collect(ticker):
            if ticker == "SPY":
                return {"ticker": ticker, "current_signal": "BUY"}
            return None  # QQQ has no result

        config = {"markets": {"us": {"defaults": {}}}}

        monkeypatch.setattr("Strategy_Auto_Trader.markov_cli.live_daemon.load_in_scope_tickers",
                          lambda market, logger: ["SPY", "QQQ"] if market == "us" else [])
        monkeypatch.setattr("Strategy_Auto_Trader.markov_cli.batch._collect_results", fake_collect)
        monkeypatch.setattr("Strategy_Auto_Trader.output.emailer.send_daily_roundup", fake_roundup)

        live_daemon._send_nightly_roundup(config, mock.Mock())

        assert len(sent) == 1
        assert len(sent[0]["results"]) == 1
        assert len(sent[0]["failed"]) == 1
        assert sent[0]["failed"][0]["ticker"] == "QQQ"


class TestExecuteSignalsWithRetry:
    """Auto-reconnect and retry on socket errors."""

    def test_success_first_attempt(self, monkeypatch):
        """Successful execution on first attempt returns immediately."""
        def fake_execute(*a, **k):
            return (["BUY"], ["SELL"], [])

        result = live_daemon.execute_signals_with_retry(
            "ftse", ["AAPL"], None, None, None, mock.Mock(),
            True, mock.Mock(),
            execute_signals=fake_execute
        )
        assert result == (["BUY"], ["SELL"], [])

    def test_socket_error_triggers_reconnect(self, monkeypatch):
        """Socket error triggers broker disconnect/reconnect and retry."""
        call_count = [0]
        broker = mock.Mock()
        logger = mock.Mock()

        def fake_execute(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("Socket disconnect: not connected to 127.0.0.1:7497")
            return (["BUY"], ["SELL"], [])

        result = live_daemon.execute_signals_with_retry(
            "ftse", ["AAPL"], None, None, None, broker,
            True, logger, max_retries=3,
            execute_signals=fake_execute
        )

        assert result == (["BUY"], ["SELL"], [])
        assert broker.disconnect.called
        assert broker.connect.called

    def test_socket_error_with_reconnect_failure(self, monkeypatch):
        """Socket error retries even if reconnect fails."""
        call_count = [0]
        broker = mock.Mock()
        broker.connect.side_effect = RuntimeError("TWS not responding")
        logger = mock.Mock()

        def fake_execute(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ConnectionError("Socket error")
            return (["BUY"], [], [])

        result = live_daemon.execute_signals_with_retry(
            "ftse", ["AAPL"], None, None, None, broker,
            True, logger, max_retries=3,
            execute_signals=fake_execute
        )

        assert result == (["BUY"], [], [])
        assert call_count[0] == 3  # Called on attempts 1, 2, 3

    def test_socket_error_max_retries_exhausted(self, monkeypatch):
        """Socket error after max retries returns empty results, doesn't raise."""
        broker = mock.Mock()
        logger = mock.Mock()

        def fake_execute(*args, **kwargs):
            raise ConnectionError("Socket disconnect")

        result = live_daemon.execute_signals_with_retry(
            "ftse", ["AAPL", "MSFT"], None, None, None, broker,
            True, logger, max_retries=2,
            execute_signals=fake_execute
        )

        assert result == ([], [], ["AAPL", "MSFT"])  # Tickers listed as skipped
        assert logger.error.called

    def test_timeout_error_triggers_retry(self, monkeypatch):
        """TimeoutError (socket-level) triggers reconnect."""
        call_count = [0]
        broker = mock.Mock()
        logger = mock.Mock()

        def fake_execute(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise TimeoutError("Socket timeout")
            return ([], ["SELL"], [])

        result = live_daemon.execute_signals_with_retry(
            "ftse", ["AAPL"], None, None, None, broker,
            True, logger,
            execute_signals=fake_execute
        )

        assert result == ([], ["SELL"], [])
        assert broker.disconnect.called

    def test_os_error_triggers_retry(self, monkeypatch):
        """OSError (socket-level) triggers reconnect."""
        call_count = [0]
        broker = mock.Mock()
        logger = mock.Mock()

        def fake_execute(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise OSError("[Errno 10054] Connection reset by peer")
            return (["BUY"], [], [])

        result = live_daemon.execute_signals_with_retry(
            "ftse", ["AAPL"], None, None, None, broker,
            True, logger,
            execute_signals=fake_execute
        )

        assert result == (["BUY"], [], [])

    def test_non_socket_error_raises_immediately(self, monkeypatch):
        """Non-connection errors are raised immediately without retry."""
        broker = mock.Mock()
        logger = mock.Mock()

        def fake_execute(*args, **kwargs):
            raise ValueError("Invalid ticker symbol")

        with pytest.raises(ValueError, match="Invalid ticker symbol"):
            live_daemon.execute_signals_with_retry(
                "ftse", ["AAPL"], None, None, None, broker,
                True, logger,
                execute_signals=fake_execute
            )

        assert not broker.disconnect.called

    def test_string_pattern_socket_error_triggers_retry(self, monkeypatch):
        """Exception with 'socket' or 'disconnect' in message triggers retry."""
        call_count = [0]
        broker = mock.Mock()
        logger = mock.Mock()

        def fake_execute(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Socket error: connection lost")
            return (["BUY"], ["SELL"], [])

        result = live_daemon.execute_signals_with_retry(
            "ftse", ["AAPL"], None, None, None, broker,
            True, logger,
            execute_signals=fake_execute
        )

        assert result == (["BUY"], ["SELL"], [])
        assert broker.disconnect.called

    def test_exponential_backoff_sleep_times(self, monkeypatch):
        """Retries use exponential backoff: 1s, 2s, 4s."""
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

        monkeypatch.setattr(time, "sleep", fake_sleep)

        result = live_daemon.execute_signals_with_retry(
            "ftse", ["AAPL"], None, None, None, broker,
            True, logger, max_retries=4,
            execute_signals=fake_execute
        )

        assert result == (["BUY"], [], [])
        # Expected: exponential backoff [1, 2, 4] + intermediate 0.5s pauses between disconnect/reconnect
        assert sleep_times == [1, 0.5, 2, 0.5, 4, 0.5]

    def test_execution_interrupted_with_orders_placed_halts_and_alerts(self, monkeypatch):
        """ExecutionInterrupted with non-empty buys/sells halts new entries and sends alert."""
        from Strategy_Auto_Trader.markov_cli import execute as execute_mod

        broker = mock.Mock()
        logger = mock.Mock()
        daemon_state = {"halt_new_entries": False}
        saved_state = []

        def fake_execute(*args, **kwargs):
            exc = execute_mod.ExecutionInterrupted(
                RuntimeError("Socket lost mid-batch"),
                buys=["AAPL x100 @ 150.0"],
                sells=[],
                skipped=[],
                unresolved=["MSFT", "GOOG"],
            )
            raise exc

        def fake_save(*args, **kwargs):
            saved_state.append(True)

        def mock_send_alert(*args, **kwargs):
            pass

        result = live_daemon.execute_signals_with_retry(
            "ftse", ["AAPL", "MSFT", "GOOG"], None, None, None, broker,
            True, logger, daemon_state=daemon_state,
            execute_signals=fake_execute,
            save_state=fake_save,
            send_interrupt_alert=mock_send_alert
        )

        assert daemon_state["halt_new_entries"] is True
        assert daemon_state["needs_reconciliation"] is True
        assert daemon_state["pending_retry_tickers"] == {"ftse": ["MSFT", "GOOG"]}
        assert result == (["AAPL x100 @ 150.0"], [], ["MSFT", "GOOG"])
        assert logger.critical.called
        assert not broker.disconnect.called  # Should not retry

    def test_execution_interrupted_with_sells_also_halts(self, monkeypatch):
        """ExecutionInterrupted with non-empty sells halts new entries."""
        from Strategy_Auto_Trader.markov_cli import execute as execute_mod

        broker = mock.Mock()
        logger = mock.Mock()
        daemon_state = {"halt_new_entries": False}
        saved_state = []

        def fake_execute(*args, **kwargs):
            exc = execute_mod.ExecutionInterrupted(
                RuntimeError("Connection lost"),
                buys=[],
                sells=["AAPL x100 @ 145.0"],
                skipped=[],
                unresolved=["MSFT"],
            )
            raise exc

        def fake_save(*args, **kwargs):
            saved_state.append(True)

        def mock_send_alert(*args, **kwargs):
            pass

        result = live_daemon.execute_signals_with_retry(
            "ftse", ["AAPL", "MSFT"], None, None, None, broker,
            True, logger, daemon_state=daemon_state,
            execute_signals=fake_execute,
            save_state=fake_save,
            send_interrupt_alert=mock_send_alert
        )

        assert daemon_state["halt_new_entries"] is True
        assert daemon_state["needs_reconciliation"] is True
        assert daemon_state["pending_retry_tickers"] == {"ftse": ["MSFT"]}
        assert result == ([], ["AAPL x100 @ 145.0"], ["MSFT"])

    def test_pending_retry_tickers_overwritten_on_second_interrupt_same_market(self, monkeypatch):
        """A second interrupt in the same market before the first is retried
        overwrites the pending list — the newest unresolved set is authoritative,
        no merge logic."""
        from Strategy_Auto_Trader.markov_cli import execute as execute_mod

        broker = mock.Mock()
        logger = mock.Mock()
        daemon_state = {"halt_new_entries": False}

        def make_fake_execute(unresolved):
            def fake_execute(*args, **kwargs):
                raise execute_mod.ExecutionInterrupted(
                    RuntimeError("Socket lost"), buys=[], sells=[], skipped=[], unresolved=unresolved,
                )
            return fake_execute

        live_daemon.execute_signals_with_retry(
            "ftse", ["AAPL"], None, None, None, broker,
            True, logger, daemon_state=daemon_state,
            execute_signals=make_fake_execute(["AAPL"]),
            save_state=lambda s: None,
            send_interrupt_alert=lambda *a, **k: None,
        )
        assert daemon_state["pending_retry_tickers"] == {"ftse": ["AAPL"]}

        live_daemon.execute_signals_with_retry(
            "ftse", ["MSFT", "GOOG"], None, None, None, broker,
            True, logger, daemon_state=daemon_state,
            execute_signals=make_fake_execute(["MSFT", "GOOG"]),
            save_state=lambda s: None,
            send_interrupt_alert=lambda *a, **k: None,
        )
        assert daemon_state["pending_retry_tickers"] == {"ftse": ["MSFT", "GOOG"]}

    def test_execution_interrupted_no_orders_still_halts(self, monkeypatch):
        """ExecutionInterrupted with empty buys/sells still halts — unresolved's
        broker-call outcome is unknown even when no confirmed fill exists yet.
        Regression test for the VOD incident: a socket disconnect mid
        place_order() must not be silently retried and dropped."""
        from Strategy_Auto_Trader.markov_cli import execute as execute_mod

        call_count = [0]
        broker = mock.Mock()
        logger = mock.Mock()
        daemon_state = {"halt_new_entries": False}
        saved_state = []
        alerts = []

        def fake_execute(*args, **kwargs):
            call_count[0] += 1
            raise execute_mod.ExecutionInterrupted(
                RuntimeError("Early interrupt"),
                buys=[],
                sells=[],
                skipped=[],
                unresolved=["AAPL", "MSFT"],
            )

        def fake_save(*args, **kwargs):
            saved_state.append(True)

        def mock_send_alert(*args, **kwargs):
            alerts.append(args)

        result = live_daemon.execute_signals_with_retry(
            "ftse", ["AAPL", "MSFT"], None, None, None, broker,
            True, logger, daemon_state=daemon_state, max_retries=2,
            execute_signals=fake_execute,
            save_state=fake_save,
            send_interrupt_alert=mock_send_alert,
        )

        assert daemon_state["halt_new_entries"] is True
        assert daemon_state["pending_retry_tickers"] == {"ftse": ["AAPL", "MSFT"]}
        assert result == ([], [], ["AAPL", "MSFT"])
        assert call_count[0] == 1  # No retry — halted on first interrupt
        assert not broker.disconnect.called  # Should not retry
        assert logger.critical.called
        assert saved_state  # daemon_state was persisted
        assert not alerts  # no immediate email — reconciliation decides

    def test_execution_interrupted_pending_retry_set_when_no_orders(self, monkeypatch):
        """ExecutionInterrupted with empty buys/sells still populates pending_retry_tickers
        so reconciliation can retry the unresolved ticker once it clears."""
        from Strategy_Auto_Trader.markov_cli import execute as execute_mod

        broker = mock.Mock()
        logger = mock.Mock()
        daemon_state = {"halt_new_entries": False}

        def fake_execute(*args, **kwargs):
            raise execute_mod.ExecutionInterrupted(
                RuntimeError("Socket disconnect"),
                buys=[],
                sells=[],
                skipped=[],
                unresolved=["VOD.L"],
            )

        live_daemon.execute_signals_with_retry(
            "ftse", ["VOD.L"], None, None, None, broker,
            True, logger, daemon_state=daemon_state,
            execute_signals=fake_execute,
            save_state=lambda s: None,
        )

        assert daemon_state["halt_new_entries"] is True
        assert daemon_state["pending_retry_tickers"] == {"ftse": ["VOD.L"]}

    def test_execution_interrupted_single_unresolved_ticker_matches_incident(self, monkeypatch):
        """Shape of the real VOD incident: one ticker in the batch, socket disconnect
        mid place_order(), no confirmed fills — must halt and alert, not retry."""
        from Strategy_Auto_Trader.markov_cli import execute as execute_mod

        broker = mock.Mock()
        logger = mock.Mock()
        daemon_state = {"halt_new_entries": False}
        call_count = [0]

        def fake_execute(*args, **kwargs):
            call_count[0] += 1
            raise execute_mod.ExecutionInterrupted(
                RuntimeError("Socket disconnect during order placement: Socket disconnect."),
                buys=[],
                sells=[],
                skipped=[],
                unresolved=["VOD.L"],
            )

        def mock_send_alert(*args, **kwargs):
            pass

        result = live_daemon.execute_signals_with_retry(
            "ftse", ["VOD.L"], None, None, None, broker,
            True, logger, daemon_state=daemon_state, max_retries=3,
            execute_signals=fake_execute,
            save_state=lambda s: None,
            send_interrupt_alert=mock_send_alert,
        )

        assert daemon_state["halt_new_entries"] is True
        assert result == ([], [], ["VOD.L"])
        assert call_count[0] == 1  # Not retried 3x like the real incident did
        assert not broker.disconnect.called

    def test_execution_interrupted_no_daemon_state_doesnt_crash(self, monkeypatch):
        """ExecutionInterrupted works without daemon_state parameter."""
        from Strategy_Auto_Trader.markov_cli import execute as execute_mod

        broker = mock.Mock()
        logger = mock.Mock()

        def fake_execute(*args, **kwargs):
            exc = execute_mod.ExecutionInterrupted(
                RuntimeError("Socket lost"),
                buys=["AAPL x100 @ 150.0"],
                sells=[],
                skipped=[],
                unresolved=["MSFT"],
            )
            raise exc

        def mock_send_alert(*args, **kwargs):
            pass

        result = live_daemon.execute_signals_with_retry(
            "ftse", ["AAPL", "MSFT"], None, None, None, broker,
            True, logger,  # No daemon_state passed
            execute_signals=fake_execute,
            send_interrupt_alert=mock_send_alert
        )

        assert result == (["AAPL x100 @ 150.0"], [], ["MSFT"])
        assert logger.critical.called


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
        pm = PortfolioManager(20_000, tmp_path / "execution_state.json")
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
        pm = PortfolioManager(20_000, tmp_path / "execution_state.json")
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
        pm = PortfolioManager(20_000, tmp_path / "execution_state.json")
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

    def test_takeover_retries_lock_after_slow_kill(self, monkeypatch):
        """Regression: killing a slow-to-die holder can itself take longer
        than the takeover deadline. That must not cause the retry to be
        abandoned before it even attempts to re-acquire the now-free lock."""
        logger = mock.Mock()
        monkeypatch.setattr(live_daemon, "_read_holder_pid", lambda: 999)
        monkeypatch.setattr(live_daemon, "_kill_daemon_process",
                             lambda pid, logger: True)

        try_lock_results = iter([False, True])
        monkeypatch.setattr(live_daemon, "_try_lock",
                             lambda handle: next(try_lock_results))

        # Kill "took" 20s — past the 15s deadline budget computed at loop start.
        clock = iter([1000.0, 1020.0])
        monkeypatch.setattr(live_daemon.time, "time", lambda: next(clock))

        assert live_daemon.acquire_process_lock(logger, takeover=True) is True
        assert not logger.critical.called


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
    """paused_by_user passes allow_new_entries=False into execute_signals."""
    from Strategy_Auto_Trader.markov_cli import batch, execute

    def fake_process_ticker(ticker_cfg, defaults, send_email):
        return {"ticker": "AAPL", "status": "OK", "time": 0.0,
                "result": {"ticker": "AAPL", "close": 100.0}}

    captured = {}

    def fake_execute_signals(tickers, data_dir, portfolio, limit_tracker,
                             broker, allow_new_entries=True, **kwargs):
        captured["allow_new_entries"] = allow_new_entries
        return [], [], []

    monkeypatch.setattr(batch, "process_ticker", fake_process_ticker)
    monkeypatch.setattr(execute, "execute_signals", fake_execute_signals)
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["AAPL"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, l: [])

    config = {
        "daytime": {"max_seconds_per_cycle": 60, "cycle_buffer_minutes": 0},
        "execution": {},
    }
    daemon_state = {"cursors": {}, "paused_by_user": True}
    live_daemon.process_cycle(
        "test_market", {}, config, daemon_state,
        portfolio=mock.Mock(), broker=mock.Mock(spec=[]), logger=mock.Mock(),
    )
    assert captured["allow_new_entries"] is False


def test_process_cycle_halt_and_paused_independent(monkeypatch):
    """halt_new_entries and paused_by_user both pass allow_new_entries=False."""
    from Strategy_Auto_Trader.markov_cli import batch, execute

    def fake_process_ticker(ticker_cfg, defaults, send_email):
        return {"ticker": "AAPL", "status": "OK", "time": 0.0,
                "result": {"ticker": "AAPL", "close": 100.0}}

    captured = []

    def fake_execute_signals(tickers, data_dir, portfolio, limit_tracker,
                             broker, allow_new_entries=True, **kwargs):
        captured.append(allow_new_entries)
        return [], [], []

    monkeypatch.setattr(batch, "process_ticker", fake_process_ticker)
    monkeypatch.setattr(execute, "execute_signals", fake_execute_signals)
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["AAPL"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, l: [])

    config = {
        "daytime": {"max_seconds_per_cycle": 60, "cycle_buffer_minutes": 0},
        "execution": {},
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

    # Both should have blocked new entries
    assert captured == [False, False]


def test_write_app_status_snapshot_includes_paused_by_user(tmp_path, monkeypatch):
    """Snapshot includes paused_by_user flag."""
    from Strategy_Auto_Trader.markov_cli import live_daemon
    from Strategy_Auto_Trader.broker.portfolio import PortfolioManager

    monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)
    pm = PortfolioManager(20_000, tmp_path / "execution_state.json")
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
    pm = PortfolioManager(20_000, tmp_path / "execution_state.json")
    daemon_state = {}  # No paused_by_user key
    config = {"execution": {"dry_run": True}, "markets": {}}

    live_daemon.write_app_status_snapshot(pm, daemon_state, config, {}, mock.Mock())

    snapshot = json.loads((tmp_path / "app_status.json").read_text(encoding="utf-8"))
    assert snapshot["paused_by_user"] is False


# -- Per-ticker strategy overrides from watchlist ----

def test_load_ticker_overrides_returns_empty_dict_when_file_missing(tmp_path, monkeypatch):
    """load_ticker_overrides returns {} when in_scope file doesn't exist."""
    monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)
    logger = mock.Mock()

    result = live_daemon.load_ticker_overrides("ftse", logger)

    assert result == {}
    assert not logger.error.called


def test_load_ticker_overrides_returns_empty_dict_when_overrides_key_missing(tmp_path, monkeypatch):
    """load_ticker_overrides returns {} when overrides key is absent from JSON."""
    monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)
    logger = mock.Mock()

    # Create in_scope file without overrides key
    scope_file = tmp_path / "in_scope_ftse.json"
    scope_file.write_text(json.dumps({"kept": ["AAPL", "MSFT"], "excluded": []}), encoding="utf-8")

    result = live_daemon.load_ticker_overrides("ftse", logger)

    assert result == {}


def test_load_ticker_overrides_returns_persisted_overrides(tmp_path, monkeypatch):
    """load_ticker_overrides returns the persisted overrides dict when present."""
    monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)
    logger = mock.Mock()

    # Create in_scope file with overrides
    scope_file = tmp_path / "in_scope_ftse.json"
    overrides_data = {
        "AAPL": {"strategy": "breakout_momentum"},
        "MSFT": {"strategy": "conservative"},
    }
    scope_file.write_text(
        json.dumps({"kept": ["AAPL", "MSFT"], "excluded": [], "overrides": overrides_data}),
        encoding="utf-8"
    )

    result = live_daemon.load_ticker_overrides("ftse", logger)

    assert result == overrides_data


def test_load_ticker_overrides_handles_json_error(tmp_path, monkeypatch):
    """load_ticker_overrides returns {} and logs error on JSON parse failure."""
    monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)
    logger = mock.Mock()

    # Create invalid JSON file
    scope_file = tmp_path / "in_scope_ftse.json"
    scope_file.write_text("{ invalid json", encoding="utf-8")

    result = live_daemon.load_ticker_overrides("ftse", logger)

    assert result == {}
    assert logger.error.called


def test_process_cycle_merges_overrides_into_must_run_ticker_cfg(monkeypatch):
    """In must-run stage, ticker_cfg is merged with overrides from watchlist."""
    from Strategy_Auto_Trader.markov_cli import batch

    captured = {}

    def fake_process_ticker(ticker_cfg, defaults, send_email):
        captured["ticker_cfg"] = ticker_cfg
        return {"ticker": ticker_cfg["ticker"], "status": "FAIL: stub", "time": 0.0}

    monkeypatch.setattr(batch, "process_ticker", fake_process_ticker)
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["AAPL"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, l: ["AAPL"])
    # Provide overrides via mock
    overrides_data = {"AAPL": {"strategy": "breakout_momentum"}}
    monkeypatch.setattr(live_daemon, "load_ticker_overrides", lambda m, l: overrides_data)

    config = {"daytime": {"max_seconds_per_cycle": 60, "cycle_buffer_minutes": 0}}
    live_daemon.process_cycle(
        "test_market", {}, config, {"cursors": {}},
        portfolio=None, broker=None, logger=mock.Mock(),
    )

    assert captured["ticker_cfg"]["ticker"] == "AAPL"
    assert captured["ticker_cfg"]["strategy"] == "breakout_momentum"


def test_process_cycle_merges_overrides_into_round_robin_ticker_cfg(monkeypatch):
    """In round-robin stage, ticker_cfg is merged with overrides from watchlist."""
    from Strategy_Auto_Trader.markov_cli import batch

    captured = {}

    def fake_process_ticker(ticker_cfg, defaults, send_email):
        captured["ticker_cfg"] = ticker_cfg
        return {"ticker": ticker_cfg["ticker"], "status": "FAIL: stub", "time": 0.0}

    monkeypatch.setattr(batch, "process_ticker", fake_process_ticker)
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["AAPL", "MSFT"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, l: [])
    # Provide overrides via mock
    overrides_data = {"MSFT": {"strategy": "conservative"}}
    monkeypatch.setattr(live_daemon, "load_ticker_overrides", lambda m, l: overrides_data)
    monkeypatch.setattr(live_daemon, "next_round_robin_slice",
                       lambda market_name, candidates, daemon_state, logger: (["MSFT"], 0))

    config = {"daytime": {"max_seconds_per_cycle": 60, "cycle_buffer_minutes": 0}}
    live_daemon.process_cycle(
        "test_market", {}, config, {"cursors": {}},
        portfolio=None, broker=None, logger=mock.Mock(),
    )

    assert captured["ticker_cfg"]["ticker"] == "MSFT"
    assert captured["ticker_cfg"]["strategy"] == "conservative"


def test_process_cycle_pins_already_open_strategy_over_watchlist_override(monkeypatch, tmp_path):
    """In must-run stage, pinned strategy from get_open_strategy() wins over watchlist override."""
    from Strategy_Auto_Trader.markov_cli import batch
    from Strategy_Auto_Trader.output import trade_state

    captured = {}

    def fake_process_ticker(ticker_cfg, defaults, send_email):
        captured["ticker_cfg"] = ticker_cfg
        return {"ticker": ticker_cfg["ticker"], "status": "FAIL: stub", "time": 0.0}

    monkeypatch.setattr(batch, "process_ticker", fake_process_ticker)
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["AAPL"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, l: ["AAPL"])
    # Override says "conservative" but pinned is "breakout_momentum"
    overrides_data = {"AAPL": {"strategy": "conservative"}}
    monkeypatch.setattr(live_daemon, "load_ticker_overrides", lambda m, l: overrides_data)

    # Mock get_open_strategy to return the pinned strategy
    monkeypatch.setattr(trade_state, "get_open_strategy", lambda t: "breakout_momentum")

    config = {"daytime": {"max_seconds_per_cycle": 60, "cycle_buffer_minutes": 0}}
    live_daemon.process_cycle(
        "test_market", {}, config, {"cursors": {}},
        portfolio=None, broker=None, logger=mock.Mock(),
    )

    # Pinned strategy should win
    assert captured["ticker_cfg"]["ticker"] == "AAPL"
    assert captured["ticker_cfg"]["strategy"] == "breakout_momentum"


def test_process_cycle_no_pin_in_round_robin_stage(monkeypatch):
    """In round-robin stage, no pin is applied (only open positions get pinned)."""
    from Strategy_Auto_Trader.markov_cli import batch
    from Strategy_Auto_Trader.output import trade_state

    captured = {}

    def fake_process_ticker(ticker_cfg, defaults, send_email):
        captured["ticker_cfg"] = ticker_cfg
        return {"ticker": ticker_cfg["ticker"], "status": "FAIL: stub", "time": 0.0}

    monkeypatch.setattr(batch, "process_ticker", fake_process_ticker)
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["MSFT"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, l: [])
    overrides_data = {"MSFT": {"strategy": "conservative"}}
    monkeypatch.setattr(live_daemon, "load_ticker_overrides", lambda m, l: overrides_data)
    monkeypatch.setattr(live_daemon, "next_round_robin_slice",
                       lambda market_name, candidates, daemon_state, logger: (["MSFT"], 0))
    # get_open_strategy should never be called for round-robin candidates
    monkeypatch.setattr(trade_state, "get_open_strategy", lambda t: "should_not_be_used")

    config = {"daytime": {"max_seconds_per_cycle": 60, "cycle_buffer_minutes": 0}}
    live_daemon.process_cycle(
        "test_market", {}, config, {"cursors": {}},
        portfolio=None, broker=None, logger=mock.Mock(),
    )

    # Override strategy should be used, not pinned
    assert captured["ticker_cfg"]["strategy"] == "conservative"


def test_process_cycle_logs_warning_on_must_run_fail_status(monkeypatch):
    """When must-run ticker returns FAIL status, logger.warning is called."""
    from Strategy_Auto_Trader.markov_cli import batch

    def fake_process_ticker(ticker_cfg, defaults, send_email):
        return {"ticker": ticker_cfg["ticker"], "status": "FAIL: bad strategy name", "time": 0.0}

    monkeypatch.setattr(batch, "process_ticker", fake_process_ticker)
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["AAPL"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, l: ["AAPL"])
    monkeypatch.setattr(live_daemon, "load_ticker_overrides", lambda m, l: {})

    logger = mock.Mock()
    config = {"daytime": {"max_seconds_per_cycle": 60, "cycle_buffer_minutes": 0}}
    live_daemon.process_cycle(
        "test_market", {}, config, {"cursors": {}},
        portfolio=None, broker=None, logger=logger,
    )

    # Check that logger.warning was called with the right message
    assert logger.warning.called
    calls = [c for c in logger.warning.call_args_list if "processing failed" in str(c)]
    assert len(calls) > 0


def test_process_cycle_logs_warning_on_round_robin_fail_status(monkeypatch):
    """When round-robin ticker returns FAIL status, logger.warning is called."""
    from Strategy_Auto_Trader.markov_cli import batch

    def fake_process_ticker(ticker_cfg, defaults, send_email):
        return {"ticker": ticker_cfg["ticker"], "status": "FAIL: data unavailable", "time": 0.0}

    monkeypatch.setattr(batch, "process_ticker", fake_process_ticker)
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["MSFT"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, l: [])
    monkeypatch.setattr(live_daemon, "load_ticker_overrides", lambda m, l: {})
    monkeypatch.setattr(live_daemon, "next_round_robin_slice",
                       lambda market_name, candidates, daemon_state, logger: (["MSFT"], 0))

    logger = mock.Mock()
    config = {"daytime": {"max_seconds_per_cycle": 60, "cycle_buffer_minutes": 0}}
    live_daemon.process_cycle(
        "test_market", {}, config, {"cursors": {}},
        portfolio=None, broker=None, logger=logger,
    )

    # Check that logger.warning was called
    assert logger.warning.called
    calls = [c for c in logger.warning.call_args_list if "processing failed" in str(c)]
    assert len(calls) > 0


def test_process_cycle_parallel_collects_all_results(monkeypatch):
    """workers>1 path still collects every candidate's result."""
    from Strategy_Auto_Trader.markov_cli import batch

    processed_tickers = []

    def fake_process_ticker(ticker_cfg, defaults, send_email, state_lock=None):
        processed_tickers.append(ticker_cfg["ticker"])
        return {"ticker": ticker_cfg["ticker"], "status": "FAIL: stub", "time": 0.0}

    monkeypatch.setattr(batch, "process_ticker", fake_process_ticker)
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["AAPL", "MSFT", "GOOG"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, l: [])

    config = {"daytime": {"max_seconds_per_cycle": 60, "cycle_buffer_minutes": 0}}
    n = live_daemon.process_cycle(
        "test_market", {}, config, {"cursors": {}},
        portfolio=None, broker=None, logger=mock.Mock(),
        workers=2,
    )

    assert n == 3
    assert sorted(processed_tickers) == ["AAPL", "GOOG", "MSFT"]


def test_process_cycle_parallel_state_lock_prevents_lost_writes(monkeypatch):
    """state_lock is acquired during shared writes when workers>1."""
    import threading
    from Strategy_Auto_Trader.markov_cli import batch

    lock_acquisitions = []

    def fake_process_ticker(ticker_cfg, defaults, send_email, state_lock=None):
        if state_lock is not None:
            lock_acquisitions.append(ticker_cfg["ticker"])
        return {"ticker": ticker_cfg["ticker"], "status": "FAIL: stub", "time": 0.0}

    monkeypatch.setattr(batch, "process_ticker", fake_process_ticker)
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["AAPL", "MSFT"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, l: [])

    config = {"daytime": {"max_seconds_per_cycle": 60, "cycle_buffer_minutes": 0}}
    live_daemon.process_cycle(
        "test_market", {}, config, {"cursors": {}},
        portfolio=None, broker=None, logger=mock.Mock(),
        workers=2,
    )

    # Both tickers must have received a non-None state_lock
    assert sorted(lock_acquisitions) == ["AAPL", "MSFT"]


def test_process_cycle_parallel_manual_commands_called_per_future(monkeypatch):
    """process_manual_commands_wrapper is called after each completed future, not just once."""
    from Strategy_Auto_Trader.markov_cli import batch

    cmd_call_count = []

    def fake_process_ticker(ticker_cfg, defaults, send_email, state_lock=None):
        return {"ticker": ticker_cfg["ticker"], "status": "FAIL: stub", "time": 0.0}

    def fake_manual_commands(config, portfolio, broker, logger, daemon_state):
        cmd_call_count.append(1)

    monkeypatch.setattr(batch, "process_ticker", fake_process_ticker)
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["AAPL", "MSFT", "GOOG"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, l: [])
    monkeypatch.setattr(live_daemon, "process_manual_commands_wrapper", fake_manual_commands)

    config = {"daytime": {"max_seconds_per_cycle": 60, "cycle_buffer_minutes": 0}}
    live_daemon.process_cycle(
        "test_market", {}, config, {"cursors": {}},
        portfolio=None, broker=None, logger=mock.Mock(),
        workers=2,
    )

    # Called at minimum once per future (3 tickers) plus before/after the executor
    assert len(cmd_call_count) >= 3


def test_process_cycle_parallel_budget_cancels_pending(monkeypatch):
    """Budget check in as_completed loop cancels queued futures and logs exhaustion."""
    import time as time_module
    from Strategy_Auto_Trader.markov_cli import batch

    # Intercept time.time used inside live_daemon: return a value that appears
    # to have blown past the budget after the cycle_start capture.
    _real_time = time_module.time
    _call_n = [0]
    _budget_base = [None]

    def mock_time():
        _call_n[0] += 1
        if _call_n[0] == 1:
            # First call is cycle_start; capture the base and return it normally.
            t = _real_time()
            _budget_base[0] = t
            return t
        # Every subsequent call returns a value 9999s past cycle_start,
        # so the budget check fires immediately after the first future completes.
        return (_budget_base[0] or _real_time()) + 9999

    def fake_process_ticker(ticker_cfg, defaults, send_email, state_lock=None):
        return {"ticker": ticker_cfg["ticker"], "status": "FAIL: stub", "time": 0.0}

    monkeypatch.setattr(time_module, "time", mock_time)
    monkeypatch.setattr(batch, "process_ticker", fake_process_ticker)
    # 6 tickers, workers=2 — some will be in the queue when budget fires
    monkeypatch.setattr(live_daemon, "load_in_scope_tickers",
                        lambda m, l: ["T1", "T2", "T3", "T4", "T5", "T6"])
    monkeypatch.setattr(live_daemon, "get_open_positions", lambda m, l: [])

    mock_logger = mock.Mock()
    config = {"daytime": {"max_seconds_per_cycle": 100, "cycle_buffer_minutes": 0}}
    n = live_daemon.process_cycle(
        "test_market", {}, config, {"cursors": {}},
        portfolio=None, broker=None, logger=mock_logger,
        workers=2,
    )

    # Fewer than all 6 collected (budget fired before draining the queue)
    assert n < 6
    # Logger should have recorded the exhaustion (either budget log line or cycle-done skipped count)
    log_text = " ".join(str(c) for c in mock_logger.info.call_args_list)
    assert "budget" in log_text.lower() or "skipped" in log_text.lower()


class TestExecuteProcessedTickersTopKGate:
    """_execute_processed_tickers top_k gate: non-top-k tickers are blocked at
    execution time (not at scope time), so all vol-screened tickers run backtests
    but only top-k ones place orders."""

    def _processed(self, tickers):
        return [{"ticker": t, "status": "OK", "result": {"close": 100.0}} for t in tickers]

    def test_non_top_k_ticker_skipped(self, monkeypatch, tmp_path):
        """Ticker not in top_k_universe.json and not an open position is blocked."""
        top_k_path = tmp_path / "top_k_universe.json"
        top_k_path.write_text(json.dumps({"tickers": ["AAPL"]}), encoding="utf-8")
        monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)

        executed = []
        monkeypatch.setattr(live_daemon, "execute_signals_with_retry",
                            lambda market, tickers, *a, **k: executed.extend(tickers) or ([], [], []))
        mock_portfolio = mock.Mock()
        mock_portfolio.get_limit_tracker.return_value = mock.Mock()

        live_daemon._execute_processed_tickers(
            "test_market",
            self._processed(["AAPL", "GOOG"]),  # GOOG not in top_k
            config={"execution": {}},
            daemon_state={"positions": {}},
            portfolio=mock_portfolio,
            broker=mock.Mock(),
            logger=mock.Mock(),
        )

        assert "AAPL" in executed
        assert "GOOG" not in executed

    def test_open_position_passes_even_if_not_in_top_k(self, monkeypatch, tmp_path):
        """Open positions always pass — needed for exits and monitoring."""
        top_k_path = tmp_path / "top_k_universe.json"
        top_k_path.write_text(json.dumps({"tickers": ["AAPL"]}), encoding="utf-8")
        monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)

        executed = []
        monkeypatch.setattr(live_daemon, "execute_signals_with_retry",
                            lambda market, tickers, *a, **k: executed.extend(tickers) or ([], [], []))
        mock_portfolio = mock.Mock()
        mock_portfolio.get_limit_tracker.return_value = mock.Mock()

        live_daemon._execute_processed_tickers(
            "test_market",
            self._processed(["GOOG"]),  # not in top_k but open position
            config={"execution": {}},
            daemon_state={"positions": {"GOOG": {}}},
            portfolio=mock_portfolio,
            broker=mock.Mock(),
            logger=mock.Mock(),
        )

        assert "GOOG" in executed

    def test_missing_top_k_file_allows_all(self, monkeypatch, tmp_path):
        """If top_k_universe.json is missing, gate is a no-op (safe fallback)."""
        monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)  # no top_k file

        executed = []
        monkeypatch.setattr(live_daemon, "execute_signals_with_retry",
                            lambda market, tickers, *a, **k: executed.extend(tickers) or ([], [], []))
        mock_portfolio = mock.Mock()
        mock_portfolio.get_limit_tracker.return_value = mock.Mock()

        live_daemon._execute_processed_tickers(
            "test_market",
            self._processed(["AAPL", "GOOG"]),
            config={"execution": {}},
            daemon_state={"positions": {}},
            portfolio=mock_portfolio,
            broker=mock.Mock(),
            logger=mock.Mock(),
        )

        assert "AAPL" in executed
        assert "GOOG" in executed


class TestRetryPendingTickers:
    """retry_pending_tickers: immediate re-evaluation of interrupted trades,
    instead of waiting for that market's next hourly cycle."""

    def _config(self):
        return {
            "markets": {"ftse": {}},
            "execution": {},
        }

    def test_noop_when_none_pending(self, monkeypatch):
        daemon_state = {}
        called = []
        monkeypatch.setattr(live_daemon, "is_trading_hours", lambda *a, **k: called.append(1) or True)

        live_daemon.retry_pending_tickers(
            self._config(), daemon_state, portfolio=mock.Mock(), broker=mock.Mock(), logger=mock.Mock(),
        )

        assert not called  # never even looked at trading hours — nothing pending

    def test_reevaluates_fresh_and_executes_via_execute_signals_with_retry(self, monkeypatch, tmp_path):
        """Must re-run process_ticker (fresh signal), and go through
        execute_signals_with_retry — not a bespoke call to execute_signals."""
        from Strategy_Auto_Trader.markov_cli import batch

        monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)  # no top_k file → gate disabled
        daemon_state = {"pending_retry_tickers": {"ftse": ["MNG.L"]}}
        process_ticker_calls = []

        def fake_process_ticker(ticker_cfg, defaults, send_email):
            process_ticker_calls.append(ticker_cfg["ticker"])
            return {"ticker": ticker_cfg["ticker"], "status": "OK", "result": {"close": 1.5}}

        exec_calls = []

        def fake_execute_signals_with_retry(*args, **kwargs):
            exec_calls.append((args, kwargs))
            return (["MNG.L x17 @ 1.5"], [], [])

        monkeypatch.setattr(batch, "process_ticker", fake_process_ticker)
        monkeypatch.setattr(live_daemon, "is_trading_hours", lambda *a, **k: True)
        monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["MNG.L"])
        monkeypatch.setattr(live_daemon, "load_ticker_overrides", lambda m, l: {})
        monkeypatch.setattr(live_daemon, "execute_signals_with_retry", fake_execute_signals_with_retry)
        monkeypatch.setattr(live_daemon, "save_daemon_state", lambda s: None)

        live_daemon.retry_pending_tickers(
            self._config(), daemon_state, portfolio=mock.Mock(), broker=mock.Mock(), logger=mock.Mock(),
        )

        assert process_ticker_calls == ["MNG.L"]
        assert len(exec_calls) == 1
        assert exec_calls[0][0][1] == ["MNG.L"]  # ticker_list positional arg

    def test_defaults_source_matches_main_cycle(self, monkeypatch):
        """The retry path shares the same IBKR-backed data source default as
        the main cycle — a retried ticker must not fall back to yfinance."""
        from Strategy_Auto_Trader.markov_cli import batch

        daemon_state = {"pending_retry_tickers": {"ftse": ["MNG.L"]}}
        captured = {}

        def fake_process_ticker(ticker_cfg, defaults, send_email):
            captured["defaults"] = defaults
            return {"ticker": ticker_cfg["ticker"], "status": "FAIL: stub"}

        monkeypatch.setattr(batch, "process_ticker", fake_process_ticker)
        monkeypatch.setattr(live_daemon, "is_trading_hours", lambda *a, **k: True)
        monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["MNG.L"])
        monkeypatch.setattr(live_daemon, "load_ticker_overrides", lambda m, l: {})
        monkeypatch.setattr(live_daemon, "save_daemon_state", lambda s: None)

        live_daemon.retry_pending_tickers(
            self._config(), daemon_state, portfolio=mock.Mock(), broker=mock.Mock(), logger=mock.Mock(),
        )

        assert captured["defaults"]["source"] == "ibkr"

    def test_skips_market_closed(self, monkeypatch):
        """Market closed by the time reconciliation clears — don't fire, and
        leave the pending record for a real next-cycle attempt."""
        from Strategy_Auto_Trader.markov_cli import batch

        daemon_state = {"pending_retry_tickers": {"ftse": ["MNG.L"]}}
        process_ticker_calls = []
        monkeypatch.setattr(batch, "process_ticker",
                            lambda *a, **k: process_ticker_calls.append(1))
        monkeypatch.setattr(live_daemon, "is_trading_hours", lambda *a, **k: False)
        monkeypatch.setattr(live_daemon, "save_daemon_state", lambda s: None)

        live_daemon.retry_pending_tickers(
            self._config(), daemon_state, portfolio=mock.Mock(), broker=mock.Mock(), logger=mock.Mock(),
        )

        assert not process_ticker_calls
        assert daemon_state["pending_retry_tickers"] == {"ftse": ["MNG.L"]}  # left in place

    def test_drops_ticker_no_longer_in_scope(self, monkeypatch):
        """Ticker removed from the watchlist since the interrupt — drop it,
        don't evaluate it, and clear the pending record (one-shot)."""
        from Strategy_Auto_Trader.markov_cli import batch

        daemon_state = {"pending_retry_tickers": {"ftse": ["MNG.L"]}}
        process_ticker_calls = []
        monkeypatch.setattr(batch, "process_ticker",
                            lambda *a, **k: process_ticker_calls.append(1))
        monkeypatch.setattr(live_daemon, "is_trading_hours", lambda *a, **k: True)
        monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: [])  # removed
        monkeypatch.setattr(live_daemon, "save_daemon_state", lambda s: None)

        live_daemon.retry_pending_tickers(
            self._config(), daemon_state, portfolio=mock.Mock(), broker=mock.Mock(), logger=mock.Mock(),
        )

        assert not process_ticker_calls
        assert daemon_state["pending_retry_tickers"] == {}

    @pytest.mark.parametrize("exec_result", [
        (["MNG.L x17 @ 1.5"], [], []),  # BUY fired
        ([], ["MNG.L x17 @ 1.6"], []),  # SELL fired
        ([], [], ["MNG.L(HOLD)"]),      # signal no longer warrants entry
    ])
    def test_clears_entry_after_attempt_regardless_of_outcome(self, monkeypatch, exec_result):
        from Strategy_Auto_Trader.markov_cli import batch

        daemon_state = {"pending_retry_tickers": {"ftse": ["MNG.L"]}}
        monkeypatch.setattr(batch, "process_ticker",
                            lambda ticker_cfg, defaults, send_email: {
                                "ticker": ticker_cfg["ticker"], "status": "OK", "result": {"close": 1.5}})
        monkeypatch.setattr(live_daemon, "is_trading_hours", lambda *a, **k: True)
        monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["MNG.L"])
        monkeypatch.setattr(live_daemon, "load_ticker_overrides", lambda m, l: {})
        monkeypatch.setattr(live_daemon, "execute_signals_with_retry", lambda *a, **k: exec_result)
        monkeypatch.setattr(live_daemon, "save_daemon_state", lambda s: None)

        live_daemon.retry_pending_tickers(
            self._config(), daemon_state, portfolio=mock.Mock(), broker=mock.Mock(), logger=mock.Mock(),
        )

        assert daemon_state["pending_retry_tickers"] == {}

    def test_reinterrupt_during_retry_rearms_without_infinite_loop(self, monkeypatch, tmp_path):
        """If the retry itself hits a fresh ExecutionInterrupted, halt/needs_reconciliation
        get re-armed with a fresh pending list — bounded by requiring reconciliation
        to clear again, not a runaway loop."""
        from Strategy_Auto_Trader.markov_cli import batch, execute as execute_mod

        monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)  # no top_k file → gate disabled
        daemon_state = {"pending_retry_tickers": {"ftse": ["MNG.L"]}, "halt_new_entries": False}
        monkeypatch.setattr(batch, "process_ticker",
                            lambda ticker_cfg, defaults, send_email: {
                                "ticker": ticker_cfg["ticker"], "status": "OK", "result": {"close": 1.5}})
        monkeypatch.setattr(live_daemon, "is_trading_hours", lambda *a, **k: True)
        monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["MNG.L"])
        monkeypatch.setattr(live_daemon, "load_ticker_overrides", lambda m, l: {})
        monkeypatch.setattr(live_daemon, "save_daemon_state", lambda s: None)

        def fake_execute_signals(*args, **kwargs):
            raise execute_mod.ExecutionInterrupted(
                RuntimeError("Socket disconnect again"), buys=[], sells=[], skipped=[],
                unresolved=["MNG.L"],
            )

        monkeypatch.setattr(execute_mod, "execute_signals", fake_execute_signals)

        live_daemon.retry_pending_tickers(
            self._config(), daemon_state, portfolio=mock.Mock(), broker=mock.Mock(), logger=mock.Mock(),
        )

        assert daemon_state["halt_new_entries"] is True
        assert daemon_state["needs_reconciliation"] is True
        assert daemon_state["pending_retry_tickers"] == {"ftse": ["MNG.L"]}  # freshly re-set, not stuck

    def test_batches_multiple_tickers_one_execute_call(self, monkeypatch, tmp_path):
        from Strategy_Auto_Trader.markov_cli import batch

        monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)  # no top_k file → gate disabled
        daemon_state = {"pending_retry_tickers": {"ftse": ["MNG.L", "AV.L"]}}
        monkeypatch.setattr(batch, "process_ticker",
                            lambda ticker_cfg, defaults, send_email: {
                                "ticker": ticker_cfg["ticker"], "status": "OK", "result": {"close": 1.5}})
        monkeypatch.setattr(live_daemon, "is_trading_hours", lambda *a, **k: True)
        monkeypatch.setattr(live_daemon, "load_in_scope_tickers", lambda m, l: ["MNG.L", "AV.L"])
        monkeypatch.setattr(live_daemon, "load_ticker_overrides", lambda m, l: {})
        monkeypatch.setattr(live_daemon, "save_daemon_state", lambda s: None)

        exec_calls = []
        monkeypatch.setattr(live_daemon, "execute_signals_with_retry",
                            lambda *a, **k: exec_calls.append(a) or ([], [], []))

        live_daemon.retry_pending_tickers(
            self._config(), daemon_state, portfolio=mock.Mock(), broker=mock.Mock(), logger=mock.Mock(),
        )

        assert len(exec_calls) == 1
        assert sorted(exec_calls[0][1]) == ["AV.L", "MNG.L"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
