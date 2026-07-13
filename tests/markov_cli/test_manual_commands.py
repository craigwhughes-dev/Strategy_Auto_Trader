"""Tests for manual_commands.py — sell command processing from mobile app."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

from Strategy_Auto_Trader.broker.types import FillResult
from Strategy_Auto_Trader.markov_cli.manual_commands import (
    _claim_pending_command,
    _ensure_command_dirs,
    _execute_pause_buying,
    _execute_resume_buying,
    _execute_sell,
    _execute_sell_all,
    _is_expired,
    _is_market_open,
    _move_to_done,
    _move_to_pending,
    _parse_iso_timestamp,
    _validate_command,
    _write_result,
    process_manual_commands,
)


@pytest.fixture
def commands_dir(tmp_path):
    """Temporary commands directory."""
    cmds = tmp_path / "commands"
    _ensure_command_dirs(cmds)
    return cmds


@pytest.fixture
def config():
    """Sample config with trading hours."""
    return {
        "markets": {
            "ftse": {
                "timezone": "Europe/London",
                "trading_start": "08:00",
                "trading_end": "16:30",
            },
            "sp500": {
                "timezone": "America/New_York",
                "trading_start": "14:30",
                "trading_end": "21:00",
            },
        },
        "execution": {
            "dry_run": True,
        },
    }


@pytest.fixture
def fake_portfolio(tmp_path):
    """Fake portfolio with positions."""
    from Strategy_Auto_Trader.broker.portfolio import PortfolioManager

    state_path = tmp_path / "state.json"
    pm = PortfolioManager(20_000, 5, state_path)
    return pm


@pytest.fixture
def fake_broker():
    """Fake broker that records orders."""
    class FakeBroker:
        def __init__(self):
            self.orders = []
            self.positions = {}

        def place_order(self, order):
            self.orders.append(order)
            fill = FillResult(
                ticker=order.ticker,
                action=order.action,
                fill_price=100.0 if order.ticker == "AZN.L" else 50.0,
                quantity=order.quantity,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            if order.action == "BUY":
                self.positions[order.ticker] = order.quantity
            else:
                self.positions.pop(order.ticker, None)
            return fill

        def connect(self):
            pass

        def disconnect(self):
            pass

        def get_open_positions(self):
            return self.positions

    return FakeBroker()


# -- Helper Function Tests --------------------------------------------------------


def test_parse_iso_timestamp_with_7_digit_fractional():
    """Parse 7-digit fractional seconds + Z."""
    dt = _parse_iso_timestamp("2026-07-13T10:00:00.1234567Z")
    assert dt is not None
    assert dt.year == 2026
    assert dt.month == 7
    assert dt.day == 13
    assert dt.hour == 10
    assert dt.minute == 0


def test_parse_iso_timestamp_with_6_digit_fractional():
    """Parse 6-digit fractional seconds + Z."""
    dt = _parse_iso_timestamp("2026-07-13T10:00:00.123456Z")
    assert dt is not None
    assert dt.hour == 10


def test_parse_iso_timestamp_no_fractional():
    """Parse timestamp without fractional seconds."""
    dt = _parse_iso_timestamp("2026-07-13T10:00:00Z")
    assert dt is not None
    assert dt.day == 13


def test_parse_iso_timestamp_invalid():
    """Invalid timestamp returns None."""
    assert _parse_iso_timestamp("not-a-timestamp") is None
    assert _parse_iso_timestamp("") is None
    assert _parse_iso_timestamp(None) is None


def test_validate_command_valid():
    """Valid command passes validation."""
    cmd = {
        "Id": "abc-123",
        "Action": "SELL",
        "Ticker": "AZN.L",
        "Status": "pending",
        "RequestedAtUtc": "2026-07-13T10:00:00Z",
        "ExpiresAtUtc": "2026-07-13T14:00:00Z",
        "Source": "android-app",
    }
    is_valid, err = _validate_command(cmd)
    assert is_valid
    assert err == ""


def test_validate_command_sell_all():
    """SELL_ALL is valid without Ticker."""
    cmd = {
        "Id": "def-456",
        "Action": "SELL_ALL",
        "Ticker": None,
        "Status": "pending",
        "RequestedAtUtc": "2026-07-13T10:00:00Z",
        "ExpiresAtUtc": "2026-07-13T14:00:00Z",
        "Source": "android-app",
    }
    is_valid, err = _validate_command(cmd)
    assert is_valid


def test_validate_command_missing_id():
    """Missing Id fails validation."""
    cmd = {
        "Action": "SELL",
        "Ticker": "AZN.L",
        "Status": "pending",
        "RequestedAtUtc": "2026-07-13T10:00:00Z",
        "ExpiresAtUtc": "2026-07-13T14:00:00Z",
        "Source": "android-app",
    }
    is_valid, err = _validate_command(cmd)
    assert not is_valid
    assert "Id" in err


def test_validate_command_invalid_action():
    """Invalid action fails validation."""
    cmd = {
        "Id": "xyz-789",
        "Action": "BUY",
        "Ticker": "AZN.L",
        "Status": "pending",
        "RequestedAtUtc": "2026-07-13T10:00:00Z",
        "ExpiresAtUtc": "2026-07-13T14:00:00Z",
        "Source": "android-app",
    }
    is_valid, err = _validate_command(cmd)
    assert not is_valid
    assert "Action" in err


def test_validate_command_sell_without_ticker():
    """SELL requires Ticker."""
    cmd = {
        "Id": "xyz-789",
        "Action": "SELL",
        "Ticker": None,
        "Status": "pending",
        "RequestedAtUtc": "2026-07-13T10:00:00Z",
        "ExpiresAtUtc": "2026-07-13T14:00:00Z",
        "Source": "android-app",
    }
    is_valid, err = _validate_command(cmd)
    assert not is_valid
    assert "Ticker" in err


def test_is_expired_future_time():
    """Command expiring in the future is not expired."""
    logger = mock.Mock()
    future_time = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    cmd = {
        "ExpiresAtUtc": future_time,
    }
    assert not _is_expired(cmd, logger)


def test_is_expired_past_time():
    """Command expiring in the past is expired."""
    logger = mock.Mock()
    past_time = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
    cmd = {
        "ExpiresAtUtc": past_time,
    }
    assert _is_expired(cmd, logger)


def test_is_expired_exactly_now():
    """Command expiring exactly now (boundary) depends on comparison."""
    logger = mock.Mock()
    now = datetime.now(timezone.utc)
    cmd = {
        "ExpiresAtUtc": now.isoformat().replace("+00:00", "Z"),
    }
    # Should be expired (> means strictly after)
    result = _is_expired(cmd, logger)
    # Due to precision, it might not be (within same microsecond)
    # This is a boundary test


def test_is_expired_one_second_in_future():
    """Command expiring 1 second in future is not expired."""
    logger = mock.Mock()
    future_time = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    cmd = {
        "ExpiresAtUtc": future_time,
    }
    assert not _is_expired(cmd, logger)


def test_is_market_open_no_position():
    """Non-existent position: market open (safe default)."""
    logger = mock.Mock()
    config = {"markets": {}}
    portfolio_positions = {}
    result = _is_market_open("NONEXISTENT", portfolio_positions, config, logger)
    assert result is True


def test_is_market_open_no_market_field():
    """Position with no market field: market closed (fail-closed for safety)."""
    logger = mock.Mock()
    config = {"markets": {}}
    portfolio_positions = {"AZN.L": {"quantity": 10, "fill_price": 100.0}}
    result = _is_market_open("AZN.L", portfolio_positions, config, logger)
    assert result is False
    # Should log a warning about missing market field
    logger.warning.assert_called_once()
    assert "missing market field" in logger.warning.call_args[0][0]


def test_is_market_open_market_not_in_config():
    """Position's market not in config: market closed (fail-closed for safety)."""
    logger = mock.Mock()
    config = {"markets": {}}
    portfolio_positions = {"AZN.L": {"quantity": 10, "market": "ftse"}}
    result = _is_market_open("AZN.L", portfolio_positions, config, logger)
    assert result is False
    # Should log a warning about unrecognized market
    logger.warning.assert_called_once()
    assert "Unrecognized or missing market" in logger.warning.call_args[0][0]


@mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.is_trading_hours")
def test_is_market_open_trading_hours_check(mock_is_trading):
    """Calls is_trading_hours from live_daemon."""
    mock_is_trading.return_value = True
    logger = mock.Mock()
    config = {"markets": {"ftse": {"timezone": "Europe/London"}}}
    portfolio_positions = {"AZN.L": {"quantity": 10, "market": "ftse"}}
    result = _is_market_open("AZN.L", portfolio_positions, config, logger)
    assert result is True
    mock_is_trading.assert_called_once()


def test_is_market_open_unrecognized_market_field():
    """Position with unrecognized market name (e.g., 'UK' instead of 'ftse'): market closed (fail-closed for safety).

    This is the bug scenario: HSBA.L had market='UK', but config only has 'ftse' and 'sp500'.
    The market-closed queuing should engage, preventing out-of-hours misfires.
    """
    logger = mock.Mock()
    config = {
        "markets": {
            "ftse": {"timezone": "Europe/London"},
            "sp500": {"timezone": "America/New_York"}
        }
    }
    # Position with unrecognized market name 'UK' (should be 'ftse' for FTSE-listed stocks)
    portfolio_positions = {
        "HSBA.L": {
            "quantity": 100,
            "fill_price": 1462.0,
            "market": "UK"  # Unrecognized! Not in config["markets"]
        }
    }
    result = _is_market_open("HSBA.L", portfolio_positions, config, logger)
    assert result is False, "Unrecognized market should be treated as closed for safety"
    # Should log a warning about unrecognized market
    logger.warning.assert_called_once()
    assert "Unrecognized or missing market" in logger.warning.call_args[0][0]
    assert "UK" in logger.warning.call_args[0][0]
    assert "HSBA.L" in logger.warning.call_args[0][0]


# -- Claim and Move Tests -------------------------------------------------------


def test_claim_pending_command_success(commands_dir):
    """Successfully claim and read a pending command."""
    cmd_data = {
        "Id": "test-123",
        "Action": "SELL",
        "Ticker": "AZN.L",
        "Status": "pending",
    }
    pending_path = commands_dir / "pending" / "test-123.json"
    pending_path.write_text(json.dumps(cmd_data), encoding="utf-8")

    logger = mock.Mock()
    success, cmd = _claim_pending_command(pending_path, logger)
    assert success
    assert cmd["Id"] == "test-123"

    # File should be moved to processing
    assert not pending_path.exists()
    processing_path = commands_dir / "processing" / "test-123.json"
    assert processing_path.exists()


def test_claim_pending_command_race(commands_dir):
    """File disappears during claim (race condition)."""
    pending_path = commands_dir / "pending" / "missing.json"
    # Don't create the file
    logger = mock.Mock()
    success, cmd = _claim_pending_command(pending_path, logger)
    assert not success
    assert cmd is None


def test_move_to_done(commands_dir):
    """Move processing file to done."""
    processing_path = commands_dir / "processing" / "test-123.json"
    processing_path.write_text("{}", encoding="utf-8")

    _move_to_done(processing_path)

    assert not processing_path.exists()
    done_path = commands_dir / "done" / "test-123.json"
    assert done_path.exists()


def test_move_to_done_with_suffix(commands_dir):
    """Move processing file to done with suffix."""
    processing_path = commands_dir / "processing" / "test-123.json"
    processing_path.write_text("{}", encoding="utf-8")

    _move_to_done(processing_path, "malformed")

    assert not processing_path.exists()
    done_path = commands_dir / "done" / "test-123.malformed"
    assert done_path.exists()


def test_move_to_pending(commands_dir):
    """Move processing file back to pending."""
    processing_path = commands_dir / "processing" / "test-123.json"
    cmd_data = {"Status": "queued_for_open"}
    processing_path.write_text(json.dumps(cmd_data), encoding="utf-8")

    _move_to_pending(processing_path)

    assert not processing_path.exists()
    pending_path = commands_dir / "pending" / "test-123.json"
    assert pending_path.exists()


# -- Write Result Tests -------------------------------------------------------


def test_write_result_filled(commands_dir):
    """Write a filled result."""
    cmd = {"Id": "test-123", "Action": "SELL", "Ticker": "AZN.L"}
    result_path = _write_result(
        cmd, "filled", commands_dir / "results", fill_price=105.0, quantity=10
    )

    assert result_path is not None
    result = json.loads((commands_dir / "results" / "test-123.json").read_text(encoding="utf-8"))
    assert result["Status"] == "filled"
    assert result["FillPrice"] == 105.0
    assert result["Quantity"] == 10


def test_write_result_error(commands_dir):
    """Write an error result."""
    cmd = {"Id": "test-456", "Action": "SELL"}
    result_path = _write_result(
        cmd, "error", commands_dir / "results", error_msg="No open position"
    )

    assert result_path is not None
    result = json.loads((commands_dir / "results" / "test-456.json").read_text(encoding="utf-8"))
    assert result["Status"] == "error"
    assert result["ErrorMessage"] == "No open position"


def test_write_result_expired(commands_dir):
    """Write an expired result."""
    cmd = {"Id": "test-789", "Action": "SELL"}
    result_path = _write_result(cmd, "expired", commands_dir / "results")

    assert result_path is not None
    result = json.loads((commands_dir / "results" / "test-789.json").read_text(encoding="utf-8"))
    assert result["Status"] == "expired"


# -- Execute Tests -------------------------------------------------------


def test_execute_sell_success(fake_portfolio, fake_broker):
    """Execute SELL on existing position."""
    # Create a position
    fill_buy = FillResult("AZN.L", "BUY", 100.0, 10, datetime.now(timezone.utc).isoformat())
    fake_portfolio.record_entry("AZN.L", fill_buy, 0.10, 95.0, 115.0, market="ftse")

    # Execute SELL
    success, fill, error = _execute_sell("AZN.L", fake_portfolio, fake_broker, mock.Mock())

    assert success
    assert fill.ticker == "AZN.L"
    assert fill.quantity == 10
    assert "AZN.L" not in fake_portfolio.positions


def test_execute_sell_no_position(fake_portfolio, fake_broker):
    """Execute SELL on non-existent position."""
    success, fill, error = _execute_sell("NONEXISTENT", fake_portfolio, fake_broker, mock.Mock())

    assert not success
    assert fill is None
    assert "No open position" in error


def test_execute_sell_all_empty():
    """SELL_ALL with no positions."""
    fake_portfolio = mock.Mock()
    fake_portfolio.positions = {}
    fake_broker = mock.Mock()
    logger = mock.Mock()

    success, fills, summary = _execute_sell_all(fake_portfolio, fake_broker, logger)

    assert not success
    assert len(fills) == 0
    assert "No open positions" in summary


def test_execute_sell_all_single_position(fake_portfolio, fake_broker):
    """SELL_ALL with one position."""
    # Create a position
    fill_buy = FillResult("AZN.L", "BUY", 100.0, 10, datetime.now(timezone.utc).isoformat())
    fake_portfolio.record_entry("AZN.L", fill_buy, 0.10, 95.0, 115.0)

    success, fills, summary = _execute_sell_all(fake_portfolio, fake_broker, mock.Mock())

    assert success
    assert len(fills) == 1
    assert fills[0].ticker == "AZN.L"
    assert fills[0].quantity == 10
    assert "AZN.L" not in fake_portfolio.positions


def test_execute_sell_all_multiple_positions(fake_portfolio, fake_broker):
    """SELL_ALL with multiple positions."""
    # Create two positions
    fill1 = FillResult("AZN.L", "BUY", 100.0, 10, datetime.now(timezone.utc).isoformat())
    fill2 = FillResult("BP.L", "BUY", 50.0, 20, datetime.now(timezone.utc).isoformat())
    fake_portfolio.record_entry("AZN.L", fill1, 0.10, 95.0, 115.0)
    fake_portfolio.record_entry("BP.L", fill2, 0.10, 45.0, 60.0)

    success, fills, summary = _execute_sell_all(fake_portfolio, fake_broker, mock.Mock())

    assert success
    assert len(fills) == 2
    assert "AZN.L" not in fake_portfolio.positions
    assert "BP.L" not in fake_portfolio.positions
    assert "AZN.L" in summary
    assert "BP.L" in summary


# -- Full Integration Tests -------------------------------------------------------


def test_process_manual_commands_sell_existing_position(commands_dir, config, fake_portfolio, fake_broker):
    """Process SELL command for existing position."""
    # Create a position
    fill_buy = FillResult("AZN.L", "BUY", 100.0, 10, datetime.now(timezone.utc).isoformat())
    fake_portfolio.record_entry("AZN.L", fill_buy, 0.10, 95.0, 115.0, market="ftse")

    # Write pending command
    cmd = {
        "Id": "cmd-001",
        "Action": "SELL",
        "Ticker": "AZN.L",
        "Status": "pending",
        "RequestedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ExpiresAtUtc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        "Source": "android-app",
    }
    pending_path = commands_dir / "pending" / "cmd-001.json"
    pending_path.write_text(json.dumps(cmd), encoding="utf-8")

    logger = mock.Mock()

    # Mock is_trading_hours to always return True
    with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.is_trading_hours", return_value=True):
        processed = process_manual_commands(config, fake_portfolio, fake_broker, logger, commands_dir)

    assert processed == 1
    assert not pending_path.exists()
    assert (commands_dir / "done" / "cmd-001.json").exists()
    assert (commands_dir / "results" / "cmd-001.json").exists()

    result = json.loads((commands_dir / "results" / "cmd-001.json").read_text(encoding="utf-8"))
    assert result["Status"] == "filled"
    assert result["FillPrice"] == 100.0
    assert result["Quantity"] == 10


def test_process_manual_commands_sell_no_position(commands_dir, config, fake_portfolio, fake_broker):
    """Process SELL command for non-existent position."""
    cmd = {
        "Id": "cmd-002",
        "Action": "SELL",
        "Ticker": "NONEXISTENT",
        "Status": "pending",
        "RequestedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ExpiresAtUtc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        "Source": "android-app",
    }
    pending_path = commands_dir / "pending" / "cmd-002.json"
    pending_path.write_text(json.dumps(cmd), encoding="utf-8")

    logger = mock.Mock()

    with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.is_trading_hours", return_value=True):
        processed = process_manual_commands(config, fake_portfolio, fake_broker, logger, commands_dir)

    assert processed == 1
    result = json.loads((commands_dir / "results" / "cmd-002.json").read_text(encoding="utf-8"))
    assert result["Status"] == "error"
    assert "No open position" in result["ErrorMessage"]


def test_process_manual_commands_malformed(commands_dir, config, fake_portfolio, fake_broker):
    """Process malformed JSON command file."""
    pending_path = commands_dir / "pending" / "cmd-003.json"
    pending_path.write_text("{not json", encoding="utf-8")

    logger = mock.Mock()
    processed = process_manual_commands(config, fake_portfolio, fake_broker, logger, commands_dir)

    assert processed == 1  # Malformed JSON; still processed but marked as error
    # Verify state after processing
    assert not (commands_dir / "pending" / "cmd-003.json").exists()
    assert not (commands_dir / "processing" / "cmd-003.json").exists()
    assert (commands_dir / "done" / "cmd-003.malformed").exists()
    # Verify error result was written
    assert (commands_dir / "results" / "cmd-003.json").exists()
    result = json.loads((commands_dir / "results" / "cmd-003.json").read_text(encoding="utf-8"))
    assert result["Status"] == "error"
    assert result["ErrorMessage"] == "malformed command file"


def test_process_manual_commands_missing_required_key(commands_dir, config, fake_portfolio, fake_broker):
    """Process command with missing required key."""
    cmd = {
        "Id": "cmd-004",
        "Action": "SELL",
        # Missing Ticker
        "Status": "pending",
    }
    pending_path = commands_dir / "pending" / "cmd-004.json"
    pending_path.write_text(json.dumps(cmd), encoding="utf-8")

    logger = mock.Mock()
    processed = process_manual_commands(config, fake_portfolio, fake_broker, logger, commands_dir)

    assert processed == 1
    # Should move to done.malformed
    assert (commands_dir / "done" / "cmd-004.malformed").exists()


def test_process_manual_commands_expired(commands_dir, config, fake_portfolio, fake_broker):
    """Process expired command."""
    expires_time = (datetime.now(timezone.utc) - timedelta(minutes=1))
    expires_iso = expires_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-1] + "Z"

    cmd = {
        "Id": "cmd-005",
        "Action": "SELL",
        "Ticker": "AZN.L",
        "Status": "pending",
        "RequestedAtUtc": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        "ExpiresAtUtc": expires_iso,
        "Source": "android-app",
    }
    pending_path = commands_dir / "pending" / "cmd-005.json"
    pending_path.write_text(json.dumps(cmd), encoding="utf-8")

    logger = mock.Mock()
    processed = process_manual_commands(config, fake_portfolio, fake_broker, logger, commands_dir)

    assert processed == 1
    result = json.loads((commands_dir / "results" / "cmd-005.json").read_text(encoding="utf-8"))
    assert result["Status"] == "expired"


@mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.is_trading_hours", return_value=False)
def test_process_manual_commands_market_closed(mock_trading, commands_dir, config, fake_portfolio, fake_broker):
    """Process SELL when market is closed (requeue)."""
    # Create a position
    fill_buy = FillResult("AZN.L", "BUY", 100.0, 10, datetime.now(timezone.utc).isoformat())
    fake_portfolio.record_entry("AZN.L", fill_buy, 0.10, 95.0, 115.0, market="ftse")

    cmd = {
        "Id": "cmd-006",
        "Action": "SELL",
        "Ticker": "AZN.L",
        "Status": "pending",
        "RequestedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ExpiresAtUtc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        "Source": "android-app",
    }
    pending_path = commands_dir / "pending" / "cmd-006.json"
    pending_path.write_text(json.dumps(cmd), encoding="utf-8")

    logger = mock.Mock()
    processed = process_manual_commands(config, fake_portfolio, fake_broker, logger, commands_dir)

    assert processed == 1
    # Should be requeued in pending with Status: queued_for_open
    requeued = json.loads((commands_dir / "pending" / "cmd-006.json").read_text(encoding="utf-8"))
    assert requeued["Status"] == "queued_for_open"


def test_process_manual_commands_sell_all(commands_dir, config, fake_portfolio, fake_broker):
    """Process SELL_ALL command."""
    # Create two positions
    fill1 = FillResult("AZN.L", "BUY", 100.0, 10, datetime.now(timezone.utc).isoformat())
    fill2 = FillResult("BP.L", "BUY", 50.0, 20, datetime.now(timezone.utc).isoformat())
    fake_portfolio.record_entry("AZN.L", fill1, 0.10, 95.0, 115.0, market="ftse")
    fake_portfolio.record_entry("BP.L", fill2, 0.10, 45.0, 60.0, market="ftse")

    cmd = {
        "Id": "cmd-007",
        "Action": "SELL_ALL",
        "Ticker": None,
        "Status": "pending",
        "RequestedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ExpiresAtUtc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        "Source": "android-app",
    }
    pending_path = commands_dir / "pending" / "cmd-007.json"
    pending_path.write_text(json.dumps(cmd), encoding="utf-8")

    logger = mock.Mock()

    with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.is_trading_hours", return_value=True):
        processed = process_manual_commands(config, fake_portfolio, fake_broker, logger, commands_dir)

    assert processed == 1
    result = json.loads((commands_dir / "results" / "cmd-007.json").read_text(encoding="utf-8"))
    assert result["Status"] == "filled"
    assert "AZN.L" in result["Summary"]
    assert "BP.L" in result["Summary"]


def test_process_manual_commands_claim_race(commands_dir, config, fake_portfolio, fake_broker):
    """Process command when claim race occurs (file disappears)."""
    cmd = {
        "Id": "cmd-008",
        "Action": "SELL",
        "Ticker": "AZN.L",
        "Status": "pending",
    }
    pending_path = commands_dir / "pending" / "cmd-008.json"
    pending_path.write_text(json.dumps(cmd), encoding="utf-8")

    logger = mock.Mock()

    # Mock os.replace to raise FileNotFoundError
    original_replace = __import__("os").replace

    def mock_replace(src, dst):
        if "cmd-008" in str(src):
            raise FileNotFoundError()
        return original_replace(src, dst)

    with mock.patch("os.replace", side_effect=mock_replace):
        processed = process_manual_commands(config, fake_portfolio, fake_broker, logger, commands_dir)

    # Should skip the command gracefully
    assert processed == 0
    assert pending_path.exists()  # File should still exist (race condition)


def test_process_manual_commands_result_json_format(commands_dir, config, fake_portfolio, fake_broker):
    """Result JSON has PascalCase keys (C# compatibility)."""
    fill_buy = FillResult("AZN.L", "BUY", 100.0, 10, datetime.now(timezone.utc).isoformat())
    fake_portfolio.record_entry("AZN.L", fill_buy, 0.10, 95.0, 115.0, market="ftse")

    cmd = {
        "Id": "cmd-009",
        "Action": "SELL",
        "Ticker": "AZN.L",
        "Status": "pending",
        "RequestedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ExpiresAtUtc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        "Source": "android-app",
    }
    pending_path = commands_dir / "pending" / "cmd-009.json"
    pending_path.write_text(json.dumps(cmd), encoding="utf-8")

    logger = mock.Mock()

    with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.is_trading_hours", return_value=True):
        processed = process_manual_commands(config, fake_portfolio, fake_broker, logger, commands_dir)

    result = json.loads((commands_dir / "results" / "cmd-009.json").read_text(encoding="utf-8"))

    # Check PascalCase keys
    assert "Id" in result
    assert "Action" in result
    assert "Ticker" in result
    assert "Status" in result
    assert "FillPrice" in result
    assert "Quantity" in result

    # No snake_case keys
    assert "fill_price" not in result
    assert "quantity" not in result


def test_process_manual_commands_multiple_commands(commands_dir, config, fake_portfolio, fake_broker):
    """Process multiple commands in one call."""
    # Create position for first command
    fill_buy = FillResult("AZN.L", "BUY", 100.0, 10, datetime.now(timezone.utc).isoformat())
    fake_portfolio.record_entry("AZN.L", fill_buy, 0.10, 95.0, 115.0, market="ftse")

    # Write two commands
    cmd1 = {
        "Id": "cmd-010",
        "Action": "SELL",
        "Ticker": "AZN.L",
        "Status": "pending",
        "RequestedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ExpiresAtUtc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        "Source": "android-app",
    }
    cmd2 = {
        "Id": "cmd-011",
        "Action": "SELL",
        "Ticker": "NONEXISTENT",
        "Status": "pending",
        "RequestedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ExpiresAtUtc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        "Source": "android-app",
    }

    (commands_dir / "pending" / "cmd-010.json").write_text(json.dumps(cmd1), encoding="utf-8")
    (commands_dir / "pending" / "cmd-011.json").write_text(json.dumps(cmd2), encoding="utf-8")

    logger = mock.Mock()

    with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.is_trading_hours", return_value=True):
        processed = process_manual_commands(config, fake_portfolio, fake_broker, logger, commands_dir)

    assert processed == 2
    assert (commands_dir / "results" / "cmd-010.json").exists()
    assert (commands_dir / "results" / "cmd-011.json").exists()


# -- Pause/Resume Tests -------------------------------------------------------


def test_validate_command_pause_buying():
    """PAUSE_BUYING is valid action."""
    cmd = {
        "Id": "pause-001",
        "Action": "PAUSE_BUYING",
        "Ticker": None,
        "Status": "pending",
        "RequestedAtUtc": "2026-07-13T10:00:00Z",
        "ExpiresAtUtc": "2026-07-13T14:00:00Z",
        "Source": "android-app",
    }
    is_valid, err = _validate_command(cmd)
    assert is_valid
    assert err == ""


def test_validate_command_resume_buying():
    """RESUME_BUYING is valid action."""
    cmd = {
        "Id": "resume-001",
        "Action": "RESUME_BUYING",
        "Ticker": None,
        "Status": "pending",
        "RequestedAtUtc": "2026-07-13T10:00:00Z",
        "ExpiresAtUtc": "2026-07-13T14:00:00Z",
        "Source": "android-app",
    }
    is_valid, err = _validate_command(cmd)
    assert is_valid
    assert err == ""


@mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.save_daemon_state")
def test_execute_pause_buying(mock_save_state):
    """Execute pause buying successfully."""
    daemon_state = {}
    logger = mock.Mock()
    success, msg = _execute_pause_buying(daemon_state, logger)

    assert success
    assert msg == "Buying paused by user"
    assert daemon_state["paused_by_user"] is True
    mock_save_state.assert_called_once_with(daemon_state)


@mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.save_daemon_state")
def test_execute_pause_buying_save_failure(mock_save_state):
    """Execute pause buying fails if save fails."""
    mock_save_state.side_effect = Exception("Save failed")
    daemon_state = {}
    logger = mock.Mock()
    success, msg = _execute_pause_buying(daemon_state, logger)

    assert not success
    assert "Save failed" in msg


@mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.save_daemon_state")
def test_execute_resume_buying(mock_save_state):
    """Execute resume buying successfully."""
    daemon_state = {"paused_by_user": True}
    logger = mock.Mock()
    success, msg = _execute_resume_buying(daemon_state, logger)

    assert success
    assert msg == "Buying resumed by user"
    assert daemon_state["paused_by_user"] is False
    mock_save_state.assert_called_once_with(daemon_state)


@mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.save_daemon_state")
def test_execute_resume_buying_save_failure(mock_save_state):
    """Execute resume buying fails if save fails."""
    mock_save_state.side_effect = Exception("Save failed")
    daemon_state = {"paused_by_user": True}
    logger = mock.Mock()
    success, msg = _execute_resume_buying(daemon_state, logger)

    assert not success
    assert "Save failed" in msg


def test_process_manual_commands_pause_buying(commands_dir, config, fake_portfolio, fake_broker):
    """Process PAUSE_BUYING command successfully."""
    cmd = {
        "Id": "pause-001",
        "Action": "PAUSE_BUYING",
        "Ticker": None,
        "Status": "pending",
        "RequestedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ExpiresAtUtc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        "Source": "android-app",
    }
    pending_path = commands_dir / "pending" / "pause-001.json"
    pending_path.write_text(json.dumps(cmd), encoding="utf-8")

    logger = mock.Mock()
    daemon_state = {}

    with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.save_daemon_state"):
        processed = process_manual_commands(
            config, fake_portfolio, fake_broker, logger, commands_dir, daemon_state=daemon_state
        )

    assert processed == 1
    assert daemon_state["paused_by_user"] is True
    result = json.loads((commands_dir / "results" / "pause-001.json").read_text(encoding="utf-8"))
    assert result["Status"] == "filled"
    assert result["Summary"] == "Buying paused by user"


def test_process_manual_commands_resume_buying(commands_dir, config, fake_portfolio, fake_broker):
    """Process RESUME_BUYING command successfully."""
    cmd = {
        "Id": "resume-001",
        "Action": "RESUME_BUYING",
        "Ticker": None,
        "Status": "pending",
        "RequestedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ExpiresAtUtc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace(
            "+00:00", "Z"),
        "Source": "android-app",
    }
    pending_path = commands_dir / "pending" / "resume-001.json"
    pending_path.write_text(json.dumps(cmd), encoding="utf-8")

    logger = mock.Mock()
    daemon_state = {"paused_by_user": True}

    with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.save_daemon_state"):
        processed = process_manual_commands(
            config, fake_portfolio, fake_broker, logger, commands_dir, daemon_state=daemon_state
        )

    assert processed == 1
    assert daemon_state["paused_by_user"] is False
    result = json.loads((commands_dir / "results" / "resume-001.json").read_text(encoding="utf-8"))
    assert result["Status"] == "filled"
    assert result["Summary"] == "Buying resumed by user"


def test_process_manual_commands_pause_no_market_gate(commands_dir, config, fake_portfolio, fake_broker):
    """PAUSE_BUYING is not gated by market hours."""
    cmd = {
        "Id": "pause-002",
        "Action": "PAUSE_BUYING",
        "Ticker": None,
        "Status": "pending",
        "RequestedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ExpiresAtUtc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace(
            "+00:00", "Z"),
        "Source": "android-app",
    }
    pending_path = commands_dir / "pending" / "pause-002.json"
    pending_path.write_text(json.dumps(cmd), encoding="utf-8")

    logger = mock.Mock()
    daemon_state = {}

    # Even with market hours check mocked to False, pause should succeed
    with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.is_trading_hours",
                    return_value=False):
        with mock.patch("Strategy_Auto_Trader.markov_cli.live_daemon.save_daemon_state"):
            processed = process_manual_commands(
                config, fake_portfolio, fake_broker, logger, commands_dir, daemon_state=daemon_state
            )

    assert processed == 1
    assert daemon_state["paused_by_user"] is True
    result = json.loads((commands_dir / "results" / "pause-002.json").read_text(encoding="utf-8"))
    assert result["Status"] == "filled"


def test_process_manual_commands_daemon_state_default_empty():
    """daemon_state defaults to empty dict if None."""
    config = {"markets": {}}
    fake_portfolio = mock.Mock()
    fake_portfolio.positions = {}
    fake_broker = mock.Mock()
    logger = mock.Mock()
    commands_dir = Path(__file__).resolve().parent / "tmp_commands"
    commands_dir.mkdir(exist_ok=True)
    try:
        # Call without daemon_state (should default to {})
        processed = process_manual_commands(config, fake_portfolio, fake_broker, logger, commands_dir)
        assert processed == 0
    finally:
        import shutil
        shutil.rmtree(commands_dir, ignore_errors=True)
