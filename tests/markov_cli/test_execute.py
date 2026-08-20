from __future__ import annotations

from unittest import mock

import pytest


class TestExecutionInterrupted:
    """ExecutionInterrupted exception for partial execution tracking."""

    def test_execution_interrupted_on_buy_exception(self, monkeypatch):
        """Exception during buy loop raises ExecutionInterrupted with partial progress."""
        from Strategy_Auto_Trader.markov_cli.execute import (
            execute_signals,
            ExecutionInterrupted,
        )
        from Strategy_Auto_Trader.broker.types import OrderRequest, FillResult

        portfolio = mock.Mock()
        limit_tracker = mock.Mock()
        broker = mock.Mock()
        data_dir = None

        # MSFT: skipped (no capacity), AAPL: succeeds, GOOG: error during place_order
        portfolio.can_open.side_effect = [False, True, True]
        portfolio.compute_quantity.return_value = 10

        # First place_order succeeds for AAPL, second fails for GOOG
        fill = FillResult("AAPL", "BUY", 150.0, 10, "2026-07-01T00:00:00Z")
        broker.place_order.side_effect = [fill, RuntimeError("Connection lost")]

        signal_reader = mock.Mock()
        signal_reader.return_value = {"flag": "BUY", "close": 150.0, "kelly_fraction": 0.1, "score": 1.0, "stop_level": 140.0, "target_level": 160.0}

        monkeypatch.setattr(
            "Strategy_Auto_Trader.broker.signal_reader.read_latest_signal",
            signal_reader
        )

        with pytest.raises(ExecutionInterrupted) as exc_info:
            execute_signals(
                ["MSFT", "AAPL", "GOOG"],
                data_dir,
                portfolio,
                limit_tracker,
                broker,
            )

        exc = exc_info.value
        assert len(exc.buys) == 1
        assert "AAPL x10 @" in exc.buys[0]  # AAPL was successfully bought
        assert exc.sells == []
        assert len(exc.skipped) == 1  # MSFT was skipped
        assert "MSFT(at capacity)" in exc.skipped
        assert "GOOG" in exc.unresolved  # GOOG's order was interrupted
        assert isinstance(exc.original, RuntimeError)

    def test_execution_interrupted_on_sell_exception(self, monkeypatch):
        """Exception during sell loop raises ExecutionInterrupted with buy results."""
        from Strategy_Auto_Trader.markov_cli.execute import (
            execute_signals,
            ExecutionInterrupted,
        )
        from Strategy_Auto_Trader.broker.types import OrderRequest, FillResult

        portfolio = mock.Mock()
        limit_tracker = mock.Mock()
        broker = mock.Mock()
        data_dir = None

        portfolio.can_open.return_value = True
        portfolio.can_sell.side_effect = [RuntimeError("Socket lost")]
        portfolio.compute_quantity.return_value = 10
        portfolio.positions = {"AAPL": {"quantity": 10}}

        # Sell signal for AAPL
        signal_reader = mock.Mock()
        signal_reader.side_effect = [
            {"flag": "SELL", "close": 150.0, "kelly_fraction": 0.1, "stop_level": 140.0, "target_level": 160.0},
        ]

        monkeypatch.setattr(
            "Strategy_Auto_Trader.broker.signal_reader.read_latest_signal",
            signal_reader
        )

        with pytest.raises(ExecutionInterrupted) as exc_info:
            execute_signals(
                ["AAPL"],
                data_dir,
                portfolio,
                limit_tracker,
                broker,
            )

        exc = exc_info.value
        assert exc.buys == []
        assert exc.sells == []  # No sells completed before exception
        assert "AAPL" in exc.unresolved  # AAPL's sell was interrupted

    def test_execution_interrupted_preserves_original_exception(self, monkeypatch):
        """ExecutionInterrupted.original is the underlying exception."""
        from Strategy_Auto_Trader.markov_cli.execute import (
            execute_signals,
            ExecutionInterrupted,
        )
        from Strategy_Auto_Trader.broker.types import FillResult

        portfolio = mock.Mock()
        limit_tracker = mock.Mock()
        broker = mock.Mock()
        data_dir = None

        portfolio.can_open.return_value = True
        portfolio.compute_quantity.return_value = 10

        fill = FillResult("AAPL", "BUY", 150.0, 10, "2026-07-01T00:00:00Z")
        original_error = TimeoutError("Socket timeout on place_order")
        broker.place_order.side_effect = original_error

        signal_reader = mock.Mock()
        signal_reader.return_value = {"flag": "BUY", "close": 150.0, "kelly_fraction": 0.1, "score": 1.0, "stop_level": 140.0, "target_level": 160.0}

        monkeypatch.setattr(
            "Strategy_Auto_Trader.broker.signal_reader.read_latest_signal",
            signal_reader
        )

        with pytest.raises(ExecutionInterrupted) as exc_info:
            execute_signals(
                ["AAPL"],
                data_dir,
                portfolio,
                limit_tracker,
                broker,
            )

        assert isinstance(exc_info.value.original, TimeoutError)
        assert str(exc_info.value.original) == "Socket timeout on place_order"

    def test_execution_interrupted_includes_skipped_entries(self, monkeypatch):
        """ExecutionInterrupted includes skipped entries when it raises."""
        from Strategy_Auto_Trader.markov_cli.execute import (
            execute_signals,
            ExecutionInterrupted,
        )
        from Strategy_Auto_Trader.broker.types import FillResult

        portfolio = mock.Mock()
        limit_tracker = mock.Mock()
        broker = mock.Mock()
        data_dir = None

        # Setup: AAPL skipped (no capacity), MSFT succeeds, GOOG error
        portfolio.can_open.side_effect = [False, True, True]
        portfolio.compute_quantity.return_value = 10

        fill = FillResult("MSFT", "BUY", 350.0, 10, "2026-07-01T00:00:00Z")
        broker.place_order.side_effect = [fill, RuntimeError("Connection lost")]

        signal_reader = mock.Mock()
        signal_reader.return_value = {"flag": "BUY", "close": 350.0, "kelly_fraction": 0.1, "score": 1.0, "stop_level": 340.0, "target_level": 360.0}

        monkeypatch.setattr(
            "Strategy_Auto_Trader.broker.signal_reader.read_latest_signal",
            signal_reader
        )

        with pytest.raises(ExecutionInterrupted) as exc_info:
            execute_signals(
                ["AAPL", "MSFT", "GOOG"],
                data_dir,
                portfolio,
                limit_tracker,
                broker,
            )

        exc = exc_info.value
        assert "AAPL(at capacity)" in exc.skipped
        assert len(exc.buys) == 1  # MSFT was bought
        assert "GOOG" in exc.unresolved  # GOOG's order was interrupted

    def test_execution_interrupted_hold_ticker_not_marked_unresolved(self, monkeypatch):
        """A HOLD ticker resolved before the batch loops must not reappear in unresolved."""
        from Strategy_Auto_Trader.markov_cli.execute import (
            execute_signals,
            ExecutionInterrupted,
        )
        from Strategy_Auto_Trader.broker.types import FillResult

        portfolio = mock.Mock()
        limit_tracker = mock.Mock()
        broker = mock.Mock()
        data_dir = None

        portfolio.can_open.return_value = True
        portfolio.compute_quantity.return_value = 10
        broker.place_order.side_effect = RuntimeError("Connection lost")

        signals = {
            "MSFT": {"flag": "HOLD", "close": 100.0},
            "AAPL": {"flag": "BUY", "close": 150.0, "kelly_fraction": 0.1, "score": 1.0,
                     "stop_level": 140.0, "target_level": 160.0},
        }
        monkeypatch.setattr(
            "Strategy_Auto_Trader.broker.signal_reader.read_latest_signal",
            lambda ticker, _dir: signals[ticker],
        )

        with pytest.raises(ExecutionInterrupted) as exc_info:
            execute_signals(
                ["MSFT", "AAPL"],
                data_dir,
                portfolio,
                limit_tracker,
                broker,
            )

        exc = exc_info.value
        assert exc.skipped == ["MSFT"]
        assert "MSFT" not in exc.unresolved
        assert exc.unresolved == ["AAPL"]


class TestExecuteSignalsNormal:
    """Verify normal execution paths still work after ExecutionInterrupted changes."""

    def test_execute_signals_success_path_unchanged(self, monkeypatch):
        """Normal success (no exception) still returns (buys, sells, skipped) tuple."""
        from Strategy_Auto_Trader.markov_cli.execute import execute_signals
        from Strategy_Auto_Trader.broker.types import FillResult

        portfolio = mock.Mock()
        limit_tracker = mock.Mock()
        broker = mock.Mock()
        data_dir = None

        portfolio.can_open.return_value = True
        portfolio.can_buy.return_value = True
        portfolio.compute_quantity.return_value = 10

        fill = FillResult("AAPL", "BUY", 150.0, 10, "2026-07-01T00:00:00Z")
        broker.place_order.return_value = fill

        signal_reader = mock.Mock()
        signal_reader.return_value = {"flag": "BUY", "close": 150.0, "kelly_fraction": 0.1, "score": 1.0, "stop_level": 140.0, "target_level": 160.0}

        monkeypatch.setattr(
            "Strategy_Auto_Trader.broker.signal_reader.read_latest_signal",
            signal_reader
        )

        buys, sells, skipped = execute_signals(
            ["AAPL"],
            data_dir,
            portfolio,
            limit_tracker,
            broker,
        )

        assert len(buys) == 1
        assert "AAPL" in buys[0]
        assert sells == []
        assert skipped == []

    def test_execute_signals_hold_signal_skipped(self, monkeypatch):
        """HOLD signals are still skipped without exception."""
        from Strategy_Auto_Trader.markov_cli.execute import execute_signals

        portfolio = mock.Mock()
        limit_tracker = mock.Mock()
        broker = mock.Mock()
        data_dir = None

        signal_reader = mock.Mock()
        signal_reader.return_value = {"flag": "HOLD", "close": 150.0}

        monkeypatch.setattr(
            "Strategy_Auto_Trader.broker.signal_reader.read_latest_signal",
            signal_reader
        )

        buys, sells, skipped = execute_signals(
            ["AAPL"],
            data_dir,
            portfolio,
            limit_tracker,
            broker,
        )

        assert buys == []
        assert sells == []
        assert skipped == ["AAPL"]


class TestInFlightMarkerIntegration:
    """Test marker write/clear around place_order() calls."""

    def test_marker_written_and_cleared_on_successful_buy(self, monkeypatch, tmp_path):
        """Marker is written before place_order and cleared after for successful BUY."""
        from Strategy_Auto_Trader.markov_cli.execute import execute_signals
        from Strategy_Auto_Trader.broker.types import FillResult
        from Strategy_Auto_Trader.broker.in_flight_marker import read_marker

        portfolio = mock.Mock()
        limit_tracker = mock.Mock()
        broker = mock.Mock()
        data_dir = None

        portfolio.can_open.return_value = True
        portfolio.compute_quantity.return_value = 10

        fill = FillResult("AAPL", "BUY", 150.0, 10, "2026-07-01T00:00:00Z")
        broker.place_order.return_value = fill

        signal_reader = mock.Mock()
        signal_reader.return_value = {"flag": "BUY", "close": 150.0, "kelly_fraction": 0.1, "score": 1.0, "stop_level": 140.0, "target_level": 160.0}

        monkeypatch.setattr(
            "Strategy_Auto_Trader.broker.signal_reader.read_latest_signal",
            signal_reader
        )

        marker_path = tmp_path / "marker.json"
        execute_signals(
            ["AAPL"],
            data_dir,
            portfolio,
            limit_tracker,
            broker,
            marker_path=marker_path,
        )

        assert not marker_path.exists()

    def test_marker_written_and_cleared_on_successful_sell(self, monkeypatch, tmp_path):
        """Marker is written before place_order and cleared after for successful SELL."""
        from Strategy_Auto_Trader.markov_cli.execute import execute_signals
        from Strategy_Auto_Trader.broker.types import FillResult

        portfolio = mock.Mock()
        limit_tracker = mock.Mock()
        broker = mock.Mock()
        data_dir = None

        portfolio.can_sell.return_value = True
        portfolio.positions = {"AAPL": {"quantity": 10}}

        fill = FillResult("AAPL", "SELL", 150.0, 10, "2026-07-01T00:00:00Z")
        broker.place_order.return_value = fill

        signal_reader = mock.Mock()
        signal_reader.return_value = {"flag": "SELL", "close": 150.0, "kelly_fraction": 0.1}

        monkeypatch.setattr(
            "Strategy_Auto_Trader.broker.signal_reader.read_latest_signal",
            signal_reader
        )

        marker_path = tmp_path / "marker.json"
        execute_signals(
            ["AAPL"],
            data_dir,
            portfolio,
            limit_tracker,
            broker,
            marker_path=marker_path,
        )

        assert not marker_path.exists()

    def test_marker_cleared_when_order_not_filled(self, monkeypatch, tmp_path):
        """Marker is cleared when place_order returns None (not filled)."""
        from Strategy_Auto_Trader.markov_cli.execute import execute_signals

        portfolio = mock.Mock()
        limit_tracker = mock.Mock()
        broker = mock.Mock()
        data_dir = None

        portfolio.can_open.return_value = True
        portfolio.compute_quantity.return_value = 10
        broker.place_order.return_value = None  # Not filled

        signal_reader = mock.Mock()
        signal_reader.return_value = {"flag": "BUY", "close": 150.0, "kelly_fraction": 0.1, "score": 1.0, "stop_level": 140.0, "target_level": 160.0}

        monkeypatch.setattr(
            "Strategy_Auto_Trader.broker.signal_reader.read_latest_signal",
            signal_reader
        )

        marker_path = tmp_path / "marker.json"
        execute_signals(
            ["AAPL"],
            data_dir,
            portfolio,
            limit_tracker,
            broker,
            marker_path=marker_path,
        )

        assert not marker_path.exists()

    def test_marker_left_in_place_on_exception(self, monkeypatch, tmp_path):
        """Marker is deliberately left in place when place_order() raises."""
        from Strategy_Auto_Trader.markov_cli.execute import execute_signals, ExecutionInterrupted
        from Strategy_Auto_Trader.broker.in_flight_marker import read_marker

        portfolio = mock.Mock()
        limit_tracker = mock.Mock()
        broker = mock.Mock()
        data_dir = None

        portfolio.can_open.return_value = True
        portfolio.compute_quantity.return_value = 10
        broker.place_order.side_effect = RuntimeError("Socket error during place_order")

        signal_reader = mock.Mock()
        signal_reader.return_value = {"flag": "BUY", "close": 150.0, "kelly_fraction": 0.1, "score": 1.0, "stop_level": 140.0, "target_level": 160.0}

        monkeypatch.setattr(
            "Strategy_Auto_Trader.broker.signal_reader.read_latest_signal",
            signal_reader
        )

        marker_path = tmp_path / "marker.json"

        with pytest.raises(ExecutionInterrupted):
            execute_signals(
                ["AAPL"],
                data_dir,
                portfolio,
                limit_tracker,
                broker,
                marker_path=marker_path,
            )

        marker = read_marker(marker_path)
        assert marker is not None
        assert marker["ticker"] == "AAPL"
        assert marker["action"] == "BUY"
        assert marker["quantity"] == 10


class TestPlaceOrderRetry:
    """Retry-on-socket-disconnect logic in _place_order_with_retry()."""

    def test_retries_on_connection_error_then_succeeds(self, monkeypatch):
        from Strategy_Auto_Trader.markov_cli.execute import _place_order_with_retry
        from Strategy_Auto_Trader.broker.types import FillResult

        broker = mock.Mock()
        broker.is_connected.return_value = False
        fill = FillResult("AAPL", "BUY", 150.0, 10, "2026-07-01T00:00:00Z")
        broker.place_order.side_effect = [
            ConnectionError("Socket disconnect during order placement: boom"),
            ConnectionError("Socket disconnect during order placement: boom"),
            fill,
        ]
        sleeps = []
        monkeypatch.setattr(
            "Strategy_Auto_Trader.markov_cli.execute.time.sleep",
            lambda s: sleeps.append(s),
        )

        result = _place_order_with_retry(broker, mock.Mock(), "AAPL")

        assert result is fill
        assert broker.place_order.call_count == 3
        assert sleeps == [30.0, 30.0]
        assert broker.connect.call_count == 2

    def test_gives_up_after_max_retries(self, monkeypatch):
        from Strategy_Auto_Trader.markov_cli.execute import _place_order_with_retry

        broker = mock.Mock()
        broker.is_connected.return_value = False
        broker.place_order.side_effect = ConnectionError("Socket disconnect: still down")
        monkeypatch.setattr(
            "Strategy_Auto_Trader.markov_cli.execute.time.sleep", lambda s: None
        )

        with pytest.raises(ConnectionError):
            _place_order_with_retry(broker, mock.Mock(), "AAPL")

        assert broker.place_order.call_count == 5

    def test_does_not_retry_non_connection_errors(self, monkeypatch):
        from Strategy_Auto_Trader.markov_cli.execute import _place_order_with_retry

        broker = mock.Mock()
        broker.place_order.side_effect = RuntimeError("order rejected")
        monkeypatch.setattr(
            "Strategy_Auto_Trader.markov_cli.execute.time.sleep",
            lambda s: pytest.fail("should not sleep/retry on non-connection errors"),
        )

        with pytest.raises(RuntimeError):
            _place_order_with_retry(broker, mock.Mock(), "AAPL")

        assert broker.place_order.call_count == 1

    def test_does_not_reconnect_if_already_connected(self, monkeypatch):
        from Strategy_Auto_Trader.markov_cli.execute import _place_order_with_retry
        from Strategy_Auto_Trader.broker.types import FillResult

        broker = mock.Mock()
        broker.is_connected.return_value = True
        fill = FillResult("AAPL", "BUY", 150.0, 10, "2026-07-01T00:00:00Z")
        broker.place_order.side_effect = [
            ConnectionError("Socket disconnect during order placement: blip"),
            fill,
        ]
        monkeypatch.setattr(
            "Strategy_Auto_Trader.markov_cli.execute.time.sleep", lambda s: None
        )

        result = _place_order_with_retry(broker, mock.Mock(), "AAPL")

        assert result is fill
        broker.connect.assert_not_called()

    def test_skips_retry_if_open_order_found_for_ticker(self, monkeypatch):
        from Strategy_Auto_Trader.markov_cli.execute import _place_order_with_retry

        broker = mock.Mock()
        broker.is_connected.return_value = True
        broker.place_order.side_effect = ConnectionError("Socket disconnect: dropped after send")
        broker.get_open_orders.return_value = [
            {"ticker": "AAPL", "action": "BUY", "status": "Submitted"}
        ]
        monkeypatch.setattr(
            "Strategy_Auto_Trader.markov_cli.execute.time.sleep", lambda s: None
        )

        result = _place_order_with_retry(broker, mock.Mock(), "AAPL")

        assert result is None
        assert broker.place_order.call_count == 1

    def test_ignores_open_order_check_failure_and_retries(self, monkeypatch):
        from Strategy_Auto_Trader.markov_cli.execute import _place_order_with_retry
        from Strategy_Auto_Trader.broker.types import FillResult

        broker = mock.Mock()
        broker.is_connected.return_value = True
        fill = FillResult("AAPL", "BUY", 150.0, 10, "2026-07-01T00:00:00Z")
        broker.place_order.side_effect = [
            ConnectionError("Socket disconnect during order placement: blip"),
            fill,
        ]
        broker.get_open_orders.side_effect = ConnectionError("still down")
        monkeypatch.setattr(
            "Strategy_Auto_Trader.markov_cli.execute.time.sleep", lambda s: None
        )

        result = _place_order_with_retry(broker, mock.Mock(), "AAPL")

        assert result is fill
        assert broker.place_order.call_count == 2
