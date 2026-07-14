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
        signal_reader.return_value = {"flag": "BUY", "close": 150.0, "kelly_fraction": 0.1, "stop_level": 140.0, "target_level": 160.0}

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
                daily_buy_limit=5,
                daily_sell_limit=None,
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
                daily_buy_limit=2,
                daily_sell_limit=5,
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
        signal_reader.return_value = {"flag": "BUY", "close": 150.0, "kelly_fraction": 0.1, "stop_level": 140.0, "target_level": 160.0}

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
                daily_buy_limit=2,
                daily_sell_limit=None,
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

        # Setup: AAPL skipped (daily limit), MSFT succeeds, GOOG error
        limit_tracker.can_buy.side_effect = [False, True, True]
        portfolio.can_open.side_effect = [True, True]
        portfolio.compute_quantity.return_value = 10

        fill = FillResult("MSFT", "BUY", 350.0, 10, "2026-07-01T00:00:00Z")
        broker.place_order.side_effect = [fill, RuntimeError("Connection lost")]

        signal_reader = mock.Mock()
        signal_reader.return_value = {"flag": "BUY", "close": 350.0, "kelly_fraction": 0.1, "stop_level": 340.0, "target_level": 360.0}

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
                daily_buy_limit=1,
                daily_sell_limit=None,
            )

        exc = exc_info.value
        assert "AAPL(daily limit reached)" in exc.skipped
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
            "AAPL": {"flag": "BUY", "close": 150.0, "kelly_fraction": 0.1,
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
                daily_buy_limit=5,
                daily_sell_limit=None,
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
        signal_reader.return_value = {"flag": "BUY", "close": 150.0, "kelly_fraction": 0.1, "stop_level": 140.0, "target_level": 160.0}

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
            daily_buy_limit=2,
            daily_sell_limit=None,
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
            daily_buy_limit=2,
            daily_sell_limit=None,
        )

        assert buys == []
        assert sells == []
        assert skipped == ["AAPL"]
