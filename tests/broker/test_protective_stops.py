"""Tests for protective stop orders feature."""

from __future__ import annotations

import json
import pytest


class TestProtectiveStops:
    """Tests for broker stop order methods and portfolio integration."""

    # -- Broker Protocol Methods -----------------------------------------------

    def test_null_broker_place_stop_order_returns_result(self):
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        from Strategy_Auto_Trader.broker.types import StopOrderRequest
        broker = NullBroker(prices={"AAPL": 195.0})
        req = StopOrderRequest("AAPL", 10, 185.0)
        result = broker.place_stop_order(req)
        assert result is not None
        assert result.perm_id > 0
        assert result.stop_price == 185.0

    def test_null_broker_place_multiple_stops_increment_perm_ids(self):
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        from Strategy_Auto_Trader.broker.types import StopOrderRequest
        broker = NullBroker(prices={"AAPL": 195.0, "MSFT": 300.0})
        req1 = StopOrderRequest("AAPL", 10, 185.0)
        req2 = StopOrderRequest("MSFT", 5, 285.0)
        r1 = broker.place_stop_order(req1)
        r2 = broker.place_stop_order(req2)
        assert r1.perm_id < r2.perm_id

    def test_null_broker_get_open_stop_orders_returns_dict(self):
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        from Strategy_Auto_Trader.broker.types import StopOrderRequest
        broker = NullBroker(prices={"AAPL": 195.0})
        req = StopOrderRequest("AAPL", 10, 185.0)
        result = broker.place_stop_order(req)
        open_stops = broker.get_open_stop_orders()
        assert result.perm_id in open_stops
        assert open_stops[result.perm_id].ticker == "AAPL"
        assert open_stops[result.perm_id].quantity == 10
        assert open_stops[result.perm_id].stop_price == 185.0

    def test_null_broker_cancel_stop_order_returns_cancelled(self):
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        from Strategy_Auto_Trader.broker.types import StopOrderRequest
        broker = NullBroker(prices={"AAPL": 195.0})
        req = StopOrderRequest("AAPL", 10, 185.0)
        result = broker.place_stop_order(req)
        outcome = broker.cancel_stop_order(result.perm_id)
        assert outcome == "Cancelled"

    def test_null_broker_cancel_nonexistent_stop_returns_notfound(self):
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        broker = NullBroker()
        outcome = broker.cancel_stop_order(999999)
        assert outcome == "NotFound"

    def test_null_broker_get_stop_fill_returns_none(self):
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        from Strategy_Auto_Trader.broker.types import StopOrderRequest
        broker = NullBroker(prices={"AAPL": 195.0})
        req = StopOrderRequest("AAPL", 10, 185.0)
        result = broker.place_stop_order(req)
        fill = broker.get_stop_fill(result.perm_id)
        assert fill is None

    # -- Portfolio State Management -------------------------------------------

    def test_portfolio_record_entry_includes_stop_fields(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        fill = FillResult("AAPL", "BUY", 195.0, 10, "2026-07-01T00:00:00+00:00")
        pm.record_entry("AAPL", fill, 0.10, 185.0, 224.0)
        assert pm.positions["AAPL"]["stop_perm_id"] is None
        assert pm.positions["AAPL"]["stop_price"] is None

    def test_portfolio_set_stop_order_updates_position(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        fill = FillResult("AAPL", "BUY", 195.0, 10, "2026-07-01T00:00:00+00:00")
        pm.record_entry("AAPL", fill, 0.10, 185.0, 224.0)
        pm.set_stop_order("AAPL", 100001, 185.0)
        assert pm.positions["AAPL"]["stop_perm_id"] == 100001
        assert pm.positions["AAPL"]["stop_price"] == 185.0

    def test_portfolio_clear_stop_order_resets_fields(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        fill = FillResult("AAPL", "BUY", 195.0, 10, "2026-07-01T00:00:00+00:00")
        pm.record_entry("AAPL", fill, 0.10, 185.0, 224.0)
        pm.set_stop_order("AAPL", 100001, 185.0)
        pm.clear_stop_order("AAPL")
        assert pm.positions["AAPL"]["stop_perm_id"] is None
        assert pm.positions["AAPL"]["stop_price"] is None

    def test_portfolio_record_exit_with_stop_loss_type(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        buy = FillResult("AAPL", "BUY", 195.0, 10, "2026-07-01T00:00:00+00:00")
        pm.record_entry("AAPL", buy, 0.10, 185.0, 224.0)
        pm.set_stop_order("AAPL", 100001, 185.0)
        sell = FillResult("AAPL", "SELL", 183.0, 10, "2026-07-02T00:00:00+00:00")
        pm.record_exit("AAPL", sell, exit_type="stop_loss")
        assert pm.trade_log[-1]["exit_type"] == "stop_loss"
        assert pm.trade_log[-1]["stop_price"] == 185.0

    def test_portfolio_record_exit_with_strategy_exit_type(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        buy = FillResult("AAPL", "BUY", 195.0, 10, "2026-07-01T00:00:00+00:00")
        pm.record_entry("AAPL", buy, 0.10, 185.0, 224.0)
        sell = FillResult("AAPL", "SELL", 210.0, 10, "2026-07-02T00:00:00+00:00")
        pm.record_exit("AAPL", sell, exit_type="strategy_exit")
        assert pm.trade_log[-1]["exit_type"] == "strategy_exit"
        assert "stop_price" not in pm.trade_log[-1]

    def test_portfolio_record_exit_default_exit_type(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        buy = FillResult("AAPL", "BUY", 195.0, 10, "2026-07-01T00:00:00+00:00")
        pm.record_entry("AAPL", buy, 0.10, 185.0, 224.0)
        sell = FillResult("AAPL", "SELL", 210.0, 10, "2026-07-02T00:00:00+00:00")
        pm.record_exit("AAPL", sell)
        assert pm.trade_log[-1]["exit_type"] == "strategy_exit"

    def test_portfolio_save_and_restore_preserves_stop_order_state(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        path = tmp_path / "state.json"
        pm = PortfolioManager(20_000, 5, path)
        fill = FillResult("AAPL", "BUY", 195.0, 10, "2026-07-01T00:00:00+00:00")
        pm.record_entry("AAPL", fill, 0.10, 185.0, 224.0)
        pm.set_stop_order("AAPL", 100001, 185.0)
        pm.save()

        pm2 = PortfolioManager(20_000, 5, path)
        assert pm2.positions["AAPL"]["stop_perm_id"] == 100001
        assert pm2.positions["AAPL"]["stop_price"] == 185.0

    def test_portfolio_backward_compat_old_state_without_stop_fields(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        path = tmp_path / "state.json"
        old_state = {
            "positions": {
                "AAPL": {
                    "entry_date": "2026-07-01",
                    "fill_price": 195.0,
                    "quantity": 10,
                    "cost_value": 1950.0,
                    "market": "",
                    "currency": "",
                    "kelly_fraction": 0.10,
                    "stop_level": 185.0,
                    "target_level": 224.0,
                }
            },
            "trade_log": [],
            "trades_today": {"date": "2026-07-02", "buys": 0, "sells": 0},
        }
        path.write_text(json.dumps(old_state), encoding="utf-8")
        pm = PortfolioManager(20_000, 5, path)
        assert pm.positions["AAPL"]["stop_perm_id"] is None
        assert pm.positions["AAPL"]["stop_price"] is None

    # -- Execute.py Integration -----------------------------------------------

    def test_execute_with_protective_stops_disabled_no_stop_placed(self, tmp_path):
        from Strategy_Auto_Trader.markov_cli.execute import execute_signals
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        import pandas as pd

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        run_dir = data_dir / "AAPL_20260701T120000Z"
        run_dir.mkdir()
        df = pd.DataFrame([{
            "close": 195.0, "kelly_fraction": 0.15,
            "stop_level": 185.0, "target_level": 224.0,
            "trade_event": "BUY",
        }])
        df.index.name = "date"
        df.to_csv(run_dir / "compositeBacktest.csv")
        (run_dir / "qualityGate.json").write_text(
            json.dumps({"flag": "BUY"}), encoding="utf-8"
        )

        state_path = tmp_path / "execution_state.json"
        portfolio = PortfolioManager(20_000, 5, state_path)
        broker = NullBroker(prices={"AAPL": 195.0})
        limit_tracker = portfolio.get_limit_tracker()

        execute_signals(
            ["AAPL"], data_dir, portfolio, limit_tracker, broker,
            protective_stops=False,
        )

        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"]["stop_perm_id"] is None

    def test_execute_with_protective_stops_enabled_places_stop(self, tmp_path):
        from Strategy_Auto_Trader.markov_cli.execute import execute_signals
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        import pandas as pd

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        run_dir = data_dir / "AAPL_20260701T120000Z"
        run_dir.mkdir()
        df = pd.DataFrame([{
            "close": 195.0, "kelly_fraction": 0.15,
            "stop_level": 185.0, "target_level": 224.0,
            "trade_event": "BUY",
        }])
        df.index.name = "date"
        df.to_csv(run_dir / "compositeBacktest.csv")
        (run_dir / "qualityGate.json").write_text(
            json.dumps({"flag": "BUY"}), encoding="utf-8"
        )

        state_path = tmp_path / "execution_state.json"
        portfolio = PortfolioManager(20_000, 5, state_path)
        broker = NullBroker(prices={"AAPL": 195.0})
        limit_tracker = portfolio.get_limit_tracker()

        execute_signals(
            ["AAPL"], data_dir, portfolio, limit_tracker, broker,
            protective_stops=True,
            stop_buffer_pct=1.5,
        )

        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"]["stop_perm_id"] is not None
        assert portfolio.positions["AAPL"]["stop_price"] is not None

    def test_execute_sell_with_protective_stops_cancels_stop_first(self, tmp_path):
        from unittest.mock import MagicMock
        from Strategy_Auto_Trader.markov_cli.execute import execute_signals
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        from Strategy_Auto_Trader.broker.types import StopOrderRequest
        import pandas as pd

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        run_dir_buy = data_dir / "AAPL_20260701T120000Z"
        run_dir_buy.mkdir()
        df_buy = pd.DataFrame([{
            "close": 195.0, "kelly_fraction": 0.15,
            "stop_level": 185.0, "target_level": 224.0,
            "trade_event": "BUY",
        }])
        df_buy.index.name = "date"
        df_buy.to_csv(run_dir_buy / "compositeBacktest.csv")
        (run_dir_buy / "qualityGate.json").write_text(
            json.dumps({"flag": "BUY"}), encoding="utf-8"
        )

        state_path = tmp_path / "execution_state.json"
        portfolio = PortfolioManager(20_000, 5, state_path)
        broker = NullBroker(prices={"AAPL": 195.0})
        limit_tracker = portfolio.get_limit_tracker()

        execute_signals(
            ["AAPL"], data_dir, portfolio, limit_tracker, broker,
            protective_stops=True,
        )

        assert "AAPL" in portfolio.positions
        perm_id = portfolio.positions["AAPL"]["stop_perm_id"]
        assert perm_id is not None

        run_dir_sell = data_dir / "AAPL_20260702T120000Z"
        run_dir_sell.mkdir()
        df_sell = pd.DataFrame([{
            "close": 210.0, "kelly_fraction": 0.15,
            "stop_level": 185.0, "target_level": 224.0,
            "trade_event": "SELL",
        }])
        df_sell.index.name = "date"
        df_sell.to_csv(run_dir_sell / "compositeBacktest.csv")
        (run_dir_sell / "qualityGate.json").write_text(
            json.dumps({"flag": "SELL"}), encoding="utf-8"
        )

        limit_tracker2 = portfolio.get_limit_tracker()
        execute_signals(
            ["AAPL"], data_dir, portfolio, limit_tracker2, broker,
            protective_stops=True,
        )

        assert "AAPL" not in portfolio.positions
        assert perm_id not in broker.get_open_stop_orders()

    # -- Live Daemon Protective Stops ------------------------------------------

    def test_check_protective_stops_re_places_missing_stop(self, tmp_path):
        from Strategy_Auto_Trader.markov_cli.live_daemon import check_protective_stops
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        from Strategy_Auto_Trader.broker.types import FillResult
        import logging

        logger = logging.getLogger(__name__)
        state_path = tmp_path / "state.json"
        portfolio = PortfolioManager(20_000, 5, state_path)
        broker = NullBroker(prices={"AAPL": 195.0})

        fill = FillResult("AAPL", "BUY", 195.0, 10, "2026-07-01T00:00:00+00:00")
        portfolio.record_entry("AAPL", fill, 0.10, 185.0, 224.0)

        check_protective_stops(portfolio, broker, logger, stop_buffer_pct=1.5)

        assert portfolio.positions["AAPL"]["stop_perm_id"] is not None

    def test_check_protective_stops_cancels_orphan_stops(self, tmp_path):
        from Strategy_Auto_Trader.markov_cli.live_daemon import check_protective_stops
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        from Strategy_Auto_Trader.broker.types import StopOrderRequest
        import logging

        logger = logging.getLogger(__name__)
        state_path = tmp_path / "state.json"
        portfolio = PortfolioManager(20_000, 5, state_path)
        broker = NullBroker(prices={"AAPL": 195.0, "MSFT": 300.0})

        req = StopOrderRequest("MSFT", 5, 285.0)
        result = broker.place_stop_order(req)

        check_protective_stops(portfolio, broker, logger, stop_buffer_pct=1.5)

        assert result.perm_id not in broker.get_open_stop_orders()

    # -- Edge Cases: Stop Fill Synthesis & Error Handling ----------------------

    def test_sell_when_cancel_returns_filled_and_get_stop_fill_returns_none(self, tmp_path):
        """Verify position exits with reconciled_stop_loss when fill is None."""
        from Strategy_Auto_Trader.markov_cli.execute import execute_signals
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        from Strategy_Auto_Trader.broker.types import StopOrderRequest
        import pandas as pd

        # Custom broker: cancel_stop_order returns "Filled", but get_stop_fill returns None
        class TestBroker(NullBroker):
            def cancel_stop_order(self, perm_id: int) -> str:
                return "Filled"
            def get_stop_fill(self, perm_id: int):
                return None

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Setup: place BUY
        run_dir_buy = data_dir / "AAPL_20260701T120000Z"
        run_dir_buy.mkdir()
        df_buy = pd.DataFrame([{
            "close": 195.0, "kelly_fraction": 0.15,
            "stop_level": 185.0, "target_level": 224.0,
            "trade_event": "BUY",
        }])
        df_buy.index.name = "date"
        df_buy.to_csv(run_dir_buy / "compositeBacktest.csv")
        (run_dir_buy / "qualityGate.json").write_text(
            json.dumps({"flag": "BUY"}), encoding="utf-8"
        )

        state_path = tmp_path / "execution_state.json"
        portfolio = PortfolioManager(20_000, 5, state_path)
        broker = TestBroker(prices={"AAPL": 195.0})
        limit_tracker = portfolio.get_limit_tracker()

        execute_signals(
            ["AAPL"], data_dir, portfolio, limit_tracker, broker,
            protective_stops=True,
        )
        assert "AAPL" in portfolio.positions
        perm_id = portfolio.positions["AAPL"]["stop_perm_id"]

        # Setup: place SELL
        run_dir_sell = data_dir / "AAPL_20260702T120000Z"
        run_dir_sell.mkdir()
        df_sell = pd.DataFrame([{
            "close": 210.0, "kelly_fraction": 0.15,
            "stop_level": 185.0, "target_level": 224.0,
            "trade_event": "SELL",
        }])
        df_sell.index.name = "date"
        df_sell.to_csv(run_dir_sell / "compositeBacktest.csv")
        (run_dir_sell / "qualityGate.json").write_text(
            json.dumps({"flag": "SELL"}), encoding="utf-8"
        )

        limit_tracker2 = portfolio.get_limit_tracker()
        execute_signals(
            ["AAPL"], data_dir, portfolio, limit_tracker2, broker,
            protective_stops=True,
        )

        # Verify: position closed, reconciled_stop_loss recorded
        assert "AAPL" not in portfolio.positions
        assert portfolio.trade_log[-1]["exit_type"] == "reconciled_stop_loss"
        # Verify: no market SELL order was placed by execute_signals
        sell_orders = [o for o in broker.orders if o.action == "SELL"]
        assert len(sell_orders) == 0

    def test_sell_when_cancel_returns_error_skips_market_sell(self, tmp_path):
        """Verify market sell is skipped when cancel_stop_order returns Error."""
        from Strategy_Auto_Trader.markov_cli.execute import execute_signals
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        import pandas as pd

        # Custom broker: cancel_stop_order returns "Error"
        class TestBroker(NullBroker):
            def cancel_stop_order(self, perm_id: int) -> str:
                return "Error"

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Setup: place BUY
        run_dir_buy = data_dir / "AAPL_20260701T120000Z"
        run_dir_buy.mkdir()
        df_buy = pd.DataFrame([{
            "close": 195.0, "kelly_fraction": 0.15,
            "stop_level": 185.0, "target_level": 224.0,
            "trade_event": "BUY",
        }])
        df_buy.index.name = "date"
        df_buy.to_csv(run_dir_buy / "compositeBacktest.csv")
        (run_dir_buy / "qualityGate.json").write_text(
            json.dumps({"flag": "BUY"}), encoding="utf-8"
        )

        state_path = tmp_path / "execution_state.json"
        portfolio = PortfolioManager(20_000, 5, state_path)
        broker = TestBroker(prices={"AAPL": 195.0})
        limit_tracker = portfolio.get_limit_tracker()

        execute_signals(
            ["AAPL"], data_dir, portfolio, limit_tracker, broker,
            protective_stops=True,
        )

        # Setup: place SELL (cancel returns Error)
        run_dir_sell = data_dir / "AAPL_20260702T120000Z"
        run_dir_sell.mkdir()
        df_sell = pd.DataFrame([{
            "close": 210.0, "kelly_fraction": 0.15,
            "stop_level": 185.0, "target_level": 224.0,
            "trade_event": "SELL",
        }])
        df_sell.index.name = "date"
        df_sell.to_csv(run_dir_sell / "compositeBacktest.csv")
        (run_dir_sell / "qualityGate.json").write_text(
            json.dumps({"flag": "SELL"}), encoding="utf-8"
        )

        limit_tracker2 = portfolio.get_limit_tracker()
        execute_signals(
            ["AAPL"], data_dir, portfolio, limit_tracker2, broker,
            protective_stops=True,
        )

        # Verify: position still open, no market SELL order
        assert "AAPL" in portfolio.positions
        sell_orders = [o for o in broker.orders if o.action == "SELL"]
        assert len(sell_orders) == 0

    def test_manual_sell_cancel_error_retains_stop_tracking(self, tmp_path):
        """After an 'Error' cancel outcome, _execute_sell must keep stop_perm_id set.

        The stop may still be live at the broker; clearing tracking would let
        check_protective_stops re-place a duplicate stop.
        """
        from Strategy_Auto_Trader.markov_cli.manual_commands import _execute_sell
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        from Strategy_Auto_Trader.broker.types import FillResult
        import logging

        class TestBroker(NullBroker):
            def cancel_stop_order(self, perm_id: int) -> str:
                return "Error"

        logger = logging.getLogger(__name__)
        state_path = tmp_path / "state.json"
        portfolio = PortfolioManager(20_000, 5, state_path)
        broker = TestBroker(prices={"AAPL": 195.0})

        fill = FillResult("AAPL", "BUY", 195.0, 10, "2026-07-01T00:00:00+00:00")
        portfolio.record_entry("AAPL", fill, 0.10, 185.0, 224.0)
        portfolio.set_stop_order("AAPL", 100001, 185.0)

        success, result_fill, error = _execute_sell("AAPL", portfolio, broker, logger)

        assert success is False
        assert result_fill is None
        assert "AAPL" in portfolio.positions
        # Tracking retained: perm_id still set so the stop is not orphaned
        assert portfolio.positions["AAPL"]["stop_perm_id"] == 100001
        # No market sell was placed
        sell_orders = [o for o in broker.orders if o.action == "SELL"]
        assert len(sell_orders) == 0

    def test_check_protective_stops_with_none_stop_price_and_stop_level(self, tmp_path):
        """Verify no crash when stop_price and stop_level are None/0."""
        from Strategy_Auto_Trader.markov_cli.live_daemon import check_protective_stops
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        from Strategy_Auto_Trader.broker.types import FillResult
        import logging

        logger = logging.getLogger(__name__)
        state_path = tmp_path / "state.json"
        portfolio = PortfolioManager(20_000, 5, state_path)
        broker = NullBroker(prices={"AAPL": 195.0})

        fill = FillResult("AAPL", "BUY", 195.0, 10, "2026-07-01T00:00:00+00:00")
        portfolio.record_entry("AAPL", fill, 0.10, stop_level=None, target_level=None)

        # Simulate position with no stop_price and no stop_level
        portfolio.positions["AAPL"]["stop_price"] = None
        portfolio.positions["AAPL"]["stop_level"] = None
        portfolio.save()

        # Should not crash
        check_protective_stops(portfolio, broker, logger, stop_buffer_pct=1.5)

        # Position should still exist (no re-place)
        assert "AAPL" in portfolio.positions

    def test_reconcile_check_stop_fills_with_none_stop_price(self, tmp_path):
        """Verify no TypeError when stop_price is None in reconcile."""
        from Strategy_Auto_Trader.broker.reconcile import check_stop_fills_for_missing_positions
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        from Strategy_Auto_Trader.broker.types import FillResult

        state_path = tmp_path / "state.json"
        portfolio = PortfolioManager(20_000, 5, state_path)
        broker = NullBroker()

        fill = FillResult("AAPL", "BUY", 195.0, 10, "2026-07-01T00:00:00+00:00")
        portfolio.record_entry("AAPL", fill, 0.10, 185.0, 224.0)

        # Simulate position missing at broker but with stop_price=None
        portfolio.positions["AAPL"]["stop_perm_id"] = 999999
        portfolio.positions["AAPL"]["stop_price"] = None
        portfolio.save()

        internal = portfolio.positions
        broker_positions = {}  # Position missing at broker

        # Should not raise TypeError
        result = check_stop_fills_for_missing_positions(
            internal, broker_positions, broker, portfolio
        )

        # Result should be empty (no stop fills found)
        assert isinstance(result, list)

    # -- NullBroker Stubs Always Match Protocol --------------------------------

    def test_null_broker_satisfies_protocol_with_stops(self):
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        from Strategy_Auto_Trader.broker.protocols import BrokerAdapterProtocol
        broker = NullBroker()
        assert isinstance(broker, BrokerAdapterProtocol)
