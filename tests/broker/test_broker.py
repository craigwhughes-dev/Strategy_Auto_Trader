from __future__ import annotations

import json
import os
import time

import pandas as pd
import pytest


class TestBroker:
    """Tests for broker/ execution layer: NullBroker, PortfolioManager, signal_reader, execute."""

    # -- BrokerAdapterProtocol ------------------------------------------------

    def test_null_broker_satisfies_protocol(self):
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        from Strategy_Auto_Trader.broker.protocols import BrokerAdapterProtocol
        assert isinstance(NullBroker(), BrokerAdapterProtocol)

    def test_null_broker_fills_at_given_price(self):
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        from Strategy_Auto_Trader.broker.types import OrderRequest
        broker = NullBroker(prices={"AAPL": 195.0})
        fill = broker.place_order(OrderRequest("AAPL", "BUY", 10))
        assert fill.fill_price == pytest.approx(195.0)
        assert fill.quantity == 10
        assert fill.action == "BUY"

    def test_null_broker_set_prices_updates_fills(self):
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        from Strategy_Auto_Trader.broker.types import OrderRequest
        broker = NullBroker(prices={})
        broker.set_prices({"AAPL": 210.5})
        fill = broker.place_order(OrderRequest("AAPL", "BUY", 3))
        assert fill.fill_price == pytest.approx(210.5)

    def test_null_broker_tracks_positions(self):
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        from Strategy_Auto_Trader.broker.types import OrderRequest
        broker = NullBroker(prices={"AAPL": 195.0})
        broker.place_order(OrderRequest("AAPL", "BUY", 5))
        assert broker.get_open_positions()["AAPL"] == 5
        broker.place_order(OrderRequest("AAPL", "SELL", 5))
        assert "AAPL" not in broker.get_open_positions()

    # -- PortfolioManager -----------------------------------------------------

    def test_portfolio_compute_quantity(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        # slot = 20000/5 = 4000, kelly=0.15, price=200 -> floor(4000*0.15/200) = 3
        qty = pm.compute_quantity(0.15, 200.0)
        assert qty == 3

    def test_portfolio_compute_quantity_min_one(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        pm = PortfolioManager(1_000, 5, tmp_path / "state.json")
        qty = pm.compute_quantity(0.01, 500.0)
        assert qty >= 1

    def test_portfolio_at_capacity_blocks_open(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 2, tmp_path / "state.json")
        fa = FillResult("A", "BUY", 100.0, 10, "2026-07-01T00:00:00+00:00")
        pm.record_entry("A", fa, 0.10, 95.0, 115.0)
        fb = FillResult("B", "BUY", 200.0, 5, "2026-07-01T00:00:00+00:00")
        pm.record_entry("B", fb, 0.10, 190.0, 230.0)
        assert not pm.can_open("C")

    def test_portfolio_same_ticker_blocks_open(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        fill = FillResult("AAPL", "BUY", 195.0, 10, "2026-07-01T00:00:00+00:00")
        pm.record_entry("AAPL", fill, 0.10, 185.0, 224.0)
        assert not pm.can_open("AAPL")

    def test_portfolio_record_entry_and_exit(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        buy = FillResult("MSFT", "BUY", 300.0, 10, "2026-07-01T00:00:00+00:00")
        pm.record_entry("MSFT", buy, 0.12, 285.0, 345.0)
        assert "MSFT" in pm.positions
        assert pm.positions["MSFT"]["quantity"] == 10
        sell = FillResult("MSFT", "SELL", 330.0, 10, "2026-07-10T00:00:00+00:00")
        pm.record_exit("MSFT", sell)
        assert "MSFT" not in pm.positions
        assert pm.trade_log[-1]["action"] == "SELL"
        assert pm.trade_log[-1]["gross_pl"] == pytest.approx(300.0)  # (330-300)*10
        assert pm.trade_log[-1]["pl"] == pytest.approx(296.85)  # gross minus IBKR commission both legs

    def test_portfolio_save_and_reload(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        path = tmp_path / "state.json"
        pm = PortfolioManager(20_000, 5, path)
        fill = FillResult("SPY", "BUY", 500.0, 4, "2026-07-01T00:00:00+00:00")
        pm.record_entry("SPY", fill, 0.10, 475.0, 575.0)
        pm.save()
        pm2 = PortfolioManager(20_000, 5, path)
        assert "SPY" in pm2.positions
        assert pm2.positions["SPY"]["fill_price"] == pytest.approx(500.0)

    # -- signal_reader --------------------------------------------------------

    def test_signal_reader_returns_none_when_no_data(self, tmp_path):
        from Strategy_Auto_Trader.broker.signal_reader import read_latest_signal
        assert read_latest_signal("FAKE", tmp_path) is None

    def test_signal_reader_reads_buy_signal(self, tmp_path):
        from Strategy_Auto_Trader.broker.signal_reader import read_latest_signal
        run_dir = tmp_path / "AAPL_20260701T120000Z"
        run_dir.mkdir()
        df = pd.DataFrame([{
            "close": 195.0, "kelly_fraction": 0.15,
            "stop_level": 185.25, "target_level": 224.25,
            "trade_event": "BUY",
        }])
        df.index.name = "date"
        df.to_csv(run_dir / "compositeBacktest.csv")
        (run_dir / "qualityGate.json").write_text(
            json.dumps({"flag": "BUY", "reason": "test"}), encoding="utf-8"
        )
        result = read_latest_signal("AAPL", tmp_path)
        assert result is not None
        assert result["flag"] == "BUY"
        assert result["close"] == pytest.approx(195.0)
        assert result["kelly_fraction"] == pytest.approx(0.15)

    def test_signal_reader_returns_none_when_stale(self, tmp_path):
        from Strategy_Auto_Trader.broker.signal_reader import read_latest_signal
        run_dir = tmp_path / "AAPL_20200101T000000Z"
        run_dir.mkdir()
        df = pd.DataFrame([{"close": 100.0, "kelly_fraction": 0.10,
                             "stop_level": 95.0, "target_level": 115.0}])
        df.index.name = "date"
        df.to_csv(run_dir / "compositeBacktest.csv")
        old_time = time.time() - 48 * 3600
        os.utime(run_dir, (old_time, old_time))
        assert read_latest_signal("AAPL", tmp_path) is None

    # -- IBKRAdapter ------------------------------------------------------------

    def test_ibkr_adapter_connect_passes_timeout(self):
        pytest.importorskip("ib_insync")
        from unittest.mock import patch, MagicMock
        from Strategy_Auto_Trader.broker.ibkr_adapter import IBKRAdapter
        adapter = IBKRAdapter(port=4002, client_id=7, connect_timeout=12.0)
        with patch("ib_insync.IB") as MockIB:
            adapter.connect()
        MockIB.return_value.connect.assert_called_once_with(
            "127.0.0.1", 4002, clientId=7, timeout=12.0)

    def test_ibkr_adapter_managed_accounts(self):
        from unittest.mock import MagicMock
        from Strategy_Auto_Trader.broker.ibkr_adapter import IBKRAdapter
        adapter = IBKRAdapter()
        adapter._ib = MagicMock()
        adapter._ib.managedAccounts.return_value = ("DU123456",)
        assert adapter.managed_accounts() == ["DU123456"]

    def test_ibkr_adapter_place_order_returns_fill_when_filled(self):
        pytest.importorskip("ib_insync")
        from unittest.mock import MagicMock
        from Strategy_Auto_Trader.broker.ibkr_adapter import IBKRAdapter
        from Strategy_Auto_Trader.broker.types import OrderRequest
        adapter = IBKRAdapter()
        adapter._ib = MagicMock()
        adapter._ib.isConnected.return_value = True

        # Mock trade object with status "Filled"
        mock_trade = MagicMock()
        mock_trade.orderStatus.status = "Filled"
        mock_trade.orderStatus.avgFillPrice = 195.5
        adapter._ib.placeOrder.return_value = mock_trade
        adapter._ib.qualifyContracts = MagicMock()
        adapter._ib.waitOnUpdate = MagicMock()

        fill = adapter.place_order(OrderRequest("AAPL", "BUY", 10))
        assert fill is not None
        assert fill.fill_price == pytest.approx(195.5)
        assert fill.quantity == 10
        assert fill.action == "BUY"
        assert fill.ticker == "AAPL"

    def test_ibkr_adapter_place_order_returns_none_when_cancelled(self):
        pytest.importorskip("ib_insync")
        from unittest.mock import MagicMock
        from Strategy_Auto_Trader.broker.ibkr_adapter import IBKRAdapter
        from Strategy_Auto_Trader.broker.types import OrderRequest
        adapter = IBKRAdapter()
        adapter._ib = MagicMock()
        adapter._ib.isConnected.return_value = True

        # Mock trade object with status "Cancelled"
        mock_trade = MagicMock()
        mock_trade.orderStatus.status = "Cancelled"
        mock_trade.orderStatus.avgFillPrice = 0.0
        adapter._ib.placeOrder.return_value = mock_trade
        adapter._ib.qualifyContracts = MagicMock()
        adapter._ib.waitOnUpdate = MagicMock()

        fill = adapter.place_order(OrderRequest("AAPL", "BUY", 10))
        assert fill is None

    def test_ibkr_adapter_place_order_returns_none_when_partially_filled(self):
        pytest.importorskip("ib_insync")
        from unittest.mock import MagicMock
        from Strategy_Auto_Trader.broker.ibkr_adapter import IBKRAdapter
        from Strategy_Auto_Trader.broker.types import OrderRequest
        adapter = IBKRAdapter()
        adapter._ib = MagicMock()
        adapter._ib.isConnected.return_value = True

        # Mock trade object with status "PartiallyFilled"
        mock_trade = MagicMock()
        mock_trade.orderStatus.status = "PartiallyFilled"
        mock_trade.orderStatus.avgFillPrice = 195.0
        adapter._ib.placeOrder.return_value = mock_trade
        adapter._ib.qualifyContracts = MagicMock()
        adapter._ib.waitOnUpdate = MagicMock()

        fill = adapter.place_order(OrderRequest("AAPL", "BUY", 10))
        assert fill is None

    def test_ibkr_adapter_place_order_avgfillprice_populated_skips_retry_loop(self):
        pytest.importorskip("ib_insync")
        from unittest.mock import MagicMock
        from Strategy_Auto_Trader.broker.ibkr_adapter import IBKRAdapter
        from Strategy_Auto_Trader.broker.types import OrderRequest
        adapter = IBKRAdapter()
        adapter._ib = MagicMock()
        adapter._ib.isConnected.return_value = True

        mock_trade = MagicMock()
        mock_trade.orderStatus.status = "Filled"
        mock_trade.orderStatus.avgFillPrice = 195.5
        adapter._ib.placeOrder.return_value = mock_trade
        adapter._ib.qualifyContracts = MagicMock()
        adapter._ib.waitOnUpdate = MagicMock()

        fill = adapter.place_order(OrderRequest("AAPL", "BUY", 10))
        assert fill.fill_price == pytest.approx(195.5)
        # Only the initial wait — avgFillPrice was already populated, no retries.
        assert adapter._ib.waitOnUpdate.call_count == 1

    def test_ibkr_adapter_place_order_falls_back_to_fills_execution_price(self):
        pytest.importorskip("ib_insync")
        from unittest.mock import MagicMock
        from Strategy_Auto_Trader.broker.ibkr_adapter import IBKRAdapter
        from Strategy_Auto_Trader.broker.types import OrderRequest
        adapter = IBKRAdapter()
        adapter._ib = MagicMock()
        adapter._ib.isConnected.return_value = True

        mock_fill = MagicMock()
        mock_fill.execution.avgPrice = 123.45
        mock_fill.execution.price = 123.45
        mock_trade = MagicMock()
        mock_trade.orderStatus.status = "Filled"
        mock_trade.orderStatus.avgFillPrice = 0.0
        mock_trade.fills = [mock_fill]
        adapter._ib.placeOrder.return_value = mock_trade
        adapter._ib.qualifyContracts = MagicMock()
        adapter._ib.waitOnUpdate = MagicMock()

        fill = adapter.place_order(OrderRequest("AAPL", "BUY", 10))
        assert fill.fill_price == pytest.approx(123.45)
        # fills was already populated on the first check — no extra polling.
        assert adapter._ib.waitOnUpdate.call_count == 1

    def test_ibkr_adapter_place_order_recovers_price_after_late_fill_event(self):
        pytest.importorskip("ib_insync")
        from unittest.mock import MagicMock
        from Strategy_Auto_Trader.broker.ibkr_adapter import IBKRAdapter
        from Strategy_Auto_Trader.broker.types import OrderRequest
        adapter = IBKRAdapter()
        adapter._ib = MagicMock()
        adapter._ib.isConnected.return_value = True

        mock_trade = MagicMock()
        mock_trade.orderStatus.status = "Filled"
        mock_trade.orderStatus.avgFillPrice = 0.0
        mock_trade.fills = []
        adapter._ib.placeOrder.return_value = mock_trade
        adapter._ib.qualifyContracts = MagicMock()

        calls = {"n": 0}

        def wait_side_effect(timeout=None):
            calls["n"] += 1
            if calls["n"] >= 2:
                late_fill = MagicMock()
                late_fill.execution.avgPrice = 88.0
                late_fill.execution.price = 88.0
                mock_trade.fills = [late_fill]

        adapter._ib.waitOnUpdate = MagicMock(side_effect=wait_side_effect)

        fill = adapter.place_order(OrderRequest("AAPL", "BUY", 10))
        assert fill.fill_price == pytest.approx(88.0)
        assert adapter._ib.waitOnUpdate.call_count == 2

    def test_ibkr_adapter_place_order_gives_up_after_bounded_retries(self):
        pytest.importorskip("ib_insync")
        from unittest.mock import MagicMock
        from Strategy_Auto_Trader.broker.ibkr_adapter import IBKRAdapter
        from Strategy_Auto_Trader.broker.types import OrderRequest
        adapter = IBKRAdapter()
        adapter._ib = MagicMock()
        adapter._ib.isConnected.return_value = True

        mock_trade = MagicMock()
        mock_trade.orderStatus.status = "Filled"
        mock_trade.orderStatus.avgFillPrice = 0.0
        mock_trade.fills = []
        adapter._ib.placeOrder.return_value = mock_trade
        adapter._ib.qualifyContracts = MagicMock()
        adapter._ib.waitOnUpdate = MagicMock()

        fill = adapter.place_order(OrderRequest("AAPL", "BUY", 10))
        # Never resolves a real price — returns what it has instead of hanging.
        assert fill.fill_price == 0.0
        # 1 initial wait + 5 bounded retries, no more.
        assert adapter._ib.waitOnUpdate.call_count == 6

    def test_ibkr_adapter_get_open_positions_single_us_position(self):
        pytest.importorskip("ib_insync")
        from unittest.mock import MagicMock
        from Strategy_Auto_Trader.broker.ibkr_adapter import IBKRAdapter
        adapter = IBKRAdapter()
        adapter._ib = MagicMock()

        mock_pos = MagicMock()
        mock_pos.contract.symbol = "AAPL"
        mock_pos.contract.currency = "USD"
        mock_pos.position = 10
        adapter._ib.positions.return_value = [mock_pos]

        positions = adapter.get_open_positions()
        assert positions == {"AAPL": 10}

    def test_ibkr_adapter_get_open_positions_single_lse_position(self):
        pytest.importorskip("ib_insync")
        from unittest.mock import MagicMock
        from Strategy_Auto_Trader.broker.ibkr_adapter import IBKRAdapter
        adapter = IBKRAdapter()
        adapter._ib = MagicMock()

        mock_pos = MagicMock()
        mock_pos.contract.symbol = "HSBA"
        mock_pos.contract.currency = "GBP"
        mock_pos.position = 200
        adapter._ib.positions.return_value = [mock_pos]

        positions = adapter.get_open_positions()
        assert positions == {"HSBA.L": 200}

    def test_ibkr_adapter_get_open_positions_cross_listed_collision_prevention(self):
        pytest.importorskip("ib_insync")
        from unittest.mock import MagicMock
        from Strategy_Auto_Trader.broker.ibkr_adapter import IBKRAdapter
        adapter = IBKRAdapter()
        adapter._ib = MagicMock()

        us_pos = MagicMock()
        us_pos.contract.symbol = "BP"
        us_pos.contract.currency = "USD"
        us_pos.position = 100

        uk_pos = MagicMock()
        uk_pos.contract.symbol = "BP"
        uk_pos.contract.currency = "GBP"
        uk_pos.position = 50

        adapter._ib.positions.return_value = [us_pos, uk_pos]

        positions = adapter.get_open_positions()
        assert positions == {"BP": 100, "BP.L": 50}
        assert "BP" in positions
        assert "BP.L" in positions
        assert positions["BP"] != positions["BP.L"]

    def test_ibkr_adapter_get_open_positions_lse_share_class(self):
        pytest.importorskip("ib_insync")
        from unittest.mock import MagicMock
        from Strategy_Auto_Trader.broker.ibkr_adapter import IBKRAdapter
        adapter = IBKRAdapter()
        adapter._ib = MagicMock()

        mock_pos = MagicMock()
        mock_pos.contract.symbol = "BT.A"
        mock_pos.contract.currency = "GBP"
        mock_pos.position = 500
        adapter._ib.positions.return_value = [mock_pos]

        positions = adapter.get_open_positions()
        assert positions == {"BT-A.L": 500}

    def test_ibkr_adapter_get_open_positions_excludes_zero_quantity(self):
        pytest.importorskip("ib_insync")
        from unittest.mock import MagicMock
        from Strategy_Auto_Trader.broker.ibkr_adapter import IBKRAdapter
        adapter = IBKRAdapter()
        adapter._ib = MagicMock()

        pos1 = MagicMock()
        pos1.contract.symbol = "AAPL"
        pos1.contract.currency = "USD"
        pos1.position = 10

        pos2 = MagicMock()
        pos2.contract.symbol = "MSFT"
        pos2.contract.currency = "USD"
        pos2.position = 0

        adapter._ib.positions.return_value = [pos1, pos2]

        positions = adapter.get_open_positions()
        assert positions == {"AAPL": 10}
        assert "MSFT" not in positions

    def test_ibkr_adapter_get_open_positions_multiple_positions(self):
        pytest.importorskip("ib_insync")
        from unittest.mock import MagicMock
        from Strategy_Auto_Trader.broker.ibkr_adapter import IBKRAdapter
        adapter = IBKRAdapter()
        adapter._ib = MagicMock()

        pos1 = MagicMock()
        pos1.contract.symbol = "SPY"
        pos1.contract.currency = "USD"
        pos1.position = 5

        pos2 = MagicMock()
        pos2.contract.symbol = "HSBA"
        pos2.contract.currency = "GBP"
        pos2.position = 200

        pos3 = MagicMock()
        pos3.contract.symbol = "TSLA"
        pos3.contract.currency = "USD"
        pos3.position = 3

        adapter._ib.positions.return_value = [pos1, pos2, pos3]

        positions = adapter.get_open_positions()
        assert positions == {"SPY": 5, "HSBA.L": 200, "TSLA": 3}

    # -- execute.py integration -----------------------------------------------

    def _make_signal_dir(self, data_dir, ticker, flag, close=200.0, kelly=0.15):
        run_dir = data_dir / f"{ticker}_20260701T120000Z"
        run_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([{
            "close": close, "kelly_fraction": kelly,
            "stop_level": close * 0.95, "target_level": close * 1.15,
            "trade_event": flag,
        }])
        df.index.name = "date"
        df.to_csv(run_dir / "compositeBacktest.csv")
        (run_dir / "qualityGate.json").write_text(
            json.dumps({"flag": flag}), encoding="utf-8"
        )

    def _make_watchlist(self, path, tickers, capital_pot=20000, max_positions=5):
        path.write_text(json.dumps({
            "defaults": {"capital_pot": capital_pot, "max_positions": max_positions},
            "tickers": [{"ticker": t} for t in tickers],
        }), encoding="utf-8")

    def test_execute_buy_places_order_and_returns_zero(self, tmp_path):
        from Strategy_Auto_Trader.markov_cli.execute import main
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        self._make_signal_dir(data_dir, "AAPL", "BUY", close=200.0, kelly=0.15)
        wl = tmp_path / "watchlist.json"
        self._make_watchlist(wl, ["AAPL"])
        rc = main([
            "--dry-run",
            "--data-dir", str(data_dir),
            "--watchlist", str(wl),
            "--state-dir", str(tmp_path),
        ])
        assert rc == 0

    def test_execute_sell_closes_existing_position(self, tmp_path):
        from Strategy_Auto_Trader.markov_cli.execute import main
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        self._make_signal_dir(data_dir, "AAPL", "SELL", close=210.0)
        (tmp_path / "execution_state.json").write_text(json.dumps({
            "positions": {"AAPL": {
                "entry_date": "2026-06-15", "fill_price": 195.0,
                "quantity": 3, "kelly_fraction": 0.15,
                "stop_level": 185.0, "target_level": 224.0,
            }},
            "trade_log": [],
        }), encoding="utf-8")
        wl = tmp_path / "watchlist.json"
        self._make_watchlist(wl, ["AAPL"])
        rc = main([
            "--dry-run",
            "--data-dir", str(data_dir),
            "--watchlist", str(wl),
            "--state-dir", str(tmp_path),
        ])
        assert rc == 0

    def test_execute_dry_run_does_not_write_state(self, tmp_path):
        from Strategy_Auto_Trader.markov_cli.execute import main
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        self._make_signal_dir(data_dir, "AAPL", "BUY")
        wl = tmp_path / "watchlist.json"
        self._make_watchlist(wl, ["AAPL"])
        state_file = tmp_path / "execution_state.json"
        assert not state_file.exists()
        main([
            "--dry-run",
            "--data-dir", str(data_dir),
            "--watchlist", str(wl),
            "--state-dir", str(tmp_path),
        ])
        assert not state_file.exists()

    def test_execute_at_max_positions_skips_buy(self, tmp_path):
        from Strategy_Auto_Trader.markov_cli.execute import main
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        self._make_signal_dir(data_dir, "AAPL", "BUY")
        (tmp_path / "execution_state.json").write_text(json.dumps({
            "positions": {"MSFT": {
                "entry_date": "2026-06-01", "fill_price": 300.0,
                "quantity": 2, "kelly_fraction": 0.10,
                "stop_level": 285.0, "target_level": 345.0,
            }},
            "trade_log": [],
            "trades_today": {"date": "2026-07-02", "buys": 0, "sells": 0},
        }), encoding="utf-8")
        wl = tmp_path / "watchlist.json"
        self._make_watchlist(wl, ["AAPL"], max_positions=1)
        rc = main([
            "--dry-run",
            "--data-dir", str(data_dir),
            "--watchlist", str(wl),
            "--state-dir", str(tmp_path),
        ])
        assert rc == 0

    # -- Daily limits ---------------------------------------------------------

    def test_daily_limit_blocks_excess_buys(self, tmp_path):
        from Strategy_Auto_Trader.markov_cli.execute import main
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        self._make_signal_dir(data_dir, "AAPL", "BUY", kelly=0.15)
        self._make_signal_dir(data_dir, "MSFT", "BUY", kelly=0.14)
        self._make_signal_dir(data_dir, "GOOGL", "BUY", kelly=0.13)
        wl = tmp_path / "watchlist.json"
        self._make_watchlist(wl, ["AAPL", "MSFT", "GOOGL"])
        (tmp_path / "execution_state.json").write_text(json.dumps({
            "positions": {},
            "trade_log": [],
            "trades_today": {"date": "2026-07-02", "buys": 0, "sells": 0},
        }), encoding="utf-8")
        rc = main([
            "--dry-run",
            "--data-dir", str(data_dir),
            "--watchlist", str(wl),
            "--state-dir", str(tmp_path),
        ])
        assert rc == 0
        state = json.loads((tmp_path / "execution_state.json").read_text())
        assert state["trades_today"]["buys"] == 0  # dry-run doesn't save

    def test_daily_limit_ranks_by_kelly_fraction(self, tmp_path):
        from Strategy_Auto_Trader.markov_cli.execute import main
        from unittest.mock import patch, MagicMock
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        self._make_signal_dir(data_dir, "WEAK", "BUY", kelly=0.08)
        self._make_signal_dir(data_dir, "STRONG", "BUY", kelly=0.20)
        self._make_signal_dir(data_dir, "MEDIUM", "BUY", kelly=0.15)
        wl = tmp_path / "watchlist.json"
        self._make_watchlist(wl, ["WEAK", "STRONG", "MEDIUM"])
        state_file = tmp_path / "execution_state.json"
        state_file.write_text(json.dumps({
            "positions": {},
            "trade_log": [],
            "trades_today": {"date": "2026-07-02", "buys": 0, "sells": 0},
        }), encoding="utf-8")
        with patch("Strategy_Auto_Trader.broker.ibkr_adapter.IBKRAdapter") as MockAdapter:
            mock_broker = MagicMock()
            mock_broker.managed_accounts.return_value = ["DU123456"]
            MockAdapter.return_value = mock_broker
            from Strategy_Auto_Trader.broker.types import FillResult
            def fill_side_effect(req):
                return FillResult(req.ticker, req.action, 200.0, req.quantity, "2026-07-02T12:00:00Z")
            mock_broker.place_order.side_effect = fill_side_effect
            rc = main([
                "--data-dir", str(data_dir),
                "--watchlist", str(wl),
                "--state-dir", str(tmp_path),
            ])
            assert rc == 0
            state = json.loads(state_file.read_text())
            buys = [t["ticker"] for t in state["trade_log"] if t["action"] == "BUY"]
            assert buys == ["STRONG", "MEDIUM"]
            assert "WEAK" not in buys

    def test_daily_limit_resets_at_midnight(self):
        from Strategy_Auto_Trader.broker.daily_limits import DailyLimitTracker
        from datetime import datetime, timezone, timedelta
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
        state = {
            "positions": {},
            "trades_today": {"date": yesterday, "buys": 5, "sells": 3},
        }
        tracker = DailyLimitTracker(state)
        assert tracker.can_buy(2)
        assert state["trades_today"]["buys"] == 0
        assert state["trades_today"]["date"] == datetime.now(timezone.utc).date().isoformat()

    def test_unlimited_sells_by_default(self, tmp_path):
        from Strategy_Auto_Trader.markov_cli.execute import main
        from unittest.mock import patch, MagicMock
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        for i in range(5):
            self._make_signal_dir(data_dir, f"T{i}", "SELL")
        wl = tmp_path / "watchlist.json"
        self._make_watchlist(wl, [f"T{i}" for i in range(5)])
        state_file = tmp_path / "execution_state.json"
        state_file.write_text(json.dumps({
            "positions": {
                f"T{i}": {
                    "entry_date": "2026-07-01",
                    "fill_price": 100.0 + i,
                    "quantity": 5,
                    "kelly_fraction": 0.10,
                    "stop_level": 90.0,
                    "target_level": 110.0,
                }
                for i in range(5)
            },
            "trade_log": [],
            "trades_today": {"date": "2026-07-02", "buys": 0, "sells": 0},
        }), encoding="utf-8")
        with patch("Strategy_Auto_Trader.broker.ibkr_adapter.IBKRAdapter") as MockAdapter:
            mock_broker = MagicMock()
            mock_broker.managed_accounts.return_value = ["DU123456"]
            MockAdapter.return_value = mock_broker
            from Strategy_Auto_Trader.broker.types import FillResult
            def fill_side_effect(req):
                return FillResult(req.ticker, req.action, 110.0, req.quantity, "2026-07-02T12:00:00Z")
            mock_broker.place_order.side_effect = fill_side_effect
            rc = main([
                "--data-dir", str(data_dir),
                "--watchlist", str(wl),
                "--state-dir", str(tmp_path),
            ])
            assert rc == 0
            state = json.loads(state_file.read_text())
            sells = [t for t in state["trade_log"] if t["action"] == "SELL"]
            assert len(sells) == 5

    def test_execute_main_refuses_real_run_when_self_check_fails(self, tmp_path):
        from unittest.mock import patch
        from Strategy_Auto_Trader.core.self_check import SelfCheckError
        from Strategy_Auto_Trader.markov_cli.execute import main
        wl = tmp_path / "watchlist.json"
        self._make_watchlist(wl, ["AAPL"])
        with patch("Strategy_Auto_Trader.core.self_check.run_startup_checks",
                   side_effect=SelfCheckError("ib_insync broken")):
            rc = main([
                "--data-dir", str(tmp_path),
                "--watchlist", str(wl),
                "--state-dir", str(tmp_path),
            ])
        assert rc == 1
        assert not (tmp_path / "execution_state.json").exists()

    def test_execute_main_dry_run_skips_broker_self_check(self, tmp_path):
        from unittest.mock import patch
        from Strategy_Auto_Trader.markov_cli.execute import main
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        wl = tmp_path / "watchlist.json"
        self._make_watchlist(wl, [])
        with patch("Strategy_Auto_Trader.core.self_check.run_startup_checks",
                   side_effect=AssertionError("must not run in dry-run")):
            rc = main([
                "--dry-run",
                "--data-dir", str(data_dir),
                "--watchlist", str(wl),
                "--state-dir", str(tmp_path),
            ])
        assert rc == 0

    # -- BVA: PortfolioManager.compute_quantity() -----

    def test_bva_compute_quantity_zero_kelly(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        qty = pm.compute_quantity(0.0, 200.0)
        assert qty == 0

    def test_bva_compute_quantity_negative_kelly(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        qty = pm.compute_quantity(-0.1, 200.0)
        assert qty == 0

    def test_bva_compute_quantity_very_high_kelly(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        qty_normal = pm.compute_quantity(0.20, 200.0)
        qty_high = pm.compute_quantity(2.0, 200.0)
        assert qty_high > qty_normal

    def test_bva_compute_quantity_zero_price(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        qty = pm.compute_quantity(0.15, 0.0)
        assert qty == 0

    def test_bva_compute_quantity_negative_price(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        qty = pm.compute_quantity(0.15, -100.0)
        assert qty == 0

    def test_bva_compute_quantity_very_small_price(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        qty = pm.compute_quantity(0.15, 0.01)
        assert qty >= 1

    def test_bva_compute_quantity_very_large_price(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        qty = pm.compute_quantity(0.15, 50_000.0)
        assert qty == 1

    # -- BVA: PortfolioManager constructor -----

    def test_bva_portfolio_zero_capital(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        pm = PortfolioManager(0.0, 5, tmp_path / "state.json")
        qty = pm.compute_quantity(0.15, 200.0)
        assert qty == 1

    def test_bva_portfolio_negative_capital(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        pm = PortfolioManager(-1000.0, 5, tmp_path / "state.json")
        assert pm._capital_pot == -1000.0

    def test_bva_portfolio_zero_max_positions(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        pm = PortfolioManager(20_000, 0, tmp_path / "state.json")
        assert not pm.can_open("AAPL")

    def test_bva_portfolio_one_position_capacity(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 1, tmp_path / "state.json")
        assert pm.can_open("AAPL")
        fill = FillResult("AAPL", "BUY", 195.0, 10, "2026-07-01T00:00:00+00:00")
        pm.record_entry("AAPL", fill, 0.10, 185.0, 224.0)
        assert not pm.can_open("MSFT")
        assert not pm.can_open("AAPL")

    def test_bva_portfolio_very_large_max_positions(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        pm = PortfolioManager(1_000_000, 1000, tmp_path / "state.json")
        assert pm._max_positions == 1000
        assert pm.can_open("AAPL")

    def test_bva_portfolio_very_large_capital_pot(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        pm = PortfolioManager(1_000_000_000.0, 5, tmp_path / "state.json")
        qty = pm.compute_quantity(0.15, 200.0)
        assert qty >= 1

    # -- BVA: DailyLimitTracker -----

    def test_bva_daily_limit_zero_buys(self):
        from Strategy_Auto_Trader.broker.daily_limits import DailyLimitTracker
        state = {"trades_today": {"date": "2026-07-02", "buys": 0, "sells": 0}}
        tracker = DailyLimitTracker(state)
        assert not tracker.can_buy(0)
        assert not tracker.can_buy(0)

    def test_bva_daily_limit_one_buy(self):
        from Strategy_Auto_Trader.broker.daily_limits import DailyLimitTracker
        state = {"trades_today": {"date": "2026-07-02", "buys": 0, "sells": 0}}
        tracker = DailyLimitTracker(state)
        assert tracker.can_buy(1)
        tracker.record_buy()
        assert not tracker.can_buy(1)

    def test_bva_daily_limit_zero_sells(self):
        from Strategy_Auto_Trader.broker.daily_limits import DailyLimitTracker
        state = {"trades_today": {"date": "2026-07-02", "buys": 0, "sells": 0}}
        tracker = DailyLimitTracker(state)
        assert not tracker.can_sell(0)

    def test_bva_daily_limit_very_high_limit(self):
        from Strategy_Auto_Trader.broker.daily_limits import DailyLimitTracker
        state = {"trades_today": {"date": "2026-07-02", "buys": 0, "sells": 0}}
        tracker = DailyLimitTracker(state)
        assert tracker.can_buy(999)
        for _ in range(100):
            tracker.record_buy()
        assert tracker.can_buy(999)

    def test_bva_daily_limit_negative_limit_treated_as_unlimited(self):
        from Strategy_Auto_Trader.broker.daily_limits import DailyLimitTracker
        state = {"trades_today": {"date": "2026-07-02", "buys": 0, "sells": 0}}
        tracker = DailyLimitTracker(state)
        assert tracker.can_buy(None)
        assert tracker.can_sell(None)

    def test_bva_daily_limit_counts_at_boundary(self):
        from Strategy_Auto_Trader.broker.daily_limits import DailyLimitTracker
        state = {"trades_today": {"date": "2026-07-02", "buys": 0, "sells": 0}}
        tracker = DailyLimitTracker(state)
        for _ in range(5):
            tracker.record_buy()
            tracker.record_sell()
        buys, sells = tracker.get_today_counts()
        assert buys == 5
        assert sells == 5

    # -- BVA: OrderRequest/FillResult -----

    def test_bva_order_request_zero_quantity(self):
        from Strategy_Auto_Trader.broker.types import OrderRequest
        req = OrderRequest("AAPL", "BUY", 0)
        assert req.quantity == 0

    def test_bva_order_request_negative_quantity(self):
        from Strategy_Auto_Trader.broker.types import OrderRequest
        req = OrderRequest("AAPL", "BUY", -10)
        assert req.quantity == -10

    def test_bva_order_request_one_share(self):
        from Strategy_Auto_Trader.broker.types import OrderRequest
        req = OrderRequest("AAPL", "BUY", 1)
        assert req.quantity == 1

    def test_bva_order_request_very_large_quantity(self):
        from Strategy_Auto_Trader.broker.types import OrderRequest
        req = OrderRequest("AAPL", "BUY", 999_999_999)
        assert req.quantity == 999_999_999

    def test_bva_fill_result_zero_fill_price(self):
        from Strategy_Auto_Trader.broker.types import FillResult
        fill = FillResult("AAPL", "BUY", 0.0, 10, "2026-07-02T12:00:00Z")
        assert fill.fill_price == 0.0

    def test_bva_fill_result_negative_fill_price(self):
        from Strategy_Auto_Trader.broker.types import FillResult
        fill = FillResult("AAPL", "BUY", -195.0, 10, "2026-07-02T12:00:00Z")
        assert fill.fill_price == -195.0

    def test_bva_fill_result_very_small_price(self):
        from Strategy_Auto_Trader.broker.types import FillResult
        fill = FillResult("AAPL", "BUY", 0.001, 10, "2026-07-02T12:00:00Z")
        assert fill.fill_price == pytest.approx(0.001)

    def test_bva_fill_result_very_large_price(self):
        from Strategy_Auto_Trader.broker.types import FillResult
        fill = FillResult("AAPL", "BUY", 999_999.99, 10, "2026-07-02T12:00:00Z")
        assert fill.fill_price == pytest.approx(999_999.99)

    def test_bva_fill_result_zero_quantity(self):
        from Strategy_Auto_Trader.broker.types import FillResult
        fill = FillResult("AAPL", "BUY", 195.0, 0, "2026-07-02T12:00:00Z")
        assert fill.quantity == 0

    def test_bva_fill_result_negative_quantity(self):
        from Strategy_Auto_Trader.broker.types import FillResult
        fill = FillResult("AAPL", "BUY", 195.0, -10, "2026-07-02T12:00:00Z")
        assert fill.quantity == -10

    # -- BVA: PortfolioManager.can_open() -----

    def test_bva_can_open_empty_portfolio(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        assert len(pm.positions) == 0
        assert pm.can_open("AAPL")

    def test_bva_can_open_at_max_minus_one(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 3, tmp_path / "state.json")
        for i, ticker in enumerate(["A", "B"]):
            fill = FillResult(ticker, "BUY", 100.0 + i, 10, "2026-07-01T00:00:00+00:00")
            pm.record_entry(ticker, fill, 0.10, 90.0, 110.0)
        assert len(pm.positions) == 2
        assert pm.can_open("C")

    def test_bva_can_open_at_max_exact(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 2, tmp_path / "state.json")
        for i, ticker in enumerate(["A", "B"]):
            fill = FillResult(ticker, "BUY", 100.0 + i, 10, "2026-07-01T00:00:00+00:00")
            pm.record_entry(ticker, fill, 0.10, 90.0, 110.0)
        assert len(pm.positions) == 2
        assert not pm.can_open("C")

    def test_bva_can_open_duplicate_ticker_in_positions(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        fill = FillResult("AAPL", "BUY", 195.0, 10, "2026-07-01T00:00:00+00:00")
        pm.record_entry("AAPL", fill, 0.10, 185.0, 224.0)
        assert not pm.can_open("AAPL")

    # -- BVA: PortfolioManager P&L calculations -----

    def test_bva_record_exit_zero_position_price(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        buy = FillResult("AAPL", "BUY", 100.0, 10, "2026-07-01T00:00:00+00:00")
        pm.record_entry("AAPL", buy, 0.10, 95.0, 105.0)
        sell = FillResult("AAPL", "SELL", 0.0, 10, "2026-07-02T00:00:00+00:00")
        pm.record_exit("AAPL", sell)
        pl = pm.trade_log[-1]["gross_pl"]
        assert pl == pytest.approx(-1000.0)

    def test_bva_record_exit_negative_price(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        buy = FillResult("AAPL", "BUY", 100.0, 10, "2026-07-01T00:00:00+00:00")
        pm.record_entry("AAPL", buy, 0.10, 95.0, 105.0)
        sell = FillResult("AAPL", "SELL", -50.0, 10, "2026-07-02T00:00:00+00:00")
        pm.record_exit("AAPL", sell)
        pl = pm.trade_log[-1]["gross_pl"]
        assert pl == pytest.approx(-1500.0)

    def test_bva_record_exit_huge_profit(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        buy = FillResult("AAPL", "BUY", 100.0, 10, "2026-07-01T00:00:00+00:00")
        pm.record_entry("AAPL", buy, 0.10, 95.0, 105.0)
        sell = FillResult("AAPL", "SELL", 10_000.0, 10, "2026-07-02T00:00:00+00:00")
        pm.record_exit("AAPL", sell)
        pl = pm.trade_log[-1]["gross_pl"]
        assert pl == pytest.approx(99_000.0)

    def test_bva_record_exit_one_share(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        buy = FillResult("AAPL", "BUY", 100.0, 1, "2026-07-01T00:00:00+00:00")
        pm.record_entry("AAPL", buy, 0.10, 95.0, 105.0)
        sell = FillResult("AAPL", "SELL", 110.0, 1, "2026-07-02T00:00:00+00:00")
        pm.record_exit("AAPL", sell)
        pl = pm.trade_log[-1]["gross_pl"]
        assert pl == pytest.approx(10.0)

    def test_bva_record_exit_many_shares(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        buy = FillResult("AAPL", "BUY", 100.0, 100_000, "2026-07-01T00:00:00+00:00")
        pm.record_entry("AAPL", buy, 0.10, 95.0, 105.0)
        sell = FillResult("AAPL", "SELL", 101.0, 100_000, "2026-07-02T00:00:00+00:00")
        pm.record_exit("AAPL", sell)
        pl = pm.trade_log[-1]["gross_pl"]
        assert pl == pytest.approx(100_000.0)

    # -- BVA: Decimal Precision & Scale -----

    def test_bva_fill_price_two_decimals(self):
        from Strategy_Auto_Trader.broker.types import FillResult
        fill = FillResult("AAPL", "BUY", 195.50, 10, "2026-07-02T12:00:00Z")
        assert fill.fill_price == pytest.approx(195.50)

    def test_bva_fill_price_three_decimals(self):
        from Strategy_Auto_Trader.broker.types import FillResult
        fill = FillResult("AAPL", "BUY", 195.505, 10, "2026-07-02T12:00:00Z")
        assert fill.fill_price == pytest.approx(195.505)

    def test_bva_fill_price_many_decimals(self):
        from Strategy_Auto_Trader.broker.types import FillResult
        fill = FillResult("AAPL", "BUY", 195.123456789, 10, "2026-07-02T12:00:00Z")
        assert fill.fill_price == pytest.approx(195.123456789)

    def test_bva_kelly_fraction_zero_decimals(self):
        from Strategy_Auto_Trader.broker.types import PositionRecord
        pos = PositionRecord("2026-07-01", 195.0, 10, 0.0, 185.0, 224.0)
        assert pos.kelly_fraction == 0.0

    def test_bva_kelly_fraction_three_decimals(self):
        from Strategy_Auto_Trader.broker.types import PositionRecord
        pos = PositionRecord("2026-07-01", 195.0, 10, 0.157, 185.0, 224.0)
        assert pos.kelly_fraction == pytest.approx(0.157)

    def test_bva_kelly_fraction_many_decimals(self):
        from Strategy_Auto_Trader.broker.types import PositionRecord
        pos = PositionRecord("2026-07-01", 195.0, 10, 0.15123456789, 185.0, 224.0)
        assert pos.kelly_fraction == pytest.approx(0.15123456789)

    def test_bva_p_l_decimal_precision(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        buy = FillResult("AAPL", "BUY", 195.555, 3, "2026-07-01T00:00:00+00:00")
        pm.record_entry("AAPL", buy, 0.10, 185.0, 224.0)
        sell = FillResult("AAPL", "SELL", 200.123, 3, "2026-07-02T00:00:00+00:00")
        pm.record_exit("AAPL", sell)
        pl = pm.trade_log[-1]["gross_pl"]
        expected_pl = (200.123 - 195.555) * 3
        assert pl == pytest.approx(expected_pl, abs=0.01)

    def test_bva_very_small_decimal_price(self):
        from Strategy_Auto_Trader.broker.types import FillResult
        fill = FillResult("PENNY", "BUY", 0.0001, 1000, "2026-07-02T12:00:00Z")
        assert fill.fill_price == pytest.approx(0.0001)

    def test_bva_very_large_decimal_price(self):
        from Strategy_Auto_Trader.broker.types import FillResult
        fill = FillResult("EXPENSIVE", "BUY", 999999.99999, 1, "2026-07-02T12:00:00Z")
        assert fill.fill_price == pytest.approx(999999.99999)

    # -- BVA: String Length Boundaries -----

    def test_bva_order_request_short_ticker(self):
        from Strategy_Auto_Trader.broker.types import OrderRequest
        req = OrderRequest("A", "BUY", 10)
        assert req.ticker == "A"

    def test_bva_order_request_normal_ticker(self):
        from Strategy_Auto_Trader.broker.types import OrderRequest
        req = OrderRequest("AAPL", "BUY", 10)
        assert req.ticker == "AAPL"

    def test_bva_order_request_long_ticker(self):
        from Strategy_Auto_Trader.broker.types import OrderRequest
        req = OrderRequest("LSEG.L", "BUY", 10)
        assert req.ticker == "LSEG.L"

    def test_bva_order_request_very_long_ticker(self):
        from Strategy_Auto_Trader.broker.types import OrderRequest
        long_ticker = "A" * 50
        req = OrderRequest(long_ticker, "BUY", 10)
        assert req.ticker == long_ticker

    def test_bva_order_action_buy(self):
        from Strategy_Auto_Trader.broker.types import OrderRequest
        req = OrderRequest("AAPL", "BUY", 10)
        assert req.action == "BUY"

    def test_bva_order_action_sell(self):
        from Strategy_Auto_Trader.broker.types import OrderRequest
        req = OrderRequest("AAPL", "SELL", 10)
        assert req.action == "SELL"

    def test_bva_order_action_long_string(self):
        from Strategy_Auto_Trader.broker.types import OrderRequest
        long_action = "B" * 100
        req = OrderRequest("AAPL", long_action, 10)
        assert req.action == long_action

    def test_bva_fill_result_timestamp_format(self):
        from Strategy_Auto_Trader.broker.types import FillResult
        timestamp = "2026-07-02T12:00:00Z"
        fill = FillResult("AAPL", "BUY", 195.0, 10, timestamp)
        assert fill.timestamp == timestamp

    def test_bva_fill_result_long_timestamp(self):
        from Strategy_Auto_Trader.broker.types import FillResult
        long_timestamp = "2026-07-02T12:00:00.123456789+00:00"
        fill = FillResult("AAPL", "BUY", 195.0, 10, long_timestamp)
        assert fill.timestamp == long_timestamp

    def test_bva_position_record_entry_date_format(self):
        from Strategy_Auto_Trader.broker.types import PositionRecord
        pos = PositionRecord("2026-07-01", 195.0, 10, 0.10, 185.0, 224.0)
        assert pos.entry_date == "2026-07-01"

    def test_bva_position_record_long_entry_date(self):
        from Strategy_Auto_Trader.broker.types import PositionRecord
        long_date = "2026-07-01T12:00:00.123456789+00:00"
        pos = PositionRecord(long_date, 195.0, 10, 0.10, 185.0, 224.0)
        assert pos.entry_date == long_date

    # -- BVA: Null/None Handling -----

    def test_bva_daily_limit_none_buy_limit(self):
        from Strategy_Auto_Trader.broker.daily_limits import DailyLimitTracker
        state = {"trades_today": {"date": "2026-07-02", "buys": 0, "sells": 0}}
        tracker = DailyLimitTracker(state)
        assert tracker.can_buy(None)
        assert tracker.can_buy(None)

    def test_bva_daily_limit_none_sell_limit(self):
        from Strategy_Auto_Trader.broker.daily_limits import DailyLimitTracker
        state = {"trades_today": {"date": "2026-07-02", "buys": 0, "sells": 0}}
        tracker = DailyLimitTracker(state)
        assert tracker.can_sell(None)
        for _ in range(100):
            tracker.record_sell()
        assert tracker.can_sell(None)

    def test_bva_portfolio_missing_state_file(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        nonexistent = tmp_path / "nonexistent" / "state.json"
        pm = PortfolioManager(20_000, 5, nonexistent)
        assert len(pm.positions) == 0
        assert len(pm.trade_log) == 0

    def test_bva_daily_tracker_missing_date_key(self):
        from Strategy_Auto_Trader.broker.daily_limits import DailyLimitTracker
        state = {}
        DailyLimitTracker(state)
        assert "trades_today" in state
        assert "date" in state["trades_today"]

    def test_bva_empty_positions_dict(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        assert isinstance(pm.positions, dict)
        assert len(pm.positions) == 0

    def test_bva_empty_trade_log_list(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        assert isinstance(pm.trade_log, list)
        assert len(pm.trade_log) == 0

    # -- BVA: Type Coercion & Edge Cases -----

    def test_bva_compute_quantity_float_kelly(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        qty1 = pm.compute_quantity(0.15, 200.0)
        qty2 = pm.compute_quantity(0.150, 200.0)
        assert qty1 == qty2

    def test_bva_compute_quantity_float_price(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        qty1 = pm.compute_quantity(0.15, 200.0)
        qty2 = pm.compute_quantity(0.15, 200.00)
        assert qty1 == qty2

    def test_bva_p_l_rounding_precision(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        buy = FillResult("AAPL", "BUY", 100.123, 7, "2026-07-01T00:00:00+00:00")
        pm.record_entry("AAPL", buy, 0.10, 95.0, 110.0)
        sell = FillResult("AAPL", "SELL", 100.456, 7, "2026-07-02T00:00:00+00:00")
        pm.record_exit("AAPL", sell)
        pl = pm.trade_log[-1]["gross_pl"]
        expected = round((100.456 - 100.123) * 7, 2)
        assert pl == expected

    def test_bva_daily_limit_boundary_transition(self):
        from Strategy_Auto_Trader.broker.daily_limits import DailyLimitTracker
        state = {"trades_today": {"date": "2026-07-02", "buys": 0, "sells": 0}}
        tracker = DailyLimitTracker(state)
        assert tracker.can_buy(2)
        tracker.record_buy()
        assert tracker.can_buy(2)
        tracker.record_buy()
        assert not tracker.can_buy(2)
        buys, _ = tracker.get_today_counts()
        assert buys == 2

    # -- Phase 0: market / currency / cost_value -----

    def test_portfolio_record_entry_with_market_currency(self, tmp_path):
        """record_entry stores market and currency fields."""
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        fill = FillResult("LLOY.L", "BUY", 450.0, 10, "2026-07-01T00:00:00+00:00")
        pm.record_entry("LLOY.L", fill, 0.12, 425.0, 515.0, market="ftse", currency="GBP")
        assert "LLOY.L" in pm.positions
        pos = pm.positions["LLOY.L"]
        assert pos["market"] == "ftse"
        assert pos["currency"] == "GBP"
        assert pos["cost_value"] == pytest.approx(4500.0)  # 450 * 10

    def test_portfolio_record_entry_cost_value_computed(self, tmp_path):
        """record_entry computes cost_value from fill_price and quantity."""
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        fill = FillResult("AAPL", "BUY", 123.45, 17, "2026-07-01T00:00:00+00:00")
        pm.record_entry("AAPL", fill, 0.10, 100.0, 150.0, market="sp500", currency="USD")
        pos = pm.positions["AAPL"]
        assert pos["cost_value"] == pytest.approx(123.45 * 17)

    # -- Commission wiring: record_entry/record_exit net real IBKR fees against pl -----

    def test_record_entry_stores_entry_cost_us_ticker(self, tmp_path):
        """US ticker routes to the USD tier: trade value 1000 -> below the
        USD 1.70 min, so entry_cost is the converted USD minimum."""
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        from Strategy_Auto_Trader.plugins.costs import USD_GBP
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        fill = FillResult("AAPL", "BUY", 100.0, 10, "2026-07-01T00:00:00+00:00")
        pm.record_entry("AAPL", fill, 0.10, 95.0, 105.0)
        assert pm.positions["AAPL"]["entry_cost"] == pytest.approx(1.70 * USD_GBP)
        assert pm.trade_log[-1]["cost"] == pytest.approx(1.70 * USD_GBP, abs=0.01)

    def test_record_entry_stores_entry_cost_uk_ticker_includes_stamp_duty(self, tmp_path):
        """UK ticker BUY adds 0.5% stamp duty on top of the tiered commission."""
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        fill = FillResult("LLOY.L", "BUY", 450.0, 10, "2026-07-01T00:00:00+00:00")
        pm.record_entry("LLOY.L", fill, 0.12, 425.0, 515.0, market="ftse", currency="GBP")
        trade_value = 450.0 * 10
        expected_cost = max(1.00, 0.0005 * trade_value) + 0.005 * trade_value
        assert pm.positions["LLOY.L"]["entry_cost"] == pytest.approx(expected_cost)

    def test_record_exit_nets_commission_off_both_legs(self, tmp_path):
        """pl is net of entry + exit commission; gross_pl keeps the raw
        (fill - entry) * qty figure for audit."""
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        from Strategy_Auto_Trader.plugins.costs import IbkrTieredCost
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        buy = FillResult("AAPL", "BUY", 300.0, 10, "2026-07-01T00:00:00+00:00")
        pm.record_entry("AAPL", buy, 0.10, 285.0, 345.0)
        sell = FillResult("AAPL", "SELL", 330.0, 10, "2026-07-10T00:00:00+00:00")
        pm.record_exit("AAPL", sell)
        entry_cost = IbkrTieredCost("AAPL").cost(3000.0, is_buy=True)
        exit_cost = IbkrTieredCost("AAPL").cost(3300.0, is_buy=False)
        last = pm.trade_log[-1]
        assert last["gross_pl"] == pytest.approx(300.0)
        assert last["cost"] == pytest.approx(entry_cost + exit_cost)
        assert last["pl"] == pytest.approx(300.0 - entry_cost - exit_cost)

    def test_portfolio_record_entry_empty_market_currency_allowed(self, tmp_path):
        """record_entry works with empty market/currency (for backward compat)."""
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000, 5, tmp_path / "state.json")
        fill = FillResult("SPY", "BUY", 500.0, 4, "2026-07-01T00:00:00+00:00")
        pm.record_entry("SPY", fill, 0.10, 475.0, 575.0)  # No market/currency
        assert "SPY" in pm.positions
        pos = pm.positions["SPY"]
        assert pos.get("market") == ""
        assert pos.get("currency") == ""
        assert pos["cost_value"] == pytest.approx(2000.0)

    def test_portfolio_save_and_reload_preserves_market_currency(self, tmp_path):
        """Positions with market/currency round-trip through save/load."""
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        path = tmp_path / "state.json"
        pm = PortfolioManager(20_000, 5, path)
        fill = FillResult("BP.L", "BUY", 350.0, 15, "2026-07-01T00:00:00+00:00")
        pm.record_entry("BP.L", fill, 0.15, 325.0, 400.0, market="ftse", currency="GBP")
        pm.save()

        pm2 = PortfolioManager(20_000, 5, path)
        assert "BP.L" in pm2.positions
        pos = pm2.positions["BP.L"]
        assert pos["market"] == "ftse"
        assert pos["currency"] == "GBP"
        assert pos["cost_value"] == pytest.approx(5250.0)

    # -- execute_signals: stop/target slippage adjustment (BUGFIX_PLAN) -----

    def _patched_signal_reader(self, monkeypatch, signal):
        """Patch signal_reader.read_latest_signal so execute_signals gets a
        fixed signal dict regardless of ticker/data_dir (its local import
        `from ..broker.signal_reader import read_latest_signal` re-reads the
        module attribute on every call, so this patch is picked up live)."""
        import Strategy_Auto_Trader.broker.signal_reader as signal_reader
        monkeypatch.setattr(signal_reader, "read_latest_signal",
                             lambda ticker, data_dir: signal)

    def test_execute_signals_mild_slippage_recomputes_stop_target(self, tmp_path, monkeypatch):
        """Normal fill close to signal price: stop/target recomputed off the
        fill price, not left as the raw signal-time levels."""
        from Strategy_Auto_Trader.markov_cli.execute import execute_signals
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        signal = {"flag": "BUY", "close": 100.0, "kelly_fraction": 0.15,
                  "stop_level": 95.0, "target_level": 115.0}
        self._patched_signal_reader(monkeypatch, signal)
        portfolio = PortfolioManager(20_000, 5, tmp_path / "state.json")
        broker = NullBroker(prices={"AAPL": 100.2})  # mild slippage above close
        buys, sells, skipped = execute_signals(
            ["AAPL"], tmp_path, portfolio, portfolio.get_limit_tracker(), broker,
        )
        pos = portfolio.positions["AAPL"]
        assert pos["stop_level"] != pytest.approx(95.0)
        assert pos["target_level"] != pytest.approx(115.0)
        assert pos["stop_level"] == pytest.approx(100.2 * 0.95)
        assert pos["target_level"] == pytest.approx(100.2 * 1.15)
        assert not any("SEVERE SLIPPAGE" in b for b in buys)

    def test_execute_signals_slippage_at_stop_boundary_triggers_immediate_stopout(self, tmp_path, monkeypatch):
        """Fill price exactly equal to the ORIGINAL signal-time stop_level
        (95.0) must trigger the same-bar stop-out path (<=), and the exit
        must log a real nonzero negative P&L (regression test for the
        zero-P&L bug: exit no longer reuses the entry fill object)."""
        from Strategy_Auto_Trader.markov_cli.execute import execute_signals
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        signal = {"flag": "BUY", "close": 100.0, "kelly_fraction": 0.15,
                  "stop_level": 95.0, "target_level": 115.0}
        self._patched_signal_reader(monkeypatch, signal)
        portfolio = PortfolioManager(20_000, 5, tmp_path / "state.json")
        broker = NullBroker(prices={"AAPL": 95.0})  # fills exactly at signal stop
        buys, sells, skipped = execute_signals(
            ["AAPL"], tmp_path, portfolio, portfolio.get_limit_tracker(), broker,
        )
        assert "AAPL" not in portfolio.positions
        assert any("SEVERE SLIPPAGE" in b for b in buys)
        last = portfolio.trade_log[-1]
        assert last["action"] == "SELL"
        assert last["pl"] < 0.0
        assert last["gross_pl"] == pytest.approx((95.0 * 0.95 - 95.0) * last["quantity"])

    def test_execute_signals_slippage_below_stop_triggers_immediate_stopout(self, tmp_path, monkeypatch):
        """Fill price well below the original stop_level (severe slippage):
        same-bar stop-out fires and the loss magnitude in the log message
        roughly matches the actual price gap from the signal close."""
        from Strategy_Auto_Trader.markov_cli.execute import execute_signals
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        signal = {"flag": "BUY", "close": 100.0, "kelly_fraction": 0.15,
                  "stop_level": 95.0, "target_level": 115.0}
        self._patched_signal_reader(monkeypatch, signal)
        portfolio = PortfolioManager(20_000, 5, tmp_path / "state.json")
        broker = NullBroker(prices={"AAPL": 90.0})  # well below the 95 stop
        buys, sells, skipped = execute_signals(
            ["AAPL"], tmp_path, portfolio, portfolio.get_limit_tracker(), broker,
        )
        assert "AAPL" not in portfolio.positions
        matches = [b for b in buys if "SEVERE SLIPPAGE" in b]
        assert matches
        assert "-10.0%" in matches[0]  # (100 - 90) / 100 * 100
        last = portfolio.trade_log[-1]
        assert last["action"] == "SELL"
        assert last["pl"] < 0.0
        assert last["gross_pl"] == pytest.approx((90.0 * 0.95 - 90.0) * last["quantity"])

    def test_execute_signals_mild_slippage_above_stop_stays_open(self, tmp_path, monkeypatch):
        """Fill price is below signal close (negative slippage) but still
        above the original stop_level: normal path, position stays open."""
        from Strategy_Auto_Trader.markov_cli.execute import execute_signals
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker
        signal = {"flag": "BUY", "close": 100.0, "kelly_fraction": 0.15,
                  "stop_level": 95.0, "target_level": 115.0}
        self._patched_signal_reader(monkeypatch, signal)
        portfolio = PortfolioManager(20_000, 5, tmp_path / "state.json")
        broker = NullBroker(prices={"AAPL": 97.0})  # below close, above stop
        buys, sells, skipped = execute_signals(
            ["AAPL"], tmp_path, portfolio, portfolio.get_limit_tracker(), broker,
        )
        assert "AAPL" in portfolio.positions
        assert not any("SEVERE SLIPPAGE" in b for b in buys)

    def test_execute_signals_buy_skipped_when_order_not_filled(self, tmp_path, monkeypatch):
        """BUY signal with order not filled (broker returns None): position
        not opened, ticker added to skipped, no entry recorded."""
        from Strategy_Auto_Trader.markov_cli.execute import execute_signals
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from unittest.mock import MagicMock
        signal = {"flag": "BUY", "close": 100.0, "kelly_fraction": 0.15,
                  "stop_level": 95.0, "target_level": 115.0}
        self._patched_signal_reader(monkeypatch, signal)
        portfolio = PortfolioManager(20_000, 5, tmp_path / "state.json")
        broker = MagicMock()
        broker.place_order.return_value = None  # Order not filled
        buys, sells, skipped = execute_signals(
            ["AAPL"], tmp_path, portfolio, portfolio.get_limit_tracker(), broker,
        )
        assert "AAPL" not in portfolio.positions
        assert any("AAPL" in s and "order not filled" in s for s in skipped)
        assert len(buys) == 0

    def test_execute_signals_sell_skipped_when_order_not_filled(self, tmp_path, monkeypatch):
        """SELL signal with existing position, but order not filled: position
        stays open, ticker added to skipped, no exit recorded."""
        from Strategy_Auto_Trader.markov_cli.execute import execute_signals
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        from unittest.mock import MagicMock
        # Pre-seed portfolio with an open position
        portfolio = PortfolioManager(20_000, 5, tmp_path / "state.json")
        entry_fill = FillResult("AAPL", "BUY", 100.0, 10, "2026-07-01T00:00:00+00:00")
        portfolio.record_entry("AAPL", entry_fill, 0.15, 95.0, 115.0)

        signal = {"flag": "SELL", "close": 105.0, "kelly_fraction": 0.15,
                  "stop_level": 95.0, "target_level": 115.0}
        self._patched_signal_reader(monkeypatch, signal)
        broker = MagicMock()
        broker.place_order.return_value = None  # Order not filled
        buys, sells, skipped = execute_signals(
            ["AAPL"], tmp_path, portfolio, portfolio.get_limit_tracker(), broker,
        )
        # Position should still be open since SELL order did not fill
        assert "AAPL" in portfolio.positions
        assert any("AAPL" in s and "order not filled" in s for s in skipped)
        assert len(sells) == 0


class TestSlippageBps:

    def _sl(self):
        from Strategy_Auto_Trader.broker.portfolio import slippage_bps
        return slippage_bps

    # -- BVA: missing prices --------------------------------------------------

    def test_zero_signal_price_none(self):
        assert self._sl()(0.0, 100.0, "BUY") is None

    def test_zero_fill_price_none(self):
        assert self._sl()(100.0, 0.0, "BUY") is None

    def test_negative_signal_price_none(self):
        assert self._sl()(-100.0, 100.0, "SELL") is None

    # -- sign conventions: positive = cost ------------------------------------

    def test_buy_above_signal_positive(self):
        assert self._sl()(100.0, 100.5, "BUY") == 50.0

    def test_buy_below_signal_negative(self):
        assert self._sl()(100.0, 99.5, "BUY") == -50.0

    def test_sell_below_signal_positive(self):
        assert self._sl()(100.0, 99.5, "SELL") == 50.0

    def test_sell_above_signal_negative(self):
        assert self._sl()(100.0, 100.5, "SELL") == -50.0

    def test_exact_fill_zero(self):
        assert self._sl()(100.0, 100.0, "BUY") == 0.0
        assert self._sl()(100.0, 100.0, "SELL") == 0.0

    def test_rounding_one_decimal(self):
        assert self._sl()(3.0, 3.001, "BUY") == 3.3

    def test_record_entry_logs_slippage(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000.0, 5, tmp_path / "state.json")
        fill = FillResult("AAPL", "BUY", 195.5, 10, "2026-07-06T00:00:00+00:00")
        pm.record_entry("AAPL", fill, 0.10, 185.0, 224.0, signal_price=195.0)
        log = pm.trade_log[-1]
        assert log["signal_price"] == 195.0
        assert log["slippage_bps"] == pytest.approx(25.6, abs=0.1)

    def test_record_exit_logs_slippage(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000.0, 5, tmp_path / "state.json")
        buy = FillResult("MSFT", "BUY", 300.0, 10, "2026-07-01T00:00:00+00:00")
        pm.record_entry("MSFT", buy, 0.12, 285.0, 345.0, signal_price=300.0)
        sell = FillResult("MSFT", "SELL", 329.0, 10, "2026-07-06T00:00:00+00:00")
        pm.record_exit("MSFT", sell, signal_price=330.0)
        log = pm.trade_log[-1]
        assert log["signal_price"] == 330.0
        assert log["slippage_bps"] == pytest.approx(30.3, abs=0.1)

    def test_record_entry_without_signal_price_none_slippage(self, tmp_path):
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        pm = PortfolioManager(20_000.0, 5, tmp_path / "state.json")
        fill = FillResult("AAPL", "BUY", 195.0, 10, "2026-07-06T00:00:00+00:00")
        pm.record_entry("AAPL", fill, 0.10, 185.0, 224.0)
        assert pm.trade_log[-1]["slippage_bps"] is None

    # -- manual_commands: order-not-filled handling --

    def test_manual_commands_execute_sell_returns_false_when_order_not_filled(self, tmp_path):
        """_execute_sell returns (False, None, error_msg) when broker.place_order
        returns None, and does not call portfolio.record_exit."""
        from Strategy_Auto_Trader.markov_cli.manual_commands import _execute_sell
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        from unittest.mock import MagicMock
        import logging

        portfolio = PortfolioManager(20_000, 5, tmp_path / "state.json")
        entry_fill = FillResult("AAPL", "BUY", 100.0, 10, "2026-07-01T00:00:00+00:00")
        portfolio.record_entry("AAPL", entry_fill, 0.15, 95.0, 115.0)

        broker = MagicMock()
        broker.place_order.return_value = None  # Order not filled
        logger = logging.getLogger("test")

        success, fill, error_msg = _execute_sell("AAPL", portfolio, broker, logger)

        assert success is False
        assert fill is None
        assert "Order not filled" in error_msg
        # Position should still be open (record_exit was not called)
        assert "AAPL" in portfolio.positions

    def test_manual_commands_execute_sell_all_skips_ticker_when_order_not_filled(self, tmp_path):
        """_execute_sell_all continues to next ticker when broker.place_order
        returns None for one ticker, and successfully processes another."""
        from Strategy_Auto_Trader.markov_cli.manual_commands import _execute_sell_all
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.types import FillResult
        from unittest.mock import MagicMock
        import logging

        portfolio = PortfolioManager(20_000, 5, tmp_path / "state.json")
        # Add two positions
        fill_aapl = FillResult("AAPL", "BUY", 100.0, 10, "2026-07-01T00:00:00+00:00")
        portfolio.record_entry("AAPL", fill_aapl, 0.15, 95.0, 115.0)
        fill_msft = FillResult("MSFT", "BUY", 300.0, 5, "2026-07-01T00:00:00+00:00")
        portfolio.record_entry("MSFT", fill_msft, 0.10, 285.0, 345.0)

        broker = MagicMock()
        # First call (AAPL) returns None, second call (MSFT) succeeds
        def side_effect(req):
            if req.ticker == "AAPL":
                return None
            else:
                return FillResult(req.ticker, "SELL", 310.0, 5, "2026-07-02T00:00:00+00:00")
        broker.place_order.side_effect = side_effect

        logger = logging.getLogger("test")
        success, fills, summary = _execute_sell_all(portfolio, broker, logger)

        # Overall should show partial failure (not all succeeded)
        assert success is False
        # MSFT should be in fills
        assert len(fills) == 1
        assert fills[0].ticker == "MSFT"
        # AAPL should be in errors in summary
        assert "AAPL" in summary and "order not filled" in summary
        # AAPL position should still be open, MSFT should be closed
        assert "AAPL" in portfolio.positions
        assert "MSFT" not in portfolio.positions
