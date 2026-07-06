from __future__ import annotations

import json
from unittest import mock

import pytest


class TestTradeState:

    def test_record_buy_and_has_open_buy(self, tmp_path):
        from Strategy_Auto_Trader.output import trade_state
        state_file = tmp_path / "trade_state.json"
        with mock.patch.object(trade_state, "STATE_FILE", state_file):
            trade_state.record_buy("AAPL")
            assert trade_state.has_open_buy("AAPL")
            assert not trade_state.has_open_buy("GOOG")

    def test_record_sell_removes_buy(self, tmp_path):
        from Strategy_Auto_Trader.output import trade_state
        state_file = tmp_path / "trade_state.json"
        with mock.patch.object(trade_state, "STATE_FILE", state_file):
            trade_state.record_buy("AAPL")
            assert trade_state.has_open_buy("AAPL")
            trade_state.record_sell("AAPL")
            assert not trade_state.has_open_buy("AAPL")

    def test_record_sell_nonexistent_is_safe(self, tmp_path):
        from Strategy_Auto_Trader.output import trade_state
        state_file = tmp_path / "trade_state.json"
        with mock.patch.object(trade_state, "STATE_FILE", state_file):
            # Selling a ticker that was never bought should not raise
            trade_state.record_sell("NOPE")
            assert not trade_state.has_open_buy("NOPE")

    def test_get_open_positions(self, tmp_path):
        from Strategy_Auto_Trader.output import trade_state
        state_file = tmp_path / "trade_state.json"
        with mock.patch.object(trade_state, "STATE_FILE", state_file):
            trade_state.record_buy("AAPL")
            trade_state.record_buy("GOOG")
            positions = trade_state.get_open_positions()
            assert "AAPL" in positions
            assert "GOOG" in positions
            assert len(positions) == 2

    def test_persistence(self, tmp_path):
        from Strategy_Auto_Trader.output import trade_state
        state_file = tmp_path / "trade_state.json"
        with mock.patch.object(trade_state, "STATE_FILE", state_file):
            trade_state.record_buy("AAPL")

        # Verify the JSON file was actually written
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert "AAPL" in data["buys"]

    def test_empty_state_file(self, tmp_path):
        from Strategy_Auto_Trader.output import trade_state
        state_file = tmp_path / "trade_state.json"
        with mock.patch.object(trade_state, "STATE_FILE", state_file):
            # No state file yet
            assert not trade_state.has_open_buy("AAPL")
            positions = trade_state.get_open_positions()
            assert positions == {}

    def test_record_sell_market_return_during_hold(self, tmp_path):
        import csv
        from unittest import mock
        from Strategy_Auto_Trader.output import trade_state
        state_file = tmp_path / "trade_state.json"
        journal_file = tmp_path / "live.csv"
        with mock.patch.object(trade_state, "STATE_FILE", state_file), \
             mock.patch.object(trade_state, "LIVE_JOURNAL", journal_file):
            trade_state.record_buy("AAPL", {
                "price": 100.0, "kelly_fraction": 0.1,
                "portfolio_value": 10000.0, "bh_return": 0.08,
            })
            trade_state.record_sell("AAPL", {
                "price": 110.0, "reason": "target", "bh_return": 0.12,
            })
        with open(journal_file, newline="") as f:
            row = next(csv.DictReader(f))
        expected = (1 + 0.12) / (1 + 0.08) - 1
        assert abs(float(row["market_ret_during_hold"]) - expected) < 1e-9

    def test_record_sell_market_return_without_entry_bh(self, tmp_path):
        import csv
        from unittest import mock
        from Strategy_Auto_Trader.output import trade_state
        state_file = tmp_path / "trade_state.json"
        journal_file = tmp_path / "live.csv"
        with mock.patch.object(trade_state, "STATE_FILE", state_file), \
             mock.patch.object(trade_state, "LIVE_JOURNAL", journal_file):
            trade_state.record_buy("AAPL", {"price": 100.0})
            trade_state.record_sell("AAPL", {"price": 105.0, "bh_return": 0.12})
        with open(journal_file, newline="") as f:
            row = next(csv.DictReader(f))
        expected = (1 + 0.12) / (1 + 0.0) - 1
        assert abs(float(row["market_ret_during_hold"]) - expected) < 1e-9
