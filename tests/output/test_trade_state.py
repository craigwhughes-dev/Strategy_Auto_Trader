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
