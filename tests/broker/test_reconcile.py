"""Tests for broker/reconcile.py."""

from __future__ import annotations

import pytest

from Strategy_Auto_Trader.broker.reconcile import reconcile_positions


def _pos(qty: int) -> dict:
    return {"entry_date": "2026-07-01", "fill_price": 100.0, "quantity": qty}


class TestReconcilePositions:
    def test_both_empty_is_clean(self):
        assert reconcile_positions({}, {}) == []

    def test_matching_positions_clean(self):
        internal = {"SPY": _pos(10), "AAPL": _pos(3)}
        broker = {"SPY": 10, "AAPL": 3}
        assert reconcile_positions(internal, broker) == []

    def test_quantity_mismatch(self):
        result = reconcile_positions({"SPY": _pos(10)}, {"SPY": 7})
        assert len(result) == 1
        assert "SPY" in result[0]
        assert "10" in result[0] and "7" in result[0]

    def test_missing_at_broker(self):
        result = reconcile_positions({"SPY": _pos(10)}, {})
        assert len(result) == 1
        assert "broker shows no position" in result[0]

    def test_unexpected_at_broker(self):
        result = reconcile_positions({}, {"TSLA": 5})
        assert len(result) == 1
        assert "no internal position" in result[0]

    def test_lse_ticker_matches_directly(self):
        internal = {"HSBA.L": _pos(200)}
        assert reconcile_positions(internal, {"HSBA.L": 200}) == []

    def test_lse_share_class_ticker_matches_directly(self):
        internal = {"BT-A.L": _pos(500)}
        assert reconcile_positions(internal, {"BT-A.L": 500}) == []

    def test_lse_ticker_mismatch_reports_ticker_name(self):
        result = reconcile_positions({"HSBA.L": _pos(200)}, {"HSBA.L": 150})
        assert len(result) == 1
        assert "HSBA.L" in result[0]

    def test_multiple_discrepancies_all_reported(self):
        internal = {"SPY": _pos(10), "AAPL": _pos(3), "MSFT": _pos(4)}
        broker = {"SPY": 10, "AAPL": 2, "TSLA": 5}
        result = reconcile_positions(internal, broker)
        joined = " | ".join(result)
        assert len(result) == 3
        assert "AAPL" in joined
        assert "MSFT" in joined
        assert "TSLA" in joined

    # -- Boundary values ------------------------------------------------------

    def test_off_by_one_share_flags(self):
        assert len(reconcile_positions({"SPY": _pos(10)}, {"SPY": 9})) == 1
        assert len(reconcile_positions({"SPY": _pos(10)}, {"SPY": 11})) == 1

    def test_single_share_position_matches(self):
        assert reconcile_positions({"SPY": _pos(1)}, {"SPY": 1}) == []

    def test_broker_short_position_vs_internal_long(self):
        result = reconcile_positions({"SPY": _pos(10)}, {"SPY": -10})
        assert len(result) == 1

    def test_internal_zero_quantity_vs_absent_broker_flags(self):
        # A zero-quantity internal position is still a state anomaly worth
        # surfacing: the ticker should not be in the positions dict at all.
        result = reconcile_positions({"SPY": _pos(0)}, {})
        assert len(result) == 1

    def test_never_mutates_inputs(self):
        internal = {"SPY": _pos(10)}
        broker = {"SPY": 7, "TSLA": 5}
        reconcile_positions(internal, broker)
        assert internal == {"SPY": _pos(10)}
        assert broker == {"SPY": 7, "TSLA": 5}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
