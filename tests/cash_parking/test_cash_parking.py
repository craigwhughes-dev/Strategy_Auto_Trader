"""Unit tests for the cash_parking module."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# strategy.tier_for
# ---------------------------------------------------------------------------

class TestTierFor:
    def test_tier_for_hy_bonds(self):
        from Strategy_Auto_Trader.cash_parking.strategy import tier_for
        # vix=10 < 12 → hy_bonds
        assert tier_for(10.0) == "hy_bonds"

    def test_tier_for_gilts(self):
        from Strategy_Auto_Trader.cash_parking.strategy import tier_for
        # vix=15 (12 <= vix < 18) → gilts
        assert tier_for(15.0) == "gilts"

    def test_tier_for_cash_high_vix(self):
        from Strategy_Auto_Trader.cash_parking.strategy import tier_for
        # vix >= 18 → cash
        assert tier_for(20.0) == "cash"

    def test_hy_bonds_boundary(self):
        from Strategy_Auto_Trader.cash_parking.strategy import tier_for
        # vix < 12 → hy_bonds
        assert tier_for(11.9) == "hy_bonds"

    def test_gilts_boundary(self):
        from Strategy_Auto_Trader.cash_parking.strategy import tier_for
        # vix < 18 → gilts (but not < 12)
        assert tier_for(17.9) == "gilts"

    def test_cash_boundary(self):
        from Strategy_Auto_Trader.cash_parking.strategy import tier_for
        # vix >= 18 → cash
        assert tier_for(18.0) == "cash"


# ---------------------------------------------------------------------------
# CashParkingManager helpers
# ---------------------------------------------------------------------------

def _make_manager(initial_cash=10_000.0, liquid_floor_pct=0.20, state_path=None):
    """Return a CashParkingManager with no persisted state."""
    from Strategy_Auto_Trader.cash_parking.manager import CashParkingManager
    if state_path is None:
        import tempfile, pathlib
        state_path = pathlib.Path(tempfile.mktemp(suffix=".json"))
    return CashParkingManager(
        initial_cash=initial_cash,
        liquid_floor_pct=liquid_floor_pct,
        state_path=state_path,
    )


def _make_broker(price: float = 5000.0) -> MagicMock:
    """NullBroker-like mock returning a fixed pence price."""
    broker = MagicMock()
    broker.get_last_price.return_value = price
    return broker


class TestLiquidFloor:
    def test_liquid_floor_respected(self):
        """available_cash=£5000, floor=0.20 → max_parkable=£4000."""
        mgr = _make_manager(initial_cash=5_000.0, liquid_floor_pct=0.20)
        # parkable = 5000 * 0.80 = 4000; at price=£40/share → 100 shares
        broker = _make_broker(price=4_000.0)  # 4000p = £40
        orders = mgr.rebalance(
            broker=broker,
            available_cash=5_000.0,
            vix_level=15.0,
            current_positions={},
            today=date.today(),
        )
        # Should place a BUY with quantity = floor(4000 / 40) = 100
        buy_orders = [o for o in orders if o.action == "BUY"]
        assert len(buy_orders) == 1
        assert buy_orders[0].quantity == 100


class TestMinParking:
    def test_no_order_below_min(self):
        """parkable < MIN_PARKING_AMOUNT → no orders."""
        from Strategy_Auto_Trader.cash_parking.strategy import MIN_PARKING_AMOUNT, LIQUID_FLOOR_PCT
        # Make available_cash just below the threshold
        tiny_cash = (MIN_PARKING_AMOUNT / (1 - LIQUID_FLOOR_PCT)) - 1.0
        mgr = _make_manager(initial_cash=tiny_cash, liquid_floor_pct=LIQUID_FLOOR_PCT)
        broker = _make_broker(price=5_000.0)
        orders = mgr.rebalance(
            broker=broker,
            available_cash=tiny_cash,
            vix_level=10.0,
            current_positions={},
            today=date.today(),
        )
        assert orders == []


class TestT2Settlement:
    def test_t2_settlement_blocks_buy(self):
        """Sell placed today → buy blocked until settling_until passes."""
        mgr = _make_manager(initial_cash=10_000.0)
        today = date.today()

        # Simulate a sell that settled today+1 (still unsettled)
        mgr.record_sell_settled("XSTR.L", today)
        # settling_until should be today + 2 business days
        from Strategy_Auto_Trader.cash_parking.manager import _add_business_days
        expected_settling = _add_business_days(today, 2)
        assert mgr._state.settling_until == expected_settling.isoformat()

        # Now try to rebalance — buy should be blocked
        broker = _make_broker(price=10_000.0)
        orders = mgr.rebalance(
            broker=broker,
            available_cash=10_000.0,
            vix_level=10.0,
            current_positions={},
            today=today,
        )
        buy_orders = [o for o in orders if o.action == "BUY"]
        assert buy_orders == []

    def test_t2_settlement_clears_after_date(self):
        """Once settling_until is past, buy is allowed."""
        mgr = _make_manager(initial_cash=10_000.0)
        old_date = date.today() - timedelta(days=10)
        mgr.record_sell_settled("XSTR.L", old_date)

        broker = _make_broker(price=10_000.0)  # 10000p = £100
        orders = mgr.rebalance(
            broker=broker,
            available_cash=10_000.0,
            vix_level=10.0,
            current_positions={},
            today=date.today(),
        )
        buy_orders = [o for o in orders if o.action == "BUY"]
        assert len(buy_orders) == 1


class TestRebalanceThreshold:
    def test_same_ticker_no_order(self):
        """Already in the desired tier/ticker → no orders."""
        mgr = _make_manager(initial_cash=10_000.0, liquid_floor_pct=0.20)
        mgr._state.tier = "hy_bonds"
        mgr._state.ticker = "ISXF.L"
        mgr._state.quantity = 80

        broker = _make_broker(price=10_000.0)
        orders = mgr.rebalance(
            broker=broker,
            available_cash=10_000.0,
            vix_level=10.0,
            current_positions={},
            today=date.today(),
        )
        assert orders == []

    def test_delta_below_threshold_no_order(self):
        """Switching would shift < 5% of pot → no order."""
        initial_cash = 100_000.0
        mgr = _make_manager(initial_cash=initial_cash, liquid_floor_pct=0.20)
        # Set current position to ISXF.L with 790 shares @ £101 ≈ £79,790
        mgr._state.tier = "hy_bonds"
        mgr._state.ticker = "ISXF.L"
        mgr._state.quantity = 790

        # parkable = 100000 * 0.80 = 80000
        # current_value = 790 * (10100p / 100) = 790 * 101 = 79790
        # delta = |80000 - 79790| = 210 < threshold (100000 * 0.05 = 5000)
        broker = _make_broker(price=10_100.0)  # 10100p = £101
        orders = mgr.rebalance(
            broker=broker,
            available_cash=100_000.0,
            vix_level=10.0,
            current_positions={},
            today=date.today(),
        )
        assert orders == []


class TestTierChange:
    def test_sell_then_buy_on_tier_change(self):
        """Currently in gilts, regime shifts to hy_bonds → SELL IGLS.L + BUY ISXF.L."""
        initial_cash = 50_000.0
        mgr = _make_manager(initial_cash=initial_cash, liquid_floor_pct=0.20)
        # Force current state to gilts with a large enough position to exceed threshold
        mgr._state.tier = "gilts"
        mgr._state.ticker = "IGLS.L"
        mgr._state.quantity = 50  # £50 × 50 = £2500 current value; parkable = 40000; delta >> threshold

        broker = _make_broker(price=500.0)  # 500p = £5/share for IGLS.L and ISXF.L
        orders = mgr.rebalance(
            broker=broker,
            available_cash=initial_cash,
            vix_level=10.0,
            current_positions={},
            today=date.today(),
        )
        actions = [o.action for o in orders]
        tickers = [o.ticker for o in orders]
        assert "SELL" in actions
        assert "BUY" in actions
        assert "IGLS.L" in tickers
        assert "ISXF.L" in tickers


class TestAppStatusDict:
    def test_app_status_dict_shape(self):
        """Keys present, settling_until serialises to ISO string or null."""
        mgr = _make_manager()
        status = mgr.app_status_dict()
        assert "tier" in status
        assert "ticker" in status
        assert "qty" in status
        assert "entry_price_gbp" in status
        assert "settling_until" in status
        # settling_until must be an ISO string or None
        sv = status["settling_until"]
        assert sv is None or (isinstance(sv, str) and len(sv) == 10)

    def test_app_status_after_rebalance(self):
        """app_status_dict reflects the current holding after a BUY."""
        mgr = _make_manager(initial_cash=10_000.0)
        broker = _make_broker(price=10_000.0)  # 10000p = £100
        mgr.rebalance(
            broker=broker,
            available_cash=10_000.0,
            vix_level=10.0,
            current_positions={},
            today=date.today(),
        )
        status = mgr.app_status_dict()
        assert status["tier"] == "hy_bonds"
        assert status["ticker"] == "ISXF.L"
        assert status["qty"] > 0
