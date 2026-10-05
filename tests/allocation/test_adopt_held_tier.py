"""Startup adoption of the held tier from broker positions (min-hold keeps every enabled tier at the floor)."""

import logging

from Strategy_Auto_Trader.allocation.allocation_manager import MultiTierAllocationManager

LOG = logging.getLogger(__name__)
PRICES = {"EQGB.L": 10.0, "VUSA": 80.0, "ISF.L": 60.0, "CSH2.L": 100.0}


def _mgr():
    return MultiTierAllocationManager(lower_tiers_enabled=False, min_hold_gbp=10.0)


class TestAdoptHeldTier:
    def test_adopts_largest_value_when_floor_positions_present(self):
        mgr = _mgr()
        # Nasdaq is the target: 1,500 units at £10 = £15k. CSH2.L sits at its £10 floor.
        mgr.adopt_held_tier({"EQGB.L": 1_500.0, "CSH2.L": 0.1}, PRICES, LOG)
        assert mgr.current_asset == "EQGB.L"

    def test_money_market_target_beats_floor_nasdaq(self):
        mgr = _mgr()
        mgr.adopt_held_tier({"EQGB.L": 1.0, "CSH2.L": 149.9}, PRICES, LOG)
        assert mgr.current_asset == "CSH2.L"

    def test_no_held_tier_leaves_default(self):
        mgr = _mgr()
        before = mgr.current_asset
        mgr.adopt_held_tier({"EQGB.L": 0, "CSH2.L": 0}, PRICES, LOG)
        assert mgr.current_asset == before

    def test_unpriced_held_tier_leaves_current_asset_unchanged(self):
        mgr = _mgr()
        before = mgr.current_asset
        mgr.adopt_held_tier({"EQGB.L": 1_500.0, "CSH2.L": 0.1}, {"EQGB.L": None, "CSH2.L": 100.0}, LOG)
        assert mgr.current_asset == before

    def test_ignores_non_tier_positions(self):
        mgr = _mgr()
        mgr.adopt_held_tier({"AAPL": 500.0, "ISF.L": 2.0}, PRICES, LOG)
        assert mgr.current_asset == "ISF.L"
