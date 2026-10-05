"""Tests for the min-hold rebalance (T212 path): every tier asset held at min_hold_gbp, target takes the rest."""

from __future__ import annotations

from datetime import date
import logging

import pytest

from Strategy_Auto_Trader.allocation.allocation_manager import MIN_HOLD_CASH_HEADROOM, MultiTierAllocationManager

TODAY = date(2026, 10, 5)
PRICES = {"EQGB.L": 586.3, "VUSA": 111.29, "ISF.L": 10.22, "CSH2.L": 1254.8}
LOG = logging.getLogger("test_min_hold")


def _mgr(min_hold: float = 10.0, current_asset: str = "VUSA") -> MultiTierAllocationManager:
    m = MultiTierAllocationManager(
        vxn_threshold=23.0, vxn_exit_threshold=24.0, lower_tiers_enabled=False,
        commission_pct=0.0, min_hold_gbp=min_hold,
    )
    m.current_asset = current_asset
    return m


def _by_ticker(orders):
    return {o.ticker: o for o in orders}


class TestBootstrap:
    def test_buys_minimum_in_each_non_target_and_rest_in_target(self):
        m = _mgr(current_asset="VUSA")
        orders = m.rebalance(TODAY, vxn=20.0, vix=None, current_price=PRICES,
                             available_cash=16000.0, positions={}, logger=LOG)
        got = _by_ticker(orders)
        assert all(o.action == "BUY" for o in orders)
        assert got["VUSA"].quantity == pytest.approx(0.09)     # £10.02
        assert got["ISF.L"].quantity == pytest.approx(0.98)   # £10.02
        assert got["CSH2.L"].quantity == pytest.approx(0.01)  # £12.55 (2 dp floor)
        others_spend = sum(got[t].quantity * PRICES[t] for t in ("VUSA", "ISF.L", "CSH2.L"))
        remainder = 16000.0 * (1 - MIN_HOLD_CASH_HEADROOM) - others_spend
        assert abs(got["EQGB.L"].quantity * PRICES["EQGB.L"] - remainder) < 3.0  # 2 dp rounding on ~£586 units

    def test_leaves_cash_headroom_for_market_buy_fills(self):
        m = _mgr(current_asset="VUSA")
        orders = m.rebalance(TODAY, vxn=20.0, vix=None, current_price=PRICES,
                             available_cash=16000.0, positions={}, logger=LOG)
        spend = sum(o.quantity * PRICES[o.ticker] for o in orders if o.action == "BUY")
        assert spend <= 16000.0 * 0.99 + 1e-6

    def test_updates_current_asset_to_target(self):
        m = _mgr(current_asset="VUSA")
        m.rebalance(TODAY, vxn=20.0, vix=None, current_price=PRICES,
                    available_cash=16000.0, positions={}, logger=LOG)
        assert m.current_asset == "EQGB.L"


class TestTopUpWithoutRotation:
    def test_empty_assets_topped_up_and_spare_cash_goes_to_target(self):
        m = _mgr(current_asset="EQGB.L")
        orders = m.rebalance(TODAY, vxn=20.0, vix=None, current_price=PRICES,
                             available_cash=1000.0, positions={"EQGB.L": 0.5}, logger=LOG)
        got = _by_ticker(orders)
        assert all(o.action == "BUY" for o in orders)
        assert got["VUSA"].quantity == pytest.approx(0.09)
        assert got["ISF.L"].quantity == pytest.approx(0.98)
        assert got["CSH2.L"].quantity == pytest.approx(0.01)
        others_spent = sum(got[t].quantity * PRICES[t] for t in ("VUSA", "ISF.L", "CSH2.L"))
        assert others_spent <= 1000.0
        # EQGB already held £293; the spare cash after the minimum top-ups is what it buys
        spare = 1000.0 * (1 - MIN_HOLD_CASH_HEADROOM) - others_spent
        assert got["EQGB.L"].quantity * PRICES["EQGB.L"] == pytest.approx(spare, abs=PRICES["EQGB.L"] * 0.01)

    def test_no_trades_on_price_drift_alone(self):
        m = _mgr(current_asset="EQGB.L")
        # Every asset already at the minimum; a price move must not trade
        positions = {"VUSA": 0.09, "ISF.L": 0.98, "CSH2.L": 0.00797, "EQGB.L": 0.02}
        drifted = {**PRICES, "EQGB.L": PRICES["EQGB.L"] * 1.02, "VUSA": PRICES["VUSA"] * 1.02}
        orders = m.rebalance(TODAY, vxn=20.0, vix=None, current_price=drifted,
                             available_cash=0.0, positions=positions, logger=LOG)
        assert orders == []


class TestRotation:
    def test_trims_old_asset_to_minimum_and_moves_rest_to_target(self):
        m = _mgr(current_asset="EQGB.L")
        positions = {"EQGB.L": 1.0, "VUSA": 0.09, "ISF.L": 0.98}
        orders = m.rebalance(TODAY, vxn=30.0, vix=None, current_price=PRICES,
                             available_cash=0.0, positions=positions, logger=LOG)
        got = _by_ticker(orders)
        assert got["EQGB.L"].action == "SELL"
        assert got["EQGB.L"].quantity == pytest.approx(0.98)
        assert got["CSH2.L"].action == "BUY"
        assert got["CSH2.L"].quantity == pytest.approx(0.45)
        assert "VUSA" not in got and "ISF.L" not in got
        assert m.current_asset == "CSH2.L"


class TestDisabledTiers:
    def test_enabled_assets_exclude_s_and_p_and_ftse_when_lower_tiers_off(self):
        assert set(_mgr().enabled_assets()) == {"EQGB.L", "CSH2.L"}

    def test_enabled_assets_include_all_when_lower_tiers_on(self):
        m = MultiTierAllocationManager(lower_tiers_enabled=True, min_hold_gbp=10.0)
        assert set(m.enabled_assets()) == {"EQGB.L", "VUSA", "ISF.L", "CSH2.L"}

    def test_disabled_tier_is_never_trimmed_on_rotation(self):
        m = _mgr(current_asset="EQGB.L")
        positions = {"EQGB.L": 1.0, "VUSA": 0.2, "ISF.L": 0.98}  # VUSA £22 is above the minimum
        orders = m.rebalance(TODAY, vxn=30.0, vix=None, current_price=PRICES,
                             available_cash=0.0, positions=positions, logger=LOG)
        assert not any(o.ticker == "VUSA" and o.action == "SELL" for o in orders)

    def test_disabled_tier_without_price_does_not_block_enabled_rebalance(self):
        m = _mgr(current_asset="EQGB.L")
        prices = {k: v for k, v in PRICES.items() if k != "VUSA"}
        orders = m.rebalance(TODAY, vxn=30.0, vix=None, current_price=prices,
                             available_cash=16000.0, positions={}, logger=LOG)
        assert orders
        assert not any(o.ticker == "VUSA" for o in orders)
        assert any(o.ticker == "CSH2.L" and o.action == "BUY" for o in orders)

    def test_missing_enabled_price_blocks_rebalance(self):
        m = _mgr(current_asset="EQGB.L")
        prices = {k: v for k, v in PRICES.items() if k != "CSH2.L"}
        orders = m.rebalance(TODAY, vxn=30.0, vix=None, current_price=prices,
                             available_cash=16000.0, positions={}, logger=LOG)
        assert orders == []


class TestConfig:
    def test_negative_minimum_rejected(self):
        with pytest.raises(ValueError, match="min_hold_gbp"):
            MultiTierAllocationManager(min_hold_gbp=-1.0)

    def test_from_config_reads_min_hold(self):
        m = MultiTierAllocationManager.from_config({"min_hold_gbp": 10, "lower_tiers_enabled": False})
        assert m.min_hold_gbp == 10.0

    def test_default_is_zero_so_whole_share_rule_is_unchanged(self):
        assert MultiTierAllocationManager().min_hold_gbp == 0.0
