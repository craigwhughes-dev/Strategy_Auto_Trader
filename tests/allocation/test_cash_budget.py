"""Min-hold cash budget: cash landing after the daily release waits for the next start."""

import pytest

from Strategy_Auto_Trader.allocation.allocation_manager import MultiTierAllocationManager


def _mgr(commission_pct=0.0):
    return MultiTierAllocationManager(commission_pct=commission_pct, min_hold_gbp=10.0)


class TestSpendableCash:
    def test_unset_budget_passes_broker_cash_through(self):
        assert _mgr().spendable_cash(1_000.0) == pytest.approx(1_000.0)

    def test_budget_caps_broker_cash(self):
        mgr = _mgr()
        mgr.release_cash(300.0)
        assert mgr.spendable_cash(1_000.0) == pytest.approx(300.0)

    def test_cash_landing_after_release_is_held_back(self):
        mgr = _mgr()
        mgr.release_cash(300.0)
        assert mgr.spendable_cash(350.0) == pytest.approx(300.0)

    def test_broker_cash_below_budget_is_the_cap(self):
        mgr = _mgr()
        mgr.release_cash(300.0)
        assert mgr.spendable_cash(120.0) == pytest.approx(120.0)

    def test_negative_release_clamped_to_zero(self):
        mgr = _mgr()
        mgr.release_cash(-5.0)
        assert mgr.spendable_cash(100.0) == 0.0


class TestNoteFill:
    def test_buy_spends_budget(self):
        mgr = _mgr()
        mgr.release_cash(300.0)
        mgr.note_fill("BUY", 120.0)
        assert mgr.cash_budget == pytest.approx(180.0)

    def test_sell_replenishes_budget(self):
        mgr = _mgr()
        mgr.release_cash(100.0)
        mgr.note_fill("SELL", 80.0)
        assert mgr.cash_budget == pytest.approx(180.0)

    def test_sell_proceeds_reinvestable_same_cycle(self):
        mgr = _mgr()
        mgr.release_cash(0.0)
        mgr.note_fill("SELL", 250.0)
        assert mgr.spendable_cash(250.0) == pytest.approx(250.0)

    def test_buy_budget_floors_at_zero(self):
        mgr = _mgr()
        mgr.release_cash(50.0)
        mgr.note_fill("BUY", 200.0)
        assert mgr.cash_budget == 0.0

    def test_commission_charged_on_buy_and_sell(self):
        mgr = _mgr(commission_pct=1.0)
        mgr.release_cash(1_000.0)
        mgr.note_fill("BUY", 100.0)   # spends 101
        assert mgr.cash_budget == pytest.approx(899.0)
        mgr.note_fill("SELL", 100.0)  # returns 99
        assert mgr.cash_budget == pytest.approx(998.0)

    def test_fills_ignored_when_budget_unset(self):
        mgr = _mgr()
        mgr.note_fill("BUY", 500.0)
        assert mgr.cash_budget is None
