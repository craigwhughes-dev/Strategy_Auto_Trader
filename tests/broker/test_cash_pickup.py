"""Start-of-day release of uninvested broker cash to the tier allocator."""

import logging

import pytest

from Strategy_Auto_Trader.allocation.allocation_manager import MultiTierAllocationManager
from Strategy_Auto_Trader.broker.cash_pickup import release_uninvested_cash

LOGGER = logging.getLogger(__name__)


class _CashBroker:
    def __init__(self, cash, raises=None):
        self._cash = cash
        self._raises = raises

    def get_available_cash(self):
        if self._raises:
            raise self._raises
        return self._cash


def _mgr():
    return MultiTierAllocationManager(commission_pct=0.0, min_hold_gbp=10.0)


class TestReleaseUninvestedCash:
    def test_releases_full_broker_cash_to_allocator(self):
        mgr = _mgr()
        released = release_uninvested_cash(mgr, _CashBroker(412.345), LOGGER, "GBP")
        assert released == pytest.approx(412.35)
        assert mgr.cash_budget == pytest.approx(412.35)

    def test_release_replaces_previous_budget(self):
        mgr = _mgr()
        release_uninvested_cash(mgr, _CashBroker(500.0), LOGGER, "GBP")
        release_uninvested_cash(mgr, _CashBroker(120.0), LOGGER, "GBP")
        assert mgr.cash_budget == pytest.approx(120.0)

    def test_non_gbp_skipped(self):
        mgr = _mgr()
        assert release_uninvested_cash(mgr, _CashBroker(900.0), LOGGER, "USD") == 0.0
        assert mgr.cash_budget is None

    def test_broker_error_propagates_without_release(self):
        mgr = _mgr()
        with pytest.raises(ConnectionError):
            release_uninvested_cash(mgr, _CashBroker(0, raises=ConnectionError("down")), LOGGER, "GBP")
        assert mgr.cash_budget is None
