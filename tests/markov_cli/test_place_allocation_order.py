"""_place_allocation_order: free-cash rejections shrink and retry, everything else skips only that order."""

from __future__ import annotations

import logging

import pytest

from Strategy_Auto_Trader.allocation.allocation_manager import AllocationOrder
from Strategy_Auto_Trader.broker.types import InsufficientFundsError
from Strategy_Auto_Trader.markov_cli.live_daemon import _place_allocation_order

LOG = logging.getLogger("test_place_allocation_order")
FILL = object()


class ScriptedBroker:
    """Records the quantity of each place_order call and replays the scripted outcomes in order."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.quantities = []

    def place_order(self, order):
        self.quantities.append(order.quantity)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _buy(qty: float = 26.88) -> AllocationOrder:
    return AllocationOrder(ticker="EQGB.L", action="BUY", quantity=qty)


def test_free_cash_rejection_retries_smaller_until_it_fills():
    broker = ScriptedBroker([InsufficientFundsError("no cash"), FILL])
    assert _place_allocation_order(broker, _buy(), LOG, "ftse") is FILL
    assert broker.quantities == [26.88, 26.07]


def test_gives_up_after_max_shrinks():
    broker = ScriptedBroker([InsufficientFundsError("no cash")] * 4)
    assert _place_allocation_order(broker, _buy(), LOG, "ftse") is None
    assert len(broker.quantities) == 4


def test_other_errors_skip_the_order_without_retry():
    broker = ScriptedBroker([RuntimeError("T212 POST failed: 500")])
    assert _place_allocation_order(broker, _buy(), LOG, "ftse") is None
    assert len(broker.quantities) == 1


def test_rejected_sell_retreats_smaller_to_part_exit():
    sell = AllocationOrder(ticker="VUSA", action="SELL", quantity=0.09)
    rejected = RuntimeError('T212 POST /api/v0/equity/orders/market failed: 400 {"type":"/api-errors/quantity-precision-mismatch"}')
    broker = ScriptedBroker([rejected, FILL])
    assert _place_allocation_order(broker, sell, LOG, "ftse") is FILL
    assert broker.quantities == [0.09, 0.08]


def test_sell_network_error_is_not_retried_since_order_may_have_been_placed():
    sell = AllocationOrder(ticker="VUSA", action="SELL", quantity=0.09)
    broker = ScriptedBroker([OSError("timed out")])
    assert _place_allocation_order(broker, sell, LOG, "ftse") is None
    assert len(broker.quantities) == 1


def test_unfilled_order_returns_none_from_broker():
    broker = ScriptedBroker([None])
    assert _place_allocation_order(broker, _buy(), LOG, "ftse") is None


def test_retry_stops_when_shrunk_quantity_falls_below_minimum():
    broker = ScriptedBroker([InsufficientFundsError("no cash")] * 4)
    assert _place_allocation_order(broker, _buy(0.01), LOG, "ftse") is None
    assert broker.quantities == [0.01]


def test_retry_warning_includes_broker_reason(caplog):
    broker = ScriptedBroker([InsufficientFundsError("insufficient-free-for-stocks-buy"), FILL])
    with caplog.at_level("WARNING", logger="test_place_allocation_order"):
        _place_allocation_order(broker, _buy(), LOG, "ftse")
    assert "rejected (insufficient-free-for-stocks-buy) — retrying 26.07" in caplog.text
