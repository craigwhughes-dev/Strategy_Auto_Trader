"""_seed_tier_price: cold-start fix. T212 only prices held positions, so a fresh account with
zero tier holdings can never price an enabled tier asset and rebalance() would block forever.
A minimal seed buy breaks the chicken-and-egg.
"""

from __future__ import annotations

import logging

from Strategy_Auto_Trader.allocation.allocation_manager import MIN_TRADE_UNITS
from Strategy_Auto_Trader.broker.types import FillResult
from Strategy_Auto_Trader.markov_cli import live_daemon
from Strategy_Auto_Trader.markov_cli.live_daemon import _seed_tier_price

LOG = logging.getLogger("test_seed_tier_price")


class ScriptedBroker:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.orders = []

    def place_order(self, order):
        self.orders.append(order)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class RefusingBroker:
    def place_order(self, order):
        raise AssertionError("place_order must not be called when the exchange is closed")


class RecordingPortfolio:
    def __init__(self):
        self.fills = []
        self.saved = False

    def record_tier_fill(self, ticker, fill, action, market_name, currency):
        self.fills.append((ticker, fill, action, market_name, currency))

    def save(self):
        self.saved = True


def test_seed_buy_fills_and_normalizes_pence_to_pounds():
    # T212 fill prices are in pence for .L tickers; with no reference price, normalize_fill_price
    # assumes pence-scale and converts.
    fill = FillResult(ticker="EQGB.L", action="BUY", fill_price=105.3, quantity=MIN_TRADE_UNITS, timestamp="t")
    broker = ScriptedBroker([fill])
    portfolio = RecordingPortfolio()

    price = _seed_tier_price("EQGB.L", broker, portfolio, config={}, market_name="ftse", logger=LOG)

    assert price == 1.053
    assert broker.orders[0].ticker == "EQGB.L"
    assert broker.orders[0].action == "BUY"
    assert broker.orders[0].quantity == MIN_TRADE_UNITS
    assert portfolio.fills[0][0] == "EQGB.L"
    assert portfolio.fills[0][1].fill_price == 1.053
    assert portfolio.saved is True


def test_seed_buy_failure_returns_none_and_does_not_touch_portfolio():
    broker = ScriptedBroker([None])  # unfilled order
    portfolio = RecordingPortfolio()

    price = _seed_tier_price("CSH2.L", broker, portfolio, config={}, market_name="ftse", logger=LOG)

    assert price is None
    assert portfolio.fills == []
    assert portfolio.saved is False


def test_seed_buy_skipped_when_exchange_closed(monkeypatch):
    monkeypatch.setattr(live_daemon, "is_trading_hours", lambda *a, **k: False)
    broker = RefusingBroker()
    portfolio = RecordingPortfolio()
    config = {"markets": {"ftse": {"some": "cfg"}}}

    price = _seed_tier_price("EQGB.L", broker, portfolio, config=config, market_name="ftse", logger=LOG)

    assert price is None
    assert portfolio.fills == []
