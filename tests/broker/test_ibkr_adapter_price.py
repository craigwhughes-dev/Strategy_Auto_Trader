"""Tests for IBKRAdapter.get_last_price — delayed-data request, historical
fallback and IBKR error capture, driven by a fake ib_async IB object."""

from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass

import pytest

from Strategy_Auto_Trader.broker.ibkr_adapter import IBKRAdapter, _first_positive


class _Event:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, h):
        self.handlers.append(h)
        return self

    def __isub__(self, h):
        self.handlers.remove(h)
        return self

    def emit(self, *args):
        for h in list(self.handlers):
            h(*args)


@dataclass
class _Ticker:
    bid: float = math.nan
    ask: float = math.nan
    last: float = math.nan
    close: float = math.nan

    def midpoint(self):
        return (self.bid + self.ask) / 2


@dataclass
class _Bar:
    date: str
    close: float


class _FakeIB:
    def __init__(self, ticker: _Ticker, bars=None, errors=None, con_id=1):
        self._ticker = ticker
        self._bars = bars or []
        self._errors = errors or []
        self._con_id = con_id
        self.errorEvent = _Event()
        self.market_data_types: list[int] = []
        self.snapshot_args = None
        self.hist_calls = 0

    def qualifyContracts(self, contract):
        contract.conId = self._con_id
        return [contract]

    def sleep(self, _s):
        pass

    def reqMarketDataType(self, t):
        self.market_data_types.append(t)

    def reqMktData(self, contract, generic, snapshot, regulatory):
        self.snapshot_args = (generic, snapshot, regulatory)
        for code, msg in self._errors:
            self.errorEvent.emit(1, code, msg, contract)
        return self._ticker

    def reqHistoricalData(self, contract, **kw):
        self.hist_calls += 1
        return list(self._bars)


class _Stock:
    def __init__(self, symbol, exchange, currency):
        self.symbol, self.exchange, self.currency = symbol, exchange, currency
        self.conId = 0


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setitem(sys.modules, "ib_async", types.SimpleNamespace(Stock=_Stock))

    def build(ticker, **kw):
        a = IBKRAdapter(port=4002)
        a._ib = _FakeIB(ticker, **kw)
        return a

    return build


class TestFirstPositive:
    def test_skips_nan_none_and_nonpositive(self):
        assert _first_positive(math.nan, None, 0.0, -1.0, 12.5, 3.0) == 12.5

    def test_none_when_nothing_usable(self):
        assert _first_positive(math.nan, None, 0.0) is None


class TestGetLastPrice:
    def test_requests_delayed_capable_snapshot(self, adapter):
        a = adapter(_Ticker(bid=100.0, ask=102.0))
        assert a.get_last_price("ISF.L") == pytest.approx(101.0)
        assert a._ib.market_data_types == [3]
        assert a._ib.snapshot_args == ("", True, False)
        assert a._ib.hist_calls == 0

    def test_midpoint_preferred_then_last_then_close(self, adapter):
        assert adapter(_Ticker(last=5.0, close=4.0)).get_last_price("SPY") == 5.0
        assert adapter(_Ticker(close=4.0)).get_last_price("SPY") == 4.0

    def test_falls_back_to_last_historical_bar_when_snapshot_empty(self, adapter):
        a = adapter(_Ticker(), bars=[_Bar("t0", 9.0), _Bar("t1", 125250.0)])
        assert a.get_last_price("CSH2.L") == 125250.0
        assert a._ib.hist_calls == 1

    def test_raises_with_ibkr_error_text_when_nothing_available(self, adapter):
        a = adapter(_Ticker(), errors=[(354, "Requested market data is not subscribed")])
        with pytest.raises(ValueError) as exc:
            a.get_last_price("CSH2.L")
        msg = str(exc.value)
        assert "CSH2.L: no valid price" in msg
        assert "354: Requested market data is not subscribed" in msg

    def test_error_handler_detached_after_call(self, adapter):
        a = adapter(_Ticker(bid=1.0, ask=1.0))
        a.get_last_price("SPY")
        assert a._ib.errorEvent.handlers == []
        a = adapter(_Ticker())
        with pytest.raises(ValueError):
            a.get_last_price("SPY")
        assert a._ib.errorEvent.handlers == []

    def test_raises_when_contract_does_not_qualify(self, adapter):
        a = adapter(_Ticker(bid=1.0, ask=1.0), con_id=0)
        with pytest.raises(ValueError, match="contract qualification failed"):
            a.get_last_price("SPY")
