"""Tests for the Trading212 adapter and its account-identity guard.

T212Adapter._request() is the sole network seam — every test replaces it
with a stub rather than hitting the real API, mirroring how ibkr_adapter
tests fake the ib_async IB object.
"""

from __future__ import annotations

import pytest

from Strategy_Auto_Trader.broker.t212_adapter import (
    T212Adapter,
    T212AccountMismatchError,
)
from Strategy_Auto_Trader.broker.protocols import BrokerAdapterProtocol
from Strategy_Auto_Trader.broker.types import OrderRequest, StopOrderRequest
from Strategy_Auto_Trader.broker import t212_symbols


SYMBOL_MAP = {"AAPL": "AAPL_US_EQ", "SPY": "SPY_US_EQ"}


def make_adapter(**kwargs):
    kwargs.setdefault("symbol_map", dict(SYMBOL_MAP))
    return T212Adapter(api_key="k", api_secret="s", **kwargs)


class FakeRequest:
    """Records calls and returns queued responses in order."""

    def __init__(self, responses):
        self.calls = []
        self._responses = list(responses)

    def __call__(self, method, path, body=None):
        self.calls.append((method, path, body))
        if not self._responses:
            raise AssertionError(f"no more stubbed responses, got {method} {path}")
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


class TestProtocolAndGuard:
    def test_satisfies_protocol(self):
        assert isinstance(make_adapter(), BrokerAdapterProtocol)

    def test_demo_is_default_and_allowed_without_real_money(self):
        adapter = make_adapter()
        assert adapter._environment == "demo"
        assert adapter._base_url.startswith("https://demo.")

    def test_live_without_allow_live_account_raises(self):
        with pytest.raises(T212AccountMismatchError, match="allow_live_account"):
            make_adapter(environment="live", allow_live_account=False)

    def test_live_with_allow_live_account_constructs(self):
        adapter = make_adapter(environment="live", allow_live_account=True)
        assert adapter._base_url.startswith("https://live.")

    def test_invalid_environment_raises(self):
        with pytest.raises(ValueError):
            make_adapter(environment="paper")


class TestPlaceOrder:
    def test_buy_fills(self, monkeypatch):
        adapter = make_adapter()
        fake = FakeRequest([
            {"id": 42},                                  # POST market order
            {"status": "FILLED", "fillPrice": 195.5},    # poll
        ])
        monkeypatch.setattr(adapter, "_request", fake)
        fill = adapter.place_order(OrderRequest("AAPL", "BUY", 10))
        assert fill.fill_price == pytest.approx(195.5)
        assert fill.quantity == 10
        method, path, body = fake.calls[0]
        assert method == "POST" and path == "/api/v0/equity/orders/market"
        assert body == {"ticker": "AAPL_US_EQ", "quantity": 10}

    def test_sell_sends_negative_quantity(self, monkeypatch):
        adapter = make_adapter()
        fake = FakeRequest([
            {"id": 7},
            {"status": "FILLED", "fillPrice": 100.0},
        ])
        monkeypatch.setattr(adapter, "_request", fake)
        adapter.place_order(OrderRequest("AAPL", "SELL", 5))
        _, _, body = fake.calls[0]
        assert body == {"ticker": "AAPL_US_EQ", "quantity": -5}

    def test_unmapped_ticker_raises(self):
        adapter = make_adapter()
        with pytest.raises(KeyError, match="no T212 instrument code"):
            adapter.place_order(OrderRequest("UNKNOWN", "BUY", 1))

    def test_rejected_order_returns_none(self, monkeypatch):
        adapter = make_adapter()
        fake = FakeRequest([
            {"id": 8},
            {"status": "REJECTED"},
        ])
        monkeypatch.setattr(adapter, "_request", fake)
        assert adapter.place_order(OrderRequest("AAPL", "BUY", 1)) is None

    def test_unfilled_at_deadline_cancels_and_registers_pending(self, monkeypatch):
        adapter = make_adapter(timeout=0.0)
        fake = FakeRequest([
            {"id": 9},
            {"status": "WORKING"},   # single poll, deadline already passed
            {},                      # DELETE cancel response
        ])
        monkeypatch.setattr(adapter, "_request", fake)
        result = adapter.place_order(OrderRequest("AAPL", "BUY", 1))
        assert result is None
        assert len(adapter._pending_cancels) == 1
        assert adapter._pending_cancels[0]["order_id"] == 9


class TestPositionsAndOrders:
    def test_get_open_positions_maps_back_to_yfinance_ticker(self, monkeypatch):
        adapter = make_adapter()
        fake = FakeRequest([
            [{"instrument": {"ticker": "AAPL_US_EQ"}, "quantity": 10}],
        ])
        monkeypatch.setattr(adapter, "_request", fake)
        assert adapter.get_open_positions() == {"AAPL": 10}

    def test_get_open_positions_skips_unmapped(self, monkeypatch):
        adapter = make_adapter()
        fake = FakeRequest([
            [{"instrument": {"ticker": "MYSTERY_EQ"}, "quantity": 3},
             {"instrument": {"ticker": "AAPL_US_EQ"}, "quantity": 1}],
        ])
        monkeypatch.setattr(adapter, "_request", fake)
        assert adapter.get_open_positions() == {"AAPL": 1}

    def test_get_open_positions_reads_positions_endpoint_and_keeps_fractions(self, monkeypatch):
        adapter = make_adapter()
        fake = FakeRequest([
            [{"instrument": {"ticker": "AAPL_US_EQ"}, "quantity": 0.34094783}],
        ])
        monkeypatch.setattr(adapter, "_request", fake)
        assert adapter.get_open_positions() == {"AAPL": pytest.approx(0.34094783)}
        assert fake.calls[0][1] == "/api/v0/equity/positions"

    def test_place_order_polls_through_transient_404(self, monkeypatch):
        adapter = make_adapter()
        monkeypatch.setattr("Strategy_Auto_Trader.broker.t212_adapter.time.sleep", lambda s: None)
        not_yet_visible = RuntimeError("T212 GET /api/v0/equity/orders/7 failed: 404 {\"detail\":\"Order not found\"}")
        fake = FakeRequest([{"id": 7}, not_yet_visible, {"status": "FILLED", "fillPrice": 201.0}])
        monkeypatch.setattr(adapter, "_request", fake)
        fill = adapter.place_order(OrderRequest("AAPL", "SELL", 5))
        assert fill is not None
        assert fill.fill_price == pytest.approx(201.0)

    def test_place_order_maps_free_cash_rejection_to_insufficient_funds(self, monkeypatch):
        from Strategy_Auto_Trader.broker.types import InsufficientFundsError
        adapter = make_adapter()
        rejected = RuntimeError('T212 POST /api/v0/equity/orders/market failed: 400 {"type":"/api-errors/insufficient-free-for-stocks-buy"}')
        monkeypatch.setattr(adapter, "_request", FakeRequest([rejected]))
        with pytest.raises(InsufficientFundsError):
            adapter.place_order(OrderRequest("AAPL", "BUY", 10))

    def test_network_error_on_order_cancels_matching_open_order_and_raises_ambiguous(self, monkeypatch):
        from Strategy_Auto_Trader.broker.types import AmbiguousOrderError
        adapter = make_adapter()
        open_orders = [{"id": 9, "ticker": "AAPL_US_EQ", "side": "BUY"}, {"id": 4, "ticker": "AAPL_US_EQ", "side": "SELL"}]
        fake = FakeRequest([TimeoutError("timed out"), open_orders, {}])
        monkeypatch.setattr(adapter, "_request", fake)
        with pytest.raises(AmbiguousOrderError):
            adapter.place_order(OrderRequest("AAPL", "BUY", 10))
        deletes = [c for c in fake.calls if c[0] == "DELETE"]
        assert [c[1] for c in deletes] == ["/api/v0/equity/orders/9"]

    def test_rejection_is_not_reconciled(self, monkeypatch):
        adapter = make_adapter()
        rejected = RuntimeError('T212 POST /api/v0/equity/orders/market failed: 400 {"type":"/api-errors/quantity-precision-mismatch"}')
        fake = FakeRequest([rejected])
        monkeypatch.setattr(adapter, "_request", fake)
        with pytest.raises(RuntimeError, match="400"):
            adapter.place_order(OrderRequest("AAPL", "SELL", 1))
        assert len(fake.calls) == 1

    def test_ambiguous_error_still_raised_when_open_orders_cannot_be_listed(self, monkeypatch):
        from Strategy_Auto_Trader.broker.types import AmbiguousOrderError
        adapter = make_adapter()
        fake = FakeRequest([TimeoutError("timed out"), TimeoutError("list timed out")])
        monkeypatch.setattr(adapter, "_request", fake)
        with pytest.raises(AmbiguousOrderError):
            adapter.place_order(OrderRequest("AAPL", "BUY", 10))

    def test_get_available_cash_reads_account_summary(self, monkeypatch):
        adapter = make_adapter()
        fake = FakeRequest([{"cash": {"availableToTrade": 15958.68}}])
        monkeypatch.setattr(adapter, "_request", fake)
        assert adapter.get_available_cash() == pytest.approx(15958.68)
        assert fake.calls[0][1] == "/api/v0/equity/account/summary"

    def test_cancel_stop_order_not_found(self, monkeypatch):
        adapter = make_adapter()
        fake = FakeRequest([None])
        monkeypatch.setattr(adapter, "_request", fake)
        assert adapter.cancel_stop_order(123) == "NotFound"

    def test_cancel_stop_order_already_filled(self, monkeypatch):
        adapter = make_adapter()
        fake = FakeRequest([{"status": "FILLED"}])
        monkeypatch.setattr(adapter, "_request", fake)
        assert adapter.cancel_stop_order(123) == "Filled"

    def test_cancel_stop_order_cancels(self, monkeypatch):
        adapter = make_adapter()
        fake = FakeRequest([{"status": "WORKING"}, {}])
        monkeypatch.setattr(adapter, "_request", fake)
        assert adapter.cancel_stop_order(123) == "Cancelled"


class TestStopOrders:
    def test_place_stop_order_sends_negative_quantity(self, monkeypatch):
        adapter = make_adapter()
        fake = FakeRequest([{"id": 55}])
        monkeypatch.setattr(adapter, "_request", fake)
        result = adapter.place_stop_order(StopOrderRequest("AAPL", 10, 190.0))
        assert result.perm_id == 55
        assert result.stop_price == 190.0
        _, _, body = fake.calls[0]
        assert body == {"ticker": "AAPL_US_EQ", "quantity": -10, "stopPrice": 190.0}


class TestPriceCache:
    def test_get_last_price_raises_when_neither_set_nor_held(self, monkeypatch):
        adapter = make_adapter()
        monkeypatch.setattr(adapter, "_request", FakeRequest([[]]))
        with pytest.raises(ValueError, match="no price available"):
            adapter.get_last_price("AAPL")

    def test_get_last_price_returns_cached_value_without_hitting_api(self, monkeypatch):
        adapter = make_adapter()
        fake = FakeRequest([])
        monkeypatch.setattr(adapter, "_request", fake)
        adapter.set_prices({"AAPL": 201.25})
        assert adapter.get_last_price("AAPL") == pytest.approx(201.25)
        assert fake.calls == []

    def test_get_last_price_falls_back_to_held_position_current_price(self, monkeypatch):
        adapter = make_adapter()
        fake = FakeRequest([[{"instrument": {"ticker": "AAPL_US_EQ"}, "quantity": 2, "currentPrice": 201.5}]])
        monkeypatch.setattr(adapter, "_request", fake)
        assert adapter.get_last_price("AAPL") == pytest.approx(201.5)

    def test_held_prices_cached_within_ttl(self, monkeypatch):
        adapter = make_adapter()
        fake = FakeRequest([[{"instrument": {"ticker": "AAPL_US_EQ"}, "quantity": 2, "currentPrice": 201.5}]])
        monkeypatch.setattr(adapter, "_request", fake)
        clock = [1000.0]
        monkeypatch.setattr("Strategy_Auto_Trader.broker.t212_adapter.time.monotonic", lambda: clock[0])
        adapter.get_last_price("AAPL")
        clock[0] += 5.0
        assert adapter.get_last_price("AAPL") == pytest.approx(201.5)
        assert len(fake.calls) == 1

    def test_set_prices_takes_precedence_over_held_price(self, monkeypatch):
        adapter = make_adapter()
        fake = FakeRequest([[{"instrument": {"ticker": "AAPL_US_EQ"}, "quantity": 2, "currentPrice": 201.5}]])
        monkeypatch.setattr(adapter, "_request", fake)
        adapter.set_prices({"AAPL": 200.0})
        assert adapter.get_last_price("AAPL") == pytest.approx(200.0)
        assert fake.calls == []


class TestSymbolMap:
    def test_load_symbol_map_missing_file_returns_empty(self, tmp_path):
        assert t212_symbols.load_symbol_map(tmp_path / "nope.json") == {}

    def test_t212_ticker_raises_when_unmapped(self):
        with pytest.raises(KeyError, match="no T212 instrument code"):
            t212_symbols.t212_ticker("ZZZZ", {})

    def test_t212_ticker_resolves(self):
        assert t212_symbols.t212_ticker("AAPL", SYMBOL_MAP) == "AAPL_US_EQ"

    def test_yfinance_ticker_from_t212_inverse(self):
        assert t212_symbols.yfinance_ticker_from_t212("AAPL_US_EQ", SYMBOL_MAP) == "AAPL"

    def test_yfinance_ticker_from_t212_raises_when_unmapped(self):
        with pytest.raises(KeyError, match="reverse lookup"):
            t212_symbols.yfinance_ticker_from_t212("UNKNOWN_EQ", SYMBOL_MAP)
