"""NullBroker — fake broker for unit tests and --dry-run mode.

Fills orders immediately at the price provided in the prices dict.
Records all fills to self.orders for assertion in tests.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .types import (
    FillResult,
    OrderRequest,
    StopOrderRequest,
    StopOrderResult,
    OpenOrderInfo,
)


class NullBroker:
    """Satisfies BrokerAdapterProtocol without any network connection."""

    def __init__(self, prices: dict[str, float] | None = None) -> None:
        self._prices: dict[str, float] = prices or {}
        self._positions: dict[str, int] = {}
        self.orders: list[FillResult] = []
        self._next_perm_id: int = 100000
        self._stop_orders: dict[int, dict] = {}

    def set_prices(self, prices: dict[str, float]) -> None:
        """Update fill prices (dry-run daemon feeds latest closes each cycle)."""
        self._prices.update(prices)

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def get_last_price(self, ticker: str) -> float:
        return self._prices.get(ticker, 0.0)

    def place_order(self, req: OrderRequest) -> FillResult:
        price = self._prices.get(req.ticker, 0.0)
        fill = FillResult(
            ticker=req.ticker,
            action=req.action,
            fill_price=price,
            quantity=req.quantity,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.orders.append(fill)
        if req.action == "BUY":
            self._positions[req.ticker] = (
                self._positions.get(req.ticker, 0) + req.quantity
            )
        elif req.action == "SELL":
            self._positions.pop(req.ticker, None)
        return fill

    def get_open_positions(self) -> dict[str, int]:
        return dict(self._positions)

    def get_open_orders(self) -> list[dict]:
        """Dry-run fills synchronously — nothing is ever left working."""
        return []

    def place_stop_order(self, req: StopOrderRequest) -> StopOrderResult | None:
        """Stub: place and track a stop order with an incrementing permId."""
        perm_id = self._next_perm_id
        self._next_perm_id += 1
        self._stop_orders[perm_id] = {
            "ticker": req.ticker,
            "quantity": req.quantity,
            "stop_price": req.stop_price,
            "status": "Submitted",
        }
        return StopOrderResult(
            perm_id=perm_id,
            stop_price=req.stop_price,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_open_stop_orders(self) -> dict[int, OpenOrderInfo]:
        """Stub: return all open stop orders."""
        result = {}
        for perm_id, order in self._stop_orders.items():
            if order["status"] == "Submitted":
                result[perm_id] = OpenOrderInfo(
                    ticker=order["ticker"],
                    quantity=order["quantity"],
                    stop_price=order["stop_price"],
                    perm_id=perm_id,
                )
        return result

    def cancel_stop_order(self, perm_id: int) -> str:
        """Stub: cancel a stop order by permId."""
        if perm_id not in self._stop_orders:
            return "NotFound"
        if self._stop_orders[perm_id]["status"] == "Filled":
            return "Filled"
        self._stop_orders[perm_id]["status"] = "Cancelled"
        return "Cancelled"

    def get_stop_fill(self, perm_id: int) -> FillResult | None:
        """Stub: no fills tracked for stops in dry-run."""
        return None
