"""NullBroker — fake broker for unit tests and --dry-run mode.

Fills orders immediately at the price provided in the prices dict.
Records all fills to self.orders for assertion in tests.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .types import FillResult, OrderRequest


class NullBroker:
    """Satisfies BrokerAdapterProtocol without any network connection."""

    def __init__(self, prices: dict[str, float] | None = None) -> None:
        self._prices: dict[str, float] = prices or {}
        self._positions: dict[str, int] = {}
        self.orders: list[FillResult] = []

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
