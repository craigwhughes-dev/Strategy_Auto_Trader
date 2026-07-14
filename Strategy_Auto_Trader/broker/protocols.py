"""Protocol contract for broker adapters.

Any concrete adapter (IBKRAdapter, NullBroker, …) must satisfy
BrokerAdapterProtocol so it can be swapped in tests or extended later.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import FillResult, OrderRequest


@runtime_checkable
class BrokerAdapterProtocol(Protocol):
    """Minimal interface for placing and querying orders."""

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_last_price(self, ticker: str) -> float: ...
    def place_order(self, req: OrderRequest) -> FillResult | None: ...
    def get_open_positions(self) -> dict[str, int]: ...
