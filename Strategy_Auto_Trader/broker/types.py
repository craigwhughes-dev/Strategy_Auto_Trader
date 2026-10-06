"""Shared dataclasses for the broker execution layer."""

from __future__ import annotations

from dataclasses import dataclass

# T212 holds fractional shares; 5 dp covers its precision without float noise in sums
QUANTITY_DECIMALS = 5


def round_quantity(quantity: float) -> float:
    """Quantity rounded to the ledger's precision, so equal holdings compare equal."""
    return round(float(quantity), QUANTITY_DECIMALS)


class AmbiguousOrderError(RuntimeError):
    """An order errored without a clear rejection, so it may have been placed. Any matching open order was
    cancelled; holdings must be re-read from the broker before acting on this ticker again."""


class InsufficientFundsError(RuntimeError):
    """The broker refused a buy for lack of free cash. Market buys can fill above the quote, so a smaller
    retry can succeed."""


@dataclass
class OrderRequest:
    ticker: str
    action: str       # "BUY" | "SELL"
    quantity: float
    order_type: str = "MKT"


@dataclass
class FillResult:
    ticker: str
    action: str
    fill_price: float
    quantity: float
    timestamp: str    # ISO-8601 UTC


@dataclass
class PositionRecord:
    entry_date: str
    fill_price: float
    quantity: float
    kelly_fraction: float
    stop_level: float
    target_level: float


@dataclass
class StopOrderRequest:
    ticker: str
    quantity: float
    stop_price: float


@dataclass
class StopOrderResult:
    perm_id: int
    stop_price: float
    timestamp: str


@dataclass
class OpenOrderInfo:
    ticker: str
    quantity: float
    stop_price: float
    perm_id: int


@dataclass
class PendingCancelEvent:
    """Result of resolving a previously-unconfirmed cancel request.

    outcome is one of: "cancelled" (confirmed, no longer working),
    "filled" (filled after the cancel was requested — a real position
    that needs reconciling), "timeout_alert" (still unconfirmed after
    the alert threshold — fires once, not every cycle)."""
    ticker: str
    action: str
    quantity: float
    outcome: str
    fill: FillResult | None = None
    elapsed_minutes: float = 0.0
