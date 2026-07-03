"""Shared dataclasses for the broker execution layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OrderRequest:
    ticker: str
    action: str       # "BUY" | "SELL"
    quantity: int
    order_type: str = "MKT"


@dataclass
class FillResult:
    ticker: str
    action: str
    fill_price: float
    quantity: int
    timestamp: str    # ISO-8601 UTC


@dataclass
class PositionRecord:
    entry_date: str
    fill_price: float
    quantity: int
    kelly_fraction: float
    stop_level: float
    target_level: float
