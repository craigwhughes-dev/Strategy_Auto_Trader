"""typing.Protocol contracts for each of the six plugin slots.

All protocols use @runtime_checkable so isinstance() checks work in tests.
Implementations need only satisfy the structural interface — no inheritance.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import BarData, ExitResult, RegimeState, TradeState


@runtime_checkable
class RegimeModelProtocol(Protocol):
    """Stateful HMM regime model.  Manages its own refit schedule and
    forward-filter cache."""

    def needs_refit(self, t: int) -> bool: ...
    def refit(self, returns: np.ndarray) -> None: ...
    def step(self, returns: np.ndarray, t: int) -> RegimeState | None: ...
    def reset(self) -> None: ...


@runtime_checkable
class SignalGeneratorProtocol(Protocol):
    """Stateless composite signal: maps (RegimeState, mom_snap) -> raw signal dict."""

    def generate(self, regime: RegimeState, mom: dict) -> dict: ...


@runtime_checkable
class QualityGateProtocol(Protocol):
    """Stateless veto layer: may downgrade BUY to HOLD or force SELL."""

    def apply(
        self, signal: dict, mom: dict, regime_signal: float, currently_in: bool,
    ) -> dict: ...


@runtime_checkable
class ExitRulesProtocol(Protocol):
    """Stateless per-bar exit checker.  Returns an ExitResult with updated
    peak/days_in_trade state for carry-forward."""

    def check(self, trade: TradeState, bar: BarData) -> ExitResult: ...


@runtime_checkable
class PositionSizerProtocol(Protocol):
    """Stateless Kelly formula: compute position size from a trade history slice."""

    def size(self, trade_history: list[float]) -> float: ...


@runtime_checkable
class ContextAdjusterProtocol(Protocol):
    """One-shot per-backtest threshold nudge (sentiment / VIX).
    Returns (entry_prob, exit_prob, stop_loss_pct) after adjustment."""

    def adjust(
        self,
        entry_prob: float,
        exit_prob: float,
        stop_loss_pct: float,
        sentiment_score: float,
        vix_signal: int,
    ) -> tuple[float, float, float]: ...


@runtime_checkable
class PrescreenProtocol(Protocol):
    """Per-ticker volatility prescreen.  Returns a profile dict or None if
    the ticker should be skipped."""

    def __call__(self, ticker: str) -> dict | None: ...
