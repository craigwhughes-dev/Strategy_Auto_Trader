"""Protocol contracts for entry and exit strategy pairs.

Each named strategy (see strategy/registry.py) must provide one class
satisfying EntryStrategyProtocol and one satisfying ExitStrategyProtocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ...plugins.types import BarData, EntryDecision, ExitResult, RegimeState, TradeState


@runtime_checkable
class EntryStrategyProtocol(Protocol):
    """Decides whether to enter a trade on each bar."""

    def evaluate(
        self,
        regime: RegimeState,
        mom: dict,
        volume_ratio: float,
        currently_in: bool = False,
    ) -> EntryDecision: ...


@runtime_checkable
class ExitStrategyProtocol(Protocol):
    """Owns stop/target levels and per-bar exit logic."""

    @property
    def stop_loss_pct(self) -> float: ...

    @property
    def take_profit_pct(self) -> float: ...

    def check(self, trade: TradeState, bar: BarData) -> ExitResult: ...
