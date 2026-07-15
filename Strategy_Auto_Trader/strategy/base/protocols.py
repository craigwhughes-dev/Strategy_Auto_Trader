"""Protocol contracts for entry and exit strategy pairs.

Each named strategy (see strategy/registry.py) must provide one class
satisfying EntryStrategyProtocol and one satisfying ExitStrategyProtocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ...plugins.types import BarData, EntryDecision, ExitResult, RegimeState, TradeState


@runtime_checkable
class EntryStrategyProtocol(Protocol):
    """Decides whether to enter a trade on each bar.

    Optional attributes (not enforced by Protocol, but supported by the engine):
    - require_flip_entry (bool): If False, allows entry on consecutive BUY bars
      (bypasses the flip-guard requirement that entry transitions from non-BUY to BUY).
      Defaults to True (requires flip). Strategies that don't declare this attribute
      fall back to the default True behavior.
    - quality_gate_enabled (bool): Whether the strategy applies
      core/quality_gate._apply_quality_gate on top of its own signal. This is
      read and used inside each strategy's own evaluate() (the engine does not
      read it directly) — documented here so it's discoverable alongside the
      other optional attributes. Defaults to True.
    """

    def evaluate(
        self,
        regime: RegimeState,
        mom: dict,
        volume_ratio: float,
        currently_in: bool = False,
    ) -> EntryDecision: ...


@runtime_checkable
class ExitStrategyProtocol(Protocol):
    """Owns stop/target levels and per-bar exit logic.

    Optional attributes (not enforced by Protocol, but supported by the engine):
    - min_hold_bars (int): Minimum bars to hold before allowing signal-based SELL.
      If declared, the engine uses it instead of the global CLI `min_hold_bars` param.
      Defaults to the engine's own `min_hold_bars` param if not declared.
    - use_kelly (bool): Whether the engine sizes positions with Kelly-fraction
      sizing (True) or a fixed/degenerate fraction (False). Read via getattr
      in consolidated_backtest; falls back to the engine's own `use_kelly` kwarg
      if not declared. Defaults to True.
    - kelly_lookback (int): Trailing trade-count window for the Kelly estimate.
      Read via getattr in consolidated_backtest; falls back to the engine's own
      `kelly_lookback` kwarg if not declared. Defaults to 20.
    """

    @property
    def stop_loss_pct(self) -> float: ...

    @property
    def take_profit_pct(self) -> float: ...

    def check(self, trade: TradeState, bar: BarData) -> ExitResult: ...
