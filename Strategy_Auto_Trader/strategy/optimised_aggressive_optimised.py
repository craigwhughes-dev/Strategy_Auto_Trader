"""Optimised-Aggressive + high-conviction entry gate.

Differentiator: Winners return 0.044%, losers 0.004% (r=0.889).
Gate: Only enter on strong composite signal score (> 6.0).
"""

from __future__ import annotations

from ..plugins.types import EntryDecision, RegimeState
from .optimised_aggressive import OptimisedAggressiveEntry as BaseOptimisedAggressiveEntry
from .optimised_aggressive import OptimisedAggressiveExit


class OptimisedAggressiveOptimisedEntry(BaseOptimisedAggressiveEntry):
    """Optimised-Aggressive entry with high-conviction signal gate."""

    min_signal_score: float = 6.0

    def evaluate(
        self,
        regime: RegimeState,
        mom: dict,
        _volume_ratio: float,
        currently_in: bool = False,
    ) -> EntryDecision:
        decision = super().evaluate(regime, mom, _volume_ratio, currently_in)

        if decision.flag == "BUY" and decision.score < self.min_signal_score:
            return EntryDecision(
                flag="HOLD", raw_flag=decision.raw_flag, score=decision.score,
                reason=f"signal_gate: weak conviction (score={decision.score:.2f}, need >={self.min_signal_score})",
            )
        return decision


class OptimisedAggressiveOptimisedExit(OptimisedAggressiveExit):
    """Identical to Optimised-Aggressive exit."""
