"""AI + HMM regime strength filter.

Differentiator: Winners r=0.604, losers r=0.348.
Gate: Only enter if HMM P(Bull) > 0.6.
"""

from __future__ import annotations

from ..plugins.types import EntryDecision, RegimeState
from .ai_strategy import AiEntry as BaseAiEntry
from .ai_strategy import AiExit


class AiOptimisedEntry(BaseAiEntry):
    """AI entry with strong bull regime gate."""

    def evaluate(
        self,
        regime: RegimeState,
        mom: dict,
        _volume_ratio: float,
        currently_in: bool = False,
    ) -> EntryDecision:
        p_bull = regime.p_bull if regime.p_bull is not None else 0.0
        if p_bull <= 0.6:
            return EntryDecision(
                flag="HOLD", raw_flag="HOLD", score=0.0,
                reason=f"regime_gate: weak bull (P_Bull={p_bull:.2f}, need >0.6)",
            )
        return super().evaluate(regime, mom, _volume_ratio, currently_in)


class AiOptimisedExit(AiExit):
    """Identical to AI exit."""
