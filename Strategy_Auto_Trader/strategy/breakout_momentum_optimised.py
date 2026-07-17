"""Breakout-Momentum + HMM regime strength filter.

Differentiator: Winners r=0.954, losers r=-0.071.
Gate: Only enter if HMM P(Bull) > 0.6.
"""

from __future__ import annotations

from ..plugins.types import EntryDecision, RegimeState
from .breakout_momentum import BreakoutMomentumEntry as BaseBreakoutMomentumEntry
from .breakout_momentum import BreakoutMomentumExit


class BreakoutMomentumOptimisedEntry(BaseBreakoutMomentumEntry):
    """Breakout-momentum entry with strong bull regime gate."""

    def evaluate(
        self,
        regime: RegimeState,
        mom: dict,
        volume_ratio: float,
        currently_in: bool = False,
    ) -> EntryDecision:
        p_bull = regime.p_bull if regime.p_bull is not None else 0.0
        if p_bull <= 0.6:
            return EntryDecision(
                flag="HOLD", raw_flag="HOLD", score=0.0,
                reason=f"regime_gate: weak bull (P_Bull={p_bull:.2f}, need >0.6)",
            )
        return super().evaluate(regime, mom, volume_ratio, currently_in)


class BreakoutMomentumOptimisedExit(BreakoutMomentumExit):
    """Identical to Breakout-Momentum exit."""
