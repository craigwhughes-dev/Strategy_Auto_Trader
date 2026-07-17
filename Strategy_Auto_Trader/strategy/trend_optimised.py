"""Trend + HMM regime strength filter.

Differentiator: Winners r=0.956, losers r=-0.069.
Gate: Only enter if HMM P(Bull) > 0.6.
"""

from __future__ import annotations

from ..plugins.types import EntryDecision, RegimeState
from .trend_follow import TrendEntry as BaseTrendEntry
from .trend_follow import TrendExit


class TrendOptimisedEntry(BaseTrendEntry):
    """Trend entry with strong bull regime gate."""

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


class TrendOptimisedExit(TrendExit):
    """Identical to Trend exit."""
