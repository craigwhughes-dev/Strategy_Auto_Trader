"""Choppy-Vol + HMM regime strength filter.

>>> NOT VALIDATED — DO NOT USE FOR LIVE CAPITAL <<<
Base choppy_vol strategy is decisively negative (199/203 losers).
This variant adds regime gate for potential future research, not live use.

Differentiator: Winners r=0.530, losers r=-0.319.
Gate: Only enter if HMM P(Bull) > 0.5 (relaxed vs other strategies).
"""

from __future__ import annotations

from ..plugins.types import EntryDecision, RegimeState
from .choppy_vol import ChoppyVolEntry as BaseChoppyVolEntry
from .choppy_vol import ChoppyVolExit


class ChoppyVolOptimisedEntry(BaseChoppyVolEntry):
    """Choppy-Vol entry with weak bull regime gate."""

    def evaluate(
        self,
        regime: RegimeState,
        mom: dict,
        _volume_ratio: float,
        currently_in: bool = False,
    ) -> EntryDecision:
        p_bull = regime.p_bull if regime.p_bull is not None else 0.0
        if not currently_in and p_bull <= 0.5:
            return EntryDecision(
                flag="HOLD", raw_flag="HOLD", score=0.0,
                reason=f"regime_gate: weak bull (P_Bull={p_bull:.2f}, need >0.5)",
            )
        return super().evaluate(regime, mom, _volume_ratio, currently_in)


class ChoppyVolOptimisedExit(ChoppyVolExit):
    """Identical to Choppy-Vol exit."""
