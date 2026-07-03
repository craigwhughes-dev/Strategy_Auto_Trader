"""Quality gate plugin implementations.

QualityGatePlugin — default, wraps core/quality_gate._apply_quality_gate.
NullQualityGate   — identity pass-through (no veto).
"""

from __future__ import annotations

from ..core.quality_gate import _apply_quality_gate


class QualityGatePlugin:
    """Default quality gate: delegates to core/quality_gate._apply_quality_gate.

    Satisfies QualityGateProtocol.
    """

    def apply(
        self, signal: dict, mom: dict, regime_signal: float, currently_in: bool,
    ) -> dict:
        return _apply_quality_gate(signal, mom, regime_signal, currently_in)


class NullQualityGate:
    """No-op quality gate — passes all signals through unchanged.

    Useful for backtests where the gate should be disabled.
    Satisfies QualityGateProtocol.
    """

    def apply(
        self, signal: dict, mom: dict, regime_signal: float, currently_in: bool,
    ) -> dict:
        return dict(signal)
