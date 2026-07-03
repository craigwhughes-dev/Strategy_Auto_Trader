"""Context adjuster plugin: sentiment / VIX threshold nudges.

SentimentAdjuster — default, wraps _compute_effective_thresholds.
NullAdjuster       — identity, returns thresholds unchanged.
"""

from __future__ import annotations

from ..quant_hmm.quant_engine import _compute_effective_thresholds


class SentimentAdjuster:
    """Applies sentiment_score and vix_signal nudges to entry/exit thresholds.

    Delegates to _compute_effective_thresholds from quant_engine.
    Satisfies ContextAdjusterProtocol.
    """

    def adjust(
        self,
        entry_prob: float,
        exit_prob: float,
        stop_loss_pct: float,
        sentiment_score: float = 0.0,
        vix_signal: int = 0,
    ) -> tuple[float, float, float]:
        return _compute_effective_thresholds(
            entry_prob, exit_prob, stop_loss_pct, sentiment_score, vix_signal,
        )


class NullAdjuster:
    """Identity context adjuster — returns thresholds unchanged.

    Use this when no sentiment or VIX data is available.
    Satisfies ContextAdjusterProtocol.
    """

    def adjust(
        self,
        entry_prob: float,
        exit_prob: float,
        stop_loss_pct: float,
        sentiment_score: float = 0.0,
        vix_signal: int = 0,
    ) -> tuple[float, float, float]:
        return entry_prob, exit_prob, stop_loss_pct
