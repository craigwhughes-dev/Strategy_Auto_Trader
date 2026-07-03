"""Composite-vote signal generator (default SignalGenerator implementation).

CompositeSignalGenerator wraps core/momentum.composite_signal, injecting
the HMM vote from RegimeState and using weights captured at construction.
"""

from __future__ import annotations

from ..core.momentum import composite_signal
from .types import RegimeState


class CompositeSignalGenerator:
    """Default signal generator: wraps composite_signal from core/momentum.

    The markov_signal is always passed as 0.0 because the consolidated engine
    zeroes the 'markov' weight slot — the HMM carry that role via hmm_state.

    Satisfies SignalGeneratorProtocol.
    """

    def __init__(
        self,
        weights: dict,
        buy_threshold: float = 3.0,
        sell_threshold: float = -3.0,
    ) -> None:
        self._weights = weights
        self._buy_threshold = buy_threshold
        self._sell_threshold = sell_threshold

    def generate(self, regime: RegimeState, mom: dict) -> dict:
        return composite_signal(
            0.0,
            mom,
            sell_threshold=self._sell_threshold,
            hmm_state=regime.hmm_vote,
            buy_threshold=self._buy_threshold,
            weights=self._weights,
        )
