"""Kelly-criterion position sizer (default PositionSizer implementation)."""

from __future__ import annotations

import numpy as np

from ..quant_hmm.quant_engine import kelly_fraction


class KellySizer:
    """Maintains the rolling trade-result history and computes Kelly fraction.

    Satisfies PositionSizerProtocol.
    """

    def __init__(
        self,
        use_kelly: bool = True,
        lookback: int = 20,
        default: float = 0.10,
        min_position: float = 0.02,
    ) -> None:
        self._use_kelly = use_kelly
        self._lookback = lookback
        self._min_position = min_position
        self._current = default
        self._trade_results: list[float] = []

    # ------------------------------------------------------------------
    # Protocol method
    # ------------------------------------------------------------------

    def size(self, trade_history: list[float]) -> float:
        """Stateless Kelly calculation from a trade-history slice.

        trade_history should be a recent window (e.g. the last ``lookback``
        closed trades), each entry a fractional P&L (positive = win).
        """
        if not trade_history:
            return 0.0
        wins = [r for r in trade_history if r > 0]
        losses = [r for r in trade_history if r < 0]
        wr = len(wins) / len(trade_history)
        aw = float(np.mean(wins)) if wins else 0.0
        al = float(np.mean(losses)) if losses else -0.05
        return kelly_fraction(wr, aw, al)

    # ------------------------------------------------------------------
    # Stateful helpers used by consolidated_engine
    # ------------------------------------------------------------------

    def record(self, trade_pl: float) -> None:
        """Append a completed trade's P&L and update the stored Kelly estimate."""
        self._trade_results.append(trade_pl)
        if self._use_kelly and len(self._trade_results) >= self._lookback:
            new_k = self.size(self._trade_results[-self._lookback:])
            self._current = new_k

    @property
    def position(self) -> float:
        """Current position fraction, floored at ``min_position``."""
        return max(self._min_position, self._current)

    @property
    def trade_results(self) -> list[float]:
        return self._trade_results

    @property
    def current_kelly(self) -> float:
        return self._current


class FixedSizer:
    """Fixed-fraction position sizer — always returns the same allocation.

    Use instead of KellySizer when Kelly variance is undesirable or when
    there is insufficient trade history for a stable Kelly estimate.
    Satisfies PositionSizerProtocol and exposes the same stateful interface
    as KellySizer so consolidated_engine can treat them interchangeably.
    """

    def __init__(self, fraction: float = 0.10) -> None:
        self._fraction = fraction
        self._trade_results: list[float] = []

    def size(self, trade_history: list[float]) -> float:
        return self._fraction

    def record(self, trade_pl: float) -> None:
        self._trade_results.append(trade_pl)

    @property
    def position(self) -> float:
        return self._fraction

    @property
    def trade_results(self) -> list[float]:
        return self._trade_results

    @property
    def current_kelly(self) -> float:
        return self._fraction
