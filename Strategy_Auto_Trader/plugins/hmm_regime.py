"""HMM regime model plugin (default RegimeModel implementation).

HMMRegimeModel encapsulates all mutable state from the original
consolidated_engine per-bar loop:
  - current_model / current_order (Gaussian HMM + state ordering)
  - _log_alpha (incremental forward-filter cache; reset on refit)
  - p_bull_history (rolling buffer for regime smoothing)

The step() method mirrors the original inline logic exactly, returning
RegimeState so the engine loop can access p_bull / p_bear / hmm_vote
without holding this state itself.
"""

from __future__ import annotations

import numpy as np

from ..quant_hmm.quant_engine import (
    _forward_step_incremental,
    discretize_p_bull,
    fit_hmm_expanding,
)
from .types import RegimeState


class HMMRegimeModel:
    """Gaussian HMM regime model with expanding-window refit schedule.

    Satisfies RegimeModelProtocol.
    """

    def __init__(
        self,
        min_train_bars: int = 500,
        refit_bars: int = 500,
        regime_smooth: int = 24,
        n_seeds: int = 3,
        n_iter: int = 50,
        bull_edge: float = 0.65,
        bear_edge: float = 0.40,
    ) -> None:
        self._min_train = min_train_bars
        self._refit_bars = refit_bars
        self._smooth = regime_smooth
        self._n_seeds = n_seeds
        self._n_iter = n_iter
        self._bull_edge = bull_edge
        self._bear_edge = bear_edge

        self._model = None
        self._order = None
        self._log_alpha: np.ndarray | None = None
        self._p_bull_history: list[float] = []

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def needs_refit(self, t: int) -> bool:
        return self._model is None or (t - self._min_train) % self._refit_bars == 0

    def refit(self, returns: np.ndarray) -> None:
        result = fit_hmm_expanding(returns, n_seeds=self._n_seeds, n_iter=self._n_iter)
        if result is not None:
            self._model, self._order = result
            self._log_alpha = None   # must reset cache after any refit

    def step(self, returns: np.ndarray, t: int) -> RegimeState | None:
        """Advance the forward filter by one bar.  Returns None if the
        model has not yet been fitted."""
        if self._model is None:
            return None

        p_bull, p_bear, self._log_alpha = _forward_step_incremental(
            self._model, self._order, returns, t, self._log_alpha,
        )
        self._p_bull_history.append(p_bull)

        if len(self._p_bull_history) >= self._smooth:
            p_bull_smooth = float(np.mean(self._p_bull_history[-self._smooth:]))
        else:
            p_bull_smooth = p_bull

        return RegimeState(
            p_bull=p_bull,
            p_bear=p_bear,
            p_bull_smooth=p_bull_smooth,
            regime_signal=p_bull_smooth - p_bear,
            hmm_vote=discretize_p_bull(p_bull_smooth, self._bull_edge, self._bear_edge),
        )

    def reset(self) -> None:
        self._model = None
        self._order = None
        self._log_alpha = None
        self._p_bull_history.clear()
