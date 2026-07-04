"""Persistent HMM regime model — carries forward-filter state across runs.

PersistentHMMRegimeModel wraps HMMRegimeModel with an on-disk cache so that
repeated runs over overlapping data windows (the live daemon's hourly cycles)
only pay for genuinely new bars instead of re-fitting and re-stepping the
whole history every time.

What is cached per ticker (one pickle file):
  - the fitted hmmlearn model + state ordering
  - the incremental forward-filter vector (_log_alpha) at the last stepped bar
  - the p_bull smoothing history tail
  - the per-bar p_bull / p_bear / p_bull_smooth series with their bar dates
  - the refit cadence counter (bars since last refit)

Cache validity: the run's dates must align as a contiguous window inside (or
extending) the cached window, the overlapping closes must match exactly, and
all model parameters must be identical. Any mismatch (data revision, changed
params, corrupt file) silently falls back to a full recompute which rebuilds
the cache.

Note on determinism: refits continue on the persisted every-`refit_bars`
cadence, whereas a cold run schedules refits by index into whatever data
window was fetched. Once windows slide, a cold rerun's refit timeline shifts
bar by bar anyway, so the persisted cadence (anchored to real elapsed bars)
is the more consistent walk-forward behaviour, not a compromise.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ..quant_hmm.quant_engine import discretize_p_bull
from .hmm_regime import HMMRegimeModel
from .types import RegimeState

_CACHE_VERSION = 1


class PersistentHMMRegimeModel(HMMRegimeModel):
    """HMMRegimeModel with an on-disk per-ticker cache of filter state.

    Satisfies RegimeModelProtocol.
    """

    def __init__(
        self,
        cache_path: str | Path,
        dates,
        closes,
        *,
        min_train_bars: int = 500,
        refit_bars: int = 500,
        regime_smooth: int = 24,
        n_seeds: int = 3,
        n_iter: int = 50,
        bull_edge: float = 0.65,
        bear_edge: float = 0.40,
    ) -> None:
        super().__init__(
            min_train_bars=min_train_bars,
            refit_bars=refit_bars,
            regime_smooth=regime_smooth,
            n_seeds=n_seeds,
            n_iter=n_iter,
            bull_edge=bull_edge,
            bear_edge=bear_edge,
        )
        self._cache_path = Path(cache_path)
        self._dates = pd.DatetimeIndex(dates).asi8.copy()
        self._closes = np.asarray(closes, dtype=float).copy()
        if len(self._dates) != len(self._closes):
            raise ValueError("dates and closes must have the same length")

        n = len(self._dates)
        self._p_bull_arr = np.full(n, np.nan)
        self._p_bear_arr = np.full(n, np.nan)
        self._p_smooth_arr = np.full(n, np.nan)

        self._bars_since_refit = 0
        self._cached_through = -1   # highest bar index served from cache (this run's indexing)
        self._last_step_idx = -1    # highest bar index with a known state (cached or computed)

        #: diagnostics — bars actually advanced through the filter this run
        self.computed_steps = 0
        #: diagnostics — bars answered straight from the cache this run
        self.cache_hits = 0

        self._load_cache()

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def _params(self) -> dict:
        return {
            "version": _CACHE_VERSION,
            "min_train": self._min_train,
            "refit_bars": self._refit_bars,
            "smooth": self._smooth,
            "n_seeds": self._n_seeds,
            "n_iter": self._n_iter,
            "bull_edge": self._bull_edge,
            "bear_edge": self._bear_edge,
        }

    def _load_cache(self) -> None:
        if not self._cache_path.exists():
            return
        try:
            with open(self._cache_path, "rb") as fh:
                cache = pickle.load(fh)
        except Exception:
            return
        if not isinstance(cache, dict) or cache.get("params") != self._params():
            return

        c_dates = cache.get("dates")
        c_closes = cache.get("closes")
        if c_dates is None or c_closes is None or len(c_dates) == 0:
            return

        # Align this run's window as a contiguous suffix-overlap of the cache.
        # A window starting before the cache would need filter state we do not
        # have, so it invalidates.
        offset = int(np.searchsorted(c_dates, self._dates[0]))
        if offset >= len(c_dates) or c_dates[offset] != self._dates[0]:
            return
        overlap = min(len(c_dates) - offset, len(self._dates))
        if overlap <= 0:
            return
        if not np.array_equal(c_dates[offset:offset + overlap], self._dates[:overlap]):
            return
        # Overlapping closes must match exactly — a data revision means every
        # downstream probability would differ, so recompute from scratch.
        if not np.allclose(c_closes[offset:offset + overlap],
                           self._closes[:overlap], rtol=0.0, atol=1e-9):
            return

        last_step_cached = int(cache.get("last_step_idx", -1))
        cached_through = last_step_cached - offset
        if cached_through < 0:
            return

        span = min(overlap, cached_through + 1)
        self._p_bull_arr[:span] = cache["p_bull"][offset:offset + span]
        self._p_bear_arr[:span] = cache["p_bear"][offset:offset + span]
        self._p_smooth_arr[:span] = cache["p_smooth"][offset:offset + span]

        self._model = cache["model"]
        self._order = cache["order"]
        self._log_alpha = cache["log_alpha"]
        self._p_bull_history = list(cache["p_bull_history"])
        self._bars_since_refit = int(cache["bars_since_refit"])
        self._cached_through = min(cached_through, len(self._dates) - 1)
        self._last_step_idx = self._cached_through

    def save(self) -> None:
        """Persist the current filter state and probability series to disk.

        No-ops when this run computed nothing new: the existing cache file
        already holds equal-or-later filter state, and overwriting it from a
        shorter data window would pair that later state with an earlier
        last-step marker, corrupting the next run's continuation point.
        """
        if self._last_step_idx < 0 or self.computed_steps == 0:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache = {
            "params": self._params(),
            "dates": self._dates,
            "closes": self._closes,
            "p_bull": self._p_bull_arr,
            "p_bear": self._p_bear_arr,
            "p_smooth": self._p_smooth_arr,
            "last_step_idx": int(self._last_step_idx),
            "model": self._model,
            "order": self._order,
            "log_alpha": self._log_alpha,
            "p_bull_history": self._p_bull_history[-max(self._smooth * 2, 50):],
            "bars_since_refit": int(self._bars_since_refit),
        }
        tmp = self._cache_path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(cache, fh)
        tmp.replace(self._cache_path)

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def needs_refit(self, t: int) -> bool:
        if t <= self._cached_through:
            return False
        if self._model is None:
            return True
        return self._bars_since_refit >= self._refit_bars

    def refit(self, returns: np.ndarray) -> None:
        super().refit(returns)
        self._bars_since_refit = 0

    def step(self, returns: np.ndarray, t: int) -> RegimeState | None:
        if t <= self._cached_through:
            p_bull = self._p_bull_arr[t]
            if np.isnan(p_bull):
                return None
            self.cache_hits += 1
            p_bear = float(self._p_bear_arr[t])
            p_smooth = float(self._p_smooth_arr[t])
            return RegimeState(
                p_bull=float(p_bull),
                p_bear=p_bear,
                p_bull_smooth=p_smooth,
                regime_signal=p_smooth - p_bear,
                hmm_vote=discretize_p_bull(p_smooth, self._bull_edge, self._bear_edge),
            )

        state = super().step(returns, t)
        if state is not None:
            self.computed_steps += 1
            self._bars_since_refit += 1
            if t < len(self._p_bull_arr):
                self._p_bull_arr[t] = state.p_bull
                self._p_bear_arr[t] = state.p_bear
                self._p_smooth_arr[t] = state.p_bull_smooth
                self._last_step_idx = max(self._last_step_idx, t)
        return state

    def reset(self) -> None:
        super().reset()
        self._bars_since_refit = 0
        self._cached_through = -1
        self._last_step_idx = -1
        self._p_bull_arr[:] = np.nan
        self._p_bear_arr[:] = np.nan
        self._p_smooth_arr[:] = np.nan
