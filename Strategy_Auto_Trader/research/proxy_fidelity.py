"""X1 of PLAN_EXIT_GENERALISATION.md: can a realized-vol proxy stand in for the deployed VXN rule?

The deployed exit is ``tiers_vxn_deadband(VXN, 23, 24)`` — implied vol, forward-looking. Any proxy
computable back to 1970 is *realized* vol, backward-looking, so it lags by construction. Before any
pre-2001 out-of-sample work (X2), this checks the proxy can reproduce the deployed rule's exit
*timing* on the 2001+ window where both exist. If it cannot, X2 does not run.

Calibration is **determined, not fitted** (keeps the trial count at 3 = one per vol window):
  - ``hi`` (RV exit level) = the RV value whose 2001-2012 percentile equals VXN 24's 2001-2012
    percentile. A concrete RV number, never the literal 24 — VXN and RV are different scales.
  - ``lo`` (RV enter level) = solved so the proxy deadband sits in cash the same fraction of
    2001-2012 days as the deployed rule. No search for agreement, no optimisation.

The deployed ``tiers_vxn_deadband`` is reused verbatim for the proxy, fed RV in place of VXN — the
deadband shape (enter <= lo, exit > hi, hold between) is identical. Research only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from Strategy_Auto_Trader.allocation.intraday_engine import _CASH, tiers_vxn_deadband

TRADING_DAYS = 252


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    """Annualised realized volatility of daily log returns over a trailing ``window``.

    NaN for the first ``window`` bars (insufficient history); the deadband treats those as
    state-preserving, and in early 2001 the deployed rule is in cash anyway, so they do not bias
    calibration.
    """
    lr = np.log(close).diff()
    return lr.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def deadband_state(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Deployed deadband applied to any vol series: 0 invested / ``_CASH`` cash. Reuses the engine."""
    return tiers_vxn_deadband(values, lo, hi)


def calibrate_thresholds(
    rv: np.ndarray,
    vxn: np.ndarray,
    train_mask: np.ndarray,
    deployed_exit_level: float = 24.0,
) -> tuple[float, float]:
    """(lo, hi) for the RV proxy, determined from the train window only. See module docstring."""
    rv_tr = rv[train_mask]
    vxn_tr = vxn[train_mask]
    rv_valid = rv_tr[np.isfinite(rv_tr)]
    vxn_valid = vxn_tr[np.isfinite(vxn_tr)]

    # hi: RV level at VXN 24's train percentile
    exit_pctl = float(np.mean(vxn_valid <= deployed_exit_level))
    hi = float(np.quantile(rv_valid, exit_pctl))

    # deployed in-cash fraction on train
    dep_state = deadband_state(vxn, 23.0, deployed_exit_level)
    target_cash = float(np.mean(dep_state[train_mask] == _CASH))

    # lo: solved so the proxy's train in-cash fraction matches. In-cash is monotone non-increasing
    # in lo (easier entry -> less cash), so bisect on [min RV, hi].
    def cash_frac(lo: float) -> float:
        st = deadband_state(rv, lo, hi)
        return float(np.mean(st[train_mask] == _CASH))

    lo_hi = hi
    lo_lo = float(np.nanmin(rv_valid))
    for _ in range(60):
        mid = 0.5 * (lo_lo + lo_hi)
        if cash_frac(mid) > target_cash:
            lo_lo = mid  # too much cash -> raise lo (easier entry)
        else:
            lo_hi = mid
    return 0.5 * (lo_lo + lo_hi), hi


def first_exit_on_or_after(state: np.ndarray, days: pd.DatetimeIndex, peak: pd.Timestamp) -> int | None:
    """Position of the first cash bar at or after ``peak``. None if never in cash through the series.

    Positional index into ``days``. A value equal to the peak's own position means 'already out at
    the peak' (exit_lag <= 0), which the caller records separately.
    """
    pk_pos = int(days.searchsorted(peak, side="left"))
    if pk_pos >= len(days):
        return None
    tail = np.flatnonzero(state[pk_pos:] == _CASH)
    return int(pk_pos + tail[0]) if tail.size else None


@dataclass
class EpisodeTiming:
    peak: pd.Timestamp
    trough: pd.Timestamp
    deployed_exit_pos: int | None
    proxy_exit_pos: int | None
    trough_pos: int
    peak_pos: int

    @property
    def lag_bars(self) -> int | None:
        """Proxy exit minus deployed exit, in trading days. None if either never exits."""
        if self.deployed_exit_pos is None or self.proxy_exit_pos is None:
            return None
        return self.proxy_exit_pos - self.deployed_exit_pos

    @property
    def proxy_precedes_trough(self) -> bool:
        return self.proxy_exit_pos is not None and self.proxy_exit_pos < self.trough_pos

    def passes(self, tol_bars: int = 10) -> bool:
        """Gate per episode: proxy exits before the trough AND within tol_bars of the deployed exit."""
        return self.proxy_precedes_trough and self.lag_bars is not None and abs(self.lag_bars) <= tol_bars


def episode_timing(
    deployed_state: np.ndarray,
    proxy_state: np.ndarray,
    days: pd.DatetimeIndex,
    peak: pd.Timestamp,
    trough: pd.Timestamp,
) -> EpisodeTiming:
    return EpisodeTiming(
        peak=peak,
        trough=trough,
        deployed_exit_pos=first_exit_on_or_after(deployed_state, days, peak),
        proxy_exit_pos=first_exit_on_or_after(proxy_state, days, peak),
        trough_pos=int(days.searchsorted(trough, side="left")),
        peak_pos=int(days.searchsorted(peak, side="left")),
    )
