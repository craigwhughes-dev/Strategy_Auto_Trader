"""X2 of PLAN_EXIT_GENERALISATION.md: does the deployed exit protect out of sample?

X1 established that a 21-day realized-vol deadband reproduces the deployed VXN 23/24 exit timing on
Nasdaq 2013-2026. X2 applies that mechanism to every market's own index across six decades and asks:
would it have been out of the market before the worst of each decline, in shocks it was never fitted
to, and at what cost in time idled in cash?

Two transfer rules are scored (owner decision 2026-09-29), separating the two interpretive readings:

  * ``fixed_pctl`` (PRIMARY - the mechanism question): exit at each market's OWN p-th realized-vol
    percentile, with p fixed from the Nasdaq calibration (enter 45.6th, exit 49.9th). Currency- and
    baseline-neutral; not per-market tuning because the percentiles are fixed, not fitted per market.
  * ``absolute`` (the reading-3 confound check): apply the single Nasdaq-calibrated RV level
    (lo=0.183, hi=0.191) to every market and era. Tests directly whether an absolute level transfers;
    expected to degenerate on high-baseline-vol markets (they sit in cash near-always), which the
    ``idle_rate`` cost metric exposes.

Metrics per episode (both pre-registered, ground rule 9 - each index in its own currency):
  * ``protection`` = fraction of the peak-to-trough decline avoided by exiting: (close_exit -
    close_trough) / (close_peak - close_trough). Reported raw AND clipped to [0, 1]. 0.0 if the gate
    never exits before the trough; category ``already_out`` (protection undefined, recorded
    separately, never scored as 1.0) if the gate is in cash at the peak.
  * ``exit_lag`` = trading days from peak to exit.
  * ``idle_rate`` = fraction of non-episode trading days in cash, per market over full history - the
    two-sided cost, without which an always-cash gate protects perfectly and earns nothing.

Aggregation is BY SHOCK, not by episode (average within a shock, then across shocks), so 1987 in six
markets is one observation. Research only; reuses the deployed deadband, changes no production code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from Strategy_Auto_Trader.allocation.intraday_engine import _CASH
from Strategy_Auto_Trader.research.proxy_fidelity import deadband_state, realized_vol

# Fixed from the X1 Nasdaq 2001-2012 calibration (see BACKTEST_LOG 2026-09-29).
NASDAQ_ENTER_PCTL = 0.4556
NASDAQ_EXIT_PCTL = 0.4985
ABS_LO = 0.1827
ABS_HI = 0.1910
VOL_WINDOW = 21
HOLDOUT_MAX_SHOCK = 13  # shock_id <= 13 => trough entirely pre-2001


@dataclass
class EpisodeProtection:
    market: str
    shock_id: int
    is_holdout: bool
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    category: str                 # "scored" | "already_out" | "never_exited"
    protection_raw: float | None
    protection_clipped: float | None
    exit_lag: int | None          # trading days peak -> exit


def percentile_thresholds(rv: pd.Series, enter_pctl: float, exit_pctl: float) -> tuple[float, float]:
    """(lo, hi) at the given percentiles of a market's own full-history RV.

    Full-history quantiles put a mild look-ahead into the *threshold only* (not the state path); this
    is retrospective characterisation, not a live signal, and is flagged in the X2 report.
    """
    valid = rv.to_numpy()
    valid = valid[np.isfinite(valid)]
    lo = float(np.quantile(valid, enter_pctl))
    hi = float(np.quantile(valid, exit_pctl))
    return lo, hi


def episode_protection(
    close: pd.Series,
    state: np.ndarray,
    peak_date: pd.Timestamp,
    trough_date: pd.Timestamp,
) -> tuple[str, float | None, float | None, int | None]:
    """(category, protection_raw, protection_clipped, exit_lag) for one episode on a market's series."""
    days = close.index
    peak_pos = int(days.searchsorted(peak_date, side="left"))
    trough_pos = int(days.searchsorted(trough_date, side="left"))
    if peak_pos >= len(days) or trough_pos >= len(days) or trough_pos <= peak_pos:
        return "never_exited", 0.0, 0.0, None

    if state[peak_pos] == _CASH:
        return "already_out", None, None, None

    tail = np.flatnonzero(state[peak_pos : trough_pos + 1] == _CASH)
    if tail.size == 0:
        return "never_exited", 0.0, 0.0, None  # gate stayed invested through the trough

    exit_pos = peak_pos + int(tail[0])
    peak_c, trough_c, exit_c = close.iloc[peak_pos], close.iloc[trough_pos], close.iloc[exit_pos]
    denom = peak_c - trough_c
    if denom <= 0:
        return "never_exited", 0.0, 0.0, None
    raw = (exit_c - trough_c) / denom
    return "scored", float(raw), float(np.clip(raw, 0.0, 1.0)), exit_pos - peak_pos


def market_idle_rate(state: np.ndarray, days: pd.DatetimeIndex, episode_windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> float:
    """Fraction of non-episode trading days the gate spends in cash.

    ``episode_windows`` are (peak, end) spans for that market; any day inside one is an episode day.
    """
    in_episode = np.zeros(len(days), dtype=bool)
    for peak, end in episode_windows:
        lo = int(days.searchsorted(peak, side="left"))
        hi = int(days.searchsorted(end, side="right"))
        in_episode[lo:hi] = True
    non_ep = ~in_episode
    if non_ep.sum() == 0:
        return float("nan")
    return float(np.mean(state[non_ep] == _CASH))


def build_state(close: pd.Series, rule: str) -> tuple[np.ndarray, float, float]:
    """Deadband state over a market's full daily series under a transfer rule. Returns (state, lo, hi)."""
    rv = realized_vol(close, VOL_WINDOW)
    if rule == "absolute":
        lo, hi = ABS_LO, ABS_HI
    elif rule == "fixed_pctl":
        lo, hi = percentile_thresholds(rv, NASDAQ_ENTER_PCTL, NASDAQ_EXIT_PCTL)
    else:
        raise ValueError(f"unknown rule {rule!r}")
    return deadband_state(rv.to_numpy(), lo, hi), lo, hi


def aggregate_by_shock(episodes: list[EpisodeProtection]) -> pd.DataFrame:
    """One row per shock: mean clipped protection over its scored episodes + holdout flag."""
    rows = []
    by_shock: dict[int, list[EpisodeProtection]] = {}
    for e in episodes:
        by_shock.setdefault(e.shock_id, []).append(e)
    for sid, eps in sorted(by_shock.items()):
        scored = [e.protection_clipped for e in eps if e.category == "scored"]
        rows.append({
            "shock_id": sid,
            "is_holdout": eps[0].is_holdout,
            "n_markets": len(eps),
            "n_scored": len(scored),
            "n_already_out": sum(e.category == "already_out" for e in eps),
            "n_never_exited": sum(e.category == "never_exited" for e in eps),
            "protection": float(np.mean(scored)) if scored else np.nan,
        })
    return pd.DataFrame(rows)


def permutation_pvalue(design: np.ndarray, holdout: np.ndarray, n: int = 10000, seed: int = 0) -> float:
    """Two-sided p for |median(design) - median(holdout)| under label shuffling.

    Equivalence framing: a HIGH p supports 'the two sets behave the same'. A low p says they differ.
    """
    design = design[np.isfinite(design)]
    holdout = holdout[np.isfinite(holdout)]
    obs = abs(np.median(design) - np.median(holdout))
    pool = np.concatenate([design, holdout])
    nd = len(design)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(np.median(pool[:nd]) - np.median(pool[nd:])) >= obs:
            count += 1
    return (count + 1) / (n + 1)
