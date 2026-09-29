"""Time-based re-entry override for the VXN deadband rule.

T1 (`reentry_override.py`) tested four price/vol triggers for re-entering after a crash; none passed, and the
realized-vol sweep (BACKTEST_LOG 2026-09-20 early hours, addendum 3) found that in 2003, 2009 and 2020 implied
AND realized vol were both elevated throughout the recovery — no observable separates "recovering" from "still
falling" at the time.

This module gives up on identifying the recovery. The override arms purely on ELAPSED TIME: once the base rule
has been continuously in cash for `wait_days`, a fraction of the pot goes back into Nasdaq, ramped in linearly
over `ramp_days`, regardless of VXN or price. It ends when the base rule returns to Nasdaq on its own.

Two deliberate differences from the T1 triggers:

- No re-arming within a cash episode. A T1 trigger could turn false and re-fire; a time rule has no trigger to
  turn false, so once it is stopped out it stays out until the base rule goes back to Nasdaq. Re-arming on
  elapsed time alone would just be a fixed-interval retry, which is a timing device wearing a disguise.
- The weight ramps rather than stepping, so the entry price is averaged across `ramp_days` instead of being one
  dated bet. That is the point of the mechanism: it cannot be right or wrong about a single day.
"""

from __future__ import annotations

import numpy as np

STOP_PCT = 0.08


def override_fractions(
    base_cash: np.ndarray,
    close: np.ndarray,
    wait_days: int,
    ramp_days: int,
    stop_pct: float = STOP_PCT,
) -> np.ndarray:
    """Per-day override fraction in [0, 1] after that day's close, before the size cap is applied.

    `base_cash[t]` is True when the base rule holds cash at day t's close. The fraction is 0 on every day the
    base rule is invested (it holds Nasdaq at full weight itself), and 0 for the first `wait_days` of each cash
    episode. It then ramps 1/ramp_days, 2/ramp_days, ... to 1.0 and holds, until the base rule re-enters or the
    trailing stop fires.
    """
    if wait_days < 1 or ramp_days < 1:
        raise ValueError("wait_days and ramp_days must both be >= 1")
    n = len(base_cash)
    frac = np.zeros(n, dtype=float)
    streak = 0          # consecutive days the base rule has been in cash
    stopped = False     # stopped out; stays out for the rest of this cash episode
    peak = 0.0
    for t in range(n):
        if not base_cash[t]:
            streak, stopped, peak = 0, False, 0.0
            continue
        streak += 1
        if stopped or streak <= wait_days:
            continue
        held = streak - wait_days               # days since the override armed
        f = min(held / ramp_days, 1.0)
        peak = close[t] if peak == 0.0 else max(peak, close[t])
        if close[t] < (1.0 - stop_pct) * peak:
            stopped = True
            continue
        frac[t] = f
    return frac


def bar_weights(base_tiers: np.ndarray, day_codes: np.ndarray, frac_by_day: np.ndarray, size: float) -> np.ndarray:
    """Nasdaq weight held after each hourly bar: 1 when the base rule is in Nasdaq, else the previous day's
    override fraction scaled by `size` (the cap on how much of the pot the override may deploy)."""
    prev = np.concatenate([[0.0], frac_by_day[:-1]])[day_codes]
    return np.where(base_tiers == 0, 1.0, prev * size)


def episodes(frac: np.ndarray) -> list[tuple[int, int]]:
    """(first day index, last day index) of each contiguous run of days with a non-zero override fraction."""
    out: list[tuple[int, int]] = []
    start = None
    for t, f in enumerate(frac):
        if f > 0 and start is None:
            start = t
        if f == 0 and start is not None:
            out.append((start, t - 1))
            start = None
    if start is not None:
        out.append((start, len(frac) - 1))
    return out
