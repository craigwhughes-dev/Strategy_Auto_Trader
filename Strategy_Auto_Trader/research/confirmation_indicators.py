"""Causal confirmation indicators for PLAN_CRASH_PHASES.md's first cut.

Every function here sees only trailing data at each point. The phase labels they are scored
against use hindsight (that is the design, see `crash_phases`), but the indicators must not:
a lookahead here would make the whole test meaningless.

Each returns a boolean array of *fresh firings* — True only on the bar where the condition
becomes true, not on every bar it stays true — so an indicator that is simply true for months
does not score as hundreds of separate calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SMA_WINDOW = 50
SLOPE_LOOKBACK = 10


def _fresh(condition: np.ndarray) -> np.ndarray:
    """True where `condition` turns from False to True. Never fires on the first bar."""
    out = np.zeros(len(condition), dtype=bool)
    out[1:] = condition[1:] & ~condition[:-1]
    return out


def price_above_rising_sma(
    close: pd.Series,
    sma_window: int = SMA_WINDOW,
    slope_lookback: int = SLOPE_LOOKBACK,
) -> np.ndarray:
    """Pre-registered indicator P: close above a *rising* moving average.

    Both clauses matter. "Close above its 50-day SMA" alone is the trigger T1 already showed
    fires constantly in bear rallies (67 times in 19 years, -15.3pp). Requiring the average
    itself to have risen over `slope_lookback` bars is what is supposed to separate a genuine
    turn from a bounce inside a downtrend; whether it does is the question being tested.
    """
    if sma_window < 2 or slope_lookback < 1:
        raise ValueError(f"bad parameters: sma_window={sma_window}, slope_lookback={slope_lookback}")
    px = close.astype(float)
    sma = px.rolling(sma_window).mean()
    rising = sma > sma.shift(slope_lookback)
    cond = ((px > sma) & rising).fillna(False).to_numpy(dtype=bool)
    return _fresh(cond)


def spread_below_falling_average(
    spread: pd.Series,
    sma_window: int = SMA_WINDOW,
    slope_lookback: int = SLOPE_LOOKBACK,
) -> np.ndarray:
    """Indicator C for the credit test: a spread series below its own falling average.

    The mirror image of indicator P — tightening credit rather than rising price. Registered
    now so the two tests use the same structural form and differ only in their input series.
    """
    if sma_window < 2 or slope_lookback < 1:
        raise ValueError(f"bad parameters: sma_window={sma_window}, slope_lookback={slope_lookback}")
    s = spread.astype(float)
    sma = s.rolling(sma_window).mean()
    falling = sma < sma.shift(slope_lookback)
    cond = ((s < sma) & falling).fillna(False).to_numpy(dtype=bool)
    return _fresh(cond)


def elapsed_time_baseline(close: pd.Series, drawdown_depth: float, wait_bars: int) -> np.ndarray:
    """Causal naive baseline: fire `wait_bars` after the drawdown first reaches `drawdown_depth`.

    This is T9's mechanism, which needs no indicator at all. Any candidate that cannot beat it
    has added nothing, so it is reported alongside the permutation null rather than instead of it.
    Knowable at the time: it uses only the running high and elapsed bars.
    """
    px = close.to_numpy(dtype=float)
    peak = np.maximum.accumulate(px)
    breached = px <= (1.0 - drawdown_depth) * peak
    out = np.zeros(len(px), dtype=bool)
    first = np.flatnonzero(breached)
    if first.size:
        target = int(first[0]) + wait_bars
        if target < len(px):
            out[target] = True
    return out
