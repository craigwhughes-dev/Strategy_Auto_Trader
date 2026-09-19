"""Cleaning for real IBKR hourly fund bars before they are spliced or used to estimate parameters.

IBKR's older LSE fund history contains bad prints (a bar quoted in the wrong unit for one or two
bars) and persistent share-count changes (a 2:1 consolidation shows up as a permanent ~ln 2
jump). A broad-index ETF does not move 25% in an hour, so both are data errors, not returns:
spikes are dropped, and remaining level shifts are back-adjusted so the series stays continuous
at its current level. Not applied to the volatility indices, which genuinely gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_SPIKE_LOG_THRESH = 0.25
_MEDIAN_WINDOW = 7


def drop_spikes(close: pd.Series, thresh: float = _SPIKE_LOG_THRESH, window: int = _MEDIAN_WINDOW) -> pd.Series:
    """Drop bars more than `thresh` (log) away from the centred rolling median."""
    median = close.rolling(window, center=True, min_periods=3).median()
    return close[(np.log(close / median).abs() <= thresh).to_numpy()]


def remove_level_shifts(close: pd.Series, thresh: float = _SPIKE_LOG_THRESH) -> pd.Series:
    """Scale history before each remaining >thresh jump so the series is continuous at its latest level."""
    jumps = np.log(close).diff()
    out = close.astype(float).copy()
    for ts, jump in jumps[jumps.abs() > thresh].items():
        out[out.index < ts] *= np.exp(jump)
    return out


def clean_fund_bars(close: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    despiked = drop_spikes(close)
    fixed = remove_level_shifts(despiked)
    n_shifts = int((np.log(despiked).diff().abs() > _SPIKE_LOG_THRESH).sum())
    return fixed, {"dropped_spikes": len(close) - len(despiked), "level_shifts": n_shifts}
