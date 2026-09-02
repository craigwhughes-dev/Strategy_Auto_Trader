"""Daily volatility from a real daily-close series.

Raw (unannualized) per-day sigma — bridge.py needs the actual day-to-day
return dispersion, not a Sharpe-style annualized figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def rolling_daily_vol(close: pd.Series, window: int = 21) -> pd.Series:
    """Rolling std of daily log returns. First `window` values are NaN
    (one lost to the log-return shift, `window - 1` more to the rolling
    warmup) — callers must drop them before use."""
    return daily_log_returns(close).rolling(window).std()
