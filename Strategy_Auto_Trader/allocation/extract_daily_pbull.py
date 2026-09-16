"""Extract daily HMM P(Bull) regime from hourly data.

Loads hourly SPY, builds HMM, computes daily close P(Bull) for 2015-2024.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


def extract_daily_pbull_from_hourly(
    hourly_df: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
) -> pd.Series:
    """Extract daily HMM P(Bull) from hourly data.

    Builds HMM on hourly closes, then resamples to daily (last value each day).

    Args:
        hourly_df: Hourly OHLCV with Close column, index must be datetime
        target_dates: Daily dates to extract P(Bull) for

    Returns:
        pd.Series with daily P(Bull) indexed by target_dates
    """
    try:
        from ..plugins.persistent_hmm import PersistentHMMRegimeModel
    except ImportError as e:
        _log.error(f"Failed to import HMM modules: {e}")
        return pd.Series(0.5, index=target_dates, name="pbull")

    # Build HMM cache path
    cache_dir = Path(__file__).resolve().parent.parent.parent / "data" / "hmm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "SPY_hourly_hmm.pkl"

    _log.info(f"Building HMM on {len(hourly_df)} hourly bars (2004-2026)...")
    try:
        hmm = PersistentHMMRegimeModel(
            cache_path=cache_path,
            dates=hourly_df.index,
            closes=hourly_df["Close"].values,
            min_train_bars=500,
            refit_bars=500,
        )

        # Step through all hourly data to compute regimes
        _log.info(f"Computing HMM regimes for all {len(hourly_df)} hourly bars...")
        closes = hourly_df["Close"].values
        for t in range(len(hourly_df)):
            if t < 500:
                continue  # Skip until min training
            returns = np.diff(np.log(closes[max(0, t - 500):t + 1]))
            if hmm.needs_refit(t):
                hmm.refit(returns)
            hmm.step(returns, t)

        _log.info("HMM regimes computed")

        # Extract daily P(Bull): last hourly value for each day
        _log.info(f"Resampling to daily P(Bull) for {len(target_dates)} dates...")
        daily_pbull = {}

        for date in target_dates:
            # Find last hourly bar on or before this date
            mask = hourly_df.index.date <= pd.Timestamp(date).date()
            if not mask.any():
                daily_pbull[date] = 0.5  # Default if no data
                continue

            idx = np.where(mask)[0][-1]  # Last True index
            pbull_val = hmm._p_smooth_arr[idx] if idx < len(hmm._p_smooth_arr) else 0.5
            daily_pbull[date] = float(pbull_val) if not np.isnan(pbull_val) else 0.5

        result = pd.Series(daily_pbull, index=target_dates, name="pbull")
        _log.info(f"Daily P(Bull): mean={result.mean():.2f}, min={result.min():.2f}, max={result.max():.2f}")
        return result

    except Exception as e:
        _log.error(f"Failed to extract P(Bull): {e}", exc_info=True)
        return pd.Series(0.5, index=target_dates, name="pbull")
