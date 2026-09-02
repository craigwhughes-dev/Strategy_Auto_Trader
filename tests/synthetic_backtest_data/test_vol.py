from __future__ import annotations

import numpy as np
import pandas as pd

from Strategy_Auto_Trader.synthetic_backtest_data import vol


def test_rolling_daily_vol_matches_hand_computed_std():
    rng = np.random.default_rng(0)
    log_returns = rng.normal(0, 0.01, size=50)
    close = pd.Series(100 * np.exp(np.cumsum(log_returns)))

    window = 21
    result = vol.rolling_daily_vol(close, window=window)

    expected_returns = vol.daily_log_returns(close)
    expected = expected_returns.iloc[1:window + 1].std()
    assert np.isclose(result.iloc[window], expected)


def test_warmup_nans_length():
    close = pd.Series(np.linspace(100, 110, 30))
    window = 21
    result = vol.rolling_daily_vol(close, window=window)
    # 1 lost to the log-return shift + (window - 1) rolling warmup
    assert result.iloc[:window].isna().all()
    assert result.iloc[window:].notna().all()


def test_daily_log_returns_first_value_is_nan():
    close = pd.Series([100.0, 101.0, 99.0])
    result = vol.daily_log_returns(close)
    assert np.isnan(result.iloc[0])
    assert np.isclose(result.iloc[1], np.log(101.0 / 100.0))
