from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from Strategy_Auto_Trader.quant_hmm import data_cache


def _hourly_df(n=10):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = np.linspace(100, 110, n)
    return pd.DataFrame({"Open": close, "High": close, "Low": close,
                          "Close": close, "Volume": np.full(n, 1000.0)}, index=idx)


@pytest.fixture(autouse=True)
def _clear_cache():
    data_cache.clear_cache()
    yield
    data_cache.clear_cache()


class TestFetchHourlyCached:
    def test_cache_miss_calls_underlying_and_caches(self):
        df = _hourly_df()
        with mock.patch.object(data_cache, "fetch_hourly", return_value=df) as m:
            result = data_cache.fetch_hourly_cached("AAPL", period="730d")
            assert result is df
            assert m.call_count == 1

    def test_cache_hit_does_not_call_underlying_again(self):
        df = _hourly_df()
        with mock.patch.object(data_cache, "fetch_hourly", return_value=df) as m:
            data_cache.fetch_hourly_cached("AAPL", period="730d")
            data_cache.fetch_hourly_cached("AAPL", period="730d")
            data_cache.fetch_hourly_cached("AAPL", period="730d")
            assert m.call_count == 1

    def test_different_ticker_is_separate_cache_entry(self):
        df = _hourly_df()
        with mock.patch.object(data_cache, "fetch_hourly", return_value=df) as m:
            data_cache.fetch_hourly_cached("AAPL", period="730d")
            data_cache.fetch_hourly_cached("MSFT", period="730d")
            assert m.call_count == 2

    def test_different_period_is_separate_cache_entry(self):
        df = _hourly_df()
        with mock.patch.object(data_cache, "fetch_hourly", return_value=df) as m:
            data_cache.fetch_hourly_cached("AAPL", period="730d")
            data_cache.fetch_hourly_cached("AAPL", period="60d")
            assert m.call_count == 2

    def test_none_result_is_not_cached_and_retries(self):
        with mock.patch.object(data_cache, "fetch_hourly", return_value=None) as m:
            r1 = data_cache.fetch_hourly_cached("BADTICKER")
            r2 = data_cache.fetch_hourly_cached("BADTICKER")
            assert r1 is None
            assert r2 is None
            assert m.call_count == 2

    def test_empty_df_result_is_not_cached_and_retries(self):
        with mock.patch.object(data_cache, "fetch_hourly", return_value=pd.DataFrame()) as m:
            data_cache.fetch_hourly_cached("BADTICKER")
            data_cache.fetch_hourly_cached("BADTICKER")
            assert m.call_count == 2

    def test_clear_cache_forces_refetch(self):
        df = _hourly_df()
        with mock.patch.object(data_cache, "fetch_hourly", return_value=df) as m:
            data_cache.fetch_hourly_cached("AAPL")
            data_cache.clear_cache()
            data_cache.fetch_hourly_cached("AAPL")
            assert m.call_count == 2


class TestVolatilityProfileCached:
    def test_cache_miss_calls_underlying_and_caches(self):
        prof = {"trend_quality": 0.5}
        with mock.patch.object(data_cache, "volatility_profile", return_value=prof) as m:
            result = data_cache.volatility_profile_cached("AAPL")
            assert result is prof
            assert m.call_count == 1

    def test_cache_hit_does_not_call_underlying_again(self):
        prof = {"trend_quality": 0.5}
        with mock.patch.object(data_cache, "volatility_profile", return_value=prof) as m:
            data_cache.volatility_profile_cached("AAPL")
            data_cache.volatility_profile_cached("AAPL")
            assert m.call_count == 1

    def test_none_result_is_not_cached_and_retries(self):
        with mock.patch.object(data_cache, "volatility_profile", return_value=None) as m:
            r1 = data_cache.volatility_profile_cached("BADTICKER")
            r2 = data_cache.volatility_profile_cached("BADTICKER")
            assert r1 is None
            assert r2 is None
            assert m.call_count == 2

    def test_independent_from_fetch_hourly_cache(self):
        df = _hourly_df()
        prof = {"trend_quality": 0.5}
        with mock.patch.object(data_cache, "fetch_hourly", return_value=df) as m_fetch, \
             mock.patch.object(data_cache, "volatility_profile", return_value=prof) as m_prof:
            data_cache.fetch_hourly_cached("AAPL")
            data_cache.volatility_profile_cached("AAPL")
            assert m_fetch.call_count == 1
            assert m_prof.call_count == 1
