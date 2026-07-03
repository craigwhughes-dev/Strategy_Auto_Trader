from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest


class TestVolScreen:

    def _daily_df(self, close, spread=1.0, start="2022-01-01"):
        close = np.asarray(close, dtype=float)
        n = len(close)
        idx = pd.date_range(start, periods=n, freq="D")
        return pd.DataFrame({
            "Open": close, "High": close + spread, "Low": close - spread,
            "Close": close, "Volume": np.full(n, 1_000_000.0),
        }, index=idx)

    def test_volatility_profile_trending_has_high_efficiency_ratio(self):
        from Strategy_Auto_Trader.quant_hmm.vol_screen import volatility_profile
        close = np.linspace(100.0, 200.0, 300)  # monotonic uptrend
        df = self._daily_df(close)
        with mock.patch("yfinance.download", return_value=df):
            profile = volatility_profile("TEST")
        assert profile is not None
        assert profile["efficiency_ratio"] > 0.95  # monotonic -> path == net change

    def test_volatility_profile_choppy_has_low_efficiency_ratio(self):
        from Strategy_Auto_Trader.quant_hmm.vol_screen import volatility_profile
        n = 300
        # Zigzag that ends near where it started -> large path length, small net change
        close = 100.0 + 10.0 * np.sin(np.arange(n) * (np.pi / 2))
        df = self._daily_df(close)
        with mock.patch("yfinance.download", return_value=df):
            profile = volatility_profile("TEST")
        assert profile is not None
        assert profile["efficiency_ratio"] < 0.1

    def test_volatility_profile_trending_beats_choppy_on_trend_quality(self):
        from Strategy_Auto_Trader.quant_hmm.vol_screen import volatility_profile
        n = 300
        trending = self._daily_df(np.linspace(100.0, 160.0, n))
        choppy = self._daily_df(100.0 + 10.0 * np.sin(np.arange(n) * (np.pi / 2)))
        with mock.patch("yfinance.download", return_value=trending):
            trend_profile = volatility_profile("TREND")
        with mock.patch("yfinance.download", return_value=choppy):
            choppy_profile = volatility_profile("CHOP")
        assert trend_profile["trend_quality"] > choppy_profile["trend_quality"]

    def test_volatility_profile_too_short_returns_none(self):
        from Strategy_Auto_Trader.quant_hmm.vol_screen import volatility_profile
        df = self._daily_df(np.full(50, 100.0))
        with mock.patch("yfinance.download", return_value=df):
            assert volatility_profile("TEST") is None

    def test_volatility_profile_empty_download_returns_none(self):
        from Strategy_Auto_Trader.quant_hmm.vol_screen import volatility_profile
        with mock.patch("yfinance.download", return_value=pd.DataFrame()):
            assert volatility_profile("TEST") is None

    def test_volatility_profile_exception_returns_none(self):
        from Strategy_Auto_Trader.quant_hmm.vol_screen import volatility_profile
        with mock.patch("yfinance.download", side_effect=Exception("network error")):
            assert volatility_profile("TEST") is None

    def test_screen_tickers_filters_by_trend_quality(self):
        from Strategy_Auto_Trader.quant_hmm import vol_screen as vs

        def fake_profile(ticker, period="2y"):
            scores = {"GOOD": 1.0, "BAD": -2.0, "BORDERLINE": 0.0}
            return {"ticker": ticker, "ann_vol": 0.2, "efficiency_ratio": 0.1,
                    "autocorr": 0.0, "choppiness_idx": 47.0, "sign_change_freq": 0.5,
                    "trend_quality": scores[ticker]}

        with mock.patch.object(vs, "volatility_profile", side_effect=fake_profile):
            kept, profiles = vs.screen_tickers(
                ["GOOD", "BAD", "BORDERLINE"], min_trend_quality=0.0, verbose=False)
        assert kept == ["GOOD", "BORDERLINE"]
        assert len(profiles) == 3

    def test_screen_tickers_skips_failed_fetches(self):
        from Strategy_Auto_Trader.quant_hmm import vol_screen as vs

        def fake_profile(ticker, period="2y"):
            if ticker == "DELISTED":
                return None
            return {"ticker": ticker, "ann_vol": 0.2, "efficiency_ratio": 0.1,
                    "autocorr": 0.0, "choppiness_idx": 47.0, "sign_change_freq": 0.5,
                    "trend_quality": 1.0}

        with mock.patch.object(vs, "volatility_profile", side_effect=fake_profile):
            kept, profiles = vs.screen_tickers(
                ["AAA", "DELISTED", "BBB"], min_trend_quality=0.0, verbose=False)
        assert "DELISTED" not in kept
        assert len(kept) == 2
        assert len(profiles) == 2
