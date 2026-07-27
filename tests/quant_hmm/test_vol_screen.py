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
            return {"ticker": ticker, "ann_vol": 0.2, "downside_vol": 0.15,
                    "efficiency_ratio": 0.1, "autocorr": 0.0, "choppiness_idx": 47.0,
                    "sign_change_freq": 0.5, "trend_quality": scores[ticker]}

        with mock.patch.object(vs, "volatility_profile", side_effect=fake_profile):
            kept, profiles = vs.screen_tickers(
                ["GOOD", "BAD", "BORDERLINE"], min_trend_quality=0.0, verbose=False)
        assert kept == ["GOOD", "BORDERLINE"]
        assert len(profiles) == 3

    def test_screen_tickers_filters_by_downside_vol(self):
        from Strategy_Auto_Trader.quant_hmm import vol_screen as vs

        def fake_profile(ticker, period="2y"):
            vols = {"LOW": 0.15, "HIGH": 0.30, "MEDIUM": 0.22}
            return {"ticker": ticker, "ann_vol": 0.35, "downside_vol": vols[ticker],
                    "efficiency_ratio": 0.1, "autocorr": 0.0, "choppiness_idx": 47.0,
                    "sign_change_freq": 0.5, "trend_quality": 0.5}

        with mock.patch.object(vs, "volatility_profile", side_effect=fake_profile):
            kept, profiles = vs.screen_tickers(
                ["LOW", "MEDIUM", "HIGH"], min_trend_quality=0.0,
                max_downside_vol=0.25, verbose=False)
        assert kept == ["LOW", "MEDIUM"]
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


class TestRollingTrendQuality:
    """rolling_trend_quality() — the daily-rescreen equivalent of
    volatility_profile()'s single snapshot, used by live_sim.py so a
    historical backtest can match live trading's actual daily overnight
    rescreen (overnight_scope.py) instead of applying today's trend_quality
    across the whole simulated window."""

    def _daily_ohlc(self, close, spread=1.0, start="2022-01-01"):
        close = np.asarray(close, dtype=float)
        n = len(close)
        idx = pd.date_range(start, periods=n, freq="D")
        return pd.DataFrame({"High": close + spread, "Low": close - spread, "Close": close}, index=idx)

    def test_nan_before_min_periods(self):
        from Strategy_Auto_Trader.quant_hmm.vol_screen import rolling_trend_quality
        close = np.linspace(100.0, 200.0, 150)
        df = self._daily_ohlc(close)
        series = rolling_trend_quality(df, window=100, min_periods=100)
        # Fewer than min_periods bars of history -> no score yet.
        assert series.iloc[:99].isna().all()

    def test_score_available_after_min_periods(self):
        from Strategy_Auto_Trader.quant_hmm.vol_screen import rolling_trend_quality
        close = np.linspace(100.0, 200.0, 150)
        df = self._daily_ohlc(close)
        series = rolling_trend_quality(df, window=100, min_periods=100)
        assert series.iloc[105:].notna().all()

    def test_tracks_a_regime_shift_trending_then_choppy(self):
        """The whole point of a rolling (not single-snapshot) score: it should
        track a real change in character partway through the series, which a
        single 'as of today' volatility_profile() call could never see for a
        historical date."""
        from Strategy_Auto_Trader.quant_hmm.vol_screen import rolling_trend_quality
        n_trend, n_chop = 200, 200
        trending = np.linspace(100.0, 200.0, n_trend)
        choppy = 150.0 + 10.0 * np.sin(np.arange(n_chop) * (np.pi / 2))
        close = np.concatenate([trending, choppy])
        df = self._daily_ohlc(close)

        series = rolling_trend_quality(df, window=100, min_periods=100)

        # Deep into the trending regime (window fully inside trending data).
        mid_trend_score = series.iloc[180]
        # Deep into the choppy regime (window fully inside choppy data).
        mid_chop_score = series.iloc[390]

        assert mid_trend_score > mid_chop_score

    def test_no_lookahead_score_unaffected_by_future_prices(self):
        """Day t's score must not change if data after day t is altered —
        the daemon screens overnight using only data through yesterday's
        close; a lookahead-contaminated score would replay the exact bug
        being fixed (applying information from outside the historical window
        available at that point in the simulation)."""
        from Strategy_Auto_Trader.quant_hmm.vol_screen import rolling_trend_quality
        close_a = np.concatenate([np.linspace(100.0, 150.0, 150), np.full(50, 150.0)])
        close_b = np.concatenate([np.linspace(100.0, 150.0, 150), np.linspace(150.0, 50.0, 50)])
        df_a = self._daily_ohlc(close_a)
        df_b = self._daily_ohlc(close_b)

        series_a = rolling_trend_quality(df_a, window=100, min_periods=100)
        series_b = rolling_trend_quality(df_b, window=100, min_periods=100)

        # Both series are identical through day 149 (the point where the two
        # price paths diverge) — a later change cannot retroactively alter it.
        pd.testing.assert_series_equal(
            series_a.iloc[:149], series_b.iloc[:149], check_names=False,
        )

    def test_shifted_by_one_day_no_same_day_lookahead(self):
        """The returned series is deliberately 1-day-shifted: day t's score
        must depend on data through day t-1 only. Changing the close on the
        LAST day must not move the last day's own score (it's masked out by
        the shift), but changing the close on the SECOND-TO-LAST day must —
        that's the day the last row's shifted score actually reads from."""
        from Strategy_Auto_Trader.quant_hmm.vol_screen import rolling_trend_quality
        base_close = np.linspace(100.0, 200.0, 150)

        df_base = self._daily_ohlc(base_close)
        series_base = rolling_trend_quality(df_base, window=100, min_periods=100)

        close_last_changed = base_close.copy()
        close_last_changed[-1] = base_close[-1] * 1.5
        series_last_changed = rolling_trend_quality(self._daily_ohlc(close_last_changed), window=100, min_periods=100)
        assert series_last_changed.iloc[-1] == pytest.approx(series_base.iloc[-1])

        close_prev_changed = base_close.copy()
        close_prev_changed[-2] = base_close[-2] * 1.5
        series_prev_changed = rolling_trend_quality(self._daily_ohlc(close_prev_changed), window=100, min_periods=100)
        assert series_prev_changed.iloc[-1] != pytest.approx(series_base.iloc[-1])
