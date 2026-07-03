from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import _dates, _rising_prices, _falling_prices


class TestMomentum:

    def test_rsi_bounded_0_100(self):
        from Strategy_Auto_Trader.core.momentum import compute_rsi
        close = _rising_prices(200)
        rsi = compute_rsi(close, period=14).dropna()
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

    def test_rsi_rising_prices_high(self):
        from Strategy_Auto_Trader.core.momentum import compute_rsi
        # A perfectly monotonic series has zero avg_loss -> RSI is NaN.
        # Build a mostly-rising series with occasional small down days so
        # avg_loss > 0 and RSI is computable.
        np.random.seed(42)
        idx = _dates(300)
        daily_rets = np.random.normal(0.005, 0.008, 300)  # positive bias, some negatives
        prices = 100.0 * np.cumprod(1 + daily_rets)
        close = pd.Series(prices, index=idx)
        rsi = compute_rsi(close, period=14)
        rsi_valid = rsi.dropna()
        assert len(rsi_valid) > 0, "RSI produced all NaN"
        # Mostly rising -> RSI should be elevated (above 50)
        assert rsi_valid.iloc[-1] > 50

    def test_rsi_falling_prices_low(self):
        from Strategy_Auto_Trader.core.momentum import compute_rsi
        close = _falling_prices(200, daily_pct=0.01)
        rsi = compute_rsi(close, period=14).dropna()
        # Consistently falling -> RSI should be low
        assert rsi.iloc[-1] < 30

    def test_sma_basic(self):
        from Strategy_Auto_Trader.core.momentum import compute_sma
        close = pd.Series([1, 2, 3, 4, 5], dtype=float)
        sma = compute_sma(close, window=3)
        assert abs(sma.iloc[-1] - 4.0) < 1e-10  # (3+4+5)/3
        assert abs(sma.iloc[-2] - 3.0) < 1e-10  # (2+3+4)/3

    def test_ema_basic(self):
        from Strategy_Auto_Trader.core.momentum import compute_ema
        close = pd.Series([10.0] * 5 + [20.0] * 5)
        ema = compute_ema(close, span=3)
        # EMA should be between 10 and 20 during transition, approach 20 at end
        assert ema.iloc[-1] > 15
        assert ema.iloc[-1] <= 20

    def test_composite_signal_buy(self):
        from Strategy_Auto_Trader.core.momentum import composite_signal
        mom = {
            "cur_rsi": 60,
            "recent_cross_above_50": True,
            "recent_cross_below_40": False,
            "above_sma20": True,
            "above_sma50": True,
        }
        result = composite_signal(0.5, mom)  # markov > 0.20 -> +1
        # Votes: markov=+1, rsi=+1 (>=50), sma20=+1, sma50=+1 = score 4
        assert result["flag"] == "BUY"
        assert result["score"] >= 2

    def test_composite_signal_sell(self):
        from Strategy_Auto_Trader.core.momentum import composite_signal
        mom = {
            "cur_rsi": 25,
            "recent_cross_above_50": False,
            "recent_cross_below_40": True,
            "above_sma20": False,
            "above_sma50": False,
        }
        result = composite_signal(-0.5, mom, sell_threshold=-3)
        # markov=-1, rsi=-1, sma20=-1, sma50=-1 = score -4
        assert result["flag"] == "SELL"
        assert result["score"] <= -3

    def test_composite_signal_hold(self):
        from Strategy_Auto_Trader.core.momentum import composite_signal
        mom = {
            "cur_rsi": 45,
            "recent_cross_above_50": False,
            "recent_cross_below_40": False,
            "above_sma20": True,
            "above_sma50": False,
        }
        result = composite_signal(0.0, mom)
        # markov=0 (|0.0| < 0.20), rsi=0 (40..50, no cross), sma20=+1, sma50=-1 = 0
        assert result["flag"] == "HOLD"

    def test_composite_signal_with_sma200_vote(self):
        from Strategy_Auto_Trader.core.momentum import composite_signal
        mom = {
            "cur_rsi": 55,
            "recent_cross_above_50": True,
            "recent_cross_below_40": False,
            "above_sma20": False,
            "above_sma50": False,
            "above_sma200": True,  # extra vote
        }
        result = composite_signal(0.5, mom)
        assert "sma200" in result["votes"]
        assert result["votes"]["sma200"] == 1
        # Weighted voting (not equal-weight): markov=1.0, rsi=1.5, trend=1.0,
        # sma200=2.0 -> max_score = 1.0+1.5+1.0+2.0 = 5.5
        assert result["max_score"] == 5.5

    def test_composite_signal_hmm_vote(self):
        from Strategy_Auto_Trader.core.momentum import composite_signal
        mom = {
            "cur_rsi": 55,
            "recent_cross_above_50": True,
            "recent_cross_below_40": False,
            "above_sma20": True,
            "above_sma50": True,
        }
        # hmm_state=0 is Bear -> vote -1
        result_bear = composite_signal(0.5, mom, hmm_state=0)
        assert result_bear["votes"]["hmm"] == -1

        # hmm_state=1 is Sideways -> vote 0
        result_sw = composite_signal(0.5, mom, hmm_state=1)
        assert result_sw["votes"]["hmm"] == 0

        # hmm_state=2 is Bull -> vote +1
        result_bull = composite_signal(0.5, mom, hmm_state=2)
        assert result_bull["votes"]["hmm"] == 1
        # Weighted voting: markov=1.0, rsi=1.5, trend=1.0, hmm=1.0 -> 4.5
        assert result_bull["max_score"] == 4.5

    def test_momentum_signals_returns_dict(self):
        from Strategy_Auto_Trader.core.momentum import momentum_signals
        close = _rising_prices(300)
        result = momentum_signals(close)
        assert "cur_rsi" in result
        assert "detail" in result
        assert isinstance(result["detail"], pd.DataFrame)
        assert "above_sma20" in result
        assert "above_sma50" in result

    def test_momentum_signals_short_series_sma200_none(self):
        from Strategy_Auto_Trader.core.momentum import momentum_signals
        close = _rising_prices(100)  # too short for SMA200
        result = momentum_signals(close)
        # SMA200 should be None with fewer than 200 data points
        assert result["cur_sma200"] is None or result["above_sma200"] is None

    def test_momentum_signals_volume_populates_ratio(self):
        from Strategy_Auto_Trader.core.momentum import momentum_signals
        close = _rising_prices(300)
        volume = pd.Series(1_000_000.0, index=close.index, name="volume")
        result = momentum_signals(close, volume=volume)
        assert result["volume_ratio"] is not None
        assert result["cur_volume"] == 1_000_000.0
        assert result["avg_volume"] == 1_000_000.0
        assert "volume_ratio" in result["detail"].columns

    def test_momentum_signals_no_volume_returns_none(self):
        from Strategy_Auto_Trader.core.momentum import momentum_signals
        close = _rising_prices(300)
        result = momentum_signals(close, volume=None)
        assert result["volume_ratio"] is None
        assert result["cur_volume"] is None
        assert result["avg_volume"] is None
        assert "volume_ratio" not in result["detail"].columns

    def test_momentum_signals_short_volume_series_returns_none(self):
        from Strategy_Auto_Trader.core.momentum import momentum_signals
        close = _rising_prices(300)
        short_volume = pd.Series(1_000_000.0, index=close.index[:15], name="volume")
        result = momentum_signals(close, volume=short_volume)
        # len(volume) <= 20 -> volume stats stay None, no lookahead-free rolling window available
        assert result["volume_ratio"] is None
