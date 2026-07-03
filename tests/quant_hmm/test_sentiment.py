from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest


class TestSentiment:

    # -- options_signals ------------------------------------------------------

    def _mock_options_ticker(self):
        tk = mock.MagicMock()
        tk.options = ("2024-01-19", "2024-02-16", "2024-03-15")
        calls = pd.DataFrame({"openInterest": [100, 200], "impliedVolatility": [0.20, 0.22],
                              "strike": [110.0, 115.0]})
        puts = pd.DataFrame({"openInterest": [300, 400], "impliedVolatility": [0.28, 0.30],
                             "strike": [90.0, 85.0]})
        tk.option_chain.return_value = SimpleNamespace(calls=calls, puts=puts)
        tk.info = {"currentPrice": 100.0}
        hist_idx = pd.date_range("2023-01-01", periods=260, freq="D")
        tk.history.return_value = pd.DataFrame(
            {"Close": 100.0 + np.sin(np.linspace(0, 20, 260))}, index=hist_idx)
        return tk

    def test_options_signals_high_put_call_is_bullish_contrarian(self):
        from Strategy_Auto_Trader.quant_hmm.sentiment import options_signals
        tk = self._mock_options_ticker()
        with mock.patch("yfinance.Ticker", return_value=tk):
            result = options_signals("TEST")
        assert abs(result["put_call_ratio"] - round(700 / 300, 3)) < 1e-9
        assert result["put_call_signal"] == 1

    def test_options_signals_iv_current_is_median(self):
        from Strategy_Auto_Trader.quant_hmm.sentiment import options_signals
        tk = self._mock_options_ticker()
        with mock.patch("yfinance.Ticker", return_value=tk):
            result = options_signals("TEST")
        assert abs(result["iv_current"] - 0.25) < 1e-9  # median of [0.20,0.22,0.28,0.30]

    def test_options_signals_skew_positive_when_puts_richer(self):
        from Strategy_Auto_Trader.quant_hmm.sentiment import options_signals
        tk = self._mock_options_ticker()
        with mock.patch("yfinance.Ticker", return_value=tk):
            result = options_signals("TEST")
        # OTM puts (strike<95) have higher IV than OTM calls (strike>105) -> positive skew
        assert result["skew"] is not None
        assert result["skew"] > 0

    def test_options_signals_no_expirations_returns_defaults(self):
        from Strategy_Auto_Trader.quant_hmm.sentiment import options_signals
        tk = mock.MagicMock()
        tk.options = ()
        with mock.patch("yfinance.Ticker", return_value=tk):
            result = options_signals("TEST")
        assert result["put_call_ratio"] is None
        assert result["put_call_signal"] == 0
        assert result["iv_current"] is None

    def test_options_signals_exception_returns_defaults(self):
        from Strategy_Auto_Trader.quant_hmm.sentiment import options_signals
        with mock.patch("yfinance.Ticker", side_effect=Exception("boom")):
            result = options_signals("TEST")
        assert result["put_call_ratio"] is None

    # -- vix_regime -------------------------------------------------------------

    def _fake_vix_download(self, vix_level, vix3m_level):
        def _download(ticker, period=None, **kwargs):
            if ticker == "^VIX":
                idx = pd.date_range("2024-01-01", periods=30, freq="D")
                return pd.DataFrame({"Close": np.full(30, vix_level)}, index=idx)
            if ticker == "^VIX3M":
                idx = pd.date_range("2024-01-01", periods=5, freq="D")
                return pd.DataFrame({"Close": np.full(5, vix3m_level)}, index=idx)
            return pd.DataFrame()
        return _download

    def test_vix_regime_low_vol(self):
        from Strategy_Auto_Trader.quant_hmm.sentiment import vix_regime
        with mock.patch("yfinance.download", side_effect=self._fake_vix_download(12.0, 14.0)):
            result = vix_regime()
        assert result["vix_regime"] == "low_vol"
        assert result["vix_signal"] == 1
        assert result["vix_term_structure"] == "contango"

    def test_vix_regime_crisis(self):
        from Strategy_Auto_Trader.quant_hmm.sentiment import vix_regime
        with mock.patch("yfinance.download", side_effect=self._fake_vix_download(45.0, 30.0)):
            result = vix_regime()
        assert result["vix_regime"] == "crisis"
        assert result["vix_signal"] == -1
        assert result["vix_term_structure"] == "backwardation"

    def test_vix_regime_normal(self):
        from Strategy_Auto_Trader.quant_hmm.sentiment import vix_regime
        with mock.patch("yfinance.download", side_effect=self._fake_vix_download(20.0, 21.0)):
            result = vix_regime()
        assert result["vix_regime"] == "normal"
        assert result["vix_signal"] == 0

    def test_vix_regime_no_data_returns_defaults(self):
        from Strategy_Auto_Trader.quant_hmm.sentiment import vix_regime
        with mock.patch("yfinance.download", return_value=pd.DataFrame()):
            result = vix_regime()
        assert result["vix_current"] is None
        assert result["vix_signal"] == 0

    # -- insider_signals ----------------------------------------------------

    def test_insider_signals_net_buying(self):
        from Strategy_Auto_Trader.quant_hmm.sentiment import insider_signals
        now = pd.Timestamp.now()
        df = pd.DataFrame({
            "Start Date": [now - pd.Timedelta(days=5), now - pd.Timedelta(days=10),
                          now - pd.Timedelta(days=20)],
            "Text": ["Purchase at price 1.0", "Purchase at price 2.0", "Sale at price 3.0"],
        })
        tk = mock.MagicMock()
        tk.insider_transactions = df
        with mock.patch("yfinance.Ticker", return_value=tk):
            result = insider_signals("TEST")
        assert result["insider_buys_90d"] == 2
        assert result["insider_sells_90d"] == 1
        assert result["insider_net"] == 1
        assert result["insider_signal"] == 1  # net_buys>=2 -> bullish

    def test_insider_signals_net_selling(self):
        from Strategy_Auto_Trader.quant_hmm.sentiment import insider_signals
        now = pd.Timestamp.now()
        df = pd.DataFrame({
            "Start Date": [now - pd.Timedelta(days=i) for i in range(4)],
            "Text": ["Sale", "Sale", "Sale", "Purchase"],
        })
        tk = mock.MagicMock()
        tk.insider_transactions = df
        with mock.patch("yfinance.Ticker", return_value=tk):
            result = insider_signals("TEST")
        assert result["insider_signal"] == -1  # net_sells>=3 -> bearish

    def test_insider_signals_no_data(self):
        from Strategy_Auto_Trader.quant_hmm.sentiment import insider_signals
        tk = mock.MagicMock()
        tk.insider_transactions = pd.DataFrame()
        with mock.patch("yfinance.Ticker", return_value=tk):
            result = insider_signals("TEST")
        assert result["insider_net"] == 0
        assert result["insider_signal"] == 0

    # -- short_interest_signal -----------------------------------------------

    def test_short_interest_signal_high_short(self):
        from Strategy_Auto_Trader.quant_hmm.sentiment import short_interest_signal
        tk = mock.MagicMock()
        tk.info = {"shortPercentOfFloat": 0.15, "shortRatio": 3.2}
        with mock.patch("yfinance.Ticker", return_value=tk):
            result = short_interest_signal("TEST")
        assert abs(result["short_pct_float"] - 15.0) < 1e-9
        assert result["short_signal"] == 1

    def test_short_interest_signal_no_data(self):
        from Strategy_Auto_Trader.quant_hmm.sentiment import short_interest_signal
        tk = mock.MagicMock()
        tk.info = {}
        with mock.patch("yfinance.Ticker", return_value=tk):
            result = short_interest_signal("TEST")
        assert result["short_pct_float"] is None
        assert result["short_signal"] == 0

    # -- composite_sentiment --------------------------------------------------

    def test_composite_sentiment_bullish(self):
        from Strategy_Auto_Trader.quant_hmm import sentiment as sm
        with mock.patch.object(sm, "options_signals", return_value={
                "put_call_ratio": 1.5, "put_call_signal": 1, "iv_signal": 0,
                "iv_current": 0.2, "iv_rank": 50, "skew": 0.0}), \
             mock.patch.object(sm, "vix_regime", return_value={
                "vix_current": 12, "vix_signal": 1, "vix_term_structure": "contango"}), \
             mock.patch.object(sm, "insider_signals", return_value={
                "insider_net": 3, "insider_signal": 1,
                "insider_buys_90d": 3, "insider_sells_90d": 0}), \
             mock.patch.object(sm, "short_interest_signal", return_value={
                "short_pct_float": 15.0, "short_signal": 1}):
            result = sm.composite_sentiment("TEST")
        assert result["sentiment_label"] == "bullish"
        assert result["sentiment_score"] > 0.3
        assert result["confidence"] == 4

    def test_composite_sentiment_bearish(self):
        from Strategy_Auto_Trader.quant_hmm import sentiment as sm
        with mock.patch.object(sm, "options_signals", return_value={
                "put_call_ratio": 0.3, "put_call_signal": -1, "iv_signal": -1,
                "iv_current": 0.5, "iv_rank": 90, "skew": 0.1}), \
             mock.patch.object(sm, "vix_regime", return_value={
                "vix_current": 40, "vix_signal": -1, "vix_term_structure": "backwardation"}), \
             mock.patch.object(sm, "insider_signals", return_value={
                "insider_net": -3, "insider_signal": -1,
                "insider_buys_90d": 0, "insider_sells_90d": 3}), \
             mock.patch.object(sm, "short_interest_signal", return_value={
                "short_pct_float": 1.0, "short_signal": 0}):
            result = sm.composite_sentiment("TEST")
        assert result["sentiment_label"] == "bearish"
        assert result["sentiment_score"] < -0.3

    def test_composite_sentiment_no_data_is_neutral(self):
        from Strategy_Auto_Trader.quant_hmm import sentiment as sm
        with mock.patch.object(sm, "options_signals", return_value={
                "put_call_ratio": None, "put_call_signal": 0, "iv_signal": 0,
                "iv_current": None, "iv_rank": None, "skew": None}), \
             mock.patch.object(sm, "vix_regime", return_value={
                "vix_current": None, "vix_signal": 0, "vix_term_structure": None}), \
             mock.patch.object(sm, "insider_signals", return_value={
                "insider_net": 0, "insider_signal": 0,
                "insider_buys_90d": 0, "insider_sells_90d": 0}), \
             mock.patch.object(sm, "short_interest_signal", return_value={
                "short_pct_float": None, "short_signal": 0}):
            result = sm.composite_sentiment("TEST")
        assert result["sentiment_label"] == "neutral"
        assert result["sentiment_score"] == 0.0
        assert result["confidence"] == 0
