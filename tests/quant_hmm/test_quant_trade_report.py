from __future__ import annotations

import argparse
from unittest import mock

import numpy as np
import pandas as pd
import pytest


class TestQuantTradeReport:

    def test_build_arg_parser_defaults(self):
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _build_arg_parser
        args = _build_arg_parser().parse_args([])
        assert args.entry_prob == 0.65
        assert args.lot_size == 100.0
        assert args.kelly is True
        assert args.sentiment is True
        assert args.vol_screen is True

    def test_load_watchlist_tickers_plain_strings(self, tmp_path):
        import json
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _load_watchlist_tickers
        wl_path = tmp_path / "watchlist.json"
        wl_path.write_text(json.dumps({"tickers": ["AAA", "BBB"]}), encoding="utf-8")
        assert _load_watchlist_tickers(wl_path) == ["AAA", "BBB"]

    def test_load_watchlist_tickers_dict_entries(self, tmp_path):
        import json
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _load_watchlist_tickers
        wl_path = tmp_path / "watchlist.json"
        wl_path.write_text(json.dumps({"tickers": [{"ticker": "AAA"}, {"ticker": "BBB"}]}), encoding="utf-8")
        assert _load_watchlist_tickers(wl_path) == ["AAA", "BBB"]

    def test_load_watchlist_tickers_empty(self, tmp_path):
        import json
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _load_watchlist_tickers
        wl_path = tmp_path / "watchlist.json"
        wl_path.write_text(json.dumps({"tickers": []}), encoding="utf-8")
        assert _load_watchlist_tickers(wl_path) == []

    def _make_args(self, **overrides):
        defaults = dict(stop_loss=0.05, take_profit=0.15, lot_size=100.0, trade_cost=1.0)
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_build_trade_row_closed_trade(self):
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _build_trade_row
        idx = pd.bdate_range("2024-01-01", periods=3, freq="h")
        detail = pd.DataFrame({
            "close": [100.0, 110.0, 111.0],
            "p_bull": [0.7, 0.3, 0.3],
            "volume_ratio": [1.5, 1.2, 1.0],
            "kelly_fraction": [0.1, 0.1, 0.1],
            "stop_level": [95.0, None, None],
            "target_level": [115.0, None, None],
            "sell_reason": ["", "stop_loss(5.0% loss)", ""],
        }, index=idx)
        buys = detail.iloc[[0]]
        sells = detail.iloc[[1]]
        args = self._make_args()
        row = _build_trade_row(
            "AAA", "Acme Co", "Tech", idx[0], buys.loc[idx[0]],
            [idx[1]], sells, detail, pd.Timestamp("2023-01-01"), args, None, 0.0,
        )
        assert row is not None
        assert row["Status"] == "CLOSED"
        assert row["Exit Reason"] == "Stop Loss"
        assert row["Entry Price"] == 100.0
        assert row["Exit Price"] == 110.0

    def test_build_trade_row_open_trade_uses_last_bar(self):
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _build_trade_row
        idx = pd.bdate_range("2024-01-01", periods=2, freq="h")
        detail = pd.DataFrame({
            "close": [100.0, 105.0],
            "p_bull": [0.7, 0.6],
            "volume_ratio": [1.5, 1.3],
            "kelly_fraction": [0.1, 0.1],
            "stop_level": [95.0, None],
            "target_level": [115.0, None],
            "sell_reason": ["", ""],
        }, index=idx)
        buys = detail.iloc[[0]]
        sells = detail.iloc[0:0]
        args = self._make_args()
        row = _build_trade_row(
            "AAA", "Acme Co", "Tech", idx[0], buys.loc[idx[0]],
            [], sells, detail, pd.Timestamp("2023-01-01"), args, None, 0.0,
        )
        assert row is not None
        assert row["Status"] == "OPEN"
        assert row["Exit Price"] == 105.0

    def test_build_trade_row_before_start_date_returns_none(self):
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _build_trade_row
        idx = pd.bdate_range("2024-01-01", periods=2, freq="h")
        detail = pd.DataFrame({
            "close": [100.0, 110.0], "p_bull": [0.7, 0.3],
            "volume_ratio": [1.5, 1.2], "kelly_fraction": [0.1, 0.1],
            "stop_level": [95.0, None], "target_level": [115.0, None],
            "sell_reason": ["", "signal"],
        }, index=idx)
        buys = detail.iloc[[0]]
        sells = detail.iloc[[1]]
        args = self._make_args()
        row = _build_trade_row(
            "AAA", "Acme Co", "Tech", idx[0], buys.loc[idx[0]],
            [idx[1]], sells, detail, pd.Timestamp("2030-01-01"), args, None, 0.0,
        )
        assert row is None

    def test_build_trade_row_includes_sentiment_when_provided(self):
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _build_trade_row
        idx = pd.bdate_range("2024-01-01", periods=2, freq="h")
        detail = pd.DataFrame({
            "close": [100.0, 110.0], "p_bull": [0.7, 0.3],
            "volume_ratio": [1.5, 1.2], "kelly_fraction": [0.1, 0.1],
            "stop_level": [95.0, None], "target_level": [115.0, None],
            "sell_reason": ["", "signal"],
        }, index=idx)
        buys = detail.iloc[[0]]
        sells = detail.iloc[[1]]
        args = self._make_args()
        sent_data = {"sentiment_label": "Bullish",
                     "options": {"put_call_ratio": 0.5, "iv_rank": 40},
                     "short_interest": {"short_pct_float": 2.0}}
        row = _build_trade_row(
            "AAA", "Acme Co", "Tech", idx[0], buys.loc[idx[0]],
            [idx[1]], sells, detail, pd.Timestamp("2023-01-01"), args, sent_data, 0.8,
        )
        assert row["Sentiment"] == "Bullish"
        assert row["P/C Ratio"] == 0.5

    def test_build_ticker_summary_row_computes_win_rate(self):
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _build_ticker_summary_row
        trades = [
            {"Net P&L": 10.0, "Status": "CLOSED", "Hours Held": 5},
            {"Net P&L": -5.0, "Status": "CLOSED", "Hours Held": 10},
            {"Net P&L": 3.0, "Status": "OPEN", "Hours Held": 2},
        ]
        bt = {"sharpe_strategy": 1.5, "max_drawdown_strategy": -0.1}
        row = _build_ticker_summary_row("AAA", "Acme Co", "Tech", trades, bt, None, 0.0, {})
        assert row["Trades"] == 3
        assert row["Wins"] == 1
        assert row["Losses"] == 1
        assert row["Open"] == 1
        assert row["Win Rate %"] == 50.0
        assert row["Total Net P&L"] == 8.0

    def test_build_ticker_summary_row_includes_vol_profile(self):
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _build_ticker_summary_row
        trades = [{"Net P&L": 10.0, "Status": "CLOSED", "Hours Held": 5}]
        bt = {"sharpe_strategy": float("nan"), "max_drawdown_strategy": float("nan")}
        vol_profiles = {"AAA": {"trend_quality": 0.8, "efficiency_ratio": 0.6}}
        row = _build_ticker_summary_row("AAA", "Acme Co", "Tech", trades, bt, None, 0.0, vol_profiles)
        assert row["Trend Quality"] == 0.8
        assert row["Sharpe"] == "N/A"

    def test_build_sentiment_row(self):
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _build_sentiment_row
        sent_data = {
            "sentiment_label": "Bullish", "confidence": 0.8,
            "options": {"put_call_ratio": 0.5, "put_call_signal": 1, "iv_current": 0.3,
                        "iv_rank": 40, "iv_signal": 0, "skew": 0.1},
            "insider": {"insider_net": 1000, "insider_signal": 1},
            "short_interest": {"short_pct_float": 2.0, "short_ratio": 1.5},
        }
        row = _build_sentiment_row("AAA", "Acme Co", sent_data, 0.8)
        assert row["P/C Signal"] == "Bullish"
        assert row["Insider Signal"] == "Buying"

    def test_build_exit_breakdown_rows(self):
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _build_exit_breakdown_rows
        closed = pd.DataFrame({
            "Exit Reason": ["Stop Loss", "Stop Loss", "Take Profit"],
            "Net P&L": [-5.0, -3.0, 10.0],
            "Hours Held": [10, 20, 5],
        })
        rows = _build_exit_breakdown_rows(closed)
        by_reason = {r["Exit Reason"]: r for r in rows}
        assert by_reason["Stop Loss"]["Count"] == 2
        assert by_reason["Stop Loss"]["Losses"] == 2
        assert by_reason["Take Profit"]["Wins"] == 1

    def test_build_stats_rows_basic(self):
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _build_stats_rows
        args = self._make_args(entry_prob=0.65, exit_prob=0.40, volume_min=1.0,
                                kelly=True, vol_screen=False, min_trend_quality=0.0,
                                start_date="2024-01-01")
        trades_df = pd.DataFrame({
            "Net P&L": [10.0, -5.0], "Costs": [1.0, 1.0], "Status": ["CLOSED", "CLOSED"],
            "Hours Held": [5, 10], "Ticker": ["AAA", "AAA"], "Entry Date": ["d1", "d2"],
        })
        closed = trades_df
        stats = _build_stats_rows(args, trades_df, closed, [{"Total Net P&L": 5.0}], [])
        stats_dict = {s["Metric"]: s["Value"] for s in stats}
        assert stats_dict["Total Trades"] == 2
        assert stats_dict["Wins"] == 1
        assert stats_dict["Losses"] == 1

    def test_print_final_summary_outputs_totals(self, capsys):
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _print_final_summary
        trades_df = pd.DataFrame({
            "Net P&L": [10.0, -5.0, 3.0],
            "Status": ["CLOSED", "CLOSED", "OPEN"],
        })
        _print_final_summary(trades_df)
        out = capsys.readouterr().out
        assert "Total trades:  3" in out
        assert "Win rate:" in out

    # -- new helper tests ---------------------------------------------------

    def test_safe_round_normal(self):
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _safe_round
        assert _safe_round(1.23456, 3) == 1.235
        assert _safe_round(0, 2) == 0.0

    def test_safe_round_none_and_nan(self):
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _safe_round
        assert _safe_round(None) is None
        assert _safe_round(float("nan")) is None
        assert _safe_round(float("inf")) is None
        assert _safe_round("bad") is None

    def test_lookup_daily_at_exact_match(self):
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _lookup_daily_at
        idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        df = pd.DataFrame({"rsi": [40.0, 50.0, 60.0]}, index=idx)
        result = _lookup_daily_at(df, pd.Timestamp("2024-01-03"))
        assert result["rsi"] == 50.0

    def test_lookup_daily_at_uses_prior_row_when_no_exact_match(self):
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _lookup_daily_at
        idx = pd.to_datetime(["2024-01-02", "2024-01-04"])
        df = pd.DataFrame({"rsi": [40.0, 60.0]}, index=idx)
        # 2024-01-03 is between rows — should return the 2024-01-02 row
        result = _lookup_daily_at(df, pd.Timestamp("2024-01-03"))
        assert result["rsi"] == 40.0

    def test_lookup_daily_at_strips_tz(self):
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _lookup_daily_at
        idx = pd.to_datetime(["2024-01-02"])
        df = pd.DataFrame({"rsi": [42.0]}, index=idx)
        ts = pd.Timestamp("2024-01-02 14:30:00", tz="UTC")
        result = _lookup_daily_at(df, ts)
        assert result["rsi"] == 42.0

    def test_lookup_daily_at_none_df_returns_empty(self):
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _lookup_daily_at
        assert _lookup_daily_at(None, pd.Timestamp("2024-01-02")) == {}

    def test_lookup_daily_at_no_prior_row_returns_empty(self):
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _lookup_daily_at
        idx = pd.to_datetime(["2024-01-10"])
        df = pd.DataFrame({"rsi": [55.0]}, index=idx)
        assert _lookup_daily_at(df, pd.Timestamp("2024-01-01")) == {}

    def test_build_trade_row_includes_hmm_extras(self):
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _build_trade_row
        idx = pd.bdate_range("2024-01-01", periods=3, freq="h")
        detail = pd.DataFrame({
            "close": [100.0, 110.0, 111.0],
            "p_bull": [0.7, 0.3, 0.3],
            "p_bull_smooth": [0.65, 0.32, 0.31],
            "p_bear": [0.1, 0.5, 0.5],
            "volume_ratio": [1.5, 1.2, 1.0],
            "kelly_fraction": [0.10, 0.12, 0.12],
            "stop_level": [95.0, None, None],
            "target_level": [115.0, None, None],
            "sell_reason": ["", "stop_loss(5.0%)", ""],
        }, index=idx)
        buys = detail.iloc[[0]]
        sells = detail.iloc[[1]]
        row = _build_trade_row(
            "AAA", "Acme", "Tech", idx[0], buys.loc[idx[0]],
            [idx[1]], sells, detail, pd.Timestamp("2023-01-01"),
            self._make_args(), None, 0.0,
        )
        assert row is not None
        assert row["P(Bull) Smooth Entry"] == 0.65
        assert row["P(Bear) Entry"] == 0.1
        assert row["P(Bull) Smooth Exit"] == 0.32
        assert row["P(Bear) Exit"] == 0.5
        assert row["Vol Ratio Exit"] == 1.2
        assert row["Kelly % Exit"] == 12.0

    def test_build_trade_row_includes_daily_indicators(self):
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _build_trade_row
        idx = pd.bdate_range("2024-01-01", periods=2, freq="h")
        detail = pd.DataFrame({
            "close": [100.0, 110.0],
            "p_bull": [0.7, 0.3],
            "p_bull_smooth": [0.65, 0.32],
            "p_bear": [0.1, 0.5],
            "volume_ratio": [1.5, 1.2],
            "kelly_fraction": [0.10, 0.12],
            "stop_level": [95.0, None],
            "target_level": [115.0, None],
            "sell_reason": ["", "regime_exit"],
        }, index=idx)
        buys = detail.iloc[[0]]
        sells = detail.iloc[[1]]

        daily_idx = pd.to_datetime(["2024-01-01"])
        daily_df = pd.DataFrame({
            "rsi": [52.0], "above_sma20": [True], "pct_from_sma20": [3.5],
            "above_sma50": [True], "pct_from_sma50": [7.2],
            "above_sma200": [True], "pct_from_sma200": [15.0],
            "macd_histogram": [0.05], "macd_trend": ["bullish"],
            "atr": [1.2], "bb_width": [0.08],
        }, index=daily_idx)

        row = _build_trade_row(
            "AAA", "Acme", "Tech", idx[0], buys.loc[idx[0]],
            [idx[1]], sells, detail, pd.Timestamp("2023-01-01"),
            self._make_args(), None, 0.0,
            daily_df=daily_df,
        )
        assert row["RSI Entry"] == 52.0
        assert row["Above SMA20"] == True  # noqa: E712 — np.True_ != True with `is`
        assert row["% from SMA20"] == 3.5
        assert row["MACD Trend Entry"] == "bullish"

    def test_build_trade_row_includes_vol_profile(self):
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _build_trade_row
        idx = pd.bdate_range("2024-01-01", periods=2, freq="h")
        detail = pd.DataFrame({
            "close": [100.0, 110.0],
            "p_bull": [0.7, 0.3],
            "p_bull_smooth": [0.65, 0.32],
            "p_bear": [0.1, 0.5],
            "volume_ratio": [1.5, 1.2],
            "kelly_fraction": [0.10, 0.12],
            "stop_level": [95.0, None],
            "target_level": [115.0, None],
            "sell_reason": ["", "regime_exit"],
        }, index=idx)
        buys = detail.iloc[[0]]
        sells = detail.iloc[[1]]
        vp = {
            "ann_vol": 0.25, "efficiency_ratio": 0.42, "autocorr": 0.18,
            "choppiness_idx": 38.5, "sign_change_freq": 0.45, "trend_quality": 0.6,
        }
        row = _build_trade_row(
            "AAA", "Acme", "Tech", idx[0], buys.loc[idx[0]],
            [idx[1]], sells, detail, pd.Timestamp("2023-01-01"),
            self._make_args(), None, 0.0, vol_profile=vp,
        )
        assert row["Ann Vol"] == 0.25
        assert row["Efficiency Ratio"] == 0.42
        assert row["Trend Quality"] == 0.6

    def test_build_trade_row_extended_sentiment(self):
        from Strategy_Auto_Trader.quant_hmm.quant_trade_report import _build_trade_row
        idx = pd.bdate_range("2024-01-01", periods=2, freq="h")
        detail = pd.DataFrame({
            "close": [100.0, 110.0],
            "p_bull": [0.7, 0.3],
            "p_bull_smooth": [0.65, 0.32],
            "p_bear": [0.1, 0.5],
            "volume_ratio": [1.5, 1.2],
            "kelly_fraction": [0.10, 0.12],
            "stop_level": [95.0, None],
            "target_level": [115.0, None],
            "sell_reason": ["", "signal"],
        }, index=idx)
        buys = detail.iloc[[0]]
        sells = detail.iloc[[1]]
        sent_data = {
            "sentiment_label": "Bearish",
            "confidence": 3,
            "options": {
                "put_call_ratio": 1.2, "iv_rank": 65, "iv_current": 0.32,
                "iv_signal": -1, "skew": 0.06,
            },
            "vix": {"vix_current": 22.5, "vix_regime": "high-vol", "vix_signal": -1},
            "insider": {"insider_net": -5, "insider_signal": -1},
            "short_interest": {"short_pct_float": 8.5, "short_ratio": 3.2},
        }
        row = _build_trade_row(
            "AAA", "Acme", "Tech", idx[0], buys.loc[idx[0]],
            [idx[1]], sells, detail, pd.Timestamp("2023-01-01"),
            self._make_args(), sent_data, -0.4,
        )
        assert row["Sent Confidence"] == 3
        assert row["IV Current"] == 0.32
        assert row["IV Signal"] == "High (risky)"
        assert row["Skew"] == 0.06
        assert row["VIX"] == 22.5
        assert row["VIX Regime"] == "high-vol"
        assert row["Insider Signal"] == "Selling"
        assert row["Short Ratio"] == 3.2
