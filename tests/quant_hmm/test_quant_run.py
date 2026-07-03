from __future__ import annotations

import argparse
from unittest import mock

import pandas as pd
import pytest


class TestQuantRun:

    def test_build_arg_parser_defaults(self):
        from Strategy_Auto_Trader.quant_hmm.quant_run import _build_arg_parser
        args = _build_arg_parser().parse_args([])
        assert args.ticker == "SPY"
        assert args.entry_prob == 0.65
        assert args.kelly is True
        assert args.sentiment is True

    def _make_args(self, **overrides):
        defaults = dict(trade_cost=10.0, entry_prob=0.65)
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def _make_detail(self, in_position=True):
        row = {
            "p_bull": 0.7, "p_bear": 0.1, "volume_ratio": 1.5,
            "position": 0.1 if in_position else 0.0,
            "entry_price": 100.0, "stop_level": 95.0, "target_level": 115.0,
        }
        return pd.DataFrame([row])

    def test_print_results_in_position(self, capsys):
        from Strategy_Auto_Trader.quant_hmm.quant_run import _print_results
        bt = {
            "sharpe_strategy": 1.2, "sharpe_bh": 0.8,
            "total_return_strategy": 0.15, "total_return_bh": 0.05,
            "max_drawdown_strategy": -0.1, "max_drawdown_bh": -0.2,
            "trade_results": [0.05, -0.02], "n_buys": 2, "n_sells": 1,
            "final_kelly": 0.12, "initial_cash": 20000.0, "final_portfolio": 21000.0,
            "total_pl": 1000.0,
        }
        args = self._make_args()
        detail = self._make_detail(in_position=True)
        _print_results(bt, args, detail)
        out = capsys.readouterr().out
        assert "Sharpe (annualised)" in out
        assert "Trades: 2" in out
        assert "IN POSITION" in out

    def test_print_results_flat_no_trades(self, capsys):
        from Strategy_Auto_Trader.quant_hmm.quant_run import _print_results
        bt = {
            "sharpe_strategy": float("nan"), "sharpe_bh": float("nan"),
            "total_return_strategy": 0.0, "total_return_bh": 0.0,
            "max_drawdown_strategy": float("nan"), "max_drawdown_bh": float("nan"),
            "trade_results": [], "n_buys": 0, "n_sells": 0,
            "final_kelly": 0.0, "initial_cash": 20000.0, "final_portfolio": 20000.0,
            "total_pl": 0.0,
        }
        args = self._make_args()
        detail = self._make_detail(in_position=False)
        _print_results(bt, args, detail)
        out = capsys.readouterr().out
        assert "No completed trades." in out
        assert "FLAT" in out

    def test_print_exit_breakdown_no_trades_prints_nothing(self, capsys):
        from Strategy_Auto_Trader.quant_hmm.quant_run import _print_exit_breakdown
        _print_exit_breakdown(pd.DataFrame(), 0)
        assert capsys.readouterr().out == ""

    def test_print_exit_breakdown_groups_by_reason(self, capsys):
        from Strategy_Auto_Trader.quant_hmm.quant_run import _print_exit_breakdown
        detail = pd.DataFrame({
            "trade_event": ["SELL", "SELL", "SELL"],
            "sell_reason": ["stop_loss(5.0%)", "take_profit(15.0%)", "stop_loss(3.0%)"],
            "close": [95.0, 115.0, 97.0],
            "entry_price": [100.0, 100.0, 100.0],
        })
        _print_exit_breakdown(detail, n_trades=3)
        out = capsys.readouterr().out
        assert "stop_loss" in out
        assert "take_profit" in out

    def test_print_sentiment_detail_outputs_composite_line(self, capsys):
        from Strategy_Auto_Trader.quant_hmm.quant_run import _print_sentiment_detail
        sent_data = {
            "sentiment_label": "bullish", "sentiment_score": 0.5, "confidence": 3,
            "options": {"put_call_ratio": 0.6, "put_call_signal": 1,
                        "iv_current": 0.3, "iv_signal": 0, "iv_rank": 40.0, "skew": 0.1},
            "vix": {"vix_current": 18.0, "vix_regime": "normal", "vix_term_structure": "contango"},
            "insider": {"insider_net": 1000, "insider_buys_90d": 3, "insider_sells_90d": 1},
            "short_interest": {"short_pct_float": 2.5, "short_ratio": 1.2},
        }
        _print_sentiment_detail(sent_data)
        out = capsys.readouterr().out
        assert "Sentiment & Alternative Data" in out
        assert "BULLISH" in out
        assert "Put/Call ratio" in out
