from __future__ import annotations

import json
from unittest import mock

import pandas as pd
import pytest


class TestRun:

    def test_build_arg_parser_defaults(self):
        from Strategy_Auto_Trader.markov_cli.run import _build_arg_parser
        args = _build_arg_parser().parse_args(["--ticker", "AAPL"])
        assert args.ticker == "AAPL"
        assert args.entry_prob == 0.65
        assert args.exit_prob == 0.40
        assert args.stop_loss_pct == 0.05
        assert args.take_profit_pct == 0.15
        assert args.use_kelly is True
        assert args.buy_threshold == 3.0
        assert args.regime_smooth == 24
        assert args.min_hold_bars == 48

    def test_fetch_company_info_success(self):
        from Strategy_Auto_Trader.markov_cli import run as run_mod

        class FakeTicker:
            def __init__(self, ticker):
                self.info = {"longName": "Apple Inc.", "sector": "Technology"}

        with mock.patch("yfinance.Ticker", FakeTicker):
            name, sector = run_mod._fetch_company_info("AAPL")
        assert name == "Apple Inc."
        assert sector == "Technology"

    def test_fetch_company_info_exception_falls_back_to_ticker(self):
        from Strategy_Auto_Trader.markov_cli import run as run_mod

        class FakeTicker:
            def __init__(self, ticker):
                raise RuntimeError("network error")

        with mock.patch("yfinance.Ticker", FakeTicker):
            name, sector = run_mod._fetch_company_info("AAPL")
        assert name == "AAPL"
        assert sector == ""

    def test_write_quality_gate_writes_default_payload(self, tmp_path):
        from Strategy_Auto_Trader.markov_cli.run import _write_quality_gate
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_quality_gate(run_dir, "HOLD", "insufficient data")
        gate_path = run_dir / "qualityGate.json"
        assert gate_path.exists()
        data = json.loads(gate_path.read_text(encoding="utf-8"))
        assert data["flag"] == "HOLD"
        assert data["reason"] == "insufficient data"

    def test_print_backtest_summary_outputs_key_lines(self, capsys):
        from Strategy_Auto_Trader.markov_cli.run import _print_backtest_summary
        detail = pd.DataFrame(
            {"trade_event": ["BUY", "HOLD", "SELL"]},
            index=pd.bdate_range("2024-01-01", periods=3),
        )
        bt = {
            "sharpe_strategy": 1.2, "sharpe_bh": 0.8,
            "max_drawdown_strategy": -0.1, "max_drawdown_bh": -0.2,
            "total_return_strategy": 0.15, "total_return_bh": 0.05,
            "n_bars": 3,
            "initial_cash": 20000.0, "final_portfolio": 21000.0, "total_pl": 1000.0,
            "trade_cost": 10.0, "n_buys": 1, "n_sells": 1,
            "final_kelly": 0.10, "detail": detail,
        }
        _print_backtest_summary(bt)
        out = capsys.readouterr().out
        assert "Sharpe (annualised)" in out
        assert "Strategy P&L" in out
