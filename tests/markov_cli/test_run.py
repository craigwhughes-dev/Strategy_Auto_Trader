from __future__ import annotations

import json
import logging
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
        # Strategy-owned tunables are argparse.SUPPRESS'd when omitted — the
        # selected strategy's own default applies instead of a CLI-wide one.
        assert not hasattr(args, "stop_loss_pct")
        assert not hasattr(args, "take_profit_pct")
        assert not hasattr(args, "buy_threshold")
        assert args.use_kelly is True
        assert args.interval == "1h"
        assert args.regime_smooth is None   # resolved by _resolve_interval_defaults
        assert args.min_hold_bars is None   # resolved by _resolve_interval_defaults
        assert args.skip_unused_indicators is True
        assert args.hmm_cache is True

    def test_build_arg_parser_no_hmm_cache_flag(self):
        from Strategy_Auto_Trader.markov_cli.run import _build_arg_parser
        args = _build_arg_parser().parse_args(["--ticker", "AAPL", "--no-hmm-cache"])
        assert args.hmm_cache is False

    def test_build_arg_parser_signal_reports_only(self):
        from Strategy_Auto_Trader.markov_cli.run import _build_arg_parser
        assert _build_arg_parser().parse_args(["--ticker", "AAPL"]).signal_reports_only is False
        args = _build_arg_parser().parse_args(["--ticker", "AAPL", "--signal-reports-only"])
        assert args.signal_reports_only is True

    def test_build_strategy_overrides_empty_when_nothing_explicit(self):
        from Strategy_Auto_Trader.markov_cli.run import _build_arg_parser, _build_strategy_overrides
        args = _build_arg_parser().parse_args(["--ticker", "AAPL", "--strategy", "optimised"])
        entry_overrides, exit_overrides = _build_strategy_overrides(args)
        assert entry_overrides == {}
        assert exit_overrides == {}

    def test_build_strategy_overrides_picks_up_explicit_exit_flags(self):
        from Strategy_Auto_Trader.markov_cli.run import _build_arg_parser, _build_strategy_overrides
        args = _build_arg_parser().parse_args([
            "--ticker", "AAPL", "--strategy", "optimised",
            "--trailing-stop", "0.15", "--profit-stop-scale", "0.3",
            "--take-profit-pct", "999", "--min-stop", "0.02",
        ])
        entry_overrides, exit_overrides = _build_strategy_overrides(args)
        assert entry_overrides == {}
        assert exit_overrides == {
            "trailing_stop": 0.15, "profit_stop_scale": 0.3,
            "take_profit_pct": 999.0, "min_stop_pct": 0.02,
        }

    def test_build_strategy_overrides_picks_up_explicit_entry_flags(self):
        from Strategy_Auto_Trader.markov_cli.run import _build_arg_parser, _build_strategy_overrides
        args = _build_arg_parser().parse_args([
            "--ticker", "AAPL", "--strategy", "optimised",
            "--buy-threshold", "1.0", "--sell-threshold", "-1.0",
        ])
        entry_overrides, exit_overrides = _build_strategy_overrides(args)
        assert entry_overrides == {"buy_threshold": 1.0, "sell_threshold": -1.0}
        assert exit_overrides == {}

    def test_build_strategy_overrides_plugin_gate_none(self):
        from Strategy_Auto_Trader.markov_cli.run import _build_arg_parser, _build_strategy_overrides
        args = _build_arg_parser().parse_args([
            "--ticker", "AAPL", "--strategy", "optimised", "--plugin-gate", "none",
        ])
        entry_overrides, _ = _build_strategy_overrides(args)
        assert entry_overrides == {"quality_gate_enabled": False}

    def test_backfill_tunable_defaults_fills_only_missing(self):
        from Strategy_Auto_Trader.markov_cli.run import (
            _build_arg_parser, _backfill_tunable_defaults, _TUNABLE_DEFAULTS,
        )
        args = _build_arg_parser().parse_args([
            "--ticker", "AAPL", "--stop-loss-pct", "0.5",
        ])
        _backfill_tunable_defaults(args)
        assert args.stop_loss_pct == 0.5  # explicit value untouched
        for key, value in _TUNABLE_DEFAULTS.items():
            if key != "stop_loss_pct":
                assert getattr(args, key) == value  # backfilled

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

    def test_print_backtest_summary_outputs_key_lines(self, caplog):
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
        caplog.set_level(logging.INFO)
        _print_backtest_summary(bt)
        assert "Sharpe (annualised)" in caplog.text
        assert "Strategy P&L" in caplog.text

    # -- daily-bar interval support -----------------------------------------

    def test_interval_defaults_to_1h(self):
        from Strategy_Auto_Trader.markov_cli.run import _build_arg_parser
        args = _build_arg_parser().parse_args(["--ticker", "SPY"])
        assert args.interval == "1h"

    def test_resolve_interval_defaults_hourly(self):
        from Strategy_Auto_Trader.markov_cli.run import (
            _build_arg_parser, _resolve_interval_defaults,
        )
        args = _build_arg_parser().parse_args(["--ticker", "SPY"])
        _resolve_interval_defaults(args)
        assert args.regime_smooth == 24
        assert args.min_hold_bars == 48

    def test_resolve_interval_defaults_daily(self):
        from Strategy_Auto_Trader.markov_cli.run import (
            _build_arg_parser, _resolve_interval_defaults,
        )
        args = _build_arg_parser().parse_args(["--ticker", "SPY", "--interval", "1d"])
        _resolve_interval_defaults(args)
        assert args.regime_smooth == 5
        assert args.min_hold_bars == 5

    def test_resolve_interval_defaults_explicit_values_respected(self):
        from Strategy_Auto_Trader.markov_cli.run import (
            _build_arg_parser, _resolve_interval_defaults,
        )
        args = _build_arg_parser().parse_args([
            "--ticker", "SPY", "--interval", "1d",
            "--regime-smooth", "10", "--min-hold-bars", "20",
        ])
        _resolve_interval_defaults(args)
        assert args.regime_smooth == 10
        assert args.min_hold_bars == 20

    def test_fetch_daily_called_for_1d_interval(self, tmp_path):
        import pandas as pd
        from Strategy_Auto_Trader.markov_cli import run as run_mod

        daily_df = pd.DataFrame({
            "Open": [100.0] * 300,
            "High": [101.0] * 300,
            "Low": [99.0] * 300,
            "Close": [100.0 + i * 0.01 for i in range(300)],
            "Volume": [1_000_000] * 300,
        }, index=pd.bdate_range("2000-01-01", periods=300))

        with mock.patch.object(run_mod, "fetch_daily", return_value=daily_df) as mock_fd, \
             mock.patch.object(run_mod, "fetch_hourly") as mock_fh, \
             mock.patch.object(run_mod, "consolidated_backtest",
                               return_value={"n_bars": 0, "detail": pd.DataFrame()}), \
             mock.patch.object(run_mod, "_make_run_dir", return_value=tmp_path), \
             mock.patch.object(run_mod, "_write_quality_gate"):
            run_mod.main(["--ticker", "SPY", "--interval", "1d"])

        mock_fd.assert_called_once()
        mock_fh.assert_not_called()
