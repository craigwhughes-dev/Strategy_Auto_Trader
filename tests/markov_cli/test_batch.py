from __future__ import annotations

import json
from unittest import mock

import pandas as pd
import pytest


class TestBatch:

    def test_build_argv_basic(self):
        from Strategy_Auto_Trader.markov_cli.batch import _build_argv
        cfg = {"ticker": "AAPL"}
        defaults = {"years": 5, "window": 20}
        argv = _build_argv(cfg, defaults)
        assert "--ticker" in argv
        assert "AAPL" in argv
        assert "--years" in argv
        assert "5" in argv
        assert "--window" in argv
        assert "20" in argv

    def test_build_argv_ticker_override(self):
        from Strategy_Auto_Trader.markov_cli.batch import _build_argv
        cfg = {"ticker": "GOOG", "years": 3}
        defaults = {"years": 5, "threshold": 0.02}
        argv = _build_argv(cfg, defaults)
        assert "GOOG" in argv
        # Per-ticker years should override default
        idx = argv.index("--years")
        assert argv[idx + 1] == "3"

    def test_build_argv_signal_reports_only(self):
        from Strategy_Auto_Trader.markov_cli.batch import _build_argv
        argv = _build_argv({"ticker": "AAPL"}, {"signal_reports_only": True})
        assert "--signal-reports-only" in argv

    def test_build_argv_signal_reports_only_absent_by_default(self):
        from Strategy_Auto_Trader.markov_cli.batch import _build_argv
        argv = _build_argv({"ticker": "AAPL"}, {})
        assert "--signal-reports-only" not in argv

    def test_build_argv_long_only_flag(self):
        from Strategy_Auto_Trader.markov_cli.batch import _build_argv
        cfg = {"ticker": "AAPL"}
        defaults = {"long_only": True}
        argv = _build_argv(cfg, defaults)
        assert "--long-only" in argv
        assert "--no-long-only" not in argv

    def test_build_argv_no_long_only(self):
        from Strategy_Auto_Trader.markov_cli.batch import _build_argv
        cfg = {"ticker": "AAPL", "long_only": False}
        defaults = {}
        argv = _build_argv(cfg, defaults)
        assert "--no-long-only" in argv
        assert "--long-only" not in argv

    def test_build_argv_sma200_flag(self):
        from Strategy_Auto_Trader.markov_cli.batch import _build_argv
        cfg = {"ticker": "AAPL", "sma200": False}
        defaults = {}
        argv = _build_argv(cfg, defaults)
        assert "--no-sma200" in argv

    def test_build_argv_no_hmm(self):
        from Strategy_Auto_Trader.markov_cli.batch import _build_argv
        cfg = {"ticker": "AAPL", "no_hmm": True}
        defaults = {}
        argv = _build_argv(cfg, defaults)
        assert "--no-hmm" in argv

    def test_build_argv_all_flags(self):
        from Strategy_Auto_Trader.markov_cli.batch import _build_argv
        cfg = {"ticker": "TEST"}
        defaults = {
            "years": 2,
            "window": 15,
            "threshold": 0.03,
            "position_mode": "state",
            "sell_threshold": -4,
            "in_sell_threshold": -2,
            "vol_stop_mult": 1.5,
            "vol_stop_window": 30,
            "profit_stop_scale": 0.3,
            "min_stop": 0.08,
            "trailing_stop": 0.25,
            "initial_cash": 50000,
            "transaction_cost": 15,
        }
        argv = _build_argv(cfg, defaults)
        assert "--position-mode" in argv
        assert "state" in argv
        assert "--vol-stop-mult" in argv
        assert "--profit-stop-scale" in argv
        assert "--initial-cash" in argv
        assert "--transaction-cost" in argv

    def test_collect_results_no_dir(self, tmp_path):
        from Strategy_Auto_Trader.markov_cli import batch
        with mock.patch.object(batch, "DATA_DIR", tmp_path):
            result = batch._collect_results("NONEXISTENT")
            assert result is None

    def test_collect_results_with_csv(self, tmp_path):
        from Strategy_Auto_Trader.markov_cli import batch
        # Create a mock data directory structure
        run_dir = tmp_path / "AAPL_20240101T120000Z"
        run_dir.mkdir()
        csv_path = run_dir / "compositeBacktest.csv"
        df = pd.DataFrame({
            "flag": ["BUY", "HOLD", "SELL"],
            "trade_event": ["BUY", "", "SELL"],
            "close": [150.0, 152.0, 148.0],
            "score": [3, 1, -3],
            "portfolio_value": [20000, 20100, 19800],
            "strategy_equity": [1.0, 1.005, 0.99],
            "bh_equity": [1.0, 1.013, 0.987],
        }, index=pd.Index(["2024-01-01", "2024-01-02", "2024-01-03"], name="date"))
        df.to_csv(csv_path)

        with mock.patch.object(batch, "DATA_DIR", tmp_path):
            result = batch._collect_results("AAPL")
            assert result is not None
            assert result["ticker"] == "AAPL"
            assert result["current_signal"] == "SELL"
            assert result["close"] == 148.0

    def test_should_send_buy_alert(self):
        from Strategy_Auto_Trader.markov_cli.batch import _should_send_buy_alert

        # quality_gate present: used as authoritative signal
        assert _should_send_buy_alert({"trade_event": "BUY", "quality_gate": "BUY"})
        assert not _should_send_buy_alert({"trade_event": "BUY", "quality_gate": "HOLD"})
        assert not _should_send_buy_alert({"trade_event": "SELL", "quality_gate": "BUY"})
        # quality_gate absent: falls back to current_signal
        assert _should_send_buy_alert({"trade_event": "BUY", "quality_gate": "", "current_signal": "BUY"})
        assert not _should_send_buy_alert({"trade_event": "BUY", "quality_gate": "", "current_signal": "HOLD"})

    def test_should_send_sell_alert(self):
        from Strategy_Auto_Trader.markov_cli.batch import _should_send_sell_alert

        with mock.patch("Strategy_Auto_Trader.output.trade_state.has_open_buy", return_value=True):
            assert _should_send_sell_alert({"trade_event": "SELL"}, "AAPL")

        with mock.patch("Strategy_Auto_Trader.output.trade_state.has_open_buy", return_value=False):
            assert not _should_send_sell_alert({"trade_event": "SELL"}, "AAPL")

        with mock.patch("Strategy_Auto_Trader.output.trade_state.has_open_buy", return_value=True):
            assert not _should_send_sell_alert({"trade_event": "BUY"}, "AAPL")

    def test_collect_results_with_quality_gate_file(self, tmp_path):
        from Strategy_Auto_Trader.markov_cli import batch

        run_dir = tmp_path / "AAPL_20240101T120000Z"
        run_dir.mkdir()
        csv_path = run_dir / "compositeBacktest.csv"
        df = pd.DataFrame({
            "flag": ["BUY", "HOLD", "SELL"],
            "trade_event": ["BUY", "", "SELL"],
            "close": [150.0, 152.0, 148.0],
            "score": [3, 1, -3],
            "portfolio_value": [20000, 20100, 19800],
            "strategy_equity": [1.0, 1.005, 0.99],
            "bh_equity": [1.0, 1.013, 0.987],
        }, index=pd.Index(["2024-01-01", "2024-01-02", "2024-01-03"], name="date"))
        df.to_csv(csv_path)
        (run_dir / "qualityGate.json").write_text(json.dumps({
            "flag": "HOLD",
            "reason": "quality_gate: weak buy context",
        }), encoding="utf-8")

        with mock.patch.object(batch, "DATA_DIR", tmp_path):
            result = batch._collect_results("AAPL")
            assert result is not None
            assert result["quality_gate"] == "HOLD"
            assert result["quality_gate_reason"] == "quality_gate: weak buy context"
            assert result["current_signal"] == "HOLD"

    def test_should_send_buy_alert_respects_quality_gate(self):
        from Strategy_Auto_Trader.markov_cli.batch import _should_send_buy_alert

        assert not _should_send_buy_alert({
            "trade_event": "BUY",
            "current_signal": "BUY",
            "quality_gate": "HOLD",
            "score": 3,
        })

    def test_fast_screen_tickers_passes_profitable(self):
        from Strategy_Auto_Trader.markov_cli import batch

        screen_result = {"profitable": True, "beats_bh": False, "ticker": "AAPL"}
        with mock.patch("Strategy_Auto_Trader.markov_cli.screen._screen_one", return_value=screen_result):
            passed, skipped = batch._fast_screen_tickers([{"ticker": "AAPL"}])
        assert len(passed) == 1
        assert passed[0]["ticker"] == "AAPL"
        assert skipped == []

    def test_fast_screen_tickers_passes_beats_bh(self):
        from Strategy_Auto_Trader.markov_cli import batch

        screen_result = {"profitable": False, "beats_bh": True, "ticker": "TSLA"}
        with mock.patch("Strategy_Auto_Trader.markov_cli.screen._screen_one", return_value=screen_result):
            passed, skipped = batch._fast_screen_tickers([{"ticker": "TSLA"}])
        assert len(passed) == 1
        assert skipped == []

    def test_fast_screen_tickers_skips_unprofitable(self):
        from Strategy_Auto_Trader.markov_cli import batch

        screen_result = {"profitable": False, "beats_bh": False, "ticker": "JUNK"}
        with mock.patch("Strategy_Auto_Trader.markov_cli.screen._screen_one", return_value=screen_result):
            passed, skipped = batch._fast_screen_tickers([{"ticker": "JUNK"}])
        assert passed == []
        assert "JUNK" in skipped

    def test_fast_screen_tickers_skips_none_result(self):
        from Strategy_Auto_Trader.markov_cli import batch

        with mock.patch("Strategy_Auto_Trader.markov_cli.screen._screen_one", return_value=None):
            passed, skipped = batch._fast_screen_tickers([{"ticker": "ERR"}])
        assert passed == []
        assert "ERR" in skipped

    def test_fast_screen_tickers_mixed(self):
        from Strategy_Auto_Trader.markov_cli import batch

        def fake_screen(ticker, years=2):
            return {
                "GOOD": {"profitable": True, "beats_bh": False},
                "BAD":  {"profitable": False, "beats_bh": False},
            }.get(ticker)

        configs = [{"ticker": "GOOD", "window": 20}, {"ticker": "BAD"}]
        with mock.patch("Strategy_Auto_Trader.markov_cli.screen._screen_one", side_effect=fake_screen):
            passed, skipped = batch._fast_screen_tickers(configs)
        assert len(passed) == 1
        assert passed[0]["ticker"] == "GOOD"
        assert passed[0]["window"] == 20  # config keys preserved
        assert "BAD" in skipped

    def test_fast_screen_preserves_all_config_keys(self):
        from Strategy_Auto_Trader.markov_cli import batch

        cfg = {"ticker": "AAPL", "window": 15, "threshold": 0.03, "no_hmm": True}
        screen_result = {"profitable": True, "beats_bh": True}
        with mock.patch("Strategy_Auto_Trader.markov_cli.screen._screen_one", return_value=screen_result):
            passed, _ = batch._fast_screen_tickers([cfg])
        assert passed[0] == cfg
