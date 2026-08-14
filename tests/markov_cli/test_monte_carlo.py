"""Tests for markov_cli/monte_carlo.py (Track A)."""

from __future__ import annotations

import json
from unittest import mock

import numpy as np
import pandas as pd
import pytest


def make_fixture_ohlcv(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.01, n)))
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    spread = np.abs(rng.normal(0, 0.005, n))
    return pd.DataFrame(
        {"Open": closes, "High": closes * (1 + spread),
         "Low": closes * (1 - spread), "Close": closes,
         "Volume": rng.integers(500, 5000, n).astype(float)},
        index=idx,
    )


@pytest.fixture
def real_df():
    return make_fixture_ohlcv(600)


class TestMonteCarloCli:

    def test_exit_zero_and_output_files_created(self, real_df, tmp_path):
        from Strategy_Auto_Trader.markov_cli import monte_carlo

        with mock.patch.object(monte_carlo, "_MC_DIR", tmp_path), \
             mock.patch("Strategy_Auto_Trader.markov_cli.monte_carlo.fetch_hourly",
                        return_value=real_df), \
             mock.patch("Strategy_Auto_Trader.markov_cli.monte_carlo.volatility_profile",
                        return_value={"trend_quality": 0.5}):
            rc = monte_carlo.main([
                "--ticker", "SPY",
                "--strategy", "default",
                "--n-paths", "3",
                "--path-bars", "150",
                "--workers", "1",
            ])

        assert rc == 0
        out_dirs = list(tmp_path.glob("SPY_default_*"))
        assert len(out_dirs) == 1, f"expected 1 output dir, got: {out_dirs}"
        out_dir = out_dirs[0]
        assert (out_dir / "mc_summary.json").exists()
        assert (out_dir / "mc_paths.csv").exists()

    def test_summary_contains_expected_keys(self, real_df, tmp_path):
        from Strategy_Auto_Trader.markov_cli import monte_carlo

        with mock.patch.object(monte_carlo, "_MC_DIR", tmp_path), \
             mock.patch("Strategy_Auto_Trader.markov_cli.monte_carlo.fetch_hourly",
                        return_value=real_df), \
             mock.patch("Strategy_Auto_Trader.markov_cli.monte_carlo.volatility_profile",
                        return_value={"trend_quality": 0.5}):
            monte_carlo.main([
                "--ticker", "SPY", "--strategy", "default",
                "--n-paths", "2", "--path-bars", "150", "--workers", "1",
            ])

        out_dir = next(tmp_path.glob("SPY_default_*"))
        summary = json.loads((out_dir / "mc_summary.json").read_text())
        assert summary["ticker"] == "SPY"
        assert summary["strategy"] == "default"
        assert summary["n_paths"] == 2
        assert "prob_of_loss" in summary

    def test_regime_model_never_passed_to_consolidated_backtest(self, real_df, tmp_path):
        """regime_model must be None on every consolidated_backtest call.
        A non-None regime_model would corrupt the real ticker's on-disk HMM cache."""
        from Strategy_Auto_Trader.markov_cli import monte_carlo

        calls: list[dict] = []
        orig_bt = None

        def capturing_bt(df, **kwargs):
            calls.append({"regime_model": kwargs.get("regime_model", "NOT_PASSED")})
            return {
                "sharpe_strategy": 0.5, "sortino_strategy": 0.6,
                "max_drawdown_strategy": -0.1, "total_return_strategy": 0.05,
                "final_portfolio": 21_000.0, "n_buys": 2, "n_sells": 2,
                "n_bars": 150,
            }

        with mock.patch.object(monte_carlo, "_MC_DIR", tmp_path), \
             mock.patch("Strategy_Auto_Trader.markov_cli.monte_carlo.fetch_hourly",
                        return_value=real_df), \
             mock.patch("Strategy_Auto_Trader.markov_cli.monte_carlo.volatility_profile",
                        return_value={"trend_quality": 0.5}), \
             mock.patch("Strategy_Auto_Trader.markov_cli.monte_carlo._run_one_path",
                        wraps=lambda *a, **kw: {
                            "sharpe_strategy": 0.5, "sortino_strategy": 0.6,
                            "max_drawdown_strategy": -0.1, "total_return_strategy": 0.05,
                            "final_portfolio": 21_000.0, "n_buys": 2,
                        }) as m_run:
            monte_carlo.main([
                "--ticker", "SPY", "--strategy", "default",
                "--n-paths", "3", "--path-bars", "150", "--workers", "1",
            ])

        # Verify _run_one_path was called, not the real consolidated_backtest
        assert m_run.call_count == 3

    def test_regime_model_none_invariant_in_run_one_path(self, real_df):
        """Direct invariant test: _run_one_path must pass regime_model=None
        to consolidated_backtest on every call."""
        from Strategy_Auto_Trader.markov_cli.monte_carlo import _run_one_path

        received_kwargs: list[dict] = []

        def capturing_bt(df, **kwargs):
            received_kwargs.append(kwargs)
            return {
                "sharpe_strategy": 0.5, "sortino_strategy": 0.6,
                "max_drawdown_strategy": -0.1, "total_return_strategy": 0.05,
                "final_portfolio": 21_000.0, "n_buys": 2, "n_sells": 2, "n_bars": 100,
            }

        backtest_kwargs = dict(
            entry_prob=0.65, exit_prob=0.40, stop_loss_pct=0.05,
            take_profit_pct=0.15, volume_min_ratio=0.8, initial_cash=20_000.0,
            trade_cost=10.0, use_kelly=True, regime_smooth=24, min_hold_bars=48,
            buy_threshold=3.0, sell_threshold=-3.0, trailing_stop=0.0,
            vol_stop_mult=0.0, vol_stop_window=20, profit_stop_scale=0.0,
            min_stop_pct=0.05, max_hold_days=0, exit_on_rsi_reversal=False,
            exit_on_macd_cross=False, exit_on_consolidation=False, use_sar_stop=False,
            sar_af_start=0.02, sar_af_step=0.02, sar_af_max=0.20,
            skip_unused_indicators=True, min_train_bars=100, hmm_refit_bars=100,
            bars_per_year=1700,
        )

        with mock.patch(
            "Strategy_Auto_Trader.markov_cli.monte_carlo.consolidated_backtest",
            side_effect=capturing_bt,
        ):
            _run_one_path(
                real_df, "default", True, {}, {}, "flat", "SPY", backtest_kwargs
            )

        assert len(received_kwargs) == 1
        assert received_kwargs[0].get("regime_model") is None, (
            "regime_model must be None — a non-None value would corrupt the real "
            "ticker's on-disk HMM cache"
        )

    def test_missing_data_returns_nonzero(self, tmp_path):
        from Strategy_Auto_Trader.markov_cli import monte_carlo

        with mock.patch.object(monte_carlo, "_MC_DIR", tmp_path), \
             mock.patch("Strategy_Auto_Trader.markov_cli.monte_carlo.fetch_hourly",
                        return_value=None):
            rc = monte_carlo.main(["--ticker", "FAKE", "--n-paths", "1", "--path-bars", "100"])

        assert rc != 0
