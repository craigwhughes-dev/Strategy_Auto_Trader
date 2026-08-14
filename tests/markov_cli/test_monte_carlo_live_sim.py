"""Tests for markov_cli/monte_carlo_live_sim.py (Track B)."""

from __future__ import annotations

import json
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from Strategy_Auto_Trader.output.journal import TradeRecord
from Strategy_Auto_Trader.quant_hmm.ticker_ranking import Candidate


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


def make_candidate(ticker: str, ts_base: pd.Timestamp) -> Candidate:
    rec = TradeRecord(
        date_opened=str(ts_base.date()),
        ticker=ticker, strategy="default",
        entry_score=1.0, kelly_fraction=0.1, return_pct=0.05,
        entry_price=100.0,
    )
    return Candidate(
        ticker=ticker,
        date_opened=ts_base,
        date_closed=ts_base + pd.Timedelta(days=5),
        entry_score=1.0, kelly_fraction=0.1, return_pct=0.05,
        record=rec,
    )


@pytest.fixture
def real_df():
    return make_fixture_ohlcv(600)


@pytest.fixture
def ts_base():
    return pd.Timestamp("2024-01-15", tz="UTC")


class TestMonteCarleLiveSimCli:

    def test_exit_zero_and_output_files_created(self, real_df, ts_base, tmp_path):
        from Strategy_Auto_Trader.markov_cli import monte_carlo_live_sim as mls

        cand = make_candidate("SPY", ts_base)
        close = pd.Series([100.0, 101.0], index=[ts_base, ts_base + pd.Timedelta(hours=1)])
        tq = pd.Series([0.5, 0.5], index=[ts_base, ts_base + pd.Timedelta(hours=1)])

        with mock.patch.object(mls, "_MC_DIR", tmp_path), \
             mock.patch("Strategy_Auto_Trader.markov_cli.monte_carlo_live_sim.generate_candidates",
                        return_value=([cand], {"SPY": close}, {"SPY": tq})), \
             mock.patch("Strategy_Auto_Trader.markov_cli.monte_carlo_live_sim.fetch_hourly_cached",
                        return_value=real_df):
            rc = mls.main([
                "--tickers", "SPY",
                "--strategies", "default",
                "--n-paths", "2",
                "--workers", "1",
                "--top-k", "0",
            ])

        assert rc == 0
        out_dirs = list(tmp_path.glob("default_portfolio_*"))
        assert len(out_dirs) == 1
        out_dir = out_dirs[0]
        assert (out_dir / "mc_summary.json").exists()
        assert (out_dir / "mc_paths.csv").exists()

    def test_regime_model_none_invariant_in_run_one_path(self, real_df, ts_base):
        """_run_one_path must pass regime_model=None to consolidated_backtest
        on every call — direct regression guard for cache-pollution invariant."""
        from Strategy_Auto_Trader.markov_cli.monte_carlo_live_sim import _run_one_path

        cand = make_candidate("SPY", ts_base)
        close = pd.Series([100.0], index=[ts_base])
        tq = pd.Series([0.5], index=[ts_base])

        received_kwargs: list[dict] = []

        def capturing_bt(df, **kwargs):
            received_kwargs.append(kwargs)
            return {
                "sharpe_strategy": 0.5, "sortino_strategy": 0.6,
                "max_drawdown_strategy": -0.1, "total_return_strategy": 0.05,
                "final_portfolio": 26_000.0, "n_buys": 1, "n_sells": 1, "n_bars": 100,
            }

        with mock.patch(
            "Strategy_Auto_Trader.quant_hmm.ticker_ranking.consolidated_backtest",
            side_effect=capturing_bt,
        ), mock.patch(
            "Strategy_Auto_Trader.quant_hmm.ticker_ranking.fetch_hourly_cached",
            return_value=real_df,
        ):
            _run_one_path(
                df_by_ticker={"SPY": real_df},
                fixed_tickers=["SPY"],
                strategy_name="default",
                pot_sizes=[25_000.0],
                start_date="2000-01-01",
                top_k=0,
                vol_weight=0.7,
                win_rate_weight=0.3,
                lookback_days=60,
                min_trend_quality=0.3,
                trade_cost=1.0,
                cost_model_name="flat",
                seasonal_volume=False,
            )

        for kw in received_kwargs:
            assert kw.get("regime_model") is None, (
                "regime_model must be None in every synthetic-path backtest call "
                "to avoid corrupting the real ticker's on-disk HMM cache"
            )

    def test_fixed_ticker_universe_reused_across_paths(self, real_df, ts_base, tmp_path):
        """The fixed_tickers list must be identical across all synthetic paths —
        the universe is selected once from real data and then reused."""
        from Strategy_Auto_Trader.markov_cli import monte_carlo_live_sim as mls

        cand = make_candidate("SPY", ts_base)
        close = pd.Series([100.0], index=[ts_base])
        tq = pd.Series([0.5], index=[ts_base])

        paths_tickers: list[list] = []
        original_run_one_path = mls._run_one_path

        def recording_path(df_by_ticker, fixed_tickers, *args, **kwargs):
            paths_tickers.append(list(fixed_tickers))
            return {25_000.0: {
                "total_return": 0.05, "final_portfolio": 26_250.0,
                "max_drawdown": -0.02, "n_admitted": 1,
            }}

        with mock.patch.object(mls, "_MC_DIR", tmp_path), \
             mock.patch.object(mls, "_run_one_path", side_effect=recording_path), \
             mock.patch("Strategy_Auto_Trader.markov_cli.monte_carlo_live_sim.generate_candidates",
                        return_value=([cand], {"SPY": close}, {"SPY": tq})), \
             mock.patch("Strategy_Auto_Trader.markov_cli.monte_carlo_live_sim.fetch_hourly_cached",
                        return_value=real_df):
            mls.main([
                "--tickers", "SPY",
                "--strategies", "default",
                "--n-paths", "3",
                "--workers", "1",
                "--top-k", "0",
            ])

        assert len(paths_tickers) == 3
        first = paths_tickers[0]
        assert all(t == first for t in paths_tickers), (
            "fixed_tickers must be identical across all synthetic paths — "
            "the universe is fixed once from real data"
        )
