from __future__ import annotations

import numpy as np
import pytest

from Strategy_Auto_Trader.synthetic_backtest_data import bridge


class TestGenerateBridgePath:
    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_last_point_lands_exactly_on_next_close(self, seed):
        rng = np.random.default_rng(seed)
        path = bridge.generate_bridge_path(
            prev_close=100.0, next_close=103.5, sigma=0.02, n_steps=7, rng=rng)
        assert path[-1] == 103.5

    def test_path_length_matches_n_steps(self):
        rng = np.random.default_rng(0)
        path = bridge.generate_bridge_path(100.0, 101.0, sigma=0.01, n_steps=7, rng=rng)
        assert len(path) == 7

    def test_zero_sigma_is_exact_log_space_interpolation(self):
        rng = np.random.default_rng(0)
        prev_close, next_close, n_steps = 100.0, 110.0, 5
        path = bridge.generate_bridge_path(prev_close, next_close, sigma=0.0, n_steps=n_steps, rng=rng)

        l0, l1 = np.log(prev_close), np.log(next_close)
        t = np.arange(1, n_steps + 1) / n_steps
        expected = np.exp(l0 + (l1 - l0) * t)
        assert np.allclose(path, expected)

    def test_deterministic_given_seeded_rng(self):
        path1 = bridge.generate_bridge_path(
            100.0, 101.0, sigma=0.02, n_steps=7, rng=np.random.default_rng(5))
        path2 = bridge.generate_bridge_path(
            100.0, 101.0, sigma=0.02, n_steps=7, rng=np.random.default_rng(5))
        assert np.array_equal(path1, path2)

    def test_single_step_returns_next_close_directly(self):
        rng = np.random.default_rng(0)
        path = bridge.generate_bridge_path(100.0, 105.0, sigma=0.05, n_steps=1, rng=rng)
        assert path[0] == 105.0


class TestBuildHourlyOhlcvForDay:
    def test_shape_and_columns(self):
        rng = np.random.default_rng(0)
        df = bridge.build_hourly_ohlcv_for_day(
            prev_close=100.0, next_close=102.0, sigma=0.01, n_bars=7, rng=rng)
        assert len(df) == 7
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_first_open_is_prev_close_last_close_is_next_close(self):
        rng = np.random.default_rng(0)
        df = bridge.build_hourly_ohlcv_for_day(
            prev_close=100.0, next_close=102.0, sigma=0.01, n_bars=7, rng=rng)
        assert df["Open"].iloc[0] == 100.0
        assert df["Close"].iloc[-1] == 102.0

    def test_high_low_bracket_open_and_close(self):
        rng = np.random.default_rng(0)
        df = bridge.build_hourly_ohlcv_for_day(
            prev_close=100.0, next_close=95.0, sigma=0.03, n_bars=7, rng=rng)
        assert (df["High"] >= df[["Open", "Close"]].max(axis=1)).all()
        assert (df["Low"] <= df[["Open", "Close"]].min(axis=1)).all()

    def test_volume_is_constant_placeholder(self):
        rng = np.random.default_rng(0)
        df = bridge.build_hourly_ohlcv_for_day(
            prev_close=100.0, next_close=102.0, sigma=0.01, n_bars=7, rng=rng)
        assert (df["Volume"] == bridge._PLACEHOLDER_VOLUME).all()
