from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Strategy_Auto_Trader.synthetic_backtest_data.correlated_bridge import (
    BridgeParams,
    accrue_deterministic,
    bridge_dataset,
    bridge_path,
    calibrate_params,
    estimate_params,
)
from Strategy_Auto_Trader.core.trading_sessions import LSE, US


class TestBridgePath:
    def test_last_bar_equals_next_close_exactly(self):
        z = np.random.default_rng(1).standard_normal(9)
        path = bridge_path(100.0, 103.5, 0.02, np.array([1.0] * 8 + [0.5]), z, 1.0)
        assert path[-1] == 103.5

    def test_zero_sigma_is_log_linear_in_elapsed_time(self):
        dur = np.array([0.5, 1.0, 1.0, 0.5])
        path = bridge_path(100.0, 110.0, 0.0, dur, np.zeros(4), 1.0)
        frac = np.cumsum(dur) / dur.sum()
        assert np.allclose(path, np.exp(np.log(100) + (np.log(110) - np.log(100)) * frac))

    def test_deterministic_for_seeded_draws(self):
        z = np.random.default_rng(3).standard_normal(6)
        a = bridge_path(100.0, 101.0, 0.01, np.ones(6), z, 0.8)
        b = bridge_path(100.0, 101.0, 0.01, np.ones(6), z, 0.8)
        assert np.array_equal(a, b)


def _daily(n: int, seed: int) -> pd.Series:
    days = pd.bdate_range("2020-01-06", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, n))), index=days)


@pytest.fixture(scope="module")
def two_series_bridged():
    daily = {"A": _daily(200, 1), "B": _daily(200, 2)}
    sessions = {"A": LSE, "B": US}
    params = BridgeParams(("A", "B"), np.array([[1.0, -0.8], [-0.8, 1.0]]), {"A": 1.0, "B": 1.0})
    since = daily["A"].index[1]
    until = {n: daily[n].index[-1] + pd.Timedelta(days=1) for n in daily}
    return daily, sessions, params, bridge_dataset(daily, sessions, params, np.random.default_rng(7), since, until)


class TestBridgeDataset:
    def test_each_days_last_bar_is_the_daily_close(self, two_series_bridged):
        daily, _, _, out = two_series_bridged
        local = out["A"].index.tz_convert(LSE.tz).tz_localize(None).normalize()
        last = out["A"]["Close"].groupby(local).last()
        assert np.allclose(last.to_numpy(), daily["A"].reindex(last.index).to_numpy())

    def test_bars_use_each_series_own_session_grid(self, two_series_bridged):
        out = two_series_bridged[3]
        assert len(out["A"]) % len(LSE.starts) == 0 and len(out["B"]) % len(US.starts) == 0

    def test_shared_bar_shocks_are_negatively_correlated_but_attenuated_by_pinning(self, two_series_bridged):
        out = two_series_bridged[3]
        r = {n: np.log(out[n]["Close"]).diff().groupby(out[n]["bar_end"]).last() for n in ("A", "B")}
        joined = pd.concat(r, axis=1, sort=True).dropna()
        assert len(joined) > 300
        assert -0.8 < joined.corr().iloc[0, 1] < -0.4

    def test_no_nonfinite_or_nonpositive_prices(self, two_series_bridged):
        for df in two_series_bridged[3].values():
            assert np.isfinite(df["Close"]).all() and (df["Close"] > 0).all()

    def test_until_bounds_are_exclusive(self):
        daily = {"A": _daily(30, 1)}
        params = BridgeParams(("A",), np.array([[1.0]]), {"A": 1.0})
        stop = daily["A"].index[20]
        out = bridge_dataset(daily, {"A": LSE}, params, np.random.default_rng(0), daily["A"].index[1], {"A": stop})
        assert out["A"].index.max().tz_convert(None).normalize() < stop


class TestEstimateParams:
    def test_recovers_correlation_sign_and_positive_scale_from_bridged_data(self, two_series_bridged):
        _, sessions, _, out = two_series_bridged
        real = {n: out[n]["Close"] for n in out}
        est = estimate_params(real, sessions, since="2020-01-01", min_pairs=50)
        assert -0.8 < est.corr[0, 1] < -0.4
        assert np.allclose(np.diag(est.corr), 1.0)
        assert all(0.5 < v < 1.3 for v in est.scale.values())

    def test_corr_matrix_is_positive_definite(self, two_series_bridged):
        _, sessions, _, out = two_series_bridged
        est = estimate_params({n: out[n]["Close"] for n in out}, sessions, since="2020-01-01", min_pairs=50)
        assert np.linalg.eigvalsh(est.corr).min() > 0


class TestAccrueDeterministic:
    def test_accrual_is_monotone_for_positive_carry_and_lands_on_daily_close(self):
        days = pd.bdate_range("2020-01-06", periods=10)
        daily = pd.Series(100 * 1.0002 ** np.arange(10), index=days)
        out = accrue_deterministic(daily, LSE, days[1], days[-1] + pd.Timedelta(days=1))
        assert out["Close"].is_monotonic_increasing
        last = out["Close"].groupby(out.index.tz_convert(LSE.tz).tz_localize(None).normalize()).last()
        assert np.allclose(last.to_numpy(), daily.iloc[1:].to_numpy())


class TestCalibrateParams:
    def test_calibrated_inputs_reproduce_the_target_correlation_and_scale(self):
        daily = {"A": _daily(200, 1), "B": _daily(200, 2)}
        sessions = {"A": LSE, "B": US}
        target = BridgeParams(("A", "B"), np.array([[1.0, -0.7], [-0.7, 1.0]]), {"A": 0.8, "B": 0.75})
        since = daily["A"].index[1]
        until = {n: daily[n].index[-1] + pd.Timedelta(days=1) for n in daily}

        cal = calibrate_params(target, daily, sessions, since, until, seed=11, iterations=3)
        out = bridge_dataset(daily, sessions, cal, np.random.default_rng(11), since, until)
        got = estimate_params({n: out[n]["Close"] for n in out}, sessions, since=str(since.date()), min_pairs=50)

        assert got.corr[0, 1] == pytest.approx(-0.7, abs=0.08)
        assert got.scale["A"] == pytest.approx(0.8, rel=0.08)
        assert got.scale["B"] == pytest.approx(0.75, rel=0.08)
        assert cal.corr[0, 1] < -0.7  # inputs must be pushed beyond the target to offset pinning attenuation
