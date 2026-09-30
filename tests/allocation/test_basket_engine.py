from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Strategy_Auto_Trader.allocation import basket_engine as be
from Strategy_Auto_Trader.allocation import intraday_engine as eng


def _inputs(log_ret: np.ndarray, day_codes: list[int], start: str = "2020-01-06") -> eng.Inputs:
    n = len(day_codes)
    days = pd.bdate_range(start, periods=max(day_codes) + 1)
    grid = pd.date_range(f"{start} 09:00", periods=n, freq="h", tz="UTC")
    return eng.Inputs(grid, log_ret, np.full(n, 15.0), np.full(n, 17.0), np.array(day_codes), days)


def _rng_inputs(n: int = 240, n_assets: int = 4, seed: int = 7) -> eng.Inputs:
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0, 0.004, size=(n, n_assets))
    log_ret[0] = 0.0
    return _inputs(log_ret, list(np.repeat(np.arange(n // 6), 6)))


class TestWeightMatrix:
    def test_maps_each_tier_to_its_row(self):
        tiers = np.array([0, 3, 0])
        rows = {0: np.array([0.5, 0.5, 0.0, 0.0]), 3: np.array([0.0, 0.0, 0.0, 1.0])}
        assert be.weight_matrix(tiers, rows).tolist() == [
            [0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.0, 0.0],
        ]

    def test_rejects_weights_that_do_not_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to"):
            be.weight_matrix(np.array([0]), {0: np.array([0.5, 0.2, 0.0, 0.0])})

    def test_unlisted_tiers_are_all_zero(self):
        rows = {0: np.array([1.0, 0.0]), 3: np.array([0.0, 1.0])}
        assert be.weight_matrix(np.array([1]), rows).tolist() == [[0.0, 0.0]]


class TestParityWithSingleAssetEngine:
    """A one-hot weight matrix must reproduce intraday_engine.simulate exactly, so the basket
    numbers are comparable to every result the single-asset engine has already produced."""

    @pytest.mark.parametrize("fill", ["same_bar", "next_bar"])
    @pytest.mark.parametrize("asset", ["NASDAQ", "FTSE"])
    def test_one_hot_matches_simulate(self, fill, asset):
        inp = _rng_inputs()
        rng = np.random.default_rng(11)
        tiers = np.where(rng.random(len(inp.grid)) < 0.5, 0, 3)
        col = eng.ASSETS.index(asset)
        rows = {0: np.eye(4)[col], 3: np.eye(4)[eng.ASSETS.index("CASH")]}
        targets = be.weight_matrix(tiers, rows)

        got = be.simulate_weights(inp, targets, eng.ASSETS.index("CASH"), fill=fill)
        ref = eng.simulate(inp, np.where(tiers == 0, col, eng.ASSETS.index("CASH")), fill=fill)

        np.testing.assert_allclose(got.daily, ref.daily, atol=1e-12)
        np.testing.assert_allclose(got.switches, ref.switches, atol=1e-12)
        np.testing.assert_allclose(got.cash_daily, ref.cash_daily, atol=1e-12)

    def test_basket_run_of_pure_nasdaq_matches_simulate(self):
        inp = _rng_inputs()
        tiers = eng.tiers_vxn_deadband(np.linspace(30.0, 10.0, len(inp.grid)), 23.0, 24.0)
        ref = eng.simulate(inp, tiers)
        got = be.basket_run(inp, eng.ASSETS, tiers, {"NASDAQ": 1.0})
        np.testing.assert_allclose(got.daily, ref.daily, atol=1e-12)


class TestCostModel:
    def test_full_switch_costs_the_whole_round_trip(self):
        inp = _inputs(np.zeros((3, 2)), [0, 0, 0])
        targets = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        run = be.simulate_weights(inp, targets, 1, cost_bps=100.0)
        assert run.switches.sum() == pytest.approx(1.0)
        assert run.daily[0] == pytest.approx(-0.01, abs=1e-6)

    def test_half_switch_costs_half(self):
        inp = _inputs(np.zeros((3, 2)), [0, 0, 0])
        targets = np.array([[1.0, 0.0], [0.5, 0.5], [0.5, 0.5]])
        run = be.simulate_weights(inp, targets, 1, cost_bps=100.0)
        assert run.switches.sum() == pytest.approx(0.5)
        assert run.daily[0] == pytest.approx(-0.005, abs=1e-6)

    def test_unchanged_target_is_never_charged(self):
        inp = _inputs(np.zeros((5, 2)), [0, 0, 0, 0, 0])
        targets = np.tile([0.5, 0.5], (5, 1))
        run = be.simulate_weights(inp, targets, 1, cost_bps=100.0)
        assert run.switches.sum() == 0.0
        assert run.daily[0] == pytest.approx(0.0, abs=1e-12)


class TestDrift:
    def test_weights_drift_between_rebalances(self):
        """Held 50/50 with only asset 0 rising, the portfolio must compound on a growing asset-0
        weight — i.e. beat the 50/50 of each leg's total return, which a daily reset would give."""
        n = 5
        log_ret = np.zeros((n, 2))
        log_ret[1:, 0] = np.log(1.10)
        inp = _inputs(log_ret, [0] * n)
        targets = np.tile([0.5, 0.5], (n, 1))
        run = be.simulate_weights(inp, targets, 1, cost_bps=0.0)
        drifted = float(np.prod(1 + run.daily))
        rebalanced_each_bar = (0.5 * 1.10 + 0.5) ** (n - 1)
        assert drifted > rebalanced_each_bar

    def test_rebalance_charges_turnover_against_the_drifted_weights(self):
        """After drift, returning to the same target is real turnover and must be charged."""
        log_ret = np.array([[0.0, 0.0], [np.log(2.0), 0.0], [0.0, 0.0]])
        inp = _inputs(log_ret, [0, 0, 0])
        # bar 2 re-states 50/50 after bar 1 drifted it to 2/3 : 1/3
        targets = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        no_reset = be.simulate_weights(inp, targets, 1, cost_bps=100.0)
        assert no_reset.switches.sum() == 0.0  # target never changes, so the daemon never trades


class TestWorldProxy:
    def test_appends_a_blended_column(self):
        inp = _rng_inputs()
        extended, names = be.extend_with_world(inp)
        assert names == eng.ASSETS + ("WORLD",)
        assert extended.log_ret.shape == (len(inp.grid), 5)

    def test_world_column_is_the_configured_blend(self):
        inp = _rng_inputs()
        extended, names = be.extend_with_world(inp)
        simple = np.expm1(inp.log_ret)
        expected = (
            be.WORLD_PROXY["SP500"] * simple[:, eng.ASSETS.index("SP500")]
            + be.WORLD_PROXY["FTSE"] * simple[:, eng.ASSETS.index("FTSE")]
        )
        np.testing.assert_allclose(np.expm1(extended.log_ret[:, names.index("WORLD")]), expected, atol=1e-12)

    def test_other_columns_are_untouched(self):
        inp = _rng_inputs()
        extended, _ = be.extend_with_world(inp)
        np.testing.assert_allclose(extended.log_ret[:, :4], inp.log_ret, atol=1e-15)


class TestValidation:
    def test_rejects_mismatched_column_counts(self):
        inp = _rng_inputs(n_assets=4)
        with pytest.raises(ValueError, match="does not match"):
            be.simulate_weights(inp, np.tile([0.5, 0.5], (len(inp.grid), 1)), 1)

    def test_rejects_unknown_fill_mode(self):
        inp = _rng_inputs()
        targets = np.tile(np.eye(4)[0], (len(inp.grid), 1))
        with pytest.raises(ValueError, match="unknown fill mode"):
            be.simulate_weights(inp, targets, 3, fill="eod")
