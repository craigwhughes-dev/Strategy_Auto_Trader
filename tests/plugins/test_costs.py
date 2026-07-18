"""Tests for plugins.costs — IBKR tiered commission model + flat parity."""

from __future__ import annotations

import pandas as pd
import pytest

from Strategy_Auto_Trader.plugins.costs import (
    COST_MODEL_CHOICES,
    USD_GBP,
    FlatCost,
    IbkrTieredCost,
    make_cost_model,
)


class TestFlatCost:

    def test_constant_regardless_of_value_and_side(self):
        m = FlatCost(10.0)
        assert m.cost(0.0, True) == 10.0
        assert m.cost(1_000_000.0, False) == 10.0


class TestIbkrTieredUk:

    def test_min_fee_binds_below_2000(self):
        m = IbkrTieredCost("SHEL.L")
        # 0.05% of 1500 = 0.75 < min 1.00
        assert m.cost(1500.0, False) == pytest.approx(1.00)

    def test_percentage_binds_above_2000(self):
        m = IbkrTieredCost("SHEL.L")
        # sell side: commission only
        assert m.cost(4000.0, False) == pytest.approx(2.00)

    def test_min_crossover_boundary(self):
        m = IbkrTieredCost("SHEL.L")
        # 0.05% x 2000 = exactly the 1.00 minimum
        assert m.cost(2000.0, False) == pytest.approx(1.00)

    def test_buy_adds_stamp_duty(self):
        m = IbkrTieredCost("SHEL.L")
        # 2000: commission 1.00 + SDRT 0.5% = 10.00
        assert m.cost(2000.0, True) == pytest.approx(11.00)

    def test_ptm_levy_above_10k(self):
        m = IbkrTieredCost("SHEL.L")
        # 12k buy: commission 6.00 + stamp 60.00 + PTM 1.00
        assert m.cost(12_000.0, True) == pytest.approx(67.00)
        # sell side: no stamp, no PTM
        assert m.cost(12_000.0, False) == pytest.approx(6.00)

    def test_spread_term(self):
        m = IbkrTieredCost("SHEL.L", include_spread=True)
        # sell 2000: commission 1.00 + 15bps spread 3.00
        assert m.cost(2000.0, False) == pytest.approx(4.00)


class TestIbkrTieredUs:

    def test_min_fee_binds_at_small_stake(self):
        m = IbkrTieredCost("AAPL")
        assert m.cost(1000.0, True) == pytest.approx(1.70 * USD_GBP)

    def test_no_stamp_duty_on_us_buys(self):
        m = IbkrTieredCost("AAPL")
        assert m.cost(4000.0, True) == m.cost(4000.0, False) == pytest.approx(2.00)

    def test_max_cap_binds_at_large_value(self):
        m = IbkrTieredCost("AAPL")
        # 0.05% x 100k = 50 > cap 39 USD
        assert m.cost(100_000.0, True) == pytest.approx(39.0 * USD_GBP)

    def test_spread_term(self):
        m = IbkrTieredCost("AAPL", include_spread=True)
        # 4000: commission 2.00 + 5bps spread 2.00
        assert m.cost(4000.0, False) == pytest.approx(4.00)


class TestFactory:

    def test_choices_all_construct(self):
        for name in COST_MODEL_CHOICES:
            assert make_cost_model(name, "AAPL", 10.0) is not None

    def test_flat_uses_trade_cost(self):
        m = make_cost_model("flat", "AAPL", 7.5)
        assert m.cost(5000.0, True) == 7.5

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            make_cost_model("nonsense", "AAPL", 10.0)

    def test_uk_routing_case_insensitive(self):
        m = make_cost_model("ibkr_tiered", "shel.l", 10.0)
        assert m.cost(2000.0, True) == pytest.approx(11.00)


class TestEngineParity:
    """Flat model through the engine seam must reproduce the historical
    flat-arithmetic path exactly."""

    @staticmethod
    def _detail():
        return pd.DataFrame({
            "trade_event": ["BUY", "", "SELL", "BUY", "SELL"],
            "strategy_return": [0.0, 0.02, -0.01, 0.0, 0.005],
            "kelly_fraction": [0.1, 0.1, 0.1, 0.2, 0.2],
        })

    def test_flat_model_matches_no_model_path(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _simulate_portfolio_value
        detail = self._detail()
        vals_none, costs_none = _simulate_portfolio_value(detail, 1000.0, 10.0)
        vals_flat, costs_flat = _simulate_portfolio_value(
            detail, 1000.0, 10.0, cost_model=FlatCost(10.0))
        assert vals_none == vals_flat
        assert costs_none == costs_flat == 40.0

    def test_ibkr_model_scales_with_stake(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _simulate_portfolio_value
        detail = self._detail()
        _, costs = _simulate_portfolio_value(
            detail, 1000.0, 10.0, cost_model=IbkrTieredCost("AAPL"))
        # 4 events, stakes ~100-200 -> min fee 1.70*FX each side
        assert costs == pytest.approx(4 * 1.70 * USD_GBP, rel=1e-6)

    def test_engine_stats_report_actual_costs(self):
        import numpy as np
        from Strategy_Auto_Trader.quant_hmm.quant_engine import (
            _build_quant_backtest_stats,
            _simulate_portfolio_value,
        )
        detail = self._detail()
        vals, costs = _simulate_portfolio_value(
            detail, 1000.0, 10.0, cost_model=IbkrTieredCost("SHEL.L"))
        ret = np.zeros(len(detail))
        eq = np.ones(len(detail))
        stats = _build_quant_backtest_stats(
            detail, ret, ret, eq, eq, 1000.0, vals, [], 0.1,
            transaction_costs_total=costs,
        )
        assert stats["transaction_costs_total"] == pytest.approx(round(costs, 2))
