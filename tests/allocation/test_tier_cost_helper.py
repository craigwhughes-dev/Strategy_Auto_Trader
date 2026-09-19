from __future__ import annotations

import math

import numpy as np
import pytest

from Strategy_Auto_Trader.allocation.tier_cost_helper import (
    POT_GBP,
    partial_rebalance_cost_bps,
    switch_cost_bps,
)


class TestSwitchCostBps:
    def test_commission_only_at_20k_is_10_bps(self):
        # £20k pot: sell £20k (max(£1, 0.05%×20k)=£10) + buy £20k (£10) = £20 / £20k = 10 bps
        bps = switch_cost_bps(20_000.0, include_spread=False)
        assert bps == pytest.approx(10.0)

    def test_with_spread_adds_3_bps_per_side(self):
        # 10 bps commission + 3 bps spread sell + 3 bps spread buy = 16 bps
        bps = switch_cost_bps(20_000.0, include_spread=True)
        assert bps == pytest.approx(16.0)

    def test_default_assumption_is_bracketed(self):
        lo = switch_cost_bps(POT_GBP, include_spread=False)
        hi = switch_cost_bps(POT_GBP, include_spread=True)
        from Strategy_Auto_Trader.allocation.intraday_engine import DEFAULT_COST_BPS
        assert lo <= DEFAULT_COST_BPS <= hi

    def test_very_small_pot_hits_minimum_floor(self):
        # At £100: 0.05%×100=£0.05, so min £1 applies each side; total £2/£100 = 200 bps
        bps = switch_cost_bps(100.0, include_spread=False)
        assert bps == pytest.approx(200.0)

    def test_large_pot_approaches_percentage(self):
        # At £1m: 0.05%×1m=£500 each side; no min applies; total £1000/£1m = 10 bps
        bps = switch_cost_bps(1_000_000.0, include_spread=False)
        assert bps == pytest.approx(10.0)

    def test_symmetry_sell_eqgb_buy_csh2_vs_reversed(self):
        a = switch_cost_bps(20_000.0, "EQGB.L", "CSH2.L", include_spread=False)
        b = switch_cost_bps(20_000.0, "CSH2.L", "EQGB.L", include_spread=False)
        assert a == pytest.approx(b)


class TestPartialRebalanceCostBps:
    def test_small_delta_w_hits_minimum_commission(self):
        # 1% of £20k = £200 traded; 0.05%×200=£0.10 < £1 min; so £1/side each, total £2/£20k = 1 bps
        bps = partial_rebalance_cost_bps(0.01, 20_000.0, include_spread=False)
        assert bps == pytest.approx(1.0)

    def test_full_rebalance_equals_switch_cost(self):
        # delta_w=1.0 means trading the entire pot — same as a full switch
        partial = partial_rebalance_cost_bps(1.0, 20_000.0, include_spread=False)
        full = switch_cost_bps(20_000.0, include_spread=False)
        assert partial == pytest.approx(full)

    def test_cost_in_bps_is_lower_for_smaller_delta(self):
        c1 = partial_rebalance_cost_bps(0.01, 20_000.0, include_spread=False)
        c5 = partial_rebalance_cost_bps(0.05, 20_000.0, include_spread=False)
        c50 = partial_rebalance_cost_bps(0.50, 20_000.0, include_spread=False)
        # As delta_w grows, the percentage commission kicks in (grows) but so does the denominator
        # — the important property is c1 <= c5 <= c50 in bps (min floor elevates small trades)
        assert c1 <= c5 <= c50

    def test_negative_delta_w_same_as_positive(self):
        pos = partial_rebalance_cost_bps(0.05, 20_000.0)
        neg = partial_rebalance_cost_bps(-0.05, 20_000.0)
        assert pos == pytest.approx(neg)

    def test_zero_delta_w_returns_zero(self):
        assert partial_rebalance_cost_bps(0.0, 20_000.0) == pytest.approx(0.0)
