from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Strategy_Auto_Trader.allocation import intraday_engine as eng
from Strategy_Auto_Trader.allocation import multi_tier_intraday_grid as grid


def _inputs(vxn: list[float], vix: list[float] | None = None) -> eng.Inputs:
    n = len(vxn)
    days = pd.bdate_range("2020-01-06", periods=n)
    return eng.Inputs(
        pd.date_range("2020-01-06 09:00", periods=n, freq="D", tz="UTC"), np.zeros((n, 4)),
        np.array(vix if vix is not None else [15.0] * n), np.array(vxn), np.arange(n), days,
    )


class TestParamGrid:
    def test_grid_sizes(self):
        assert len(grid.param_grid("vix_only")) == 969  # C(19, 3)
        assert len(grid.param_grid("vxn_vix")) == 17 * 171  # 17 VXN cuts x C(19, 2) VIX pairs
        assert len(grid.param_grid("vxn_deadband")) == 17 * 11

    def test_ladders_are_strictly_ordered(self):
        assert all(a < b < c for a, b, c in grid.param_grid("vix_only"))
        assert all(b < c for _, b, c in grid.param_grid("vxn_vix"))

    def test_deadband_exit_is_never_below_enter(self):
        assert all(hi >= lo for lo, hi, _ in grid.param_grid("vxn_deadband"))


class TestTiersFor:
    def test_deadband_variant_uses_the_deadband_rule(self):
        inp = _inputs([20.0, 24.0, 27.0, 24.0])
        assert grid.tiers_for(inp, "vxn_deadband", (22.0, 26.0, 0.0)).tolist() == [0, 0, 3, 3]

    def test_vxn_vix_variant_uses_the_ladder(self):
        inp = _inputs([30.0], [14.0])
        assert grid.tiers_for(inp, "vxn_vix", (18.0, 15.0, 17.5, )).tolist() == [1]


class TestNeighbourMean:
    def test_averages_over_existing_neighbours_only(self):
        df = pd.DataFrame({"p1": [1.0, 2.0, 3.0], "p2": [1.0, 1.0, 1.0], "p3": [0.0, 0.0, 0.0],
                           "train_xsharpe": [0.0, 3.0, 0.0]})
        out = grid.add_neighbour_mean(df)
        assert out["train_xsharpe_nbr"].tolist() == pytest.approx([1.5, 1.0, 1.5])

    def test_isolated_spike_scores_lower_than_a_plateau(self):
        # cell 3 is a lone spike; cells 6-8 are a plateau of the same height
        p1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        x = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        df = pd.DataFrame({"p1": p1, "p2": 1.0, "p3": 0.0, "train_xsharpe": x})
        out = grid.add_neighbour_mean(df)["train_xsharpe_nbr"].tolist()
        assert out[6] > out[2]


def test_selection_diagnostics_reports_rank_agreement():
    df = pd.DataFrame({"train_xsharpe": np.arange(20.0), "test_xsharpe": np.arange(20.0) * 0.5})
    diag = grid.selection_diagnostics(df)
    assert diag["spearman_train_vs_test"] == pytest.approx(1.0)
    assert diag["cells"] == 20 and diag["share_test_positive"] == pytest.approx(0.95)


def test_band_width_table_groups_by_exit_minus_enter():
    rows = [{"p1": lo, "p2": lo + w, "p3": 0.0, "train_xsharpe": 1.0 - 0.1 * w, "test_xsharpe": 0.2 + 0.1 * w,
             "test_ret_pct": 10.0 * w, "test_max_dd_pct": -10.0, "test_sw_per_yr": 20.0 - 4 * w}
            for lo in (20.0, 22.0) for w in range(3)]
    lines = grid.band_width_table(pd.DataFrame(rows))
    assert len([ln for ln in lines if ln.strip()[:1].isdigit()]) == 3
    assert "20.0" in lines[-3] and "12.0" in lines[-1]
