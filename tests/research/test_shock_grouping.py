"""Tests for shock grouping — the count P4's stop-gate is measured on."""

from __future__ import annotations

import pandas as pd
import pytest

from Strategy_Auto_Trader.research import crash_phases as cp


def register(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame({"market": [r[0] for r in rows], "trough_date": pd.to_datetime([r[1] for r in rows], utc=True)})


def test_empty_register_yields_no_shocks():
    out = cp.group_into_shocks(register([]))
    assert out.empty
    assert cp.count_independent_shocks(register([])) == 0


def test_same_crash_in_different_markets_is_one_shock():
    """The panel's central objection: 2008 in Tokyo and 2008 in New York must not count as two."""
    reg = register([("^GSPC", "2009-03-09"), ("^N225", "2008-10-27"), ("^FTSE", "2009-03-03")])
    assert cp.count_independent_shocks(reg) == 1


def test_crashes_further_apart_than_the_window_are_separate_shocks():
    reg = register([("^GSPC", "2009-03-09"), ("^GSPC", "2020-03-23")])
    assert cp.count_independent_shocks(reg) == 2


def test_grouping_is_transitive_across_a_long_spread():
    """Troughs 100 days apart chain into one shock even though the ends are 300 days apart."""
    reg = register([("A", "2008-06-01"), ("B", "2008-09-09"), ("C", "2008-12-18"), ("D", "2009-03-28")])
    assert cp.count_independent_shocks(reg, window_days=182) == 1


def test_window_boundary_is_inclusive_at_exactly_the_window():
    reg = register([("A", "2000-01-01"), ("B", "2000-07-01")])  # 182 days apart
    assert cp.count_independent_shocks(reg, window_days=182) == 1
    assert cp.count_independent_shocks(reg, window_days=181) == 2


def test_shock_ids_are_ordered_by_trough_date_regardless_of_input_order():
    reg = register([("A", "2020-03-23"), ("B", "1987-10-19"), ("C", "2009-03-09")])
    out = cp.group_into_shocks(reg)
    assert list(out["trough_date"].dt.year) == [1987, 2009, 2020]
    assert list(out["shock_id"]) == [0, 1, 2]


def test_declared_window_default_is_the_pre_registered_six_months():
    assert cp.SHOCK_WINDOW_DAYS == 182


def test_count_matches_the_number_of_distinct_ids():
    reg = register([("A", "1998-08-31"), ("B", "1998-10-08"), ("C", "2008-10-27"), ("D", "2020-03-23")])
    out = cp.group_into_shocks(reg)
    assert out["shock_id"].nunique() == cp.count_independent_shocks(reg) == 3
