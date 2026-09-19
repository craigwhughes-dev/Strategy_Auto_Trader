from __future__ import annotations

import pandas as pd

from Strategy_Auto_Trader.allocation.tier_filters import (
    apply_asymmetric_hysteresis,
    apply_buy_cadence,
    apply_hysteresis,
)


def _s(vals):
    return pd.Series(vals, index=pd.date_range("2024-01-01", periods=len(vals)))


def test_hysteresis_one_day_is_identity():
    raw = _s([2, 3, 2, 4, 4, 1])
    assert list(apply_hysteresis(raw, 1)) == list(raw)


def test_hysteresis_needs_same_target_consecutively():
    # 3 requested, then 4, then 3 again: never 2 in a row of the same target -> stay on 2
    raw = _s([2, 3, 4, 3, 3, 3])
    assert list(apply_hysteresis(raw, 2)) == [2, 2, 2, 2, 3, 3]


def test_hysteresis_interrupted_streak_resets():
    raw = _s([2, 3, 2, 3, 3])
    assert list(apply_hysteresis(raw, 2)) == [2, 2, 2, 2, 3]


def test_asymmetric_defensive_immediate_offensive_delayed():
    raw = _s([1, 4, 1, 1, 1, 1])
    assert list(apply_asymmetric_hysteresis(raw, 3)) == [1, 4, 4, 4, 1, 1]


def test_buy_cadence_blocks_offensive_until_days_since_switch():
    raw = _s([2, 4, 2, 2, 2, 2, 2])
    # defensive at i=1; offensive allowed once >=3 days since that switch (i=4)
    assert list(apply_buy_cadence(raw, 3)) == [2, 4, 4, 4, 2, 2, 2]


def test_buy_cadence_does_not_block_first_offensive_move():
    raw = _s([4, 1, 1])
    assert list(apply_buy_cadence(raw, 5)) == [4, 1, 1]


def test_only_raw_tiers_and_index_preserved():
    raw = _s([1, 2, 3, 4, 3, 2, 1, 1, 2])
    for f in (apply_hysteresis(raw, 3), apply_asymmetric_hysteresis(raw, 3), apply_buy_cadence(raw, 3)):
        assert set(f) <= set(raw)
        assert f.index.equals(raw.index)


def test_empty_series():
    assert apply_hysteresis(pd.Series([], dtype=int), 3).empty
