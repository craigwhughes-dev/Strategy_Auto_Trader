from __future__ import annotations

import numpy as np
import pandas as pd

from Strategy_Auto_Trader.allocation.multi_tier_split_sweep_lse_lag import (
    INF,
    banded_tiers,
    ftse_ablation,
    split_grid,
    sweep,
)
from Strategy_Auto_Trader.allocation.multi_tier_threshold_sweep_lse_lag import tiers_for

DATES = pd.bdate_range("2018-01-02", periods=8)
READINGS = [(15.0, 12.0), (22.0, 12.0), (22.0, 19.0), (30.0, 12.0), (30.0, 19.0), (30.0, 40.0), (None, 12.0), (20.0, None)]


def test_infinite_band_matches_deployed_allocator():
    for vxn1, vix1, vix2 in ((18.0, 15.0, 17.5), (25.0, 20.0, 30.0)):
        deployed = tiers_for(DATES, READINGS, vxn1, vix1, vix2)
        banded = banded_tiers(DATES, READINGS, vxn1, INF, vix1, vix2)
        assert list(banded) == list(deployed)


def test_spy_gets_its_own_vxn_band():
    # vxn1=18, vxn2=25, vix1=15, vix2=20
    t = banded_tiers(DATES, READINGS, 18.0, 25.0, 15.0, 20.0)
    assert list(t) == [1, 2, 3, 3, 3, 4, 2, 4]
    # (22,12): in band + VIX<=15 -> SPY;  (22,19): band but VIX>15 -> FTSE;  (30,12): above band -> FTSE (VIX<=20)
    # VXN missing counts as inside the band (row 7 -> SPY); VIX missing -> cash (row 8)


def test_grid_is_ordered():
    assert all(a < b and c < d for a, b, c, d in split_grid())


def test_sweep_and_ablation_shapes():
    n = 300
    dates = pd.bdate_range("2017-06-01", periods=n)
    rng = np.random.default_rng(7)
    rets = pd.DataFrame(rng.normal(0.0002, 0.01, (n, 4)), index=dates, columns=[1, 2, 3, 4])
    readings = [(float(a), float(b)) for a, b in zip(rng.uniform(12, 35, n), rng.uniform(11, 30, n))]
    cfg = (18.0, 25.0, 20.0, 30.0)
    df = sweep(dates, readings, rets, [cfg], bps=13.0)
    assert len(df) == 2
    np.testing.assert_allclose(df[["t1", "t2", "t3", "t4"]].sum(axis=1), 1.0)
    ab = ftse_ablation(dates, readings, rets, [cfg], bps=13.0)
    assert list(ab["variant"]) == ["as is", "FTSE->SPY", "FTSE->cash", "FTSE->Nasdaq"]
