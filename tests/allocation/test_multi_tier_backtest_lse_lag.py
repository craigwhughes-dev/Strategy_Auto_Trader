from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from Strategy_Auto_Trader.allocation.multi_tier_allocator_4tier import MultiTierAllocator4Tier
from Strategy_Auto_Trader.allocation.multi_tier_backtest_lse_lag import (
    bars_by_end_time,
    compare,
    load_lse_cutoff,
    net_of_switch_cost,
    strategy_returns,
    switch_flags,
    tier_series,
    value_asof,
)

TZ = "Europe/London"
CUTOFF = time(16, 30)


def _bars(rows: list[tuple[str, float]]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([r[0] for r in rows], tz="UTC")
    return pd.DataFrame({"Close": [r[1] for r in rows]}, index=idx)


def test_cutoff_read_from_ftse_config():
    tz, cutoff = load_lse_cutoff()
    assert tz == "Europe/London"
    assert cutoff == time(16, 30)


def test_bar_only_counts_once_ended_gmt():
    # GMT (winter): London == UTC. 15:00 bar ends 16:00 (<= 16:30 cutoff), 16:00 bar ends 17:00 (> cutoff).
    s = bars_by_end_time(_bars([("2024-01-10 15:00", 20.0), ("2024-01-10 16:00", 30.0), ("2024-01-10 17:00", 30.0)]))
    assert value_asof(s, pd.Timestamp("2024-01-10"), TZ, CUTOFF) == 20.0
    assert value_asof(s, pd.Timestamp("2024-01-10"), TZ, None) == 30.0


def test_cutoff_shifts_with_bst():
    # BST (summer): 16:30 London == 15:30 UTC. 14:00 UTC bar ends 15:00 UTC (in), 15:00 UTC bar ends 16:00 UTC (out).
    s = bars_by_end_time(_bars([("2024-07-10 14:00", 20.0), ("2024-07-10 15:00", 30.0), ("2024-07-10 16:00", 30.0)]))
    assert value_asof(s, pd.Timestamp("2024-07-10"), TZ, CUTOFF) == 20.0


def test_stale_value_carried_when_no_bar_yet_today():
    s = bars_by_end_time(_bars([("2024-01-09 19:00", 22.0), ("2024-01-09 20:00", 22.0), ("2024-01-10 14:30", 40.0), ("2024-01-10 15:00", 40.0)]))
    # LSE morning cutoff-style lookup before any bar today sees yesterday's close
    assert value_asof(s, pd.Timestamp("2024-01-10"), TZ, time(9, 0)) == 22.0


def test_weekend_day_uses_friday_value():
    s = bars_by_end_time(_bars([("2024-01-12 19:00", 25.0), ("2024-01-12 20:00", 25.0)]))
    assert value_asof(s, pd.Timestamp("2024-01-13"), TZ, CUTOFF) == 25.0


def test_tier_series_uses_allocator_signal():
    dates = pd.DatetimeIndex(["2024-01-10"])
    vxn = bars_by_end_time(_bars([("2024-01-10 14:30", 30.0), ("2024-01-10 15:00", 30.0)]))
    vix = bars_by_end_time(_bars([("2024-01-10 14:30", 12.0), ("2024-01-10 15:00", 12.0)]))
    assert tier_series(dates, vxn, vix, TZ, None).iloc[0] == 2  # VXN high, VIX<=15 -> SPY


def test_strategy_returns_lag_no_lookahead():
    dates = pd.date_range("2024-01-01", periods=4)
    rets = pd.DataFrame({1: [0.0] * 4, 2: [0.1, 0.2, 0.3, 0.4], 3: [0.0] * 4, 4: [0.0] * 4}, index=dates)
    tiers = pd.Series([2, 4, 4, 4], index=dates)
    same_day = strategy_returns(tiers, rets, lag=0)
    assert same_day.iloc[0] == 0.1  # day-0 tier earns day-0 return
    lagged = strategy_returns(tiers, rets, lag=1)
    assert list(lagged.index) == list(dates[1:])
    assert lagged.iloc[0] == 0.2  # day-0 tier (SPY) earns day-1 return
    assert lagged.iloc[1] == 0.0  # day-1 tier (CSH2) earns day-2 return


def test_lookahead_run_matches_existing_allocator_backtest():
    dates = pd.bdate_range("2024-01-02", periods=30)
    rng = np.random.default_rng(0)
    px = pd.DataFrame(
        {t: 100 * np.cumprod(1 + rng.normal(0, 0.01, len(dates))) for t in (1, 2, 3, 4)}, index=dates
    )
    rets = px.pct_change().dropna()
    vix_vals = np.linspace(12, 22, len(dates))
    vxn_vals = np.linspace(25, 15, len(dates))

    def hourly(vals):
        rows = []
        for d, v in zip(dates, vals):
            rows += [(f"{d.date()} 15:00", v), (f"{d.date()} 19:00", v)]
        return _bars(rows)

    result = compare(rets, hourly(vxn_vals), hourly(vix_vals), TZ, CUTOFF)

    daily = lambda s: pd.DataFrame({"Close": s}, index=dates)  # noqa: E731
    ref = MultiTierAllocator4Tier().backtest(
        nasdaq_df=daily(px[1]), spy_df=daily(px[2]), isfl_df=daily(px[3]), csh2_df=daily(px[4]),
        vxn_df=daily(pd.Series(vxn_vals, index=dates)), vix_df=daily(pd.Series(vix_vals, index=dates)),
    )
    mine = strategy_returns(result["close_tiers"], rets, lag=0) * 100
    theirs = ref["daily_nav"].set_index("date")["daily_return_pct"].loc[mine.index]
    np.testing.assert_allclose(mine.values, theirs.values, atol=1e-9)


def test_switch_flags_count_changes_in_held_tier_only():
    dates = pd.date_range("2024-01-01", periods=6)
    tiers = pd.Series([2, 2, 4, 4, 2, 2], index=dates)
    flags = switch_flags(tiers, dates)
    # held (lag 1) = [nan,2,2,4,4,2]; changes on day 3 (2->4) and day 5 (4->2); day 1 has no prior
    assert list(flags) == [False, False, False, True, False, True]


def test_net_of_switch_cost_only_hits_switch_days():
    dates = pd.date_range("2024-01-01", periods=3)
    r = pd.Series([0.01, 0.01, 0.01], index=dates)
    sw = pd.Series([False, True, False], index=dates)
    net = net_of_switch_cost(r, sw, round_trip_bps=100)
    np.testing.assert_allclose(net.iloc[[0, 2]], 0.01)
    np.testing.assert_allclose(net.iloc[1], 1.01 * 0.99 - 1)


def test_compare_reports_benchmarks_and_costs_monotonic():
    dates = pd.bdate_range("2024-01-02", periods=40)
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0.0003, 0.01, (len(dates), 4)), index=dates, columns=[1, 2, 3, 4])
    vix_vals = np.tile([12.0, 22.0], len(dates) // 2)
    vxn_vals = np.tile([25.0, 15.0], len(dates) // 2)

    def hourly(vals):
        rows = []
        for d, v in zip(dates, vals):
            rows += [(f"{d.date()} 15:00", v), (f"{d.date()} 19:00", v)]
        return _bars(rows)

    res = compare(rets, hourly(vxn_vals), hourly(vix_vals), TZ, CUTOFF)
    common = res["summaries"]["B_live"]["n_days"]
    assert {"B&H SPY", "B&H CSH2.L"} <= set(res["benchmarks"])
    assert res["benchmarks"]["B&H SPY"]["n_days"] == common
    assert res["n_switches_b"] > 0
    rets_by_cost = [s["total_return_pct"] for s in res["net_summaries"].values()]
    assert rets_by_cost == sorted(rets_by_cost, reverse=True)
    assert rets_by_cost[0] == res["summaries"]["B_live"]["total_return_pct"]
