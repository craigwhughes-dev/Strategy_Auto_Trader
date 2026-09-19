"""VXN / VIX threshold sweep on run B (16:30 London cutoff, next-day return), net of switch cost.

Re-tests the tier thresholds (vxn_threshold, vix_tier1, vix_tier2) that were tuned on the look-ahead
daily model. tier2 is the VIX level above which the strategy goes to cash (CSH2.L), so pushing it up
tests "stay invested longer". Each threshold set is evaluated raw and with the asymmetric-hysteresis
filter. To limit selection bias the grid winner is picked on pre-2019 data only and scored on 2019+.
"""

from __future__ import annotations

import argparse
import itertools
import logging

import pandas as pd

from .multi_tier_allocator_4tier import MultiTierAllocator4Tier
from .multi_tier_backtest_lse_lag import (
    bars_by_end_time,
    load_inputs,
    net_of_switch_cost,
    strategy_returns,
    summarise,
    switch_flags,
    value_asof,
)
from .tier_filters import apply_asymmetric_hysteresis

_MEASURED_BPS = 13.0
_SPLIT = pd.Timestamp("2019-01-01")
_FILTER_DAYS = 10
DEFAULTS = (18.0, 15.0, 17.5)  # vxn_threshold, vix_tier1, vix_tier2 as deployed
NO_VXN = 0.0                   # VXN never <= 0 -> Nasdaq tier disabled
NO_CASH = 1e9                  # tier2 so high the cash tier is never chosen on VIX
ALWAYS_VXN = 1e9               # VXN always <= thr -> pure Nasdaq (= buy-and-hold EQGB)


def cutoff_readings(dates, vxn_end, vix_end, tz, cutoff) -> list[tuple[float | None, float | None]]:
    return [(value_asof(vxn_end, d, tz, cutoff), value_asof(vix_end, d, tz, cutoff)) for d in dates]


def tiers_for(dates, readings, vxn_thr: float, vix1: float, vix2: float) -> pd.Series:
    alloc = MultiTierAllocator4Tier(vxn_threshold=vxn_thr, vix_tier1=vix1, vix_tier2=vix2)
    return pd.Series([alloc.signal(d, vxn, vix).tier for d, (vxn, vix) in zip(dates, readings)], index=dates)


def _stats(returns: pd.Series, switches: pd.Series, bps: float) -> dict:
    s = summarise(net_of_switch_cost(returns, switches, bps), 100_000.0)
    return {"sharpe": s["sharpe"], "ret": s["total_return_pct"], "dd": s["max_drawdown_pct"]}


def evaluate_one(tiers: pd.Series, tier_returns: pd.DataFrame, bps: float) -> dict:
    rets = strategy_returns(tiers, tier_returns, lag=1)
    sw = switch_flags(tiers, rets.index)
    early, late = rets[rets.index < _SPLIT], rets[rets.index >= _SPLIT]
    full = _stats(rets, sw, bps)
    return {
        "sw/yr": sw.sum() / (len(rets) / 252),
        "net_sharpe": full["sharpe"],
        "net_ret": full["ret"],
        "net_dd": full["dd"],
        "sharpe_pre2019": _stats(early, sw, bps)["sharpe"],
        "sharpe_2019+": _stats(late, sw, bps)["sharpe"],
        "share_t1": float((tiers == 1).mean()),
        "share_t4": float((tiers == 4).mean()),
    }


def sweep(dates, readings, tier_returns, grid, bps: float) -> pd.DataFrame:
    rows = []
    for vxn_thr, vix1, vix2 in grid:
        raw = tiers_for(dates, readings, vxn_thr, vix1, vix2)
        for label, tiers in (("raw", raw), (f"asym{_FILTER_DAYS}d", apply_asymmetric_hysteresis(raw, _FILTER_DAYS))):
            rows.append({"vxn": vxn_thr, "vix1": vix1, "vix2": vix2, "filter": label, **evaluate_one(tiers, tier_returns, bps)})
    return pd.DataFrame(rows)


def default_grid() -> list[tuple[float, float, float]]:
    vxn = (NO_VXN, 16.0, 18.0, 20.0, 22.0, 25.0, 28.0, 30.0, 35.0, ALWAYS_VXN)
    vix1 = (12.0, 15.0, 17.5, 20.0)
    vix2 = (17.5, 20.0, 22.5, 25.0, 30.0, 35.0, NO_CASH)
    return [(a, b, c) for a, b, c in itertools.product(vxn, vix1, vix2) if b < c]


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda v: f"{v:.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", default="data_synthetic/hourly")
    p.add_argument("--hourly-cache", default="data/cache/ibkr_hourly")
    p.add_argument("--start-date", default="2007-11-20")
    p.add_argument("--end-date", default="2026-09-15")
    p.add_argument("--cost-bps", type=float, default=_MEASURED_BPS)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    tz, cutoff, vix, vxn, tier_returns = load_inputs(args.data_dir, args.hourly_cache, args.start_date, args.end_date)
    dates = tier_returns.index
    readings = cutoff_readings(dates, bars_by_end_time(vxn), bars_by_end_time(vix), tz, cutoff)
    grid = default_grid()
    df = sweep(dates, readings, tier_returns, grid, args.cost_bps)
    print(f"\nTHRESHOLD SWEEP run B, net {args.cost_bps:g} bps/switch, {len(grid)} threshold sets x 2 filters")

    base = df[(df.vxn == DEFAULTS[0]) & (df.vix1 == DEFAULTS[1]) & (df.vix2 == DEFAULTS[2])]
    print("\nDeployed thresholds (18 / 15 / 17.5):\n" + _fmt(base))

    for label in ("raw", f"asym{_FILTER_DAYS}d"):
        sub = df[df["filter"] == label]
        print(f"\nTop 12 by full-period net Sharpe, filter={label}:\n" + _fmt(sub.sort_values("net_sharpe", ascending=False).head(12)))
        pick = sub.sort_values("sharpe_pre2019", ascending=False).iloc[0]
        print(f"\nWalk-forward pick for filter={label} (best pre-2019 Sharpe):\n" + _fmt(pick.to_frame().T))

    for axis, others in (("vix2", ("vxn", "vix1")), ("vxn", ("vix1", "vix2")), ("vix1", ("vxn", "vix2"))):
        fixed = {"vxn": DEFAULTS[0], "vix1": DEFAULTS[1], "vix2": DEFAULTS[2]}
        mask = pd.Series(True, index=df.index)
        for o in others:
            mask &= df[o] == fixed[o]
        print(f"\nMarginal on {axis} (others at default), both filters:\n" + _fmt(df[mask].sort_values(["filter", axis])))


if __name__ == "__main__":
    main()
