"""Nasdaq / SPY / FTSE / cash split sweep on run B, with SPY given its own VXN band.

The deployed allocator picks Nasdaq if VXN <= vxn_threshold, else SPY only if VIX <= vix_tier1. Raising
vxn_threshold therefore takes low-vol days *from* SPY (Nasdaq wins first). Here the tiers are banded
instead, so every asset owns a regime and the days split between them:

  Nasdaq (1)  VXN <= vxn1
  SPY    (2)  vxn1 < VXN <= vxn2  and VIX <= vix1     (VXN missing counts as inside the band)
  FTSE   (3)  otherwise, VIX <= vix2                  (ISF.L)
  cash   (4)  otherwise                               (CSH2.L)

vxn2 = inf reproduces the deployed allocator exactly. Run B timing/cost as in multi_tier_backtest_lse_lag.
Configs are ranked on full-period net Sharpe among those where Nasdaq, SPY and FTSE each hold at least
--min-share of days; the split is scored again on pre-2019 / 2019+ halves.
"""

from __future__ import annotations

import argparse
import itertools
import logging

import pandas as pd

from .multi_tier_backtest_lse_lag import bars_by_end_time, load_inputs
from .multi_tier_threshold_sweep_lse_lag import DEFAULTS, cutoff_readings, evaluate_one
from .tier_filters import apply_asymmetric_hysteresis

_MEASURED_BPS = 13.0
_FILTER_DAYS = 10
INF = float("inf")


def banded_tiers(dates, readings, vxn1: float, vxn2: float, vix1: float, vix2: float) -> pd.Series:
    tiers = []
    for vxn, vix in readings:
        if vxn is not None and vxn <= vxn1:
            tiers.append(1)
        elif vix is None:
            tiers.append(4)
        elif vix <= vix1 and (vxn is None or vxn <= vxn2):
            tiers.append(2)
        elif vix <= vix2:
            tiers.append(3)
        else:
            tiers.append(4)
    return pd.Series(tiers, index=dates)


def split_grid() -> list[tuple[float, float, float, float]]:
    vxn1 = (16.0, 18.0, 20.0, 22.0, 25.0)
    vxn2 = (20.0, 22.0, 25.0, 28.0, 30.0, 35.0, INF)
    vix1 = (15.0, 17.5, 20.0, 25.0)
    vix2 = (17.5, 20.0, 25.0, 35.0)
    return [g for g in itertools.product(vxn1, vxn2, vix1, vix2) if g[0] < g[1] and g[2] < g[3]]


def sweep(dates, readings, tier_returns, grid, bps: float) -> pd.DataFrame:
    rows = []
    for vxn1, vxn2, vix1, vix2 in grid:
        raw = banded_tiers(dates, readings, vxn1, vxn2, vix1, vix2)
        for label, tiers in (("raw", raw), (f"asym{_FILTER_DAYS}d", apply_asymmetric_hysteresis(raw, _FILTER_DAYS))):
            shares = {f"t{t}": float((tiers == t).mean()) for t in (1, 2, 3, 4)}
            m = evaluate_one(tiers, tier_returns, bps)
            m.pop("share_t1"), m.pop("share_t4")
            rows.append({"vxn1": vxn1, "vxn2": vxn2, "vix1": vix1, "vix2": vix2, "filter": label, **shares, **m})
    return pd.DataFrame(rows)


def ftse_ablation(dates, readings, tier_returns, configs, bps: float) -> pd.DataFrame:
    """Does the FTSE tier earn its slot? Re-map its days to SPY / cash / Nasdaq and compare (asym filter)."""
    rows = []
    for cfg in configs:
        raw = banded_tiers(dates, readings, *cfg)
        for name, remap in (("as is", {}), ("FTSE->SPY", {3: 2}), ("FTSE->cash", {3: 4}), ("FTSE->Nasdaq", {3: 1})):
            m = evaluate_one(apply_asymmetric_hysteresis(raw.replace(remap), _FILTER_DAYS), tier_returns, bps)
            m.pop("share_t1"), m.pop("share_t4")
            rows.append({"vxn1": cfg[0], "vxn2": cfg[1], "vix1": cfg[2], "vix2": cfg[3], "variant": name, **m})
    return pd.DataFrame(rows)


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda v: f"{v:.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", default="data_synthetic/hourly")
    p.add_argument("--hourly-cache", default="data/cache/ibkr_hourly")
    p.add_argument("--start-date", default="2007-11-20")
    p.add_argument("--end-date", default="2026-09-15")
    p.add_argument("--cost-bps", type=float, default=_MEASURED_BPS)
    p.add_argument("--min-share", type=float, default=0.15, help="min share of days for each of Nasdaq/SPY/FTSE")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    tz, cutoff, vix, vxn, tier_returns = load_inputs(args.data_dir, args.hourly_cache, args.start_date, args.end_date)
    dates = tier_returns.index
    readings = cutoff_readings(dates, bars_by_end_time(vxn), bars_by_end_time(vix), tz, cutoff)
    grid = split_grid()
    df = sweep(dates, readings, tier_returns, grid, args.cost_bps)
    print(f"\nSPLIT SWEEP run B, net {args.cost_bps:g} bps/switch, {len(grid)} configs x 2 filters")

    d = DEFAULTS
    ref = df[(df.vxn1 == d[0]) & (df.vxn2 == INF) & (df.vix1 == d[1]) & (df.vix2 == d[2])]
    print("\nDeployed structure (vxn 18, no SPY band, 15 / 17.5):\n" + _fmt(ref))
    print("\nDeployed structure with VXN 25 (SPY squeezed out), 15 / 17.5:\n"
          + _fmt(df[(df.vxn1 == 25.0) & (df.vxn2 == INF) & (df.vix1 == 15.0) & (df.vix2 == 17.5)]))

    for label in ("raw", f"asym{_FILTER_DAYS}d"):
        sub = df[(df["filter"] == label) & (df[["t1", "t2", "t3"]].min(axis=1) >= args.min_share)]
        print(f"\nfilter={label}: {len(sub)} configs with Nasdaq/SPY/FTSE each >= {args.min_share:.0%} of days")
        print("Top 10 by net Sharpe:\n" + _fmt(sub.sort_values("net_sharpe", ascending=False).head(10)))
        print("Top 5 by (Sharpe of the weaker half):\n"
              + _fmt(sub.assign(worst=sub[["sharpe_pre2019", "sharpe_2019+"]].min(axis=1)).sort_values("worst", ascending=False).head(5)))
        print("Lowest-drawdown 5 with net Sharpe >= 0.6:\n"
              + _fmt(sub[sub.net_sharpe >= 0.6].sort_values("net_dd", ascending=False).head(5)))

    top = df[(df["filter"] == f"asym{_FILTER_DAYS}d") & (df[["t1", "t2", "t3"]].min(axis=1) >= args.min_share)]
    top = top.sort_values("net_sharpe", ascending=False).head(3)
    configs = list(zip(top.vxn1, top.vxn2, top.vix1, top.vix2))
    print("\nFTSE-tier ablation on the top 3 balanced asym configs:\n"
          + _fmt(ftse_ablation(dates, readings, tier_returns, configs, args.cost_bps)))


if __name__ == "__main__":
    main()
