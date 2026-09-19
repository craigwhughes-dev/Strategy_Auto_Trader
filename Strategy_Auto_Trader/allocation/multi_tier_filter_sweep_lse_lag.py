"""Whipsaw-filter sweep on the live-faithful run B (16:30 London cutoff, next-day return), net of switch cost.

Re-tests the 2026-09-16 hysteresis/cadence rejection, which was made on the look-ahead daily model.
Filters live in tier_filters.py; run B tiers/returns/cost come from multi_tier_backtest_lse_lag.py.
Sub-period columns guard against picking a filter that only fits one regime.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from .multi_tier_backtest_lse_lag import (
    _COST_BPS,
    bars_by_end_time,
    load_inputs,
    net_of_switch_cost,
    strategy_returns,
    summarise,
    switch_flags,
    tier_series,
)
from .tier_filters import apply_asymmetric_hysteresis, apply_buy_cadence, apply_hysteresis

_MEASURED_BPS = 13.0
_SPLIT = pd.Timestamp("2019-01-01")


def filter_grid() -> dict:
    grid = {"raw (no filter)": lambda t: t}
    for n in (2, 3, 5, 10, 15, 20, 30, 60):
        grid[f"hysteresis {n}d"] = lambda t, n=n: apply_hysteresis(t, n)
    for n in (2, 3, 5, 10, 15, 20, 30, 60):
        grid[f"asym hysteresis {n}d (defensive immediate)"] = lambda t, n=n: apply_asymmetric_hysteresis(t, n)
    for n in (3, 5, 10, 20, 40, 60):
        grid[f"buy cadence {n}d"] = lambda t, n=n: apply_buy_cadence(t, n)
    return grid


def _row(returns: pd.Series, switches: pd.Series, bps: float) -> dict:
    net = net_of_switch_cost(returns, switches, bps)
    s = summarise(net, 100_000.0)
    return {"sharpe": s["sharpe"], "ret": s["total_return_pct"], "dd": s["max_drawdown_pct"]}


def evaluate(raw_tiers: pd.Series, tier_returns: pd.DataFrame, bps: float = _MEASURED_BPS) -> pd.DataFrame:
    rows = []
    for name, fn in filter_grid().items():
        tiers = fn(raw_tiers)
        rets = strategy_returns(tiers, tier_returns, lag=1)
        sw = switch_flags(tiers, rets.index)
        years = len(rets) / 252
        early, late = rets[rets.index < _SPLIT], rets[rets.index >= _SPLIT]
        rows.append(
            {
                "filter": name,
                "switches/yr": sw.sum() / years,
                **{f"gross_{k}": v for k, v in _row(rets, sw, 0.0).items()},
                **{f"net_{k}": v for k, v in _row(rets, sw, bps).items()},
                "net_sharpe_pre2019": _row(early, sw, bps)["sharpe"],
                "net_sharpe_2019+": _row(late, sw, bps)["sharpe"],
            }
        )
    return pd.DataFrame(rows)


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
    raw = tier_series(tier_returns.index, bars_by_end_time(vxn), bars_by_end_time(vix), tz, cutoff)
    df = evaluate(raw, tier_returns, args.cost_bps)
    print(f"\nRUN B FILTER SWEEP  {args.start_date} .. {args.end_date}  net cost {args.cost_bps:g} bps/switch")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n(_COST_BPS grid in lse_lag script: {_COST_BPS})")


if __name__ == "__main__":
    main()
