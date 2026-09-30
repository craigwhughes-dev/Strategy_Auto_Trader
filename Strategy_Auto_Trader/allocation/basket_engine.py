"""Weighted-basket variant of the 4-tier rotation: a tier holds a blend, not one fund.

The deployed rule holds exactly one asset per tier (see intraday_engine.simulate, which indexes
a single column per bar). This module generalises that to a weight vector per bar, so tier 1 can
be e.g. 34% Nasdaq / 33% world / 33% FTSE instead of 100% Nasdaq.

Rebalancing follows what the daemon would actually do: weights DRIFT with returns while the target
is unchanged, and are reset to target only when the target changes (a tier switch). Charging a
rebalance every bar would invent turnover the daemon never generates.

Cost is turnover-based: a full one-asset-to-another switch moves sum|dw| = 2.0 and is charged
cost_bps (the same round-trip figure intraday_engine uses), so a one-hot weight matrix reproduces
intraday_engine.simulate exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import intraday_engine as eng
from .intraday_engine import Inputs, Run

# World-equity proxy weights over (NASDAQ, SP500, FTSE). FTSE stands in for non-US developed:
# the spliced dataset has no all-world fund, and FTSE 100 is the only non-US series in it.
# Deliberately crude — see the caveat in the report.
WORLD_PROXY = {"SP500": 0.65, "FTSE": 0.35}


def extend_with_world(inp: Inputs) -> tuple[Inputs, tuple[str, ...]]:
    """Append a synthetic WORLD column (daily-rebalanced SP500/FTSE blend) to the asset matrix."""
    simple = np.expm1(inp.log_ret)
    w = np.zeros(len(eng.ASSETS))
    for name, weight in WORLD_PROXY.items():
        w[eng.ASSETS.index(name)] = weight
    world = np.log1p(simple @ w)
    extended = np.column_stack([inp.log_ret, world])
    names = eng.ASSETS + ("WORLD",)
    return Inputs(inp.grid, extended, inp.vix, inp.vxn, inp.day_codes, inp.days), names


def weight_matrix(tiers: np.ndarray, tier_weights: dict[int, np.ndarray]) -> np.ndarray:
    """(n, n_assets) target weights per bar, from a tier index array and one weight row per tier."""
    n_assets = len(next(iter(tier_weights.values())))
    table = np.zeros((max(tier_weights) + 1, n_assets))
    for tier, row in tier_weights.items():
        if not np.isclose(row.sum(), 1.0):
            raise ValueError(f"tier {tier} weights sum to {row.sum()}, expected 1.0")
        table[tier] = row
    return table[tiers]


def simulate_weights(
    inp: Inputs,
    targets: np.ndarray,
    cash_col: int,
    fill: str = "same_bar",
    cost_bps: float = eng.DEFAULT_COST_BPS,
) -> Run:
    """Portfolio run for a per-bar target weight matrix.

    Args:
        inp: asset matrix whose log_ret has the same column count as `targets`
        targets: (n, n_assets) target weights, rows summing to 1
        cash_col: column holding the cash asset (the pre-first-signal state, and the
            excess-Sharpe benchmark)
        fill: "same_bar" acts on the signal read at that bar end; "next_bar" one bar later
        cost_bps: round-trip cost of a full switch, charged pro-rata on turnover
    """
    n, n_assets = targets.shape
    if inp.log_ret.shape != (n, n_assets):
        raise ValueError(f"log_ret {inp.log_ret.shape} does not match targets {targets.shape}")
    if fill == "next_bar":
        held = np.vstack([np.eye(n_assets)[cash_col], targets[:-1]])
    elif fill == "same_bar":
        held = targets
    else:
        raise ValueError(f"unknown fill mode {fill!r}")

    simple = np.expm1(inp.log_ret)
    changed = np.zeros(n, dtype=bool)
    changed[1:] = np.abs(np.diff(held, axis=0)).sum(axis=1) > 1e-12

    step = np.zeros(n)
    switched = np.zeros(n)
    actual = held[0].copy()
    for k in range(1, n):
        gross = actual * (1.0 + simple[k])
        total = gross.sum()
        step[k] = np.log(total) if total > 0 else 0.0
        actual = gross / total if total > 0 else held[k].copy()
        if changed[k]:
            turnover = np.abs(held[k] - actual).sum()
            step[k] += np.log1p(-turnover / 2.0 * cost_bps / 1e4)
            switched[k] = turnover / 2.0
            actual = held[k].copy()

    daily = eng._daily_from_steps(inp, step)
    switches = np.bincount(inp.day_codes, weights=switched, minlength=len(inp.days))
    cash_step = inp.log_ret[:, cash_col].copy()
    cash_step[0] = 0.0
    return Run(daily, switches, inp.days, eng._daily_from_steps(inp, cash_step))


def basket_run(
    inp: Inputs,
    names: tuple[str, ...],
    tiers: np.ndarray,
    basket: dict[str, float],
    cost_bps: float = eng.DEFAULT_COST_BPS,
    fill: str = "same_bar",
) -> Run:
    """Run the given tier path with tier 0 held as `basket` (name -> weight) and tier 3 as cash."""
    cash_col = names.index("CASH")
    row = np.zeros(len(names))
    for name, weight in basket.items():
        row[names.index(name)] = weight
    cash_row = np.zeros(len(names))
    cash_row[cash_col] = 1.0
    targets = weight_matrix(tiers, {0: row, 3: cash_row})
    return simulate_weights(inp, targets, cash_col, fill=fill, cost_bps=cost_bps)
