"""Per-switch bps helper and cost-sensitivity sweep for the tier strategy (Task B).

The flat 13 bps default in intraday_engine.DEFAULT_COST_BPS was derived from
IBKR's 0.05%/side commission and measured half-spreads (~3 bps ETF half-spread).
This module exposes `switch_cost_bps` and `partial_rebalance_cost_bps` so that
cost sensitivity analyses don't require manual arithmetic.

Key numbers at £20k pot (EQGB.L → CSH2.L, UCITS ETF):
  commission only (0.05%/side, min £1): 10.0 bps
  + ETF half-spread (~3 bps/side):      16.0 bps
  flat assumption (DEFAULT_COST_BPS):   13.0 bps   ← bracketed by the two

Run: uv run python -m Strategy_Auto_Trader.allocation.tier_cost_helper
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..plugins.costs import IbkrTieredCost
from .intraday_engine import (
    DEFAULT_COST_BPS,
    WINDOWS,
    buy_and_hold,
    load_inputs,
    simulate,
    tiers_vxn_deadband,
    window_stats,
)

POT_GBP: float = 20_000.0
_COST_LEVELS = (10.0, 13.0, 16.0, 26.0)
_WINDOWS_REF = [
    ("26yr (incl. bridged)", "y26"),
    ("since VXN data 2007-11-20", "vxn_era"),
    ("recent real 2024-03-25", "recent"),
]


def switch_cost_bps(
    pot_gbp: float = POT_GBP,
    sell_ticker: str = "EQGB.L",
    buy_ticker: str = "CSH2.L",
    include_spread: bool = False,
) -> float:
    """Cost in bps of NAV for one full UCITS ETF switch (sell pot_gbp + buy pot_gbp).

    Assumes the full pot is in the sold fund — valid for the tier strategy because
    the rule is binary (100% Nasdaq or 100% cash) and the pot is near-fully allocated.
    Both funds are UCITS ETFs: no stamp duty or PTM levy applies.
    """
    c_sell = IbkrTieredCost(sell_ticker, include_spread=include_spread).cost(pot_gbp, is_buy=False)
    c_buy = IbkrTieredCost(buy_ticker, include_spread=include_spread).cost(pot_gbp, is_buy=True)
    return (c_sell + c_buy) / pot_gbp * 1e4


def partial_rebalance_cost_bps(
    delta_w: float,
    pot_gbp: float = POT_GBP,
    sell_ticker: str = "EQGB.L",
    buy_ticker: str = "CSH2.L",
    include_spread: bool = False,
) -> float:
    """Cost in bps of total NAV when rebalancing by |delta_w| fraction of the pot.

    Used for continuous-weight strategies (vol-target) where each rebalance trades
    only a fraction of the total portfolio. The IBKR minimum (£1/side) dominates
    for small trades at £20k — e.g. |Δw|=5% trades £1k, commission = £1/side = 1 bps NAV.
    """
    if delta_w == 0.0:
        return 0.0
    traded = abs(delta_w) * pot_gbp
    c_sell = IbkrTieredCost(sell_ticker, include_spread=include_spread).cost(traded, is_buy=False)
    c_buy = IbkrTieredCost(buy_ticker, include_spread=include_spread).cost(traded, is_buy=True)
    return (c_sell + c_buy) / pot_gbp * 1e4


def _register_windows() -> None:
    WINDOWS.setdefault("y26", (None, None))
    WINDOWS.setdefault("vxn_era", ("2007-11-20", None))
    WINDOWS.setdefault("recent", ("2024-03-25", None))


def cost_sensitivity(
    inp=None,
    cost_levels: tuple[float, ...] = _COST_LEVELS,
) -> dict[str, pd.DataFrame]:
    """Cost-sensitivity table over the three reference windows.

    Returns a dict keyed by window label; each value is a DataFrame with columns:
    rule, cost_bps, n_days, ret_pct, cagr_pct, xsharpe, max_dd_pct, sw_per_yr.
    """
    _register_windows()
    if inp is None:
        inp = load_inputs()

    tiers_deployed = tiers_vxn_deadband(inp.vxn, 23.0, 24.0)
    tiers_plain = tiers_vxn_deadband(inp.vxn, 24.0, 24.0)
    bh = buy_and_hold(inp, "NASDAQ")

    result = {}
    for win_label, win_key in _WINDOWS_REF:
        bh_s = window_stats(bh, win_key)
        years = bh_s["n_days"] / 252
        cagr_bh = ((1 + bh_s["ret_pct"] / 100) ** (1 / years) - 1) * 100 if years > 0 else float("nan")
        rows = [{"rule": "B&H Nasdaq", "cost_bps": float("nan"), "cagr_pct": cagr_bh, **bh_s}]
        for c in cost_levels:
            for label, tiers in [
                ("two-state VXN 23/24", tiers_deployed),
                ("plain VXN<=24", tiers_plain),
            ]:
                run = simulate(inp, tiers, cost_bps=c)
                s = window_stats(run, win_key)
                cagr = ((1 + s["ret_pct"] / 100) ** (1 / years) - 1) * 100 if years > 0 else float("nan")
                rows.append({"rule": label, "cost_bps": c, "cagr_pct": cagr, **s})
        result[win_label] = pd.DataFrame(rows)
    return result


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Cost sensitivity sweep for the tier strategy")
    p.add_argument("--pot-gbp", type=float, default=POT_GBP)
    args = p.parse_args()

    print(f"\nPer-switch bps at £{args.pot_gbp:,.0f} pot (EQGB.L -> CSH2.L, UCITS ETF):")
    print(f"  Commission only (0.05%/side, min £1):  {switch_cost_bps(args.pot_gbp, include_spread=False):.1f} bps")
    print(f"  + ETF spread (~3 bps/side measured):   {switch_cost_bps(args.pot_gbp, include_spread=True):.1f} bps")
    print(f"  Flat assumption (DEFAULT_COST_BPS):     {DEFAULT_COST_BPS:.1f} bps")
    print()

    print(f"Partial rebalance cost at £{args.pot_gbp:,.0f} pot (commission-only, no spread):")
    for dw in (0.01, 0.02, 0.05, 0.10, 0.20, 1.00):
        c = partial_rebalance_cost_bps(dw, args.pot_gbp, include_spread=False)
        c_sp = partial_rebalance_cost_bps(dw, args.pot_gbp, include_spread=True)
        traded = abs(dw) * args.pot_gbp
        print(f"  |dw|={dw:.0%}  (£{traded:,.0f} traded): {c:.2f} bps NAV  (w/spread: {c_sp:.2f} bps)")

    print("\n... loading 26yr dataset ...")
    _register_windows()
    inp = load_inputs()
    tables = cost_sensitivity(inp)

    print("\nCOST SENSITIVITY (sweep: 10 / 13 / 16 / 26 bps per switch)")
    print("  xSharpe = excess-over-cash Sharpe  |  13 bps = DEFAULT_COST_BPS baseline")
    hdr = f"  {'Rule':<24} {'bps':>4} {'Ret%':>7} {'CAGR%':>6} {'xSh':>6} {'maxDD%':>7} {'sw/yr':>6}"
    sep = "  " + "-" * (len(hdr) - 2)

    for win_label, df in tables.items():
        print(f"\n--- {win_label} ---")
        print(hdr)
        print(sep)
        prev_rule = None
        for _, row in df.iterrows():
            rule = str(row["rule"])
            if rule != prev_rule and prev_rule is not None:
                print()
            prev_rule = rule
            bps_str = "-" if np.isnan(row["cost_bps"]) else f"{row['cost_bps']:.0f}"
            xsh = row["xsharpe"]
            xsh_str = f"{xsh:+.2f}" if not np.isnan(xsh) else "  nan"
            mark = " *" if not np.isnan(row["cost_bps"]) and abs(row["cost_bps"] - DEFAULT_COST_BPS) < 0.1 else "  "
            print(
                f"{mark}{rule:<24} {bps_str:>4} "
                f"{row['ret_pct']:>7.1f} {row['cagr_pct']:>6.1f} "
                f"{xsh_str:>6} {row['max_dd_pct']:>7.1f} "
                f"{row['sw_per_yr']:>6.1f}"
            )


if __name__ == "__main__":
    main()
