"""Tier 1 as a weighted basket instead of 100% Nasdaq: does splitting reduce drawdown?

Run: uv run python scripts/tier_analysis/basket_tier1.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from Strategy_Auto_Trader.allocation import basket_engine as be
from Strategy_Auto_Trader.allocation import intraday_engine as eng

inp0 = eng.load_inputs()
inp, NAMES = be.extend_with_world(inp0)
TIERS = eng.tiers_vxn_deadband(inp.vxn, 23.0, 24.0)  # deployed rule: VXN enter 23, hold to 24

BASKETS = {
    "100 NAS (deployed)":      {"NASDAQ": 1.00},
    "50/50 NAS+WORLD":         {"NASDAQ": 0.50, "WORLD": 0.50},
    "34/33/33 NAS/WORLD/FTSE": {"NASDAQ": 0.34, "WORLD": 0.33, "FTSE": 0.33},
    "34/33/33 NAS/SP500/FTSE": {"NASDAQ": 0.34, "SP500": 0.33, "FTSE": 0.33},
    "50/25/25 NAS/WORLD/FTSE": {"NASDAQ": 0.50, "WORLD": 0.25, "FTSE": 0.25},
    "60/20/20 NAS/WORLD/FTSE": {"NASDAQ": 0.60, "WORLD": 0.20, "FTSE": 0.20},
    "80/10/10 NAS/WORLD/FTSE": {"NASDAQ": 0.80, "WORLD": 0.10, "FTSE": 0.10},
    "50/50 NAS+FTSE":          {"NASDAQ": 0.50, "FTSE": 0.50},
    "100 WORLD":               {"WORLD": 1.00},
    "100 SP500":               {"SP500": 1.00},
    "100 FTSE":                {"FTSE": 1.00},
    # Control: the incumbent diluted with cash, not with other equities. Any basket must beat
    # the NAS+CASH blend at matched volatility, or the diversification is adding nothing a
    # smaller Nasdaq position would not.
    "80/20 NAS+CASH":          {"NASDAQ": 0.80, "CASH": 0.20},
    "65/35 NAS+CASH":          {"NASDAQ": 0.65, "CASH": 0.35},
    "50/50 NAS+CASH":          {"NASDAQ": 0.50, "CASH": 0.50},
}
WINDOWS = ("all", "real", "train", "test")


def correlations() -> None:
    daily = np.column_stack([eng._daily_from_steps(inp, inp.log_ret[:, i]) for i in range(len(NAMES))])
    print("\n== Daily return correlation (whole sample)")
    equity = [i for i, n in enumerate(NAMES) if n != "CASH"]
    print("          " + "".join(f"{NAMES[j]:>8}" for j in equity))
    for i in equity:
        print(f"  {NAMES[i]:7s} " + "".join(f"{np.corrcoef(daily[:, i], daily[:, j])[0, 1]:8.2f}" for j in equity))
    worst = daily[:, equity[0]] < np.percentile(daily[:, equity[0]], 5)
    print()
    print(f"  same, restricted to the worst 5 pct of Nasdaq days (n={worst.sum()}):")
    print("          " + "".join(f"{NAMES[j]:>8}" for j in equity))
    for i in equity:
        print(f"  {NAMES[i]:7s} " + "".join(f"{np.corrcoef(daily[worst, i], daily[worst, j])[0, 1]:8.2f}" for j in equity))


def sweep() -> None:
    runs = {label: be.basket_run(inp, NAMES, TIERS, basket) for label, basket in BASKETS.items()}
    for window in WINDOWS:
        lo, hi = eng.WINDOWS[window]
        print(f"\n== Tier 1 basket, VXN 23/24 deadband, 13bps/switch  [{window}: {lo or 'start'}..{hi or 'end'}]")
        print(f"  {'basket':26s} {'xSharpe':>8} {'CAGR%':>7} {'maxDD%':>8} {'ret%':>9} {'sw/yr':>6}")
        for label, run in runs.items():
            s = eng.window_stats(run, window)
            yrs = s["n_days"] / 252
            cagr = ((1 + s["ret_pct"] / 100) ** (1 / yrs) - 1) * 100
            print(f"  {label:26s} {s['xsharpe']:+8.2f} {cagr:7.1f} {s['max_dd_pct']:8.1f} {s['ret_pct']:9.0f} {s['sw_per_yr']:6.1f}")


if __name__ == "__main__":
    correlations()
    sweep()
