"""At the SAME drawdown, does a VXN gate on an existing holding beat just holding less of it?

The gate trades return for drawdown. So does simply holding x% of the fund and x-1% cash. The
only fair question for an owner whose objective is drawdown is: at matched max drawdown, which
delivers more return? Reports the equity/cash blend whose max DD matches each gate's, and the
CAGR gap between them. Ranked on TRAIN, matched and compared on the 2020+ holdout.

Run: uv run python scripts/tier_analysis/gate_vs_cash_at_matched_dd.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from Strategy_Auto_Trader.allocation import basket_engine as be
from Strategy_Auto_Trader.allocation import intraday_engine as eng

inp0 = eng.load_inputs()
inp, NAMES = be.extend_with_world(inp0)
ALWAYS_IN = np.zeros(len(inp.grid), dtype=int)
BLEND_WEIGHTS = np.round(np.arange(0.05, 1.001, 0.05), 2)

# Gates picked on TRAIN by gate_index_matrix.py; restated here so this script is self-contained.
GATES = {
    "NASDAQ": ("VXN", 22.0, 23.0),
    "SP500":  ("VXN", 24.0, 25.0),
    "WORLD":  ("VXN", 22.0, 24.0),
    "FTSE":   ("VXN", 22.0, 26.0),
}


def stats(run, window):
    s = eng.window_stats(run, window)
    s["cagr"] = ((1 + s["ret_pct"] / 100) ** (252 / s["n_days"]) - 1) * 100
    return s


def blend_curve(holding: str, window: str):
    """(weight, cagr, maxDD, xsharpe) for each equity/cash blend, held throughout."""
    out = []
    for w in BLEND_WEIGHTS:
        basket = {holding: float(w), "CASH": float(round(1 - w, 2))}
        s = stats(be.basket_run(inp, NAMES, ALWAYS_IN, basket), window)
        out.append((float(w), s["cagr"], s["max_dd_pct"], s["xsharpe"]))
    return out


def matched(curve, target_dd: float):
    """Blend whose max DD is closest to target_dd (both negative)."""
    return min(curve, key=lambda row: abs(row[2] - target_dd))


for window in ("test", "real"):
    print(f"\n{'=' * 104}")
    print(f"== Gate vs holding-less-of-it at MATCHED max drawdown  [{window}]")
    print(f"{'=' * 104}")
    print(f"  {'holding':8s} {'gate':11s} | {'gate CAGR':>9} {'gate DD':>8} {'gate xSh':>8} "
          f"| {'matched blend':>14} {'CAGR':>6} {'DD':>7} {'xSh':>6} | {'CAGR edge':>9}")
    for holding, (index_name, enter, exit_) in GATES.items():
        index = {"VIX": inp.vix, "VXN": inp.vxn}[index_name]
        g = stats(be.basket_run(inp, NAMES, eng.tiers_vxn_deadband(index, enter, exit_), {holding: 1.0}), window)
        curve = blend_curve(holding, window)
        w, cagr, dd, xsh = matched(curve, g["max_dd_pct"])
        edge = g["cagr"] - cagr
        verdict = "GATE WINS" if edge > 0 else "cash wins"
        print(f"  {holding:8s} {index_name} {enter:g}/{exit_:g}  | {g['cagr']:9.1f} {g['max_dd_pct']:8.1f} {g['xsharpe']:+8.2f} "
              f"| {f'{w:.0%} {holding}':>14} {cagr:6.1f} {dd:7.1f} {xsh:+6.2f} | {edge:+9.1f}  {verdict}")
    bh = {h: stats(be.basket_run(inp, NAMES, ALWAYS_IN, {h: 1.0}), window) for h in GATES}
    print(f"\n  buy-and-hold reference: " + "  ".join(
        f"{h} {bh[h]['cagr']:.1f}%/{bh[h]['max_dd_pct']:.1f}%" for h in GATES))
