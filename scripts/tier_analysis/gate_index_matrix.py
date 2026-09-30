"""Which gate index, on which holding, actually beats just holding it?

2x2+ matrix: gate index (VIX / VXN) x holding (WORLD / SP500 / FTSE / NASDAQ). Same protocol for
every cell so the deployed Nasdaq/VXN rule is judged by the same standard as the owner's proposed
VWRL/VIX rule: sweep enter x deadband width, RANK ON TRAIN ONLY (2007-11-20..2019-12-31), report
the picked gate on the 2020+ holdout against buy-and-hold of that same holding.

Benchmark is buy-and-hold of the holding, not cash and not the deployed strategy: the owner
already owns these funds, so holding them is the real alternative.

Run: uv run python scripts/tier_analysis/gate_index_matrix.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from Strategy_Auto_Trader.allocation import basket_engine as be
from Strategy_Auto_Trader.allocation import intraday_engine as eng

inp0 = eng.load_inputs()
inp, NAMES = be.extend_with_world(inp0)

HOLDINGS = ("NASDAQ", "SP500", "WORLD", "FTSE")
GATES = {"VIX": inp.vix, "VXN": inp.vxn}
ENTERS = tuple(float(x) for x in range(12, 34, 2))
WIDTHS = (0.0, 1.0, 2.0, 4.0)
ALWAYS_IN = np.zeros(len(inp.grid), dtype=int)


def stats(run, window):
    s = eng.window_stats(run, window)
    s["cagr"] = ((1 + s["ret_pct"] / 100) ** (252 / s["n_days"]) - 1) * 100
    return s


def best_on_train(holding: str, index: np.ndarray):
    best = None
    for width in WIDTHS:
        for enter in ENTERS:
            run = be.basket_run(inp, NAMES, eng.tiers_vxn_deadband(index, enter, enter + width), {holding: 1.0})
            train = stats(run, "train")
            if best is None or train["xsharpe"] > best[2]["xsharpe"]:
                best = ((enter, enter + width), run, train)
    (enter, exit_), run, train = best
    return f"{enter:g}/{exit_:g}", run, train


print(f"\n{'=' * 118}")
print("Gate index x holding. Threshold picked on TRAIN (2007-11..2019-12); test (2020+) is a holdout.")
print("xSharpe = annualised Sharpe of excess return over the cash asset. 13bps/switch.")
print(f"{'=' * 118}")
print(f"  {'holding':8s} {'gate':5s} {'picked':9s} "
      f"| {'TRAIN xSh':>9} {'vs B&H':>7} | {'TEST xSh':>8} {'vs B&H':>7} "
      f"| {'test CAGR':>9} {'B&H':>6} | {'test DD':>7} {'B&H':>7} | {'sw/yr':>5}")
for holding in HOLDINGS:
    bh_run = be.basket_run(inp, NAMES, ALWAYS_IN, {holding: 1.0})
    bh = {w: stats(bh_run, w) for w in ("train", "test")}
    print(f"  {holding:8s} {'--':5s} {'BUY+HOLD':9s} "
          f"| {bh['train']['xsharpe']:+9.2f} {'':7s} | {bh['test']['xsharpe']:+8.2f} {'':7s} "
          f"| {bh['test']['cagr']:9.1f} {'':6s} | {bh['test']['max_dd_pct']:7.1f} {'':7s} | {0.0:5.1f}")
    for gate_name, index in GATES.items():
        label, run, train = best_on_train(holding, index)
        test = stats(run, "test")
        flag = "  <-- beats B&H out of sample" if test["xsharpe"] > bh["test"]["xsharpe"] else ""
        print(f"  {'':8s} {gate_name:5s} {label:9s} "
              f"| {train['xsharpe']:+9.2f} {train['xsharpe'] - bh['train']['xsharpe']:+7.2f} "
              f"| {test['xsharpe']:+8.2f} {test['xsharpe'] - bh['test']['xsharpe']:+7.2f} "
              f"| {test['cagr']:9.1f} {bh['test']['cagr']:6.1f} "
              f"| {test['max_dd_pct']:7.1f} {bh['test']['max_dd_pct']:7.1f} | {test['sw_per_yr']:5.1f}{flag}")
    print()

print("Deployed rule for reference (VXN 23/24 on NASDAQ, not re-selected):")
dep = be.basket_run(inp, NAMES, eng.tiers_vxn_deadband(inp.vxn, 23.0, 24.0), {"NASDAQ": 1.0})
bh_nas = be.basket_run(inp, NAMES, ALWAYS_IN, {"NASDAQ": 1.0})
for w in ("train", "test", "real"):
    d, b = stats(dep, w), stats(bh_nas, w)
    print(f"  {w:6s} xSharpe {d['xsharpe']:+.2f} vs B&H {b['xsharpe']:+.2f} | "
          f"CAGR {d['cagr']:5.1f} vs {b['cagr']:5.1f} | maxDD {d['max_dd_pct']:6.1f} vs {b['max_dd_pct']:6.1f}")
