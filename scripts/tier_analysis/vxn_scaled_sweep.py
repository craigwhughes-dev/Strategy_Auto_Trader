"""Experiment B: VXN-scaled position size sweep.

Sweeps enter_at in [22, 23, 24] and full_weight_at in [12, 14, 15, 17, 20].
Reports xSharpe, maxDD%, avg_weight, sw/yr, ret% across three windows and
compares against the binary current rule.

Run: uv run python scripts/tier_analysis/vxn_scaled_sweep.py
"""
from __future__ import annotations

import numpy as np

from Strategy_Auto_Trader.allocation import intraday_comparators as cmp
from Strategy_Auto_Trader.allocation import intraday_engine as eng

inp = eng.load_inputs()
eng.WINDOWS.setdefault("y26", (None, None))
eng.WINDOWS.setdefault("vxn_era", ("2007-11-20", None))
eng.WINDOWS.setdefault("recent", ("2024-03-25", None))
eng.WINDOWS.setdefault("train", ("2007-11-20", "2019-12-31"))
eng.WINDOWS.setdefault("test", ("2020-01-01", None))

COST_BPS = 13.0
ENTER_AT_RANGE = [22.0, 23.0, 24.0]
FULL_WEIGHT_AT_RANGE = [12.0, 14.0, 15.0, 17.0, 20.0]

WINDOWS_SPEC = [
    ("26yr",       "y26"),
    ("since-2007", "vxn_era"),
    ("train",      "train"),
    ("test",       "test"),
    ("recent",     "recent"),
]

# Reference: binary current rule
tiers_current = eng.tiers_vxn_deadband(inp.vxn, 23.0, 24.0)
run_current = eng.simulate(inp, tiers_current, cost_bps=COST_BPS)

rows = []
for enter_at in ENTER_AT_RANGE:
    for full_weight_at in FULL_WEIGHT_AT_RANGE:
        run = cmp.vxn_scaled_blend(inp, enter_at=enter_at, full_weight_at=full_weight_at)
        row = {"enter_at": enter_at, "full_weight_at": full_weight_at}
        for wname, wkey in WINDOWS_SPEC:
            s = eng.window_stats(run, wkey)
            row[f"{wname}_xsh"] = s["xsharpe"]
            row[f"{wname}_maxdd"] = s["max_dd_pct"]
            row[f"{wname}_sw"] = s["sw_per_yr"]
            row[f"{wname}_ret"] = s["ret_pct"]
        rows.append(row)

# Reference stats
ref = {}
for wname, wkey in WINDOWS_SPEC:
    ref[wname] = eng.window_stats(run_current, wkey)

hdr = f"  {'enter':>5} {'fw_at':>5}  {'xSh':>7}  {'maxDD%':>7}  {'sw/yr':>6}  {'ret%':>8}"
sep = "  " + "-" * (len(hdr) - 2)

for wname, _ in WINDOWS_SPEC:
    print(f"\n=== {wname} ===")
    ref_s = ref[wname]
    print(f"  REFERENCE (binary 23/24 13bps): xSh={ref_s['xsharpe']:+.3f}  maxDD={ref_s['max_dd_pct']:.1f}%  sw/yr={ref_s['sw_per_yr']:.1f}  ret={ref_s['ret_pct']:.1f}%")
    print(hdr)
    print(sep)
    for r in rows:
        xsh = r[f"{wname}_xsh"]
        dd = r[f"{wname}_maxdd"]
        sw = r[f"{wname}_sw"]
        ret = r[f"{wname}_ret"]
        xsh_str = f"{xsh:+.3f}" if xsh == xsh else "    nan"
        # Mark rows where xSh >= reference
        ref_xsh = ref_s["xsharpe"]
        marker = " >" if xsh >= ref_xsh else "  "
        print(f"{marker}{r['enter_at']:>5.0f} {r['full_weight_at']:>5.0f}  {xsh_str:>7}  {dd:>7.1f}  {sw:>6.1f}  {ret:>8.1f}")

# Summary: rows beating reference on both test and 26yr xSh
print("\n=== Beats binary 23/24 on BOTH 26yr xSh AND test xSh ===")
ref_y26 = ref["26yr"]["xsharpe"]
ref_test = ref["test"]["xsharpe"]
winners = [r for r in rows if r["26yr_xsh"] >= ref_y26 and r["test_xsh"] >= ref_test]
if winners:
    print(hdr)
    print(sep)
    for r in sorted(winners, key=lambda x: x["test_xsh"], reverse=True):
        xsh_26 = r["26yr_xsh"]
        xsh_t = r["test_xsh"]
        xsh_r = r["recent_xsh"]
        print(f"  enter={r['enter_at']:.0f} fw={r['full_weight_at']:.0f}  26yr={xsh_26:+.3f}  test={xsh_t:+.3f}  recent={xsh_r:+.3f}  26yr-DD={r['26yr_maxdd']:.1f}%  test-DD={r['test_maxdd']:.1f}%")
else:
    print("  None.")

print(f"\n  Reference (binary 23/24): 26yr={ref_y26:+.3f}  test={ref_test:+.3f}  recent={ref['recent']['xsharpe']:+.3f}")
print(f"  Drawdown improvement (if any): binary 26yr={ref['26yr']['max_dd_pct']:.1f}%  test={ref['test']['max_dd_pct']:.1f}%")
