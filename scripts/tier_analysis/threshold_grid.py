"""Experiment C: Threshold grid search with train/test split.

Sweeps enter_at in [19..25] and deadband in [0..10] (exit_above = enter_at + deadband).
Optimises on TRAIN (2007-2019) only; reports TEST (2020+) and RECENT (2024+) separately.

Selection rule: winner must beat current 23/24 on BOTH train_xsh AND test_xsh.

Run: uv run python scripts/tier_analysis/threshold_grid.py
"""
from __future__ import annotations

from itertools import product

import pandas as pd

from Strategy_Auto_Trader.allocation import intraday_engine as eng

inp = eng.load_inputs()
eng.WINDOWS.setdefault("y26", (None, None))
eng.WINDOWS.setdefault("vxn_era", ("2007-11-20", None))
eng.WINDOWS.setdefault("train", ("2007-11-20", "2019-12-31"))
eng.WINDOWS.setdefault("test", ("2020-01-01", None))
eng.WINDOWS.setdefault("recent", ("2024-03-25", None))

COST_BPS = 13.0
ENTER_AT_RANGE = [19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
DEADBAND_RANGE = range(0, 11)

rows = []
for enter_at, deadband in product(ENTER_AT_RANGE, DEADBAND_RANGE):
    exit_above = enter_at + deadband
    tiers = eng.tiers_vxn_deadband(inp.vxn, enter_at, exit_above)
    run = eng.simulate(inp, tiers, cost_bps=COST_BPS)
    row = {"enter_at": enter_at, "exit_above": exit_above, "deadband": deadband}
    for wname, wkey in [("train", "train"), ("test", "test"), ("recent", "recent"), ("y26", "y26")]:
        s = eng.window_stats(run, wkey)
        row[f"{wname}_xsh"] = s["xsharpe"]
        row[f"{wname}_maxdd"] = s["max_dd_pct"]
        row[f"{wname}_sw"] = s["sw_per_yr"]
        row[f"{wname}_ret"] = s["ret_pct"]
    rows.append(row)

df = pd.DataFrame(rows)

# Reference row: current 23/24
ref = df[(df.enter_at == 23.0) & (df.exit_above == 24.0)].iloc[0]
ref_train_xsh = ref["train_xsh"]
ref_test_xsh = ref["test_xsh"]
ref_y26_xsh = ref["y26_xsh"]

print(f"\nThreshold grid: {len(df)} combinations  (enter_at x deadband)  cost={COST_BPS} bps")
print(f"Reference (23/24): train={ref_train_xsh:+.3f}  test={ref_test_xsh:+.3f}  recent={ref['recent_xsh']:+.3f}  26yr={ref_y26_xsh:+.3f}\n")

# Top 10 by test xSh
hdr = f"  {'enter':>5} {'exit':>5} {'db':>3}  {'train_xsh':>9}  {'test_xsh':>8}  {'recent_xsh':>10}  {'y26_xsh':>7}  {'test_DD':>7}  {'test_sw':>7}"
sep = "  " + "-" * (len(hdr) - 2)

print("=== Top 10 by TEST xSharpe ===")
top_test = df.nlargest(10, "test_xsh")
print(hdr)
print(sep)
for _, r in top_test.iterrows():
    marker = " *" if r["enter_at"] == 23 and r["exit_above"] == 24 else "  "
    beat_train = r["train_xsh"] >= ref_train_xsh
    flag = "T" if beat_train else " "
    print(f"{marker}{r['enter_at']:>5.0f} {r['exit_above']:>5.0f} {r['deadband']:>3.0f}  "
          f"{r['train_xsh']:>9.3f}  {r['test_xsh']:>8.3f}  {r['recent_xsh']:>10.3f}  "
          f"{r['y26_xsh']:>7.3f}  {r['test_maxdd']:>7.1f}  {r['test_sw']:>7.1f}  [{flag}]")
print("  [T] = also beats reference on train_xsh")

print("\n=== Top 10 by TEST maxDD (lowest drawdown) ===")
top_dd = df.nsmallest(10, "test_maxdd")
print(hdr)
print(sep)
for _, r in top_dd.iterrows():
    marker = " *" if r["enter_at"] == 23 and r["exit_above"] == 24 else "  "
    beat_train = r["train_xsh"] >= ref_train_xsh
    flag = "T" if beat_train else " "
    print(f"{marker}{r['enter_at']:>5.0f} {r['exit_above']:>5.0f} {r['deadband']:>3.0f}  "
          f"{r['train_xsh']:>9.3f}  {r['test_xsh']:>8.3f}  {r['recent_xsh']:>10.3f}  "
          f"{r['y26_xsh']:>7.3f}  {r['test_maxdd']:>7.1f}  {r['test_sw']:>7.1f}  [{flag}]")

print("\n=== Pareto frontier: non-dominated on (test_xsh, -test_maxdd) ===")
# A row is dominated if another row has >= test_xsh AND <= test_maxdd (less negative = better DD)
dominated = set()
records = df[["enter_at", "exit_above", "deadband", "test_xsh", "test_maxdd", "train_xsh",
              "recent_xsh", "y26_xsh", "test_sw"]].to_dict("records")
for i, a in enumerate(records):
    for j, b in enumerate(records):
        if i == j:
            continue
        if b["test_xsh"] >= a["test_xsh"] and b["test_maxdd"] <= a["test_maxdd"]:
            if b["test_xsh"] > a["test_xsh"] or b["test_maxdd"] < a["test_maxdd"]:
                dominated.add(i)
                break
pareto = [r for i, r in enumerate(records) if i not in dominated]
pareto.sort(key=lambda r: r["test_xsh"], reverse=True)
print(hdr)
print(sep)
for r in pareto:
    marker = " *" if r["enter_at"] == 23 and r["exit_above"] == 24 else "  "
    beat_train = r["train_xsh"] >= ref_train_xsh
    flag = "T" if beat_train else " "
    print(f"{marker}{r['enter_at']:>5.0f} {r['exit_above']:>5.0f} {r['deadband']:>3.0f}  "
          f"{r['train_xsh']:>9.3f}  {r['test_xsh']:>8.3f}  {r['recent_xsh']:>10.3f}  "
          f"{r['y26_xsh']:>7.3f}  {r['test_maxdd']:>7.1f}  {r['test_sw']:>7.1f}  [{flag}]")

print("\n=== Must beat reference on BOTH train AND test xSh ===")
winners = df[(df.train_xsh >= ref_train_xsh) & (df.test_xsh >= ref_test_xsh)]
if len(winners) > 0:
    print(hdr)
    print(sep)
    for _, r in winners.sort_values("test_xsh", ascending=False).iterrows():
        marker = " *" if r["enter_at"] == 23 and r["exit_above"] == 24 else "  "
        print(f"{marker}{r['enter_at']:>5.0f} {r['exit_above']:>5.0f} {r['deadband']:>3.0f}  "
              f"{r['train_xsh']:>9.3f}  {r['test_xsh']:>8.3f}  {r['recent_xsh']:>10.3f}  "
              f"{r['y26_xsh']:>7.3f}  {r['test_maxdd']:>7.1f}  {r['test_sw']:>7.1f}")
else:
    print("  None — current 23/24 is on the Pareto frontier among validated thresholds.")

print(f"\n  Reference (23/24): * row above  |  train={ref_train_xsh:+.3f}  test={ref_test_xsh:+.3f}")
