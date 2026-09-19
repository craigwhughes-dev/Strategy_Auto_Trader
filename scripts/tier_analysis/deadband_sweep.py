"""Experiment A: wider exit deadband sweep.

Fixes enter_at=23 and sweeps exit_above from 24 to 33 (deadband 1..10).
Reports xSharpe, maxDD%, sw/yr, ret% across five windows: 26yr, since-2007,
train (2007-2019), test (2020+), recent (2024-03-25+).

Run: uv run python scripts/tier_analysis/deadband_sweep.py
"""
from __future__ import annotations

from Strategy_Auto_Trader.allocation import intraday_engine as eng

inp = eng.load_inputs()
eng.WINDOWS.setdefault("y26", (None, None))
eng.WINDOWS.setdefault("vxn_era", ("2007-11-20", None))
eng.WINDOWS.setdefault("recent", ("2024-03-25", None))
eng.WINDOWS.setdefault("train", ("2007-11-20", "2019-12-31"))
eng.WINDOWS.setdefault("test", ("2020-01-01", None))

ENTER_AT = 23.0
COST_BPS = 13.0
EXIT_RANGE = [24.0, 25.0, 26.0, 27.0, 28.0, 30.0, 33.0]

WINDOWS_SPEC = [
    ("26yr",       "y26"),
    ("since-2007", "vxn_era"),
    ("train",      "train"),
    ("test",       "test"),
    ("recent",     "recent"),
]

rows = []
for exit_above in EXIT_RANGE:
    tiers = eng.tiers_vxn_deadband(inp.vxn, ENTER_AT, exit_above)
    run = eng.simulate(inp, tiers, cost_bps=COST_BPS)
    for wname, wkey in WINDOWS_SPEC:
        s = eng.window_stats(run, wkey)
        rows.append({"exit_above": exit_above, "window": wname, **s})

# Print pivot: one block per window
print(f"\nDeadband sweep: enter_at={ENTER_AT}, cost={COST_BPS} bps")
print(f"Current deployed: exit_above=24  (deadband 1)\n")

hdr = f"  {'exit_above':>10}  {'xSharpe':>8}  {'maxDD%':>7}  {'sw/yr':>6}  {'ret%':>8}"
sep = "  " + "-" * (len(hdr) - 2)

for wname, _ in WINDOWS_SPEC:
    print(f"=== {wname} ===")
    print(hdr)
    print(sep)
    for r in rows:
        if r["window"] != wname:
            continue
        marker = " *" if r["exit_above"] == 24.0 else "  "
        xsh = r["xsharpe"]
        dd = r["max_dd_pct"]
        sw = r["sw_per_yr"]
        ret = r["ret_pct"]
        xsh_str = f"{xsh:+.3f}" if xsh == xsh else "   nan"
        print(f"{marker}{r['exit_above']:>10.0f}  {xsh_str:>8}  {dd:>7.1f}  {sw:>6.1f}  {ret:>8.1f}")
    print()

# Summary: Pareto-dominant rows on test window (best xSh without being dominated on maxDD)
test_rows = [r for r in rows if r["window"] == "test"]
test_rows.sort(key=lambda r: r["xsharpe"], reverse=True)
print("=== TEST window: sorted by xSharpe ===")
print(hdr)
print(sep)
for r in test_rows:
    marker = " *" if r["exit_above"] == 24.0 else "  "
    xsh = r["xsharpe"]
    xsh_str = f"{xsh:+.3f}" if xsh == xsh else "   nan"
    print(f"{marker}{r['exit_above']:>10.0f}  {xsh_str:>8}  {r['max_dd_pct']:>7.1f}  {r['sw_per_yr']:>6.1f}  {r['ret_pct']:>8.1f}")
print()

recent_rows = [r for r in rows if r["window"] == "recent"]
recent_rows.sort(key=lambda r: r["xsharpe"], reverse=True)
print("=== RECENT window: sorted by xSharpe ===")
print(hdr)
print(sep)
for r in recent_rows:
    marker = " *" if r["exit_above"] == 24.0 else "  "
    xsh = r["xsharpe"]
    xsh_str = f"{xsh:+.3f}" if xsh == xsh else "   nan"
    print(f"{marker}{r['exit_above']:>10.0f}  {xsh_str:>8}  {r['max_dd_pct']:>7.1f}  {r['sw_per_yr']:>6.1f}  {r['ret_pct']:>8.1f}")
