"""VVIX (vol-of-VIX) gate on top of the deployed VXN 23/24 deadband: does it help?

Real VVIXCLS only (FRED, 2006-01-04+, no pre-2006 bridge — unlike VXN there is no published
earlier-methodology VVIX to anchor a synthetic bridge to). real/train/test windows all start
2007-11-20+, fully inside VVIX coverage, so they are the primary comparison. The `all` window
(1999-03-11+) is shown for reference only: pre-2006 VVIX is NaN, which never forces cash (see
tiers_vxn_vvix_deadband), so every threshold is identical to deployed before 2006 — `all` proves
nothing about VVIX, it just should not differ much from deployed pre-2006.

VVIX trades on a different scale than VIX/VXN (median ~91, range ~60-208 over the real-data
window, checked via `load_vvix_daily().describe()` before picking thresholds) — 30-70 (a VIX-scale
guess) would breach almost permanently and was rejected before this sweep was written. Thresholds
below span roughly the 10th-99th percentile of the real series instead.

Run: uv run python scripts/tier_analysis/test_vvix_gate.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from Strategy_Auto_Trader.allocation import intraday_engine as eng

VVIX_THRESHOLDS = (80.0, 90.0, 100.0, 110.0, 120.0, 130.0)
WINDOWS = ("real", "train", "test", "all")


def stats(run, window):
    s = eng.window_stats(run, window)
    s["cagr"] = ((1 + s["ret_pct"] / 100) ** (252 / s["n_days"]) - 1) * 100 if s["n_days"] else float("nan")
    return s


inp = eng.load_inputs()
vvix = eng.vvix_for_grid(inp.grid)

run_deployed = eng.simulate(inp, eng.tiers_vxn_deadband(inp.vxn, 23.0, 24.0))
runs = {t: eng.simulate(inp, eng.tiers_vxn_vvix_deadband(inp.vxn, vvix, 23.0, 24.0, t)) for t in VVIX_THRESHOLDS}

print(f"\n{'=' * 100}")
print("VXN 23/24 deployed vs. VXN 23/24 + VVIX hard-cash-gate, swept threshold. 13bps/switch.")
print("real/train/test are inside VVIX's real-data coverage (2006+); `all` is reference only (pre-2006 VVIX gate is inert).")
print(f"{'=' * 100}")
print(f"  {'window':7s} {'rule':22s} | {'xSharpe':>8} {'CAGR%':>7} {'maxDD%':>7} {'sw/yr':>6}")
for w in WINDOWS:
    d = stats(run_deployed, w)
    print(f"  {w:7s} {'VXN 23/24 (deployed)':22s} | {d['xsharpe']:+8.3f} {d['cagr']:7.1f} {d['max_dd_pct']:7.1f} {d['sw_per_yr']:6.1f}")
    for t in VVIX_THRESHOLDS:
        s = stats(runs[t], w)
        flag = "  <-- beats deployed" if s["xsharpe"] > d["xsharpe"] else ""
        print(f"  {'':7s} {'+ VVIX<' + str(int(t)):22s} | {s['xsharpe']:+8.3f} {s['cagr']:7.1f} {s['max_dd_pct']:7.1f} {s['sw_per_yr']:6.1f}{flag}")
    print()
