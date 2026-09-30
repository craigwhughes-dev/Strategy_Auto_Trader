"""VVIX-banded VXN deadband: widen (permissive) when VVIX says calm, tighten (exit sooner) when
VVIX says stress, both legs moving together — vs. holding deployed VXN 23/24 fixed.

Follow-up to the 2026-09-30 16:20 BACKTEST_LOG.md entry, which tested a hard VVIX veto (force
cash above a VVIX threshold, otherwise defer to VXN 23/24) and rejected it (monotone-negative on
every window). That was the wrong mechanism — this script tests the one actually intended:
VVIX doesn't veto, it moves the VXN threshold pair itself.

VVIX quantiles used to pick the edges/deltas below (real-data window, `load_vvix_daily().quantile()`):
5%=71.7  10%=76.4  25%=82.9  50%=91.2  75%=102.3  90%=115.3  95%=122.2  99%=143.1

Primary config declared before running, to avoid picking the flattering result post-hoc:
83/102 edges (quartile split, symmetric), delta=2.

Run: uv run python scripts/tier_analysis/test_vvix_banded.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from Strategy_Auto_Trader.allocation import intraday_engine as eng

BASE_PAIR = (23.0, 24.0)
BASE_BAND = 1
CONFIGS = [
    ("80/105 d2", (80.0, 105.0), ((25.0, 26.0), BASE_PAIR, (21.0, 22.0))),
    ("83/102 d2 [primary]", (83.0, 102.0), ((25.0, 26.0), BASE_PAIR, (21.0, 22.0))),
    ("76/122 d2", (76.0, 122.0), ((25.0, 26.0), BASE_PAIR, (21.0, 22.0))),
    ("83/102 d1", (83.0, 102.0), ((24.0, 25.0), BASE_PAIR, (22.0, 23.0))),
    ("83/102 d4", (83.0, 102.0), ((27.0, 28.0), BASE_PAIR, (19.0, 20.0))),
]
WINDOWS = ("real", "train", "test", "all")


def stats(run, window):
    s = eng.window_stats(run, window)
    s["cagr"] = ((1 + s["ret_pct"] / 100) ** (252 / s["n_days"]) - 1) * 100 if s["n_days"] else float("nan")
    return s


inp = eng.load_inputs()
vvix = eng.vvix_for_grid(inp.grid)

run_deployed = eng.simulate(inp, eng.tiers_vxn_deadband(inp.vxn, *BASE_PAIR))
runs = {
    label: eng.simulate(inp, eng.tiers_vxn_vvix_banded(inp.vxn, vvix, edges, pairs, BASE_BAND))
    for label, edges, pairs in CONFIGS
}

print(f"\n{'=' * 100}")
print("VXN 23/24 deployed (fixed) vs. VVIX-banded VXN deadband (widen calm / tighten stressed). 13bps/switch.")
print("real/train/test are inside VVIX's real-data coverage (2007+); `all` is reference only (pre-2007 VVIX gate is inert).")
print(f"{'=' * 100}")
print(f"  {'window':7s} {'rule':22s} | {'xSharpe':>8} {'CAGR%':>7} {'maxDD%':>7} {'sw/yr':>6}")
for w in WINDOWS:
    d = stats(run_deployed, w)
    print(f"  {w:7s} {'VXN 23/24 (deployed)':22s} | {d['xsharpe']:+8.3f} {d['cagr']:7.1f} {d['max_dd_pct']:7.1f} {d['sw_per_yr']:6.1f}")
    for label, _, _ in CONFIGS:
        s = stats(runs[label], w)
        flag = "  <-- beats deployed" if s["xsharpe"] > d["xsharpe"] else ""
        print(f"  {'':7s} {label:22s} | {s['xsharpe']:+8.3f} {s['cagr']:7.1f} {s['max_dd_pct']:7.1f} {s['sw_per_yr']:6.1f}{flag}")
    print()
