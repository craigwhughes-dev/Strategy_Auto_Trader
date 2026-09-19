"""Does FTSE add value as a middle tier? Asset returns by VXN bucket, and FTSE in place of cash.

Run: uv run python scripts/tier_analysis/ftse_role.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, so the package imports work when run as a script

import numpy as np
import pandas as pd

from Strategy_Auto_Trader.allocation import intraday_engine as eng

inp = eng.load_inputs()
eng.WINDOWS.update({"recent": ("2024-03-25", None)})
n = len(inp.grid)

# ---- 1) what each asset earns, by the VXN regime in force at the START of the bar (no look-ahead)
vxn_prev = np.concatenate([[np.nan], inp.vxn[:-1]])
day = pd.Series(inp.days[inp.day_codes])
bounds = {w: eng._bounds(inp.days, w) for w in ("train", "test", "recent")}
day_idx = inp.day_codes


def window_mask(w):
    sl = bounds[w]
    return (day_idx >= sl.start) & (day_idx < sl.stop)


buckets = [(0, 20), (20, 24), (24, 28), (28, 35), (35, 200)]
print("== annualised return (%) by prior-bar VXN bucket; share of time in bucket. Cash = CSH2 accrual.")
for w in ("train", "test"):
    m = window_mask(w)
    n_days = len(np.unique(day_idx[m]))
    print(f"\n  window {w} ({n_days} days)")
    print("    VXN bucket | time  | Nasdaq   S&P     FTSE    cash  | FTSE-cash | FTSE hit rate vs cash (daily)")
    for lo, hi in buckets:
        sel = m & (vxn_prev > lo) & (vxn_prev <= hi) if lo else m & (vxn_prev <= hi)
        if sel.sum() < 50:
            continue
        days_in = len(np.unique(day_idx[sel]))
        ann = lambda col: (np.exp(inp.log_ret[sel, col].sum() * 252 / days_in) - 1) * 100
        # daily FTSE vs cash hit rate (days entirely in the bucket are approximated by summing bars in it)
        d_ftse = np.bincount(day_idx[sel], weights=inp.log_ret[sel, 2], minlength=len(inp.days))
        d_cash = np.bincount(day_idx[sel], weights=inp.log_ret[sel, 3], minlength=len(inp.days))
        present = np.bincount(day_idx[sel], minlength=len(inp.days)) > 0
        hit = float((d_ftse[present] > d_cash[present]).mean())
        label = f"<= {hi}" if not lo else f"{lo}-{hi if hi < 200 else '+'}"
        print(f"    {label:10s} | {sel.sum() / m.sum():4.0%}  | {ann(0):+7.1f} {ann(1):+7.1f} {ann(2):+7.1f} {ann(3):+6.1f} | {ann(2) - ann(3):+8.1f}  | {hit:.0%}")

# ---- 2) Nasdaq deadband 23/24, and when NOT in Nasdaq hold FTSE while VXN <= f, else cash
nasdaq = eng.tiers_vxn_deadband(inp.vxn, 23.0, 24.0) == 0
print("\n== Nasdaq (VXN enter<=23 / hold<=24) else FTSE while VXN<=f else cash   (same-bar, 13 bps)")
print("  f           | time N/F/C          | train x   ret%  dd%  | test x   ret%   dd%   sw/yr | recent x ret%  dd%  sw/yr | 2007+ x  ret%  dd%")
rows = [("cash only", None)] + [(f"f={f}", float(f)) for f in (26, 28, 30, 35, 40)] + [("FTSE always", 1e9)]
for label, f in rows:
    tiers = np.full(n, 3)
    if f is not None:
        tiers[np.nan_to_num(inp.vxn, nan=1e9) <= f] = 2
    tiers[nasdaq] = 0
    occ = np.bincount(tiers, minlength=4) / n
    run = eng.simulate(inp, tiers)
    s = {w: eng.window_stats(run, w) for w in ("train", "test", "recent", "real")}
    print(f"  {label:11s} | {occ[0]:.0%}/{occ[2]:.0%}/{occ[3]:.0%}".ljust(31)
          + f"| {s['train']['xsharpe']:+.2f} {s['train']['ret_pct']:+6.0f} {s['train']['max_dd_pct']:+6.1f} "
          + f"| {s['test']['xsharpe']:+.2f} {s['test']['ret_pct']:+5.0f} {s['test']['max_dd_pct']:+6.1f} {s['test']['sw_per_yr']:5.1f} "
          + f"| {s['recent']['xsharpe']:+.2f} {s['recent']['ret_pct']:+4.0f} {s['recent']['max_dd_pct']:+6.1f} {s['recent']['sw_per_yr']:5.1f} "
          + f"| {s['real']['xsharpe']:+.2f} {s['real']['ret_pct']:+5.0f} {s['real']['max_dd_pct']:+6.1f}")

print("\n== FTSE buy and hold for reference (xSharpe / return / dd)")
for w in ("train", "test", "recent"):
    s = eng.window_stats(eng.buy_and_hold(inp, "FTSE"), w)
    print(f"  {w:7s} {s['xsharpe']:+.2f} / {s['ret_pct']:+.0f}% / {s['max_dd_pct']:+.1f}%")
