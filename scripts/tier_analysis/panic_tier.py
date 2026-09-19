"""Panic tier test: hold an equity fund once VXN spikes above g. Grid over asset x g x width, plus episode list.

Run: uv run python scripts/tier_analysis/panic_tier.py
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
ASSET_IDX = {"NASDAQ": 0, "SP500": 1, "FTSE": 2}
base_nasdaq = eng.tiers_vxn_deadband(inp.vxn, 23.0, 24.0) == 0


def panic_state(vxn, enter_above, exit_at_or_below):
    """In panic once VXN > enter_above; stays until VXN <= exit_at_or_below. NaN keeps the state."""
    ev = np.zeros(len(vxn), dtype=np.int8)
    ev[vxn > enter_above] = 1
    ev[vxn <= exit_at_or_below] = -1
    last = np.maximum.accumulate(np.where(ev != 0, np.arange(len(vxn)), -1))
    return (last >= 0) & (ev[np.maximum(last, 0)] == 1)


def build(asset, g, width):
    tiers = np.full(n, 3)
    st = panic_state(inp.vxn, g, g - width)
    tiers[st & ~base_nasdaq] = ASSET_IDX[asset]
    tiers[base_nasdaq] = 0
    return tiers, st


def episodes(state):
    d = np.diff(state.astype(int), prepend=0, append=0)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def count_in(state, window):
    sl = eng._bounds(inp.days, window)
    eps = [(a, b) for a, b in episodes(state) if sl.start <= inp.day_codes[a] < sl.stop]
    return len(eps)


base_tiers = np.where(base_nasdaq, 0, 3)
b = {w: eng.window_stats(eng.simulate(inp, base_tiers), w) for w in ("train", "test", "recent", "bridged", "real")}
print("== baseline: Nasdaq deadband 23/24 else cash")
print(f"  train x{b['train']['xsharpe']:+.2f} ret {b['train']['ret_pct']:+.0f}% dd {b['train']['max_dd_pct']:+.1f} | test x{b['test']['xsharpe']:+.2f} ret {b['test']['ret_pct']:+.0f}% dd {b['test']['max_dd_pct']:+.1f} "
      f"sw {b['test']['sw_per_yr']:.1f} | bridged x{b['bridged']['xsharpe']:+.2f} | 2007+ x{b['real']['xsharpe']:+.2f} ret {b['real']['ret_pct']:+.0f}% dd {b['real']['max_dd_pct']:+.1f}")

print("\n== panic tier: hold <asset> once VXN > g, back to cash when VXN <= g-width   (Nasdaq deadband 23/24 unchanged; same-bar, 13 bps)")
print("  asset   g  width | time  eps tr/te | train x  ret%    dd%  | test x   ret%    dd%  sw/yr | bridged x | 2007+ x  ret%     dd%")
results = []
for asset in ("NASDAQ", "SP500", "FTSE"):
    for g in (30, 35, 40, 45):
        for width in (0, 3, 6):
            tiers, st = build(asset, g, width)
            run = eng.simulate(inp, tiers)
            s = {w: eng.window_stats(run, w) for w in ("train", "test", "bridged", "real")}
            occ = float((tiers == ASSET_IDX[asset]).mean() - (base_nasdaq & (ASSET_IDX[asset] == 0)).mean()) if asset != "NASDAQ" else float((st & ~base_nasdaq).mean())
            results.append((asset, g, width, s))
            print(f"  {asset:7s} {g:2d}  {width:3d}   | {st.mean():4.1%}  {count_in(st, 'train'):2d}/{count_in(st, 'test'):2d} | {s['train']['xsharpe']:+.2f} {s['train']['ret_pct']:+6.0f} {s['train']['max_dd_pct']:+6.1f} "
                  f"| {s['test']['xsharpe']:+.2f} {s['test']['ret_pct']:+6.0f} {s['test']['max_dd_pct']:+6.1f} {s['test']['sw_per_yr']:5.1f} | {s['bridged']['xsharpe']:+.2f}     "
                  f"| {s['real']['xsharpe']:+.2f} {s['real']['ret_pct']:+6.0f} {s['real']['max_dd_pct']:+7.1f}")

print("\n== how the panic tier fares across the grid vs baseline (share of cells better than baseline)")
for asset in ("NASDAQ", "SP500", "FTSE"):
    cells = [r for r in results if r[0] == asset]
    for w in ("train", "test", "bridged"):
        better = np.mean([c[3][w]["xsharpe"] > b[w]["xsharpe"] for c in cells])
        print(f"  {asset:7s} {w:8s}: {better:.0%} of {len(cells)} cells beat baseline xSharpe ({b[w]['xsharpe']:+.2f})")

print("\n== each panic episode for one setting: hold NASDAQ, g=35, width=3")
tiers, st = build("NASDAQ", 35, 3)
day = pd.DatetimeIndex(inp.days[inp.day_codes])
print("  start       end         bars  Nasdaq ret%   worst dip inside%   S&P ret%  FTSE ret%")
for a, z in episodes(st):
    if a == 0:
        continue
    seg = slice(a + 1, z + 1)  # position taken at bar a, earns bars a+1..z
    def path(col):
        c = np.cumsum(inp.log_ret[seg, col])
        return (np.exp(c[-1]) - 1) * 100, (np.exp(c.min() if c.min() < 0 else 0) - 1) * 100
    nr, nd = path(0)
    print(f"  {day[a].date()}  {day[min(z, n - 1)].date()}  {z - a:5d}  {nr:+9.1f}    {nd:+9.1f}      {path(1)[0]:+7.1f}   {path(2)[0]:+7.1f}")
