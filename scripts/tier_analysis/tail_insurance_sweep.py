"""Cheapest tail insurance on a fund you intend to hold 100% of anyway.

Owner's stated position: 100% invested in the fund, no protection, and NOT willing to hold a
permanent cash sleeve (tax, drag). So the benchmark is buy-and-hold of that fund, and the
objective is NOT xSharpe — it is drawdown removed per point of CAGR given up, while staying
in-market as much of the time as possible.

That is a different selection criterion from gate_index_matrix.py, which ranked on train xSharpe
and therefore picked a mid-level gate that trades often. Here the sweep runs out to VXN 45 so the
"only exit in genuinely extreme vol" end of the range is actually covered.

Run: uv run python scripts/tier_analysis/tail_insurance_sweep.py [HOLDING]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from Strategy_Auto_Trader.allocation import basket_engine as be
from Strategy_Auto_Trader.allocation import intraday_engine as eng

HOLDING = sys.argv[1] if len(sys.argv) > 1 else "WORLD"
ENTERS = tuple(float(x) for x in range(20, 46, 2))
WIDTHS = (0.0, 2.0, 4.0, 6.0)
WINDOWS = ("real", "test")

inp0 = eng.load_inputs()
inp, NAMES = be.extend_with_world(inp0)
ALWAYS_IN = np.zeros(len(inp.grid), dtype=int)


def stats(run, window):
    s = eng.window_stats(run, window)
    s["cagr"] = ((1 + s["ret_pct"] / 100) ** (252 / s["n_days"]) - 1) * 100
    return s


def in_market_pct(tiers: np.ndarray, window: str) -> float:
    """Share of BARS in the window spent holding the fund rather than cash."""
    sl = eng._bounds(inp.days, window)
    lo_day, hi_day = inp.days[sl][0], inp.days[sl][-1]
    london = inp.grid.tz_convert("Europe/London").tz_localize(None).normalize()
    mask = (london >= lo_day) & (london <= hi_day)
    return float((tiers[mask] == 0).mean() * 100)


bh = {w: stats(be.basket_run(inp, NAMES, ALWAYS_IN, {HOLDING: 1.0}), w) for w in WINDOWS}
print(f"\n{'=' * 114}")
print(f"== Tail insurance on {HOLDING}: VXN gate vs staying 100% invested")
print(f"   buy-and-hold  " + "  ".join(
    f"[{w}] CAGR {bh[w]['cagr']:.1f}%  maxDD {bh[w]['max_dd_pct']:.1f}%  xSharpe {bh[w]['xsharpe']:+.2f}" for w in WINDOWS))
print(f"{'=' * 114}")
print(f"  {'gate':12s} {'in-mkt%':>7} {'sw/yr':>6} "
      f"| {'real: CAGR':>10} {'give-up':>8} {'maxDD':>7} {'DD saved':>9} {'ratio':>6} "
      f"| {'test: CAGR':>10} {'give-up':>8} {'maxDD':>7} {'DD saved':>9} {'ratio':>6}")

rows = []
for width in WIDTHS:
    for enter in ENTERS:
        tiers = eng.tiers_vxn_deadband(inp.vxn, enter, enter + width)
        run = be.basket_run(inp, NAMES, tiers, {HOLDING: 1.0})
        s = {w: stats(run, w) for w in WINDOWS}
        cells = []
        for w in WINDOWS:
            give_up = bh[w]["cagr"] - s[w]["cagr"]
            dd_saved = s[w]["max_dd_pct"] - bh[w]["max_dd_pct"]  # both negative; positive = shallower
            ratio = dd_saved / give_up if give_up > 0.05 else float("inf")
            cells.append((s[w]["cagr"], give_up, s[w]["max_dd_pct"], dd_saved, ratio))
        label = f"VXN<={enter:g}" + (f"/{enter + width:g}" if width else "")
        rows.append((label, in_market_pct(tiers, "real"), s["real"]["sw_per_yr"], cells))

for label, inmkt, sw, cells in rows:
    line = f"  {label:12s} {inmkt:7.1f} {sw:6.1f} "
    for cagr, give_up, dd, dd_saved, ratio in cells:
        r = "   inf" if ratio == float("inf") else f"{ratio:6.2f}"
        line += f"| {cagr:10.1f} {give_up:8.1f} {dd:7.1f} {dd_saved:9.1f} {r} "
    print(line)

print(f"\n  give-up = CAGR points sacrificed vs buy-and-hold.  DD saved = drawdown points removed.")
print(f"  ratio   = DD points saved per CAGR point given up. Higher is cheaper insurance.")
print(f"  in-mkt% = share of bars holding {HOLDING} (2007-11-20 onward).")
