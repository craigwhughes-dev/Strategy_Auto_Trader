"""VIX gate on an ALREADY-HELD index fund: does gating beat just holding it?

Different question from basket_tier1.py. There the benchmark was the deployed Nasdaq tier and
the basket lost. Here the holding already exists outside the pot (owner holds VWRL etc.), so the
benchmark is BUY-AND-HOLD OF THAT FUND — the only alternative the owner actually has.

Gate is VIX (broad-market implied vol), not VXN: VXN is Nasdaq-specific and has no claim on a
world or FTSE fund. Thresholds are swept rather than inherited, since 23/24 was fitted to
Nasdaq/VXN and carries no information about where a VIX gate on world equity belongs.

Run: uv run python scripts/tier_analysis/vix_gate_existing_holding.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from Strategy_Auto_Trader.allocation import basket_engine as be
from Strategy_Auto_Trader.allocation import intraday_engine as eng

inp0 = eng.load_inputs()
inp, NAMES = be.extend_with_world(inp0)

HOLDINGS = ("WORLD", "SP500", "FTSE", "NASDAQ")
ENTERS = (14.0, 16.0, 18.0, 20.0, 22.0, 25.0, 28.0, 30.0)
WIDTHS = (0.0, 2.0)
WINDOWS = ("real", "train", "test")
ALWAYS_IN = np.zeros(len(inp.grid), dtype=int)  # tier 0 every bar = buy and hold, no switches


def stats(run, window):
    s = eng.window_stats(run, window)
    yrs = s["n_days"] / 252
    s["cagr"] = ((1 + s["ret_pct"] / 100) ** (1 / yrs) - 1) * 100
    return s


def run_holding(name: str) -> None:
    bh = {w: stats(be.basket_run(inp, NAMES, ALWAYS_IN, {name: 1.0}), w) for w in WINDOWS}
    print(f"\n{'=' * 104}")
    print(f"== {name}: VIX gate vs simply holding it  (13bps/switch, deadband enter<=E hold until >E+W)")
    print(f"{'=' * 104}")
    print(f"  {'gate':16s} " + "".join(f"|{w:>10s} xSh  CAGR   maxDD " for w in WINDOWS) + "| sw/yr")
    b = bh["real"]
    print(f"  {'BUY AND HOLD':16s} " + "".join(
        f"|{'':11s}{bh[w]['xsharpe']:+5.2f} {bh[w]['cagr']:5.1f} {bh[w]['max_dd_pct']:7.1f} " for w in WINDOWS
    ) + "|   0.0")
    best = None
    for width in WIDTHS:
        for enter in ENTERS:
            tiers = eng.tiers_vxn_deadband(inp.vix, enter, enter + width)
            run = be.basket_run(inp, NAMES, tiers, {name: 1.0})
            s = {w: stats(run, w) for w in WINDOWS}
            label = f"VIX<={enter:g}" + (f" /{enter + width:g}" if width else "")
            print(f"  {label:16s} " + "".join(
                f"|{'':11s}{s[w]['xsharpe']:+5.2f} {s[w]['cagr']:5.1f} {s[w]['max_dd_pct']:7.1f} " for w in WINDOWS
            ) + f"|{s['real']['sw_per_yr']:6.1f}")
            # rank on TRAIN only; test stays a holdout
            if best is None or s["train"]["xsharpe"] > best[1]["train"]["xsharpe"]:
                best = (label, s)
    print(f"\n  best-on-train: {best[0]}")
    for w in WINDOWS:
        g, h = best[1][w], bh[w]
        print(f"    {w:6s} xSharpe {g['xsharpe']:+.2f} vs B&H {h['xsharpe']:+.2f}  |  "
              f"CAGR {g['cagr']:5.1f} vs {h['cagr']:5.1f}  |  maxDD {g['max_dd_pct']:6.1f} vs {h['max_dd_pct']:6.1f}")


if __name__ == "__main__":
    for holding in HOLDINGS:
        run_holding(holding)
