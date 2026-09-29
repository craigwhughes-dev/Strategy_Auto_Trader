"""P4a feasibility check for PLAN_CRASH_PHASES.md: can 10 independent shocks be assembled for free?

Run: uv run python scripts/tier_analysis/p4a_event_register.py [--refresh]

This answers the owner-ratified stop-gate and nothing else. It builds the cross-market event
register from free daily index closes, groups episodes into global shocks, and reports the
independent-shock count. It runs no strategy, tests no indicator, and makes no claim about
detectability — that is the first cut, which only starts if this gate clears.

Depth is measured on each index **in its own currency**, which is why the register is built from
local-currency indices rather than the GBP legs the strategy trades: a threshold applied to a
foreign index in GBP is partly an FX threshold, as COVID reading -21.6% on the GBP Nasdaq leg
against the USD index's ~-30% showed.

Writes `data/monte_carlo/../` nothing; the register goes to data/cache/yahoo_index/event_register.csv.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, so the package imports work when run as a script

import pandas as pd

from Strategy_Auto_Trader.research import crash_phases as cp
from Strategy_Auto_Trader.research import index_history as ih

POST_TROUGH_DAYS = 60
DESIGN_MARKETS = ("^GSPC", "^IXIC")   # US: already contaminated by T1/T9/realized-vol work
GATE = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-fetch every index instead of using the cache")
    args = ap.parse_args()

    frames, meta = [], []
    for sym, (name, ccy) in ih.MARKETS.items():
        try:
            close = ih.load_daily_closes(sym, refresh=args.refresh)
        except Exception as exc:  # a single unavailable index must not abort the gate check
            meta.append((sym, name, ccy, 0, "-", "-", f"FAILED {type(exc).__name__}"))
            continue
        reg = cp.episode_register(close, market=sym, post_trough_days=POST_TROUGH_DAYS)
        reg["name"], reg["currency"] = name, ccy
        frames.append(reg)
        meta.append((sym, name, ccy, len(close), str(close.index[0].date()), str(close.index[-1].date()), f"{len(reg)} episodes"))

    print("=== DATA COVERAGE (Yahoo chart endpoint; local-currency indices)")
    print(f"  {'symbol':<8s} {'index':<20s} {'ccy':<4s} {'bars':>6s} {'from':>11s} {'to':>11s}  episodes")
    for sym, name, ccy, n, f, l, note in meta:
        print(f"  {sym:<8s} {name:<20s} {ccy:<4s} {n:6d} {f:>11s} {l:>11s}  {note}")
    print()

    if not frames:
        print("NO DATA — gate fails outright.")
        return

    reg = cp.group_into_shocks(pd.concat(frames, ignore_index=True))
    reg["is_design"] = reg["market"].isin(DESIGN_MARKETS)
    out = ih.CACHE_DIR / "event_register.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    reg.to_csv(out, index=False)

    print(f"=== EVENT REGISTER: {len(reg)} episodes, {reg['market'].nunique()} markets -> {out}")
    print(f"  {'shock':>5s} {'trough span':<25s} {'mkts':>4s} {'depth% med/worst':>18s}  markets")
    for sid, grp in reg.groupby("shock_id"):
        lo, hi = grp["trough_date"].min().date(), grp["trough_date"].max().date()
        syms = ",".join(sorted(s.lstrip("^") for s in grp["market"]))
        print(
            f"  {sid:5d} {str(lo) + ' .. ' + str(hi):<25s} {grp['market'].nunique():4d} "
            f"{grp['depth_pct'].median():8.1f} /{grp['depth_pct'].min():8.1f}  {syms}"
        )
    print()

    n_shocks = cp.count_independent_shocks(reg)
    holdout = reg[~reg["is_design"]]
    n_holdout = cp.count_independent_shocks(holdout)
    deep = reg[reg["depth_pct"] <= -25.0]
    print("=== GATE")
    print(f"  independent shocks (window {cp.SHOCK_WINDOW_DAYS}d): {n_shocks}")
    print(f"  of which have at least one non-US market (holdout-usable): {n_holdout}")
    print(f"  shocks containing a >=25% drawdown somewhere: {cp.count_independent_shocks(deep)}")
    print(f"  episodes in design set (US): {int(reg['is_design'].sum())}, holdout: {len(holdout)}")
    print(f"  GATE (>= {GATE} independent shocks): {'PASS' if n_shocks >= GATE else 'FAIL'}")
    print(f"  GATE on holdout-usable shocks alone: {'PASS' if n_holdout >= GATE else 'FAIL'}")


if __name__ == "__main__":
    main()
