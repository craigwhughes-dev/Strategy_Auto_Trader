"""T9: time-based re-entry override for the VXN 23/24 deadband rule.

Run: uv run python scripts/tier_analysis/t9_time_based_reentry.py

Follow-on from T1 (`t1_override_rules.py`), which tested four price/vol triggers and found no finalist, and from
the realized-vol sweep that found no observable distinguishes a recovery from a continuing crash at the time.
This tests the opposite approach: arm on elapsed time alone, ramp in, never try to pick the bottom.

Same real-data window, cost, engine and gate as T1, so the two are directly comparable.
Grid stated up front: wait_days (21, 63, 126, 252) x ramp_days (1, 63) x size (1.0, 0.5) = 16 trials, 8% trailing
stop fixed at T1's value.

READ THE EPISODE COUNTS BEFORE THE GATE. The whole question concerns three events (2003 is outside the real
window; 2009 and 2020 are inside it). Three observations cannot validate a rule, and the design/confirm/holdout
gate is reported for comparability with T1, not because passing it would establish anything. The descriptive
output — how often each configuration fires, and what it did in the recovery years — is the usable result.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, so the package imports work when run as a script

import numpy as np
import pandas as pd

from Strategy_Auto_Trader.allocation import intraday_engine as eng
from Strategy_Auto_Trader.allocation.real_window import day_mask, last_bar_per_day, slice_from
from Strategy_Auto_Trader.allocation import matched_blend as mb  # perf_stats only
from Strategy_Auto_Trader.allocation import reentry_override as ro  # simulate_weighted only
from Strategy_Auto_Trader.allocation import time_based_reentry as tbr

REAL_START = "2007-11-20"
WAITS = (21, 63, 126, 252)
RAMPS = (1, 63)
SIZES = (1.0, 0.5)
WINDOWS = {"design": ("2007-11-20", "2019-12-31"), "confirm": ("2020-01-01", "2022-12-31"), "holdout": ("2023-01-01", None)}
BEAR_RALLIES = {"2008 Q4": ("2008-09-01", "2008-12-31"), "2022": ("2022-01-01", "2022-12-31")}
RECOVERIES = {"2009": ("2009-01-01", "2009-12-31"), "2020": ("2020-01-01", "2020-12-31")}
STABILITY_WAITS = (100, 126, 150, 175, 200, 225, 240, 252, 265, 280, 300, 330)
CAGR_MARGIN = 0.5
DD_MARGIN = 3.0
BEAR_RALLY_LIMIT = -5.0


def period_return(daily: np.ndarray, m: np.ndarray) -> float:
    return float(np.prod(1 + daily[m]) - 1) * 100


def name_of(wait: int, ramp: int, size: float) -> str:
    return f"w{wait}r{ramp}{'F' if size == 1.0 else 'H'}"


def main() -> None:
    full = eng.load_inputs()
    inp = slice_from(full, REAL_START)
    days = inp.days
    rc = eng.buy_and_hold(inp, "CASH").daily

    last_full = last_bar_per_day(full.day_codes, len(full.days))
    close_full = pd.Series(np.exp(np.cumsum(full.log_ret[:, 0]))[last_full], index=full.days)
    pos = full.days.get_indexer(days)
    assert (pos >= 0).all()
    close_d = close_full.to_numpy()[pos]

    tiers = eng.tiers_vxn_deadband(inp.vxn, 23.0, 24.0)
    last = last_bar_per_day(inp.day_codes, len(days))
    base_cash = tiers[last] == 3
    n_days = len(days)

    def run(weights: np.ndarray) -> np.ndarray:
        return ro.simulate_weighted(inp.log_ret, weights, inp.day_codes, n_days, eng.DEFAULT_COST_BPS)

    base = run((tiers == 0).astype(float))
    engine = eng.simulate(inp, tiers).daily
    print(f"real window {days[0].date()} to {days[-1].date()}; sanity: weighted sim vs engine, max abs daily diff {np.abs(base - engine).max():.2e}")
    print(f"base rule in cash on {base_cash.sum()} of {n_days} days ({base_cash.mean()*100:.1f}%)\n")

    # ---- how long are the cash episodes? -----------------------------------
    print("=== CASH EPISODES OF THE BASE RULE (this is what the wait_days grid has to work with)")
    eps = tbr.episodes(base_cash.astype(float))
    lengths = np.array([e - s + 1 for s, e in eps])
    print(f"  {len(eps)} episodes, median {np.median(lengths):.0f} days, longest {lengths.max()} days")
    for w in WAITS:
        qualifying = [(s, e) for s, e in eps if (e - s + 1) > w]
        yrs = sorted({days[s].year for s, _ in qualifying})
        print(f"  wait {w:3d}d: {len(qualifying):2d} episodes exceed it ({len(qualifying)/(n_days/252):.2f}/yr) -> starts in {yrs}")
    print()

    variants: dict[str, np.ndarray] = {}
    fracs: dict[str, np.ndarray] = {}
    for wait in WAITS:
        for ramp in RAMPS:
            f = tbr.override_fractions(base_cash, close_d, wait, ramp)
            for size in SIZES:
                nm = name_of(wait, ramp, size)
                fracs[nm] = f
                variants[nm] = run(tbr.bar_weights(tiers, inp.day_codes, f, size))
    print(f"trials in this workstream: {len(variants)} (wait x ramp x size; stop fixed at {tbr.STOP_PCT:.0%})\n")

    # ---- descriptive first --------------------------------------------------
    print("=== FIRING RATE AND RECOVERY-YEAR EFFECT (descriptive — the usable result)")
    hdr = f"  {'rule':10s} {'episodes':>8s} {'per decade':>11s} {'days on%':>9s} {'avg w%':>7s}"
    hdr += "".join(f" {k:>9s}" for k in RECOVERIES) + "".join(f" {k:>9s}" for k in BEAR_RALLIES) + f" {'whole pd':>9s}"
    print(hdr)
    years = n_days / 252
    for nm, d in variants.items():
        f = fracs[nm]
        size = 1.0 if nm.endswith("F") else 0.5
        n_ep = len(tbr.episodes(f))
        row = f"  {nm:10s} {n_ep:8d} {n_ep/years*10:11.1f} {(f > 0).mean()*100:8.1f}% {f[f > 0].mean()*size*100 if (f > 0).any() else 0.0:6.1f}%"
        for lo, hi in list(RECOVERIES.values()) + list(BEAR_RALLIES.values()):
            m = day_mask(days, lo, hi)
            row += f" {period_return(d, m) - period_return(base, m):+9.2f}"
        row += f" {period_return(d, np.ones(n_days, dtype=bool)) - period_return(base, np.ones(n_days, dtype=bool)):+9.1f}"
        print(row)
    print("  (columns after 'avg w%' are override return minus base return in pp; last column is the whole real window)")
    print()

    # ---- windows ------------------------------------------------------------
    print("=== RESULTS BY WINDOW (CAGR%, max DD%, xSharpe)")
    stats = {}
    for wname, (lo, hi) in WINDOWS.items():
        m = day_mask(days, lo, hi)
        stats[wname] = {"BASE": mb.perf_stats(base[m], rc[m])}
        for nm, d in variants.items():
            stats[wname][nm] = mb.perf_stats(d[m], rc[m])
    for wname in ("design", "confirm"):
        print(f"--- {wname} {WINDOWS[wname][0]}..{WINDOWS[wname][1]}")
        b = stats[wname]["BASE"]
        print(f"  {'rule':10s} {'CAGR%':>7s} {'maxDD%':>7s} {'xSh':>6s} | {'dCAGR':>6s} {'dDD':>6s} {'dxSh':>6s}")
        for nm, s in stats[wname].items():
            print(f"  {nm:10s} {s['cagr_pct']:+7.1f} {s['max_dd_pct']:+7.1f} {s['xsharpe']:+6.2f} | "
                  f"{s['cagr_pct'] - b['cagr_pct']:+6.1f} {s['max_dd_pct'] - b['max_dd_pct']:+6.1f} {s['xsharpe'] - b['xsharpe']:+6.2f}")
    print()

    # ---- gate (comparability with T1 only) ----------------------------------
    print(f"=== GATE, same as T1 (design + confirm): xSharpe >= base and CAGR >= base + {CAGR_MARGIN}pp in both; "
          f"max DD no worse than base by > {DD_MARGIN}pp; bear-rally sum > {BEAR_RALLY_LIMIT}pp")
    finalists = []
    for nm in variants:
        bear_sum = sum(period_return(variants[nm], day_mask(days, lo, hi)) - period_return(base, day_mask(days, lo, hi))
                       for lo, hi in BEAR_RALLIES.values())
        c1 = all(stats[w][nm]["xsharpe"] >= stats[w]["BASE"]["xsharpe"]
                 and stats[w][nm]["cagr_pct"] >= stats[w]["BASE"]["cagr_pct"] + CAGR_MARGIN for w in ("design", "confirm"))
        c2 = all(stats[w][nm]["max_dd_pct"] >= stats[w]["BASE"]["max_dd_pct"] - DD_MARGIN for w in ("design", "confirm"))
        c3 = bear_sum > BEAR_RALLY_LIMIT
        ok = c1 and c2 and c3
        if ok:
            finalists.append(nm)
        print(f"  {nm:10s} return/xSh {'PASS' if c1 else 'fail':4s} | drawdown {'PASS' if c2 else 'fail':4s} | "
              f"bear rallies {'PASS' if c3 else 'fail':4s} -> {'FINALIST' if ok else 'not a finalist'}")
    print()

    print("=== HOLDOUT 2023+ (evaluated once, finalists only)")
    if not finalists:
        print("  no rule passed the design + confirm gate; the holdout was not evaluated for any rule.\n")
    else:
        b = stats["holdout"]["BASE"]
        print(f"  BASE: CAGR {b['cagr_pct']:+.1f}% maxDD {b['max_dd_pct']:+.1f}% xSharpe {b['xsharpe']:+.2f}")
        for nm in finalists:
            s = stats["holdout"][nm]
            print(f"  {nm}: CAGR {s['cagr_pct']:+.1f}% maxDD {s['max_dd_pct']:+.1f}% xSharpe {s['xsharpe']:+.2f} "
                  f"-> xSharpe not worse than base: {s['xsharpe'] >= b['xsharpe']}")
        print()

    # ---- stability of the wait parameter -----------------------------------
    # A gate PASS on three events means nothing if the parameter that produced it is a spike. T2 required the
    # deployed VXN thresholds to sit on a plateau (within 0.05 xSharpe of their neighbours' median); the same
    # question has to be asked here before any configuration is taken seriously.
    print("=== WAIT-PARAMETER STABILITY (full size, no ramp) — is any passing configuration on a plateau?")
    print(f"  {'wait':>5s} {'eps':>4s} {'whole pp':>9s} {'xSh':>6s} " + " ".join(f"{y:>7d}" for y in (2009, 2020, 2021, 2022, 2023)))
    allm = np.ones(n_days, dtype=bool)
    base_whole = period_return(base, allm)
    for w in STABILITY_WAITS:
        f = tbr.override_fractions(base_cash, close_d, w, 1)
        d = run(tbr.bar_weights(tiers, inp.day_codes, f, 1.0))
        yr = " ".join(f"{period_return(d, days.year == y) - period_return(base, days.year == y):+7.1f}" for y in (2009, 2020, 2021, 2022, 2023))
        print(f"  {w:5d} {len(tbr.episodes(f)):4d} {period_return(d, allm) - base_whole:+9.1f} {mb.perf_stats(d, rc)['xsharpe']:+6.2f} {yr}")
    print(f"  BASE xSharpe {mb.perf_stats(base, rc)['xsharpe']:+.2f} over the whole real window")
    print()

    # ---- year by year -------------------------------------------------------
    print("=== DESCRIPTIVE ONLY: year-by-year override return minus base return, pp")
    names = list(variants)
    print(f"  {'year':>5s} " + " ".join(f"{n:>8s}" for n in names))
    for y in sorted(set(days.year)):
        m = days.year == y
        bret = period_return(base, m)
        row = " ".join(f"{period_return(variants[n], m) - bret:+8.1f}" for n in names)
        print(f"  {y:>5d} " + row)


if __name__ == "__main__":
    main()
