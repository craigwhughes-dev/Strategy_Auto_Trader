"""P0 precheck for PLAN_CRASH_PHASES.md: how much of the recovery miss is recoverable at all?

Run: uv run python scripts/tier_analysis/p0_oracle_lag_curve.py

THIS IS NOT A STRATEGY AND CANNOT BE TRADED. It is given the drawdown trough by hindsight
and re-enters Nasdaq `lag` trading days after it, holding until the base VXN 23/24 rule takes
over on its own. Every number below is therefore an upper bound that no causal detector can
reach, not a candidate result. Reporting it as a strategy would be a look-ahead error.

Why it is worth computing. The plan's P5 pass criterion (median absolute error <= 15 trading
days from a phase boundary) was chosen for statistical convenience, not from any economic
requirement. This curve supplies the missing calibration: it shows what a detector that is
late by L days can still recover of the 2009 and 2020 gaps, so the day tolerance can be set
from the pp requirement instead of guessed. If the curve is already flat and small at L=0,
the detection question is not worth answering and the plan should stop before P1.

Risk columns follow T4's convention: the owner's stated 15% loss tolerance governs the **worst
k-day loss while invested** (T4 used 1/3/5 days and noted max DD is the less clean read of it).
The accepted max drawdown is about -24%, agreed when the VXN 23/24 rule was kept on 2026-09-20.
Do not compare a max-DD figure against the 15% number; they are different quantities.

Real data only (2007-11-20 onward), 13 bps per switch, same engine and cost as T1/T9 so the
numbers are directly comparable. Episode detection runs on the full spliced series so that
the 2007 Nasdaq peak (31 Oct 2007, just before the real window opens) is located correctly;
that peak date rests on bridged data and is used only to bound the drawdown, never scored.
Only episodes whose trough falls inside the real window are used.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, so the package imports work when run as a script

import numpy as np
import pandas as pd

from Strategy_Auto_Trader.allocation import intraday_engine as eng
from Strategy_Auto_Trader.allocation import matched_blend as mb  # perf_stats only
from Strategy_Auto_Trader.allocation import reentry_override as ro  # simulate_weighted only
from Strategy_Auto_Trader.allocation.real_window import last_bar_per_day, slice_from
from Strategy_Auto_Trader.research import crash_phases as cp

REAL_START = "2007-11-20"
COST_BPS = 13.0
LAGS = (0, 5, 10, 15, 20, 30, 40, 60, 90)
SIZES = (1.0, 0.5)
RECOVERY_YEARS = (2009, 2020)
BEAR_RALLY_YEARS = (2008, 2022)


def period_return(daily: np.ndarray, m: np.ndarray) -> float:
    return float(np.prod(1 + daily[m]) - 1) * 100


def rolling_worst(daily: np.ndarray, k: int) -> float:
    """Worst k-day compounded return (%). T4's measure, which is what the owner's 15% loss tolerance governs."""
    eq = np.cumprod(1.0 + daily)
    if len(eq) <= k:
        return float((eq[-1] - 1.0) * 100.0)
    return float((eq[k:] / eq[:-k] - 1.0).min() * 100.0)


def oracle_active(base_cash: np.ndarray, trough_pos: int, lag: int) -> np.ndarray:
    """Days on which the oracle holds Nasdaq: from trough+lag until the base rule re-enters by itself."""
    n = len(base_cash)
    active = np.zeros(n, dtype=bool)
    start = trough_pos + lag
    if start >= n:
        return active
    stop = start
    while stop < n and base_cash[stop]:
        stop += 1
    active[start:stop] = True
    return active


def main() -> None:
    full = eng.load_inputs()
    inp = slice_from(full, REAL_START)
    days = inp.days
    n_days = len(days)
    rc = eng.buy_and_hold(inp, "CASH").daily

    last_full = last_bar_per_day(full.day_codes, len(full.days))
    close_full = pd.Series(np.exp(np.cumsum(full.log_ret[:, 0]))[last_full], index=full.days)
    pos = full.days.get_indexer(days)
    assert (pos >= 0).all()

    tiers = eng.tiers_vxn_deadband(inp.vxn, 23.0, 24.0)
    last = last_bar_per_day(inp.day_codes, n_days)
    base_cash = tiers[last] == 3

    def run(active: np.ndarray, size: float) -> np.ndarray:
        weights = ro.bar_weights(tiers, inp.day_codes, active, size)
        return ro.simulate_weighted(inp.log_ret, weights, inp.day_codes, n_days, COST_BPS)

    base = run(np.zeros(n_days, dtype=bool), 1.0)
    nasdaq = eng.buy_and_hold(inp, "NASDAQ").daily
    allm = np.ones(n_days, dtype=bool)

    # ---- episodes ------------------------------------------------------------
    eps = cp.find_episodes(close_full, min_depth=cp.MIN_DEPTH, post_trough_days=60)
    in_window = [e for e in eps if full.days[e.trough_idx] >= days[0]]
    print(f"=== EPISODES on Nasdaq daily close (>={cp.MIN_DEPTH * 100:.0f}% drawdown, trough >=60 bars before the data ends)")
    print("    NB depth is measured on EQQQ in GBP, unhedged — sterling moves compress it against the USD index")
    print(f"  {'peak':>10s} {'trough':>10s} {'recovery':>10s} {'depth%':>7s} {'t->rec':>7s}  in real window")
    for e in eps:
        rec = full.days[e.recovery_idx].date() if e.recovery_idx is not None else None
        t2r = (e.recovery_idx - e.trough_idx) if e.recovery_idx is not None else -1
        flag = "yes" if e in in_window else "no (trough pre-2007-11-20, bridged)"
        print(
            f"  {full.days[e.peak_idx].date()!s:>10s} {full.days[e.trough_idx].date()!s:>10s} "
            f"{str(rec):>10s} {e.depth_pct * 100:7.1f} {t2r:7d}  {flag}"
        )
    print(f"  usable episodes with trough inside the real window: {len(in_window)}")
    print()

    # ---- the gap being closed ------------------------------------------------
    print("=== THE GAP, measured on the real window (base rule vs Nasdaq, calendar-year return %)")
    print(f"  {'year':>5s} {'base':>8s} {'nasdaq':>8s} {'gap pp':>8s}")
    for y in RECOVERY_YEARS:
        m = days.year == y
        b, nd = period_return(base, m), period_return(nasdaq, m)
        print(f"  {y:5d} {b:+8.1f} {nd:+8.1f} {b - nd:+8.1f}")
    print()

    # ---- the curve -----------------------------------------------------------
    for size in SIZES:
        print(f"=== ORACLE LAG CURVE, size {size:.2f} — pp added to the base rule (UPPER BOUND, uses hindsight)")
        head = " ".join(f"{y:>9d}" for y in RECOVERY_YEARS) + " " + " ".join(f"{y:>9d}" for y in BEAR_RALLY_YEARS)
        print(f"  {'lag':>4s} {'days held':>9s} {head} {'whole pp':>9s} {'xSh':>6s} {'maxDD%':>7s} {'w1d%':>6s} {'w3d%':>6s} {'w5d%':>6s}")
        base_whole = period_return(base, allm)
        base_dd = mb.perf_stats(base, rc)["max_dd_pct"]
        troughs = [int(np.flatnonzero(pos == e.trough_idx)[0]) for e in in_window]
        for lag in LAGS:
            active = np.zeros(n_days, dtype=bool)
            for tpos in troughs:
                active |= oracle_active(base_cash, tpos, lag)
            d = run(active, size)
            st = mb.perf_stats(d, rc)
            yrs = " ".join(
                f"{period_return(d, days.year == y) - period_return(base, days.year == y):+9.1f}"
                for y in RECOVERY_YEARS + BEAR_RALLY_YEARS
            )
            print(
                f"  {lag:4d} {int(active.sum()):9d} {yrs} "
                f"{period_return(d, allm) - base_whole:+9.1f} {st['xsharpe']:+6.2f} {st['max_dd_pct']:7.1f} "
                f"{rolling_worst(d, 1):+6.1f} {rolling_worst(d, 3):+6.1f} {rolling_worst(d, 5):+6.1f}"
            )
        print(
            f"  BASE: whole-window {base_whole:+.1f}%, xSharpe {mb.perf_stats(base, rc)['xsharpe']:+.2f}, "
            f"max DD {base_dd:.1f}%, worst 1d/3d/5d {rolling_worst(base, 1):.1f}/{rolling_worst(base, 3):.1f}/{rolling_worst(base, 5):.1f}%"
        )
        print()

    # ---- per-episode decay ---------------------------------------------------
    # Measured episode-relative, not by calendar year. The GFC trough is 2008-10-16 (EQQQ is
    # GBP-unhedged, so it bottomed with sterling in Oct 2008, not with the USD index in Mar 2009),
    # so a calendar-year split puts every lag difference in 2008 and reports a flat 2009.
    print("=== PER-EPISODE DECAY, size 1.00 - over the override's own span, not the calendar year")
    print("    span = trough+lag until the base rule re-enters Nasdaq by itself")
    for e in in_window:
        tpos = int(np.flatnonzero(pos == e.trough_idx)[0])
        if not oracle_active(base_cash, tpos, 0).any():
            continue
        print(f"  trough {full.days[e.trough_idx].date()}  depth {e.depth_pct * 100:.1f}%")
        print(f"    {'lag':>4s} {'held':>5s} {'from':>11s} {'to':>11s} {'oracle%':>8s} {'base%':>8s} {'pp added':>9s} {'% of L0':>8s}")
        at_zero = None
        for lag in LAGS:
            active = oracle_active(base_cash, tpos, lag)
            if not active.any():
                continue
            d = run(active, 1.0)
            o_ret, b_ret = period_return(d, active), period_return(base, active)
            added = o_ret - b_ret
            at_zero = added if at_zero is None else at_zero
            span = np.flatnonzero(active)
            print(
                f"    {lag:4d} {int(active.sum()):5d} {days[span[0]].date()!s:>11s} {days[span[-1]].date()!s:>11s} "
                f"{o_ret:+8.1f} {b_ret:+8.1f} {added:+9.1f} {added / at_zero * 100 if at_zero else float('nan'):+8.0f}"
            )
        print()

if __name__ == "__main__":
    main()
