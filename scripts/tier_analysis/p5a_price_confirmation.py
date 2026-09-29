"""P5 first cut, Test A: does a causal price confirmation signal mark recoveries and not the way down?

Run: uv run python scripts/tier_analysis/p5a_price_confirmation.py

PRE-REGISTERED BEFORE RUNNING (PLAN_CRASH_PHASES.md, and logged in BACKTEST_LOG.md):

  Indicator     P = close above a *rising* 50-day SMA (fresh crossings only), per market, causal.
  Unit          the shock, not the episode. Per-market flags averaged within a shock, then across
                shocks, so 2008 in twelve markets is one observation, not twelve.
  Statistic     edge = hit rate - false-alarm rate.
                hit         = fires within `window` trading days after the trough.
                false alarm = fires earlier in the episode at a point from which the market still
                              fell >= 10% before bottoming.
  Split         design  = shocks containing any trough from 2007-01-01 (contaminated by T1, T9 and
                          the realized-vol sweep, all run on the 2007+ window).
                holdout = shocks entirely before 2007. Scored ONCE, all 9 variations together.
  Trials        9 = min_depth {15%, 20%, 25%} x window {21, 42, 63} bars. Bonferroni alpha = 0.05/9
                = 0.00556.
  Criteria      primary   holdout edge > 0.30 with permutation p < 0.00556
                secondary hit rate >= 0.70 and false-alarm rate <= 0.333
                stability holds across all 9 variations
                baseline  must beat the causal elapsed-time baseline (T9's mechanism)
  Null          circular rotation of the firing pattern within each episode, 10,000 draws. Preserves
                the indicator's own firing count and spacing; destroys only where they sit.

NOTE ON THE PRE-REGISTERED GRID. The plan's original 9 variations were min_depth x confirm_retrace.
`confirm_retrace` has no effect on this test, which uses only the peak and the trough, so sweeping
it would be theatre. It is replaced by the hit `window`, which the test does depend on. Same trial
count, same Bonferroni denominator; the substitution is recorded in BACKTEST_LOG.md.

Test B (credit, HY OAS) is registered separately and is blocked: keyless FRED serves only ~3 years
and ignores every range parameter, so the series cannot be obtained at all without an API key.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, so the package imports work when run as a script

import numpy as np
import pandas as pd

from Strategy_Auto_Trader.research import confirmation_indicators as ci
from Strategy_Auto_Trader.research import crash_phases as cp
from Strategy_Auto_Trader.research import index_history as ih
from Strategy_Auto_Trader.research import phase_scoring as ps

DEPTHS = (0.15, 0.20, 0.25)
WINDOWS = (21, 42, 63)
DESIGN_FROM = pd.Timestamp("2007-01-01", tz="UTC")
POST_TROUGH_DAYS = 60
BASELINE_WAIT = 126
ALPHA = 0.05 / 9


def build(closes: dict[str, pd.Series], min_depth: float):
    """Episodes across all markets, grouped into shocks, with each market's firing array attached."""
    frames, eps_by_market = [], {}
    for sym, close in closes.items():
        eps = cp.find_episodes(close, min_depth=min_depth, post_trough_days=POST_TROUGH_DAYS)
        eps_by_market[sym] = eps
        if eps:
            frames.append(pd.DataFrame({
                "market": sym,
                "idx": range(len(eps)),
                "trough_date": [close.index[e.trough_idx] for e in eps],
            }))
    reg = cp.group_into_shocks(pd.concat(frames, ignore_index=True))
    shock_first = reg.groupby("shock_id")["trough_date"].min()
    design_shocks = set(shock_first[shock_first >= DESIGN_FROM].index)
    return reg, eps_by_market, design_shocks


def rows_for(reg, eps_by_market, closes, firings, window, min_depth):
    """Per-(market, episode) scored rows plus the per-episode payload the permutation test needs."""
    rows, payload = [], []
    for r in reg.itertuples():
        ep = eps_by_market[r.market][r.idx]
        close = closes[r.market]
        sc = ps.score_episode(firings[r.market], close, ep, hit_window=window)
        rows.append({"shock_id": r.shock_id, "market": r.market, "hit": sc.hit,
                     "false_alarm": sc.false_alarm, "n_firings": sc.n_firings})
        payload.append((firings[r.market], close, ep, r.shock_id))
    return pd.DataFrame(rows), payload


def main() -> None:
    closes = {}
    for sym in ih.MARKETS:
        try:
            closes[sym] = ih.load_daily_closes(sym)
        except Exception as exc:
            print(f"  WARNING {sym} unavailable: {type(exc).__name__}")
    print(f"=== TEST A — price confirmation (indicator P), {len(closes)} markets, local currency")
    print(f"    pre-registered: 9 trials, Bonferroni alpha = {ALPHA:.5f}; holdout scored once")
    print()

    firings_P = {sym: ci.price_above_rising_sma(c) for sym, c in closes.items()}

    print(f"  {'depth':>6s} {'win':>4s} {'set':>8s} {'shocks':>7s} {'hit':>6s} {'FA':>6s} {'edge':>7s} "
          f"{'p':>9s} {'null med':>9s} {'base edge':>9s}")
    results = []
    for min_depth in DEPTHS:
        reg, eps_by_market, design_shocks = build(closes, min_depth)
        base_fire = {sym: ci.elapsed_time_baseline(c, min_depth, BASELINE_WAIT) for sym, c in closes.items()}
        for window in WINDOWS:
            all_rows, all_payload = rows_for(reg, eps_by_market, closes, firings_P, window, min_depth)
            base_rows, _ = rows_for(reg, eps_by_market, closes, base_fire, window, min_depth)
            for label, keep in (("design", True), ("holdout", False)):
                mask = all_rows["shock_id"].isin(design_shocks) == keep
                sub, bsub = all_rows[mask], base_rows[mask]
                sc = ps.aggregate(sub)
                bsc = ps.aggregate(bsub)
                payload = [p for p, m in zip(all_payload, mask) if m]
                p_val, null = ps.permutation_p_value(sc.edge, payload, hit_window=window)
                print(f"  {min_depth * 100:5.0f}% {window:4d} {label:>8s} {sc.n_shocks:7d} "
                      f"{sc.hit_rate:6.3f} {sc.false_alarm_rate:6.3f} {sc.edge:+7.3f} "
                      f"{p_val:9.5f} {np.median(null):+9.3f} {bsc.edge:+9.3f}")
                results.append({"depth": min_depth, "window": window, "set": label, "shocks": sc.n_shocks,
                                "hit": sc.hit_rate, "fa": sc.false_alarm_rate, "edge": sc.edge,
                                "p": p_val, "base_edge": bsc.edge})
    print()

    res = pd.DataFrame(results)
    hold = res[res["set"] == "holdout"]
    print("=== CRITERIA, holdout only (all 9 variations, scored in one pass)")
    prim = (hold["edge"] > 0.30) & (hold["p"] < ALPHA)
    sec = (hold["hit"] >= 0.70) & (hold["fa"] <= 0.3333)
    beats = hold["edge"] > hold["base_edge"]
    print(f"  primary   (edge > 0.30 and p < {ALPHA:.5f}):  {int(prim.sum())} of 9 variations")
    print(f"  secondary (hit >= 0.70 and FA <= 0.333):    {int(sec.sum())} of 9")
    print(f"  beats the causal elapsed-time baseline:     {int(beats.sum())} of 9")
    print(f"  stability (primary in ALL 9):               {'PASS' if bool(prim.all()) else 'FAIL'}")
    print(f"  edge range across variations: {hold['edge'].min():+.3f} .. {hold['edge'].max():+.3f}")
    print()
    verdict = "PASS" if bool(prim.all() and sec.all() and beats.all()) else "FAIL"
    print(f"  *** TEST A VERDICT: {verdict} ***")
    if verdict == "FAIL":
        print("  Per the plan: stop. Do not iterate on the indicator set.")


if __name__ == "__main__":
    main()
