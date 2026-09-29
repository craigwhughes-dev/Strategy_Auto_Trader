"""X1 driver: proxy-fidelity hard gate for PLAN_EXIT_GENERALISATION.md.

Run:  uv run python scripts/tier_analysis/x1_proxy_fidelity.py

Calibrates a realized-vol deadband to the deployed VXN 23/24 rule on 2001-2012 (thresholds
determined, not fitted), then measures exit timing on the unseen 2013-2026 half against the
post-2013 Nasdaq episodes. Trials = 3 (vol window 10/21/63). Gate is exit-timing only.

GATE (fixed pre-run, reframed by the 2026-09-29 panel): for at least one window, in >=80% of the
post-2013 ^IXIC episodes the proxy exit PRECEDES the trough AND is within +/-10 trading days of the
deployed rule's exit. If no window clears, STOP: the exit is not testable on free pre-2001 data.

Research only. Reads the deployed deadband; changes no engine/production code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from Strategy_Auto_Trader.allocation.intraday_engine import _CASH
from Strategy_Auto_Trader.research import proxy_fidelity as PF
from Strategy_Auto_Trader.research.crash_phases import find_episodes
from Strategy_Auto_Trader.research.index_history import cache_path

VXN_CSV = Path("data_synthetic/vxn_daily_fred_2001_2026.csv")
WINDOWS = [10, 21, 63]
TRAIN_END = pd.Timestamp("2013-01-01", tz="UTC")   # calibration = 2001-2012; measurement = 2013+
GATE_TOL_BARS = 10
GATE_MIN_FRAC = 0.80


def _load_ixic() -> pd.Series:
    df = pd.read_csv(cache_path("^IXIC"), index_col=0, comment="#")
    idx = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True)).normalize()
    return pd.Series(df["close"].to_numpy(dtype=float), index=idx, name="^IXIC").sort_index()


def _load_vxn() -> pd.Series:
    df = pd.read_csv(VXN_CSV)
    idx = pd.DatetimeIndex(pd.to_datetime(df["observation_date"], utc=True)).normalize()
    return pd.Series(df["VXNCLS"].to_numpy(dtype=float), index=idx, name="VXN").sort_index()


def main() -> None:
    ixic = _load_ixic()
    vxn_raw = _load_vxn()

    # Common daily grid: ^IXIC trading days from VXN's start, VXN ffilled onto it (as the live rule reads it).
    start = vxn_raw.index.min()
    days = ixic.index[ixic.index >= start]
    close = ixic.reindex(days)
    vxn = vxn_raw.reindex(days, method="ffill")

    train_mask = np.asarray(days < TRAIN_END)
    test_mask = ~train_mask
    print(f"grid: {days.min().date()}..{days.max().date()} | train(2001-2012) {train_mask.sum()}d | test(2013+) {test_mask.sum()}d")

    deployed_state = PF.deadband_state(vxn.to_numpy(), 23.0, 24.0)

    # Post-2013 ^IXIC episodes (troughs from 2013), the gate's unit of observation.
    eps = [e for e in find_episodes(ixic, post_trough_days=252)
           if ixic.index[e.trough_idx].year >= 2013]
    episodes = [(ixic.index[e.peak_idx], ixic.index[e.trough_idx], round(e.depth_pct * 100, 1)) for e in eps]
    print(f"post-2013 ^IXIC episodes: {len(episodes)}")

    vxn_dep_cash = float(np.mean(deployed_state[train_mask] == _CASH))
    print(f"deployed in-cash fraction, train: {vxn_dep_cash:.3f}\n")

    any_pass = False
    window_summaries = []
    for w in WINDOWS:
        rv = PF.realized_vol(close, w).to_numpy()
        lo, hi = PF.calibrate_thresholds(rv, vxn.to_numpy(), train_mask)
        proxy_state = PF.deadband_state(rv, lo, hi)

        proxy_train_cash = float(np.mean(proxy_state[train_mask] == _CASH))
        agree_test = float(np.mean(deployed_state[test_mask] == proxy_state[test_mask]))

        print(f"{'='*72}\nWINDOW {w}d | lo(enter RV)={lo:.2f}  hi(exit RV)={hi:.2f}")
        print(f"  train in-cash: deployed {vxn_dep_cash:.3f} vs proxy {proxy_train_cash:.3f} (matched by construction)")
        print(f"  [descriptive] daily state agreement, 2013+: {agree_test:.3f}")

        rows = []
        n_pass = 0
        for peak, trough, depth in episodes:
            t = PF.episode_timing(deployed_state, proxy_state, days, peak, trough)
            dep_exit = days[t.deployed_exit_pos].date() if t.deployed_exit_pos is not None else None
            px_exit = days[t.proxy_exit_pos].date() if t.proxy_exit_pos is not None else None
            passed = t.passes(GATE_TOL_BARS)
            n_pass += passed
            dep_lag = None if t.deployed_exit_pos is None else t.deployed_exit_pos - t.peak_pos
            px_lag = None if t.proxy_exit_pos is None else t.proxy_exit_pos - t.peak_pos
            rows.append({
                "peak": peak.date(), "trough": trough.date(), "depth%": depth,
                "dep_exit": dep_exit, "dep_lag_bars": dep_lag,
                "proxy_exit": px_exit, "proxy_lag_bars": px_lag,
                "px_precedes_trough": t.proxy_precedes_trough,
                "px_vs_dep_bars": t.lag_bars, "PASS": passed,
            })
        frac = n_pass / len(episodes) if episodes else 0.0
        cleared = frac >= GATE_MIN_FRAC
        any_pass = any_pass or cleared
        window_summaries.append((w, lo, hi, agree_test, n_pass, len(episodes), cleared))

        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(pd.DataFrame(rows).to_string(index=False))
        print(f"  --> episodes passing gate: {n_pass}/{len(episodes)} ({frac:.0%})  window clears: {cleared}\n")

    print(f"{'='*72}\nX1 GATE VERDICT\n{'='*72}")
    for w, lo, hi, ag, np_, nt, cleared in window_summaries:
        print(f"  window {w:>2}d: lo={lo:.2f} hi={hi:.2f} agree={ag:.3f} pass={np_}/{nt} {'CLEARS' if cleared else 'no'}")
    if any_pass:
        print("\nGATE PASSES. At least one window reproduces the deployed exit timing out of sample.")
        print("Proceed to X2 (pending owner ratification of X2 metric thresholds).")
    else:
        print("\nGATE FAILS. No window reproduces the deployed exit timing on 2013-2026.")
        print("STOP per plan: a backward-looking proxy cannot stand in for VXN; the exit's")
        print("generalisation is not testable on free pre-2001 data. Do not widen the tolerance.")


if __name__ == "__main__":
    main()
