"""Summarise item #5 of the exit-parameter audit: profit_stop_scale/
min_stop_pct full-universe validation (current 0.30/0.03 vs off 0.0/0.04,
optimised's original) from run_exit_param_audit.ps1.

The module docstring in optimised_new.py flags the current values as
validated only on a single ticker (AAPL, Sharpe 1.96 vs 1.21) — never
generalized across the universe. This closes that gap.

Usage:
    uv run python scripts/analyze_ratchet_sweep.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LABELS = ["ratchet_current", "ratchet_off"]
VALUE_MAP = {
    "ratchet_current": "pss=0.30, msp=0.03 (current)",
    "ratchet_off": "pss=0.00, msp=0.04 (off, optimised's original)",
}
BASELINE_LABEL = "ratchet_current"
PREFIX = "ratchet_sweep"

JOURNALS_DIR = Path("data/journals")


def load_summary(label: str, window: str) -> dict | None:
    path = JOURNALS_DIR / f"{PREFIX}_{label}_{window}_equity.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    summary = df[df["date"] == "SUMMARY"]
    if summary.empty:
        return None
    return summary.iloc[0].to_dict()


def exit_reason_breakdown(label: str, window: str) -> pd.DataFrame | None:
    path = JOURNALS_DIR / f"{PREFIX}_{label}_{window}.csv"
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    df["cat"] = df["exit_reason"].astype(str).str.split("(").str[0].str.strip()
    return df.groupby("cat").agg(n=("pnl_usd", "count"), pnl=("pnl_usd", "sum"))


def pct(val: float) -> str:
    return f"{val * 100:+.1f}%"


def print_window_table(window: str, title: str) -> dict:
    print(f"\n=== profit_stop_scale/min_stop_pct Sweep - {title} ===\n")
    print(f"{'Label':<16} {'Config':>34} {'Return':>10} {'Max DD':>10} {'Admitted':>10}")
    print("-" * 82)

    rows = {}
    for label in LABELS:
        row = load_summary(label, window)
        if row is None:
            print(f"{label:<16} {'N/A':>34}")
            continue
        initial = 100_000.0
        final = float(row.get("cash", initial))
        ret = (final - initial) / initial
        max_dd = float(row.get("max_drawdown", 0.0))
        admitted = int(row.get("n_admitted", 0))
        rows[label] = {"ret": ret, "max_dd": max_dd, "admitted": admitted}
        print(f"{label:<16} {VALUE_MAP[label]:>34} {pct(ret):>10} {pct(max_dd):>10} {admitted:>10}")

    print(f"\n--- exit_reason breakdown, {title} ---")
    for label in LABELS:
        bd = exit_reason_breakdown(label, window)
        if bd is None:
            continue
        print(f"\n  {label}:")
        for cat, r in bd.sort_values("pnl").iterrows():
            print(f"    {cat:<15} n={int(r['n']):>4}  pnl=£{r['pnl']:>12,.2f}")

    return rows


def main() -> None:
    crash_rows = print_window_table("synthetic", "Crash window: synthetic Jan2008-Jul2009")
    real_rows = print_window_table("real", "Normal window: real prev-2yr (2024-09-03-present)")

    if BASELINE_LABEL in crash_rows and BASELINE_LABEL in real_rows:
        print("\n=== Delta vs ratchet_current ===\n")
        b_crash = crash_rows[BASELINE_LABEL]
        b_real = real_rows[BASELINE_LABEL]
        c = crash_rows.get("ratchet_off")
        r = real_rows.get("ratchet_off")
        if c and r:
            d_crash_ret = c["ret"] - b_crash["ret"]
            d_crash_dd = c["max_dd"] - b_crash["max_dd"]
            d_real_ret = r["ret"] - b_real["ret"]
            print(f"  ratchet_off vs ratchet_current: crash ret {pct(d_crash_ret)}, "
                  f"crash DD {pct(d_crash_dd)}, real ret {pct(d_real_ret)}")


if __name__ == "__main__":
    main()
