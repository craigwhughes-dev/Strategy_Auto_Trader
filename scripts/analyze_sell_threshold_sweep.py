"""Summarise item #3 of the exit-parameter audit: sell_threshold sweep
({-6.0, -4.5 (current), -3.0, -1.5}, buy_threshold fixed at 6.0) from
run_exit_param_audit.ps1.

Same shape as analyze_vol_stop_mult_sweep.py — return/max-DD/admitted per
window + exit_reason breakdown. sell_threshold directly gates the composite-
signal SELL path (currently_in=True), so its exit_reason effect should show
up most clearly in the 'signal' category's count and pnl.

Usage:
    uv run python scripts/analyze_sell_threshold_sweep.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LABELS = ["st6", "st45", "st3", "st15"]
VALUE_MAP = {"st6": "-6.0", "st45": "-4.5 (current)", "st3": "-3.0", "st15": "-1.5"}
BASELINE_LABEL = "st45"
PREFIX = "sell_thresh_sweep"

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
    print(f"\n=== sell_threshold Sweep - {title} ===\n")
    print(f"{'Label':<8} {'sell_threshold':>16} {'Return':>10} {'Max DD':>10} {'Admitted':>10}")
    print("-" * 60)

    rows = {}
    for label in LABELS:
        row = load_summary(label, window)
        if row is None:
            print(f"{label:<8} {'N/A':>16}")
            continue
        initial = 100_000.0
        final = float(row.get("cash", initial))
        ret = (final - initial) / initial
        max_dd = float(row.get("max_drawdown", 0.0))
        admitted = int(row.get("n_admitted", 0))
        rows[label] = {"ret": ret, "max_dd": max_dd, "admitted": admitted}
        print(f"{label:<8} {VALUE_MAP[label]:>16} {pct(ret):>10} {pct(max_dd):>10} {admitted:>10}")

    print(f"\n--- exit_reason breakdown, {title} ---")
    for label in LABELS:
        bd = exit_reason_breakdown(label, window)
        if bd is None:
            continue
        print(f"\n  {label} (sell_threshold={VALUE_MAP[label]}):")
        for cat, r in bd.sort_values("pnl").iterrows():
            print(f"    {cat:<15} n={int(r['n']):>4}  pnl=£{r['pnl']:>12,.2f}")

    return rows


def main() -> None:
    crash_rows = print_window_table("synthetic", "Crash window: synthetic Jan2008-Jul2009")
    real_rows = print_window_table("real", "Normal window: real prev-2yr (2024-09-03-present)")

    if BASELINE_LABEL in crash_rows and BASELINE_LABEL in real_rows:
        print("\n=== Delta vs st45 (current, -4.5) ===\n")
        print(f"{'Label':<8} {'sell_threshold':>16} {'Crash ret d':>14} {'Crash DD d':>12} {'Real ret d':>12}")
        print("-" * 68)
        b_crash = crash_rows[BASELINE_LABEL]
        b_real = real_rows[BASELINE_LABEL]
        for label in LABELS:
            if label == BASELINE_LABEL:
                continue
            c = crash_rows.get(label)
            r = real_rows.get(label)
            if c is None or r is None:
                print(f"{label:<8} {'N/A':>16}")
                continue
            d_crash_ret = c["ret"] - b_crash["ret"]
            d_crash_dd = c["max_dd"] - b_crash["max_dd"]
            d_real_ret = r["ret"] - b_real["ret"]
            print(f"{label:<8} {VALUE_MAP[label]:>16} {pct(d_crash_ret):>14} {pct(d_crash_dd):>12} {pct(d_real_ret):>12}")


if __name__ == "__main__":
    main()
