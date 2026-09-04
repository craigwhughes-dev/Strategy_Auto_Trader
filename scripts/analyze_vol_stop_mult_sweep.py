"""Summarise the vol_stop_mult sweep results from run_vol_stop_mult_sweep.ps1.

Reads per-(label, window) equity + trade-journal CSVs from data/journals/ and
prints return/max-DD/admitted (mirrors analyze_vix_gate_sweep.py's shape)
PLUS an exit_reason breakdown (rr_stop_loss / trailing_stop / signal — count
and total pnl) per label/window. The breakdown is the actual mechanism check:
item #2 of the exit-parameter audit exists because the vol-scaled trailing
stop widens during high-vol periods (effective_stop = vol_stop_mult *
realised_vol * sqrt(vol_stop_window)) — a tighter multiplier is only a real
fix if trailing_stop's share of exits/losses grows and rr_stop_loss's share
shrinks, not just if headline return moves.

mult20 (2.0) is the current/baseline value — deltas are reported against it.

Usage:
    uv run python scripts/analyze_vol_stop_mult_sweep.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LABELS = ["mult10", "mult15", "mult20", "mult25"]
VALUE_MAP = {"mult10": "1.0", "mult15": "1.5", "mult20": "2.0 (current)", "mult25": "2.5"}
BASELINE_LABEL = "mult20"

JOURNALS_DIR = Path("data/journals")


def load_summary(label: str, window: str) -> dict | None:
    path = JOURNALS_DIR / f"vol_stop_mult_sweep_{label}_{window}_equity.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    summary = df[df["date"] == "SUMMARY"]
    if summary.empty:
        return None
    return summary.iloc[0].to_dict()


def exit_reason_breakdown(label: str, window: str) -> pd.DataFrame | None:
    path = JOURNALS_DIR / f"vol_stop_mult_sweep_{label}_{window}.csv"
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
    print(f"\n=== vol_stop_mult Sweep — {title} ===\n")
    print(f"{'Label':<10} {'vol_stop_mult':>14} {'Return':>10} {'Max DD':>10} {'Admitted':>10}")
    print("-" * 60)

    rows = {}
    for label in LABELS:
        row = load_summary(label, window)
        if row is None:
            print(f"{label:<10} {'N/A':>14}")
            continue
        initial = 100_000.0
        final = float(row.get("cash", initial))
        ret = (final - initial) / initial
        max_dd = float(row.get("max_drawdown", 0.0))
        admitted = int(row.get("n_admitted", 0))
        rows[label] = {"ret": ret, "max_dd": max_dd, "admitted": admitted}
        print(f"{label:<10} {VALUE_MAP[label]:>14} {pct(ret):>10} {pct(max_dd):>10} {admitted:>10}")

    print(f"\n--- exit_reason breakdown, {title} ---")
    for label in LABELS:
        bd = exit_reason_breakdown(label, window)
        if bd is None:
            continue
        print(f"\n  {label} (vol_stop_mult={VALUE_MAP[label]}):")
        for cat, r in bd.sort_values("pnl").iterrows():
            print(f"    {cat:<15} n={int(r['n']):>4}  pnl=£{r['pnl']:>12,.2f}")

    return rows


def main() -> None:
    crash_rows = print_window_table("synthetic", "Crash window: synthetic Jan2008-Jul2009")
    real_rows = print_window_table("real", "Normal window: real prev-2yr (2024-09-03-present)")

    if BASELINE_LABEL in crash_rows and BASELINE_LABEL in real_rows:
        print("\n=== Delta vs mult20 (current, 2.0) ===\n")
        print(f"{'Label':<10} {'vol_stop_mult':>14} {'Crash ret d':>14} {'Crash DD d':>12} {'Real ret d':>12}")
        print("-" * 66)
        b_crash = crash_rows[BASELINE_LABEL]
        b_real = real_rows[BASELINE_LABEL]
        for label in LABELS:
            if label == BASELINE_LABEL:
                continue
            c = crash_rows.get(label)
            r = real_rows.get(label)
            if c is None or r is None:
                print(f"{label:<10} {'N/A':>14}")
                continue
            d_crash_ret = c["ret"] - b_crash["ret"]
            d_crash_dd = c["max_dd"] - b_crash["max_dd"]
            d_real_ret = r["ret"] - b_real["ret"]
            print(f"{label:<10} {VALUE_MAP[label]:>14} {pct(d_crash_ret):>14} {pct(d_crash_dd):>12} {pct(d_real_ret):>12}")


if __name__ == "__main__":
    main()
