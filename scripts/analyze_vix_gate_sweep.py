"""Summarise the VIX gate sweep results from run_vix_gate_sweep.ps1.

Reads per-(label, window) equity CSVs from data/journals/ and prints a
table comparing crash protection (synthetic 2008) vs normal upside cost
(real 2024-Nov) across threshold levels.

Usage:
    uv run python scripts/analyze_vix_gate_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

LABELS = ["baseline", "vix20", "vix25", "vix30", "vix35", "vix40"]
THRESHOLD_MAP = {
    "baseline": "None (gate off)",
    "vix20": "20",
    "vix25": "25",
    "vix30": "30",
    "vix35": "35",
    "vix40": "40",
}

JOURNALS_DIR = Path("data/journals")


def load_summary(label: str, window: str) -> dict | None:
    """Return the SUMMARY row from the equity CSV for this label/window."""
    path = JOURNALS_DIR / f"vix_sweep_{label}_{window}_equity.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    summary = df[df["date"] == "SUMMARY"]
    if summary.empty:
        return None
    return summary.iloc[0].to_dict()


def pct(val: float) -> str:
    return f"{val * 100:+.1f}%"


def fmt_cash(val: float) -> str:
    return f"£{val:,.0f}"


def main() -> None:
    print("\n=== VIX Gate Sweep — Crash window: synthetic Jan2008–Jul2009 ===\n")
    print(f"{'Label':<12} {'Threshold':>10} {'Return':>10} {'Max DD':>10} {'Admitted':>10} {'VIX-rejected':>14}")
    print("-" * 68)

    crash_rows = {}
    for label in LABELS:
        row = load_summary(label, "synthetic")
        if row is None:
            print(f"{label:<12} {'N/A':>10}")
            continue
        initial = 100_000.0
        final = float(row.get("cash", initial))
        ret = (final - initial) / initial
        max_dd = float(row.get("max_drawdown", 0.0))
        admitted = int(row.get("n_admitted", 0))
        vix_rej = int(row.get("n_rejected_vix", 0))
        crash_rows[label] = {"ret": ret, "max_dd": max_dd, "admitted": admitted, "vix_rej": vix_rej}
        print(f"{label:<12} {THRESHOLD_MAP[label]:>10} {pct(ret):>10} {pct(max_dd):>10} {admitted:>10} {vix_rej:>14}")

    print("\n=== VIX Gate Sweep — Normal window: real Nov2024–present ===\n")
    print(f"{'Label':<12} {'Threshold':>10} {'Return':>10} {'Max DD':>10} {'Admitted':>10} {'VIX-rejected':>14}")
    print("-" * 68)

    real_rows = {}
    for label in LABELS:
        row = load_summary(label, "real")
        if row is None:
            print(f"{label:<12} {'N/A':>10}")
            continue
        initial = 100_000.0
        final = float(row.get("cash", initial))
        ret = (final - initial) / initial
        max_dd = float(row.get("max_drawdown", 0.0))
        admitted = int(row.get("n_admitted", 0))
        vix_rej = int(row.get("n_rejected_vix", 0))
        real_rows[label] = {"ret": ret, "max_dd": max_dd, "admitted": admitted, "vix_rej": vix_rej}
        print(f"{label:<12} {THRESHOLD_MAP[label]:>10} {pct(ret):>10} {pct(max_dd):>10} {admitted:>10} {vix_rej:>14}")

    if "baseline" in crash_rows and "baseline" in real_rows:
        print("\n=== Delta vs baseline ===\n")
        print(f"{'Label':<12} {'Threshold':>10} {'Crash ret d':>14} {'Crash DD d':>12} {'Real ret d':>12}")
        print("-" * 64)
        b_crash = crash_rows["baseline"]
        b_real = real_rows["baseline"]
        for label in LABELS[1:]:
            c = crash_rows.get(label)
            r = real_rows.get(label)
            if c is None or r is None:
                print(f"{label:<12} {'N/A':>10}")
                continue
            d_crash_ret = c["ret"] - b_crash["ret"]
            d_crash_dd = c["max_dd"] - b_crash["max_dd"]
            d_real_ret = r["ret"] - b_real["ret"]
            print(f"{label:<12} {THRESHOLD_MAP[label]:>10} {pct(d_crash_ret):>14} {pct(d_crash_dd):>12} {pct(d_real_ret):>12}")


if __name__ == "__main__":
    main()
