#!/usr/bin/env python3
"""Analyze vol_window sweep results.

Reads journals from data/journals/vol_window_<N>_{crash,real}.csv and
position-summary _equity.csv files, prints a results table per window.
"""

import sys
from pathlib import Path

import pandas as pd

JOURNAL_DIR = Path("data/journals")
WINDOWS = [126, 252, 378, 504, 756]


def _equity_stats(eq_path: Path) -> dict:
    if not eq_path.exists():
        return {}
    df = pd.read_csv(eq_path, parse_dates=["date"])
    df = df[df["date"] != "SUMMARY"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["pot_size"] == 100000]
    if df.empty:
        return {}
    df = df.sort_values("date")
    final_val = df["portfolio_value"].iloc[-1]
    net_pnl = final_val - 100_000
    pct_return = net_pnl / 100_000 * 100
    # max drawdown from equity curve
    roll_max = df["portfolio_value"].cummax()
    dd = (df["portfolio_value"] - roll_max) / roll_max * 100
    max_dd = dd.min()
    return {"return_pct": pct_return, "max_dd_pct": max_dd, "final_val": final_val}


def _journal_stats(journal_path: Path) -> dict:
    if not journal_path.exists():
        return {}
    df = pd.read_csv(journal_path)
    if df.empty or "entry_date" not in df.columns:
        return {}
    admitted = len(df)
    return {"admitted": admitted}


def main():
    print("\n=== VOL WINDOW SWEEP RESULTS ===\n")

    for window_label, suffix in [("CRASH (synthetic 2008–09)", "crash"), ("REAL (ibkr 2024–present)", "real")]:
        print(f"--- {window_label} ---")
        header = f"{'vol_window':>12} | {'return %':>10} | {'max DD %':>10} | {'admitted':>10}"
        print(header)
        print("-" * len(header))
        for w in WINDOWS:
            label = f"vol_window_{w}_{suffix}"
            eq = _equity_stats(JOURNAL_DIR / f"{label}_equity.csv")
            jrn = _journal_stats(JOURNAL_DIR / f"{label}.csv")
            tag = " (current)" if w == 504 else ""
            ret = f"{eq['return_pct']:+.1f}%" if eq else "N/A"
            dd = f"{eq['max_dd_pct']:.1f}%" if eq else "N/A"
            adm = str(jrn.get("admitted", "N/A"))
            print(f"{w:>10}{tag:<10} | {ret:>10} | {dd:>10} | {adm:>10}")
        print()


if __name__ == "__main__":
    main()
