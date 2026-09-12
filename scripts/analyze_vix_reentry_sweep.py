#!/usr/bin/env python3
"""Analyze vix_gate_allow_reentry sweep results."""

from pathlib import Path
import pandas as pd

JOURNAL_DIR = Path("data/journals")


def _equity_stats(eq_path: Path) -> dict:
    if not eq_path.exists():
        return {}
    df = pd.read_csv(eq_path, parse_dates=["date"])
    df = df[df["date"] != "SUMMARY"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df[df["pot_size"] == 100000]
    if df.empty:
        return {}
    final_val = df["portfolio_value"].iloc[-1]
    pct_return = (final_val - 100_000) / 100_000 * 100
    roll_max = df["portfolio_value"].cummax()
    max_dd = ((df["portfolio_value"] - roll_max) / roll_max * 100).min()
    return {"return_pct": pct_return, "max_dd_pct": max_dd}


def _journal_stats(journal_path: Path) -> dict:
    if not journal_path.exists():
        return {}
    df = pd.read_csv(journal_path)
    if df.empty or "entry_date" not in df.columns:
        return {}
    return {"admitted": len(df)}


def main():
    print("\n=== VIX RE-ENTRY SWEEP RESULTS ===\n")
    for suffix, title in [("crash", "CRASH (synthetic 2008–09)"), ("real", "REAL (ibkr 2024–present)")]:
        print(f"--- {title} ---")
        hdr = f"{'allow_reentry':>16} | {'return %':>10} {'max DD %':>10} {'admitted':>10}"
        print(hdr)
        print("-" * len(hdr))
        for reentry in ["False", "True"]:
            label = f"vix_reentry_{reentry}_{suffix}"
            eq = _equity_stats(JOURNAL_DIR / f"{label}_equity.csv")
            jrn = _journal_stats(JOURNAL_DIR / f"{label}.csv")
            ret = f"{eq['return_pct']:+.1f}%" if eq else "N/A"
            dd = f"{eq['max_dd_pct']:.1f}%" if eq else "N/A"
            adm = str(jrn.get("admitted", "N/A"))
            print(f"{reentry:>16} | {ret:>10} {dd:>10} {adm:>10}")
        print()


if __name__ == "__main__":
    main()
