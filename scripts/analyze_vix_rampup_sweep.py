#!/usr/bin/env python3
"""Analyze vix_recovery_window_days x vix_recovery_kelly_mult sweep results."""

from pathlib import Path
import pandas as pd

JOURNAL_DIR = Path("data/journals")
WINDOW_DAYS = [30, 60, 90]
KELLY_MULTS = [0.3, 0.5, 0.7]


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


def _row(label: str, suffix: str) -> tuple:
    eq = _equity_stats(JOURNAL_DIR / f"{label}_{suffix}_equity.csv")
    ret = f"{eq['return_pct']:+.1f}%" if eq else "N/A"
    dd = f"{eq['max_dd_pct']:.1f}%" if eq else "N/A"
    return ret, dd


def main():
    print("\n=== VIX RAMP-UP SWEEP RESULTS ===\n")
    for suffix, title in [("crash", "CRASH (synthetic 2008–09)"), ("real", "REAL (ibkr 2024–present)")]:
        print(f"--- {title} ---")
        hdr = f"{'window_days':>14} {'kelly_mult':>12} | {'return %':>10} {'max DD %':>10}"
        print(hdr)
        print("-" * len(hdr))
        base_ret, base_dd = _row("vix_rampup_baseline", suffix)
        print(f"{'baseline (off)':>14} {'—':>12} | {base_ret:>10} {base_dd:>10}")
        for days in WINDOW_DAYS:
            for mult in KELLY_MULTS:
                label = f"vix_rampup_d{days}_k{int(mult * 10)}"
                ret, dd = _row(label, suffix)
                print(f"{days:>14} {mult:>12.1f} | {ret:>10} {dd:>10}")
        print()


if __name__ == "__main__":
    main()
