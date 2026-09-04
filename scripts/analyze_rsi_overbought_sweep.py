"""Summarise item #6 of the exit-parameter audit: _RSI_OVERBOUGHT sweep
({60, 65, 70 (current), 75, 999 (no veto)}, real prev-2yr window only) from
run_exit_param_audit.ps1.

_RSI_OVERBOUGHT is an entry veto (RSI > threshold blocks new BUYs), not an
exit parameter — included in the "exit-parameter audit" batch because it's
the same never-swept/hand-picked-since-day-one class as the others.

Usage:
    uv run python scripts/analyze_rsi_overbought_sweep.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LABELS = ["rsi60", "rsi65", "rsi70", "rsi75", "rsiOff"]
VALUE_MAP = {"rsi60": "60", "rsi65": "65", "rsi70": "70 (current)", "rsi75": "75", "rsiOff": "off (no veto)"}
BASELINE_LABEL = "rsi70"
PREFIX = "rsi_overbought_sweep"

JOURNALS_DIR = Path("data/journals")


def load_summary(label: str) -> dict | None:
    path = JOURNALS_DIR / f"{PREFIX}_{label}_real_equity.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    summary = df[df["date"] == "SUMMARY"]
    if summary.empty:
        return None
    return summary.iloc[0].to_dict()


def pct(val: float) -> str:
    return f"{val * 100:+.1f}%"


def main() -> None:
    print("\n=== _RSI_OVERBOUGHT Sweep - Normal window: real prev-2yr (2024-09-03-present) ===\n")
    print(f"{'Label':<8} {'threshold':>14} {'Return':>10} {'Max DD':>10} {'Admitted':>10}")
    print("-" * 58)

    rows = {}
    for label in LABELS:
        row = load_summary(label)
        if row is None:
            print(f"{label:<8} {'N/A':>14}")
            continue
        initial = 100_000.0
        final = float(row.get("cash", initial))
        ret = (final - initial) / initial
        max_dd = float(row.get("max_drawdown", 0.0))
        admitted = int(row.get("n_admitted", 0))
        rows[label] = {"ret": ret, "max_dd": max_dd, "admitted": admitted}
        print(f"{label:<8} {VALUE_MAP[label]:>14} {pct(ret):>10} {pct(max_dd):>10} {admitted:>10}")

    if BASELINE_LABEL in rows:
        print("\n=== Delta vs rsi70 (current) ===\n")
        b = rows[BASELINE_LABEL]
        for label in LABELS:
            if label == BASELINE_LABEL:
                continue
            r = rows.get(label)
            if r is None:
                print(f"  {label}: N/A")
                continue
            d_ret = r["ret"] - b["ret"]
            d_dd = r["max_dd"] - b["max_dd"]
            print(f"  {label} ({VALUE_MAP[label]}): return {pct(d_ret)}, max DD {pct(d_dd)}, admitted {r['admitted']} (vs {b['admitted']})")


if __name__ == "__main__":
    main()
