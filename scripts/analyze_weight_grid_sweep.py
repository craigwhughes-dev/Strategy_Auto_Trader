"""Summarise item #4 of the exit-parameter audit: trend/sma200 entry-weight
grid ({1,2,3} x {2,3,4}, real prev-2yr window only) from
run_exit_param_audit.ps1.

Prints two 3x3 grids (return, max DD) — trend as rows, sma200 as columns —
so the highest-leverage entry weights' effect is visible at a glance rather
than as 9 flat rows.

Usage:
    uv run python scripts/analyze_weight_grid_sweep.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TREND_VALS = [1, 2, 3]
SMA200_VALS = [2, 3, 4]
BASELINE = (2, 3)  # current: trend=2.0, sma200=3.0
PREFIX = "weight_grid_sweep"

JOURNALS_DIR = Path("data/journals")


def load_summary(trend: int, sma200: int) -> dict | None:
    label = f"t{trend}s{sma200}"
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
    results = {}
    for t in TREND_VALS:
        for s in SMA200_VALS:
            row = load_summary(t, s)
            if row is None:
                results[(t, s)] = None
                continue
            initial = 100_000.0
            final = float(row.get("cash", initial))
            ret = (final - initial) / initial
            max_dd = float(row.get("max_drawdown", 0.0))
            admitted = int(row.get("n_admitted", 0))
            results[(t, s)] = {"ret": ret, "max_dd": max_dd, "admitted": admitted}

    print("\n=== trend/sma200 weight grid - Return (real prev-2yr window) ===")
    print(f"trend \\ sma200 {'':>4}" + "".join(f"{s:>12}" for s in SMA200_VALS))
    for t in TREND_VALS:
        row_str = f"{t:>14} " + "".join(
            f"{pct(results[(t, s)]['ret']):>12}" if results[(t, s)] else f"{'N/A':>12}"
            for s in SMA200_VALS
        )
        print(row_str)

    print("\n=== trend/sma200 weight grid - Max Drawdown (real prev-2yr window) ===")
    print(f"trend \\ sma200 {'':>4}" + "".join(f"{s:>12}" for s in SMA200_VALS))
    for t in TREND_VALS:
        row_str = f"{t:>14} " + "".join(
            f"{pct(results[(t, s)]['max_dd']):>12}" if results[(t, s)] else f"{'N/A':>12}"
            for s in SMA200_VALS
        )
        print(row_str)

    print("\n=== trend/sma200 weight grid - Admitted candidates (real prev-2yr window) ===")
    print(f"trend \\ sma200 {'':>4}" + "".join(f"{s:>12}" for s in SMA200_VALS))
    for t in TREND_VALS:
        row_str = f"{t:>14} " + "".join(
            f"{results[(t, s)]['admitted']:>12}" if results[(t, s)] else f"{'N/A':>12}"
            for s in SMA200_VALS
        )
        print(row_str)

    base = results.get(BASELINE)
    if base:
        print(f"\n=== Delta vs current (trend=2.0, sma200=3.0): return {pct(base['ret'])}, max DD {pct(base['max_dd'])} ===")
        for t in TREND_VALS:
            for s in SMA200_VALS:
                if (t, s) == BASELINE:
                    continue
                r = results.get((t, s))
                if r is None:
                    print(f"  t{t}s{s}: N/A")
                    continue
                d_ret = r["ret"] - base["ret"]
                d_dd = r["max_dd"] - base["max_dd"]
                print(f"  t{t}s{s}: return {pct(d_ret)}, max DD {pct(d_dd)}")


if __name__ == "__main__":
    main()
