#!/usr/bin/env python3
"""Analyze score_lookback_days sweep results.

Reads data/journals/score_lookback_sweep_summary.csv and per-label equity CSVs.
Reports:
  - Return % and max drawdown per (window, score_lookback_days)
  - Which tickers enter/exit top-K at each lookback value
  - Whether AI/momentum names (NVDA, META, etc.) appear at shorter lookbacks

Usage:
    uv run python scripts/analyze_score_lookback_sweep.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

JOURNAL_DIR = Path("data/journals")
SUMMARY_CSV = JOURNAL_DIR / "score_lookback_sweep_summary.csv"
AI_NAMES = {"NVDA", "META", "MSFT", "TSLA", "AAPL", "AMZN", "GOOGL", "GOOG"}


def main():
    if not SUMMARY_CSV.exists():
        print(f"Summary CSV not found: {SUMMARY_CSV}")
        print("Run: uv run python scripts/run_score_lookback_sweep.py")
        return

    df = pd.read_csv(SUMMARY_CSV)
    print(f"Loaded {len(df)} rows from {SUMMARY_CSV}\n")

    for window in df["window"].unique():
        sub = df[df["window"] == window].copy()
        sub = sub.sort_values("score_lookback_days", na_position="last")

        print(f"=== Window: {window} ===")
        print(f"{'score_lookback':>15} | {'return%':>8} | {'max_dd':>7} | {'admitted':>8}")
        print("-" * 50)
        for _, row in sub.iterrows():
            sld = row["score_lookback_days"]
            sld_str = str(int(sld)) if pd.notna(sld) else "all-time"
            print(f"{sld_str:>15} | {row['return_pct']:>8.1f} | "
                  f"{row['max_dd']:>7.1%} | {row['n_admitted']:>8.0f}")
        print()

        # AI names presence
        print(f"  AI/momentum names in top-K (by lookback):")
        print(f"  {'score_lookback':>15} | {'AI names found'}")
        for _, row in sub.iterrows():
            sld = row["score_lookback_days"]
            sld_str = str(int(sld)) if pd.notna(sld) else "all-time"
            top20_str = row.get("top_tickers_20", "")
            if isinstance(top20_str, str) and top20_str:
                top20 = set(top20_str.split(","))
            else:
                top20 = set()
            found = sorted(AI_NAMES & top20)
            print(f"  {sld_str:>15} | {', '.join(found) if found else '—'}")
        print()

        # Top-10 ticker composition at each lookback
        print(f"  Top-10 tickers by lookback (sample):")
        for _, row in sub.iterrows():
            sld = row["score_lookback_days"]
            sld_str = str(int(sld)) if pd.notna(sld) else "all-time"
            sample = row.get("top_tickers_sample", "")
            if pd.isna(sample):
                sample = ""
            print(f"  {sld_str:>15}: {sample}")
        print()


if __name__ == "__main__":
    main()
