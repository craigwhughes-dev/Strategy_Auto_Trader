#!/usr/bin/env python3
"""Analyze momentum_weight sweep results.

Reads data/journals/momentum_sweep_summary.csv.
Reports:
  - Return % and max drawdown per (window, momentum_weight, momentum_lookback)
  - Which AI/momentum names appear in top-K at each weight
  - Ticker composition changes vs baseline

Usage:
    uv run python scripts/analyze_momentum_sweep.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

JOURNAL_DIR = Path("data/journals")
SUMMARY_CSV = JOURNAL_DIR / "momentum_sweep_summary.csv"
AI_NAMES = {"NVDA", "META", "MSFT", "TSLA", "AAPL", "AMZN", "GOOGL", "GOOG"}


def main():
    if not SUMMARY_CSV.exists():
        print(f"Summary CSV not found: {SUMMARY_CSV}")
        print("Run: uv run python scripts/run_momentum_sweep.py")
        return

    df = pd.read_csv(SUMMARY_CSV)
    print(f"Loaded {len(df)} rows from {SUMMARY_CSV}\n")

    for window in df["window"].unique():
        sub = df[df["window"] == window].copy()
        sub = sub.sort_values(["momentum_weight", "momentum_lookback_days"])

        print(f"=== Window: {window} ===")
        print(f"{'mom_weight':>10} | {'mom_lb':>6} | {'return%':>8} | {'max_dd':>7} | {'admitted':>8} | AI names")
        print("-" * 75)
        for _, row in sub.iterrows():
            top20_str = row.get("top_tickers_20", "")
            if isinstance(top20_str, str) and top20_str:
                top20 = set(top20_str.split(","))
            else:
                top20 = set()
            found = sorted(AI_NAMES & top20)
            print(f"{row['momentum_weight']:>10.1f} | {row['momentum_lookback_days']:>6.0f} | "
                  f"{row['return_pct']:>8.1f} | {row['max_dd']:>7.1%} | "
                  f"{row['n_admitted']:>8.0f} | {', '.join(found) if found else '—'}")
        print()

        print(f"  Top-10 tickers (sample):")
        for _, row in sub.iterrows():
            sample = row.get("top_tickers_sample", "")
            if pd.isna(sample):
                sample = ""
            print(f"  mw={row['momentum_weight']:.1f} lb={row['momentum_lookback_days']:.0f}: {sample}")
        print()


if __name__ == "__main__":
    main()
