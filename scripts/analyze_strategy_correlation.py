#!/usr/bin/env python3
"""Correlation analysis across strategies and FTSE100/S&P500 markets.

Tests whether combining strategies (or the same strategy across markets)
would raise portfolio Sharpe the way multi-strategy funds do — that only
works if the underlying return streams are genuinely uncorrelated. Reads
existing reports/full_scan/ output; makes no changes to any strategy or
live_sim code.

Usage: python scripts/analyze_strategy_correlation.py [--min-tickers N]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
SUMMARY_CSV = ROOT / "reports" / "full_scan" / "summary.csv"
SCAN_DIR = ROOT / "reports" / "full_scan"
UNIVERSE_JSON = ROOT / "config" / "universe_sp_ftse.json"


def market_of(ticker: str) -> str:
    return "FTSE" if ticker.endswith(".L") else "SP500"


def load_universe_tags() -> dict[str, str]:
    tickers = json.loads(UNIVERSE_JSON.read_text())["tickers"]
    return {t: market_of(t) for t in tickers}


def bucket_return_series(strategy: str, tickers: list[str], min_tickers: int) -> pd.Series | None:
    """Equal-weighted average daily pct-change of portfolio_value across
    tickers, for one (strategy, market) bucket. None if too few tickers."""
    series = []
    for ticker in tickers:
        safe = ticker.replace("/", "-").replace("\\", "-")
        daily_path = SCAN_DIR / strategy / "daily" / f"{safe}.csv"
        if not daily_path.exists():
            continue
        df = pd.read_csv(daily_path, index_col=0, parse_dates=True)
        if "portfolio_value" not in df.columns:
            continue
        ret = df["portfolio_value"].pct_change().replace([np.inf, -np.inf], np.nan)
        series.append(ret.rename(ticker))
    if len(series) < min_tickers:
        return None
    aligned = pd.concat(series, axis=1)
    return aligned.mean(axis=1, skipna=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-tickers", type=int, default=5,
                         help="Skip a (strategy, market) bucket with fewer completed tickers than this")
    args = parser.parse_args()

    if not SUMMARY_CSV.exists():
        print(f"No summary.csv at {SUMMARY_CSV} — run full_scan first.")
        return 1

    df = pd.read_csv(SUMMARY_CSV)
    df_ok = df[df["status"] == "ok"].copy()
    tags = load_universe_tags()
    df_ok["market"] = df_ok["ticker"].map(tags)
    df_ok = df_ok.dropna(subset=["market"])

    buckets: dict[str, pd.Series] = {}
    avg_sharpe: dict[str, float] = {}
    n_tickers: dict[str, int] = {}

    for (strategy, market), group in df_ok.groupby(["strategy", "market"]):
        key = f"{strategy}@{market}"
        ret = bucket_return_series(strategy, group["ticker"].tolist(), args.min_tickers)
        if ret is None:
            continue
        buckets[key] = ret
        sharpe = group["sharpe_strategy"].replace([np.inf, -np.inf], np.nan)
        avg_sharpe[key] = sharpe.mean()
        n_tickers[key] = len(group)

    if len(buckets) < 2:
        print(f"Only {len(buckets)} bucket(s) had >= {args.min_tickers} tickers — nothing to correlate.")
        return 1

    matrix = pd.concat(buckets, axis=1).corr()
    out_path = SCAN_DIR / "strategy_correlation_matrix.csv"
    matrix.to_csv(out_path)

    print("=" * 88)
    print(f"Strategy correlation analysis: {len(buckets)} (strategy, market) buckets, "
          f"min {args.min_tickers} tickers each")
    print(f"Full matrix written to {out_path}")
    print("=" * 88)

    strategies = sorted({k.split("@")[0] for k in buckets})
    markets = sorted({k.split("@")[1] for k in buckets})

    print()
    print("SAME-STRATEGY, CROSS-MARKET correlation (does FTSE vs S&P decorrelate a strategy?):")
    print("-" * 88)
    for strat in strategies:
        pair = [f"{strat}@{m}" for m in markets if f"{strat}@{m}" in buckets]
        if len(pair) < 2:
            continue
        a, b = pair[0], pair[1]
        corr = matrix.loc[a, b]
        print(f"  {strat:20s} corr={corr:+.3f}  "
              f"({a}: Sharpe {avg_sharpe[a]:.2f}, n={n_tickers[a]}  |  "
              f"{b}: Sharpe {avg_sharpe[b]:.2f}, n={n_tickers[b]})")

    print()
    print("SAME-MARKET, CROSS-STRATEGY correlation (do strategy variants decorrelate?):")
    print("-" * 88)
    for market in markets:
        keys = [f"{s}@{market}" for s in strategies if f"{s}@{market}" in buckets]
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                corr = matrix.loc[a, b]
                print(f"  [{market}] {a} vs {b}: corr={corr:+.3f}")

    print()
    print("=" * 88)
    print("Reminder: correlation near 0 is required (not sufficient) for the Sharpe-boost math.")
    print("Also check bucket Sharpe isn't near-zero/negative — combining noise doesn't help.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
