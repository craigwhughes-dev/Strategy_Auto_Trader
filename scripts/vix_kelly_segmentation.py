"""
VIX segmentation of historical trades.

For each bin (equal-count quintiles by default, or custom --vix-splits),
computes: win_rate, avg_win, avg_loss, payoff_ratio, implied_kelly, n_trades

Answers: does win_rate degrade monotonically with VIX?
If yes → conditional kelly is coherent. If payoff also rises → net effect ambiguous.

Usage:
    uv run python -m scripts.vix_kelly_segmentation --strategy optimised_new
    uv run python -m scripts.vix_kelly_segmentation --strategy optimised_new --vix-splits 14.2 15.8 17.1 19.1 25
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Strategy_Auto_Trader.quant_hmm.quant_engine import fetch_daily


def load_journal(path: str, strategy: str | None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date_opened", "date_closed"])
    if strategy:
        df = df[df["strategy"] == strategy]
    df = df.dropna(subset=["return_pct", "date_opened"])
    print(f"Loaded {len(df)} trades" + (f" (strategy={strategy})" if strategy else ""))
    return df


def fetch_vix_series() -> pd.Series:
    vix_df = fetch_daily("^VIX")
    if vix_df is None or "Close" not in vix_df.columns:
        raise RuntimeError("^VIX fetch failed")
    s = vix_df["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.sort_index()


def join_vix(df: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(df["date_opened"], utc=True).dt.tz_localize(None)
    df = df.copy()
    df["vix_at_entry"] = dates.map(lambda d: vix.asof(d))
    before = len(df)
    df = df.dropna(subset=["vix_at_entry"])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} trades with no VIX observation")
    return df


def implied_kelly(win_rate: float, payoff: float) -> float:
    if payoff <= 0:
        return 0.0
    return win_rate - (1 - win_rate) / payoff


def bin_stats(df: pd.DataFrame, bins: list[float], bin_labels: list[str]) -> pd.DataFrame:
    rows = []
    for i, label in enumerate(bin_labels):
        lo = bins[i]
        hi = bins[i + 1]
        grp = df[(df["vix_at_entry"] >= lo) & (df["vix_at_entry"] < hi)]
        if len(grp) == 0:
            continue
        vix_lo = grp["vix_at_entry"].min()
        vix_hi = grp["vix_at_entry"].max()
        wins = grp[grp["return_pct"] > 0]["return_pct"]
        losses = grp[grp["return_pct"] <= 0]["return_pct"]
        n = len(grp)
        wr = len(wins) / n
        avg_win = wins.mean() if len(wins) else 0.0
        avg_loss = abs(losses.mean()) if len(losses) else 0.0
        pr = avg_win / avg_loss if avg_loss > 0 else float("inf")
        k = implied_kelly(wr, pr)
        rows.append({
            "bin": label,
            "vix_actual": f"{vix_lo:.1f}–{vix_hi:.1f}",
            "n_trades": n,
            "win_rate": round(wr, 3),
            "avg_win_pct": round(avg_win, 3),
            "avg_loss_pct": round(avg_loss, 3),
            "payoff_ratio": round(pr, 3),
            "implied_kelly": round(k, 4),
        })
    return pd.DataFrame(rows)


def run_quintiles(df: pd.DataFrame) -> pd.DataFrame:
    """Equal-count quintile binning."""
    boundaries = [df["vix_at_entry"].quantile(q) for q in [0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    boundaries[-1] += 0.01  # make upper bound exclusive-safe
    labels = [f"Q{i+1}" for i in range(5)]
    return bin_stats(df, boundaries, labels)


def run_custom_splits(df: pd.DataFrame, splits: list[float]) -> pd.DataFrame:
    """Custom boundary binning. splits are interior breakpoints; edges clamped to data range."""
    lo = df["vix_at_entry"].min() - 0.01
    hi = df["vix_at_entry"].max() + 0.01
    boundaries = [lo] + sorted(splits) + [hi]
    labels = []
    for i in range(len(boundaries) - 1):
        labels.append(f"{boundaries[i+1-1] if i==0 else splits[i-1]:.0f}–{splits[i] if i < len(splits) else 'inf'}")
    # Simpler label: just the breakpoint range
    labels = []
    prev = boundaries[0]
    for b in boundaries[1:]:
        labels.append(f"VIX {prev:.1f}–{b:.1f}")
        prev = b
    return bin_stats(df, boundaries, labels)


def print_stats(stats: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print("=" * 85)
    print(stats.to_string(index=False))
    xs = np.arange(1, len(stats) + 1)
    wr_trend = np.polyfit(xs, stats["win_rate"], 1)[0]
    k_trend = np.polyfit(xs, stats["implied_kelly"], 1)[0]
    pr_trend = np.polyfit(xs, stats["payoff_ratio"], 1)[0]
    print("\nLinear trends (low to high VIX):")
    print(f"  win_rate:      {wr_trend:+.4f}  ({'degrades' if wr_trend < 0 else 'improves'})")
    print(f"  payoff_ratio:  {pr_trend:+.4f}  ({'improves' if pr_trend > 0 else 'degrades'})")
    print(f"  implied_kelly: {k_trend:+.4f}  ({'degrades' if k_trend < 0 else 'improves'})")
    monotone_wr = all(stats["win_rate"].iloc[i] >= stats["win_rate"].iloc[i+1]
                      for i in range(len(stats)-1))
    monotone_k  = all(stats["implied_kelly"].iloc[i] >= stats["implied_kelly"].iloc[i+1]
                      for i in range(len(stats)-1))
    print(f"  win_rate monotone-degrading: {monotone_wr}")
    print(f"  kelly    monotone-degrading: {monotone_k}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--journal", default="data/journals/backtest.csv")
    parser.add_argument(
        "--vix-splits", nargs="+", type=float, default=None,
        help="Custom interior VIX breakpoints (e.g. 14.2 15.8 17.1 19.1 25). "
             "Default: equal-count quintiles."
    )
    args = parser.parse_args()

    df = load_journal(args.journal, args.strategy)
    vix = fetch_vix_series()
    print(f"VIX series: {vix.index[0].date()} to {vix.index[-1].date()}, {len(vix)} obs")

    df = join_vix(df, vix)
    print(f"VIX range in trades: {df['vix_at_entry'].min():.1f} – {df['vix_at_entry'].max():.1f}")

    if args.vix_splits:
        stats = run_custom_splits(df, args.vix_splits)
        print_stats(stats, f"Custom VIX Splits: {args.vix_splits}")
    else:
        # Run both: default quintiles AND with Q5 split at 25
        stats_q = run_quintiles(df)
        print_stats(stats_q, "Equal-Count Quintiles")

        splits_with_gate = [
            df["vix_at_entry"].quantile(0.2),
            df["vix_at_entry"].quantile(0.4),
            df["vix_at_entry"].quantile(0.6),
            df["vix_at_entry"].quantile(0.8),
            25.0,
        ]
        stats_split = run_custom_splits(df, splits_with_gate)
        print_stats(stats_split, "Quintile Boundaries + Q5 Split at VIX=25 (live gate)")


if __name__ == "__main__":
    main()
