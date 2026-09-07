"""Regime-split P&L analysis of the 26-year synthetic run.

Separates trading P&L (from trade journal) from interest income
(portfolio_value growth) and reports per regime period.

Calm regimes:    2003-07, 2012-19, 2023-26
Volatile regimes: 2000-02, 2008-09, 2022
Intermediate:    2010-11, 2020-21

Usage:
    uv run python scripts/analyse_regime_split.py
    uv run python scripts/analyse_regime_split.py --journal data_synthetic/journals/live_sim_synthetic.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_JOURNAL = Path("data_synthetic/journals/live_sim_synthetic.csv")
_DEFAULT_EQUITY  = Path("data_synthetic/journals/synth_26yr_equity.csv")

_REGIMES = {
    "dot-com crash 2000-02":   ("2000-01-01", "2002-12-31", "volatile"),
    "bull 2003-07":            ("2003-01-01", "2007-12-31", "calm"),
    "GFC 2008-09":             ("2008-01-01", "2009-12-31", "volatile"),
    "recovery 2010-11":        ("2010-01-01", "2011-12-31", "intermediate"),
    "bull 2012-19":            ("2012-01-01", "2019-12-31", "calm"),
    "COVID/recovery 2020-21":  ("2020-01-01", "2021-12-31", "intermediate"),
    "rate-hike 2022":          ("2022-01-01", "2022-12-31", "volatile"),
    "recovery 2023-26":        ("2023-01-01", "2026-12-31", "calm"),
}


def _load_journal(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["date_opened"] != "SUMMARY"]
    df["date_opened"] = pd.to_datetime(df["date_opened"], utc=True).dt.tz_localize(None).dt.normalize()
    df["return_pct"] = pd.to_numeric(df["return_pct"], errors="coerce")
    df["pnl_usd"] = pd.to_numeric(df["pnl_usd"], errors="coerce")
    return df.dropna(subset=["date_opened", "return_pct"])


def _load_equity(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["date"] != "SUMMARY"]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["portfolio_value"] = pd.to_numeric(df["portfolio_value"], errors="coerce")
    return df.sort_values("date")


def _regime_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}
    rets = trades["return_pct"].values
    n = len(rets)
    win_mask = rets > 0
    mean_ret = rets.mean()
    win_rate = win_mask.mean()
    total_pnl = trades["pnl_usd"].sum()
    return dict(n=n, win_rate=win_rate, mean_ret=mean_ret, total_pnl=total_pnl)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", default=str(_DEFAULT_JOURNAL))
    parser.add_argument("--equity",  default=str(_DEFAULT_EQUITY))
    args = parser.parse_args()

    journal_path = Path(args.journal)
    equity_path  = Path(args.equity)

    if not journal_path.exists():
        raise SystemExit(f"Journal not found: {journal_path}")

    trades = _load_journal(journal_path)
    equity = _load_equity(equity_path)

    print(f"\nRegime-split P&L -- {len(trades)} total trades\n")
    print(f"{'Regime':<28} {'Type':<14} {'Trades':>7} {'WinRate':>8} {'MeanRet':>9} {'TradePnL':>12}")
    print("-" * 84)

    regime_totals: dict[str, dict] = {"volatile": [], "calm": [], "intermediate": []}

    for label, (start, end, kind) in _REGIMES.items():
        mask = (trades["date_opened"] >= start) & (trades["date_opened"] <= end)
        subset = trades[mask]
        stats = _regime_stats(subset)
        if not stats:
            print(f"  {label:<26} {kind:<14} {'(no trades)':>7}")
            continue
        regime_totals[kind].append(stats)
        print(f"  {label:<26} {kind:<14} "
              f"{stats['n']:>7} "
              f"{stats['win_rate']:>8.1%} "
              f"{stats['mean_ret']:>+9.2%} "
              f"{stats['total_pnl']:>+12,.0f}")

    print("-" * 84)
    print("\nBy regime type:")
    for kind in ("calm", "volatile", "intermediate"):
        rows = regime_totals[kind]
        if not rows:
            continue
        n = sum(r["n"] for r in rows)
        wr = np.mean([r["win_rate"] for r in rows])
        mr = np.mean([r["mean_ret"] for r in rows])
        pnl = sum(r["total_pnl"] for r in rows)
        print(f"  {kind:<14} {n:>7} trades  WinRate {wr:.1%}  MeanRet {mr:+.2%}  TradePnL {pnl:>+12,.0f}")

    if equity is not None:
        print("\nEquity curve summary (portfolio_value growth incl. interest):")
        for (strategy, pot_size), grp in equity.groupby(["strategy", "pot_size"]):
            grp = grp.set_index("date").sort_index()
            pv = grp["portfolio_value"]
            first, last = pv.iloc[0], pv.iloc[-1]
            total_growth = last - pot_size
            # Approximate trading P&L from journal
            mask = trades["strategy"] == strategy if "strategy" in trades.columns else slice(None)
            t_pnl = trades.loc[mask]["pnl_usd"].sum() if isinstance(mask, pd.Series) else trades["pnl_usd"].sum()
            interest_est = total_growth - t_pnl
            print(f"  {strategy} pot={pot_size:,.0f}  final={last:,.0f}  "
                  f"total_growth={total_growth:+,.0f}  "
                  f"trade_pnl={t_pnl:+,.0f}  interest_est={interest_est:+,.0f}")


if __name__ == "__main__":
    main()
