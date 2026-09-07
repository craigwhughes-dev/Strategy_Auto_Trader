"""Annual return decomposition for the 26-year synthetic out-of-sample run.

Usage:
    uv run python scripts/analyse_synth_26yr.py
    uv run python scripts/analyse_synth_26yr.py --equity data_synthetic/journals/synth_26yr_equity.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_EQUITY = Path("data_synthetic/journals/synth_26yr_equity.csv")


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["date"] != "SUMMARY"]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("date")
    return df


def _annual_table(group: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    """Given a single (strategy, pot_size) group, return per-year stats."""
    group = group.set_index("date").sort_index()
    pv = group["portfolio_value"]

    # Last recorded portfolio_value per calendar year (sparse data: forward-fill is wrong
    # here — we want the last actual observation, since the series is already a snapshot
    # at trade-event days; the year-end value is the last known portfolio state that year).
    year_end = pv.resample("YE").last().dropna()

    # Prepend t=0 row so year-1 return is computed against initial capital.
    t0 = pd.Series([initial_cash], index=[pd.Timestamp(f"{year_end.index[0].year - 1}-12-31")])
    pv_anchored = pd.concat([t0, year_end])

    annual_ret = pv_anchored.pct_change().dropna()
    annual_ret.index = annual_ret.index.year

    rows = []
    for yr, ret in annual_ret.items():
        yr_pv = pv.loc[str(yr)] if str(yr) in pv.index.strftime("%Y") else pd.Series(dtype=float)
        if yr_pv.empty:
            dd = float("nan")
        else:
            rolling_max = yr_pv.cummax()
            dd = ((yr_pv - rolling_max) / rolling_max).min()
        rows.append({"year": yr, "return_pct": round(ret * 100, 2), "max_drawdown_pct": round(dd * 100, 2) if not np.isnan(dd) else float("nan")})

    return pd.DataFrame(rows).set_index("year")


def _summary(ann: pd.DataFrame, initial_cash: float, n_years: int) -> dict:
    rets = ann["return_pct"] / 100
    final_value = initial_cash * np.prod(1 + rets)
    # n_years = actual calendar span, not count of years-with-trades
    cagr = (final_value / initial_cash) ** (1 / n_years) - 1
    sharpe = rets.mean() / rets.std() if rets.std() > 0 else float("nan")
    return {
        "CAGR": f"{cagr * 100:.2f}%",
        "Final value": f"{final_value:,.0f}",
        "Best year": f"{ann['return_pct'].max():.1f}% ({ann['return_pct'].idxmax()})",
        "Worst year": f"{ann['return_pct'].min():.1f}% ({ann['return_pct'].idxmin()})",
        "% years positive": f"{(rets > 0).mean() * 100:.0f}%",
        "Annual Sharpe": f"{sharpe:.2f}",
        "Worst intra-year DD": f"{ann['max_drawdown_pct'].min():.1f}%",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equity", default=str(_DEFAULT_EQUITY))
    args = parser.parse_args()

    path = Path(args.equity)
    if not path.exists():
        raise SystemExit(f"File not found: {path}\nRun scripts/run_synth_26yr.ps1 first.")

    df = _load(path)

    for (strategy, pot_size), group in df.groupby(["strategy", "pot_size"]):
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy}  |  Pot: £{pot_size:,.0f}")
        print(f"{'='*60}")

        ann = _annual_table(group, float(pot_size))
        actual_span = ann.index.max() - ann.index.min() + 1

        print(ann.to_string())
        print()

        for k, v in _summary(ann, float(pot_size), actual_span).items():
            print(f"  {k:<25} {v}")
        print(f"  {'Span (yrs)':<25} {actual_span} ({ann.index.min()}–{ann.index.max()})")


if __name__ == "__main__":
    main()
