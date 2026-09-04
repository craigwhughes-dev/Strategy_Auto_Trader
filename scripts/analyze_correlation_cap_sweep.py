"""Compare rolling-30d Sharpe volatility across the max_correlation_to_admitted_today
sweep produced by scripts/run_correlation_cap_sweep.ps1.

Same reporting shape as scripts/analyze_same_day_cap_sweep.py (the rejected
$-cap's analysis script) for a direct apples-to-apples comparison, plus one
extra check that script didn't need: whether the gate actually moves the
number this whole investigation is about — corr(max same-day trade-event
count in trailing 30d, rolling Sharpe), the -0.37 correlation from the
2026-09-02 diagnosis. A gate that shrinks rolling_sharpe_std without moving
this correlation toward zero isn't fixing the mechanism, just correlated
noise.

For each threshold label, reads
data/journals/correlation_cap_sweep_<label>_equity.csv (the pot_size=100000
position-summary output) and data/journals/correlation_cap_sweep_<label>.csv
(the trade journal), reconstructs a daily equity curve (forward-filled from
the sparse trade-event samples, per cli.md's position_summary contract),
computes the rolling 30-calendar-day Sharpe series the same way the original
investigation did, and reports:
  - std dev of the rolling Sharpe series (primary metric: lower = less
    correlated-cluster volatility)
  - total return, max drawdown (from the SUMMARY row)
  - n_rejected_correlation (sanity check: gate should bind mainly on the
    known clustering days, not indiscriminately)
  - corr(trailing-30d max same-day trade-event count, rolling Sharpe) — the
    actual number that has to move toward zero for this gate to be worth
    keeping

Prints a comparison table and writes it to
reports/correlation_cap_validation_<timestamp>.csv.

Usage:
  uv run python scripts/analyze_correlation_cap_sweep.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
JOURNALS = BASE / "data" / "journals"
REPORTS = BASE / "reports"

LABELS = ["baseline", "corr30", "corr50", "corr70"]


def _daily_equity(equity_path: Path) -> tuple[pd.Series, pd.Series]:
    """Returns (daily forward-filled portfolio_value, rolling-30d Sharpe)."""
    df = pd.read_csv(equity_path)
    df = df[df["pot_size"].astype(str) == "100000.0"].copy()
    df = df[df["date"] != "SUMMARY"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.drop_duplicates(subset="date", keep="last")

    eq = df.set_index("date")["portfolio_value"].astype(float)
    if eq.empty:
        return eq, pd.Series(dtype=float)

    daily_idx = pd.date_range(eq.index.min(), eq.index.max(), freq="D")
    eq_daily = eq.reindex(daily_idx).ffill()
    ret = eq_daily.pct_change().dropna()

    roll_mean = ret.rolling(30).mean()
    roll_std = ret.rolling(30).std()
    rolling_sharpe = ((roll_mean / roll_std) * np.sqrt(252)).dropna()
    return eq_daily, rolling_sharpe


def _cluster_correlation(journal_path: Path, rolling_sharpe: pd.Series) -> float | None:
    """corr(trailing-30d max same-day trade-event count, rolling Sharpe) —
    the mechanism metric from the 2026-09-02 diagnosis (-0.37 at baseline).
    A trade contributes an event to both its open day and its close day,
    matching how the original investigation counted same-day clustering."""
    if not journal_path.exists() or rolling_sharpe.empty:
        return None
    trades = pd.read_csv(journal_path)
    if trades.empty:
        return None

    opens = pd.to_datetime(trades["date_opened"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    closes = pd.to_datetime(trades["date_closed"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    events = pd.concat([opens, closes]).dropna()
    daily_counts = events.value_counts().sort_index()

    daily_idx = pd.date_range(rolling_sharpe.index.min() - pd.Timedelta(days=30),
                               rolling_sharpe.index.max(), freq="D")
    daily_counts = daily_counts.reindex(daily_idx, fill_value=0)
    rolling_max_cluster = daily_counts.rolling(30).max()

    aligned = pd.DataFrame({
        "cluster": rolling_max_cluster.reindex(rolling_sharpe.index),
        "sharpe": rolling_sharpe,
    }).dropna()
    if len(aligned) < 10:
        return None
    return float(aligned["cluster"].corr(aligned["sharpe"]))


def rolling_sharpe_stats(equity_path: Path, journal_path: Path) -> dict:
    df = pd.read_csv(equity_path)
    df_full = df.copy()
    summary_row = df_full[df_full["pot_size"].astype(str) == "100000.0"]
    summary_row = summary_row[summary_row["date"] == "SUMMARY"]

    eq_daily, rolling_sharpe = _daily_equity(equity_path)
    if eq_daily.empty:
        return {"error": "no equity rows found for pot_size=100000"}

    result = {
        "rolling_sharpe_std": rolling_sharpe.std(),
        "rolling_sharpe_min": rolling_sharpe.min(),
        "rolling_sharpe_max": rolling_sharpe.max(),
        "cluster_sharpe_corr": _cluster_correlation(journal_path, rolling_sharpe),
        "final_portfolio_value": eq_daily.iloc[-1],
        "start_date": eq_daily.index.min().date().isoformat(),
        "end_date": eq_daily.index.max().date().isoformat(),
    }
    if not summary_row.empty:
        s = summary_row.iloc[0]
        result["realized_pnl_cum"] = s.get("realized_pnl_cum")
        result["max_drawdown"] = s.get("max_drawdown")
        result["n_admitted"] = s.get("n_admitted")
        result["n_rejected_cash"] = s.get("n_rejected_cash")
        result["n_rejected_kelly"] = s.get("n_rejected_kelly")
        result["n_rejected_correlation"] = s.get("n_rejected_correlation")
    return result


def main() -> None:
    rows = []
    for label in LABELS:
        equity_path = JOURNALS / f"correlation_cap_sweep_{label}_equity.csv"
        journal_path = JOURNALS / f"correlation_cap_sweep_{label}.csv"
        if not equity_path.exists():
            print(f"SKIP {label}: {equity_path} not found (run not completed?)")
            continue
        stats = rolling_sharpe_stats(equity_path, journal_path)
        stats["label"] = label
        rows.append(stats)

    if not rows:
        print("No sweep results found under data/journals/correlation_cap_sweep_*_equity.csv")
        return

    out = pd.DataFrame(rows).set_index("label")
    cols = ["rolling_sharpe_std", "rolling_sharpe_min", "rolling_sharpe_max",
            "cluster_sharpe_corr", "realized_pnl_cum", "max_drawdown", "n_admitted",
            "n_rejected_cash", "n_rejected_kelly", "n_rejected_correlation",
            "final_portfolio_value", "start_date", "end_date"]
    out = out[[c for c in cols if c in out.columns]]

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print(out.to_string())

    if "baseline" in out.index:
        base_std = out.loc["baseline", "rolling_sharpe_std"]
        base_pnl = out.loc["baseline", "realized_pnl_cum"]
        base_corr = out.loc["baseline", "cluster_sharpe_corr"]
        print(f"\nvs baseline (rolling_sharpe_std={base_std:.4f}, "
              f"cluster_sharpe_corr={base_corr}, realized_pnl_cum={base_pnl:,.2f}):")
        for label in out.index:
            if label == "baseline":
                continue
            std_delta_pct = (out.loc[label, "rolling_sharpe_std"] - base_std) / base_std * 100
            pnl_delta_pct = (out.loc[label, "realized_pnl_cum"] - base_pnl) / base_pnl * 100
            print(f"  {label}: rolling_sharpe_std {std_delta_pct:+.1f}%, "
                  f"cluster_sharpe_corr {out.loc[label, 'cluster_sharpe_corr']}, "
                  f"realized_pnl_cum {pnl_delta_pct:+.1f}%")

    REPORTS.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = REPORTS / f"correlation_cap_validation_{timestamp}.csv"
    out.to_csv(out_path)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
