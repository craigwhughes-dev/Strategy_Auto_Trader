"""Compare rolling-30d Sharpe volatility across the same_day_deployment_cap_pct
sweep produced by scripts/run_same_day_cap_sweep.ps1.

For each cap_pct label, reads data/journals/same_day_cap_sweep_<label>_equity.csv
(the pot_size=100000 position-summary output), reconstructs a daily equity
curve (forward-filled from the sparse trade-event samples, per cli.md's
position_summary contract), computes the rolling 30-calendar-day Sharpe
series the same way the original 2026-09-02 investigation did, and reports:
  - std dev of the rolling Sharpe series (primary metric: lower = less
    correlated-cluster volatility)
  - total return, max drawdown (from the SUMMARY row)
  - n_rejected_concentration (sanity check: cap should bind on clustered
    days, not indiscriminately)

Prints a comparison table and writes it to
reports/same_day_cap_validation_<timestamp>.csv.

Usage:
  uv run python scripts/analyze_same_day_cap_sweep.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
JOURNALS = BASE / "data" / "journals"
REPORTS = BASE / "reports"

LABELS = ["baseline", "cap10", "cap20", "cap35", "cap50"]


def rolling_sharpe_stats(equity_path: Path) -> dict:
    df = pd.read_csv(equity_path)
    df = df[df["pot_size"].astype(str) == "100000.0"].copy()
    summary_row = df[df["date"] == "SUMMARY"]
    df = df[df["date"] != "SUMMARY"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.drop_duplicates(subset="date", keep="last")

    eq = df.set_index("date")["portfolio_value"].astype(float)
    if eq.empty:
        return {"error": "no equity rows found for pot_size=100000"}

    daily_idx = pd.date_range(eq.index.min(), eq.index.max(), freq="D")
    eq_daily = eq.reindex(daily_idx).ffill()
    ret = eq_daily.pct_change().dropna()

    roll_mean = ret.rolling(30).mean()
    roll_std = ret.rolling(30).std()
    rolling_sharpe = ((roll_mean / roll_std) * np.sqrt(252)).dropna()

    result = {
        "rolling_sharpe_std": rolling_sharpe.std(),
        "rolling_sharpe_min": rolling_sharpe.min(),
        "rolling_sharpe_max": rolling_sharpe.max(),
        "final_portfolio_value": eq.iloc[-1],
        "start_date": eq.index.min().date().isoformat(),
        "end_date": eq.index.max().date().isoformat(),
    }
    if not summary_row.empty:
        s = summary_row.iloc[0]
        result["realized_pnl_cum"] = s.get("realized_pnl_cum")
        result["max_drawdown"] = s.get("max_drawdown")
        result["n_admitted"] = s.get("n_admitted")
        result["n_rejected_cash"] = s.get("n_rejected_cash")
        result["n_rejected_kelly"] = s.get("n_rejected_kelly")
        result["n_rejected_concentration"] = s.get("n_rejected_concentration")
    return result


def main() -> None:
    rows = []
    for label in LABELS:
        path = JOURNALS / f"same_day_cap_sweep_{label}_equity.csv"
        if not path.exists():
            print(f"SKIP {label}: {path} not found (run not completed?)")
            continue
        stats = rolling_sharpe_stats(path)
        stats["label"] = label
        rows.append(stats)

    if not rows:
        print("No sweep results found under data/journals/same_day_cap_sweep_*_equity.csv")
        return

    out = pd.DataFrame(rows).set_index("label")
    cols = ["rolling_sharpe_std", "rolling_sharpe_min", "rolling_sharpe_max",
            "realized_pnl_cum", "max_drawdown", "n_admitted",
            "n_rejected_cash", "n_rejected_kelly", "n_rejected_concentration",
            "final_portfolio_value", "start_date", "end_date"]
    out = out[[c for c in cols if c in out.columns]]

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print(out.to_string())

    if "baseline" in out.index:
        base_std = out.loc["baseline", "rolling_sharpe_std"]
        base_pnl = out.loc["baseline", "realized_pnl_cum"]
        print(f"\nvs baseline (rolling_sharpe_std={base_std:.4f}, realized_pnl_cum={base_pnl:,.2f}):")
        for label in out.index:
            if label == "baseline":
                continue
            std_delta_pct = (out.loc[label, "rolling_sharpe_std"] - base_std) / base_std * 100
            pnl_delta_pct = (out.loc[label, "realized_pnl_cum"] - base_pnl) / base_pnl * 100
            print(f"  {label}: rolling_sharpe_std {std_delta_pct:+.1f}%, realized_pnl_cum {pnl_delta_pct:+.1f}%")

    REPORTS.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = REPORTS / f"same_day_cap_validation_{timestamp}.csv"
    out.to_csv(out_path)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
