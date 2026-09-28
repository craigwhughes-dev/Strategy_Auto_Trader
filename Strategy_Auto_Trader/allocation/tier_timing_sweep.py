"""Timing sensitivity sweep for the deployed 4-tier allocation strategy (run B, asym10d).

Sweeps monthly start dates from --start-from to --start-to, end date fixed.
Uses hourly_spliced VXN/VIX so the full 26-year synthetic history is available.

Strategy-owned: asym10d filter, 13 bps/switch cost — matches the deployed run B config.

Run:
  uv run python -m Strategy_Auto_Trader.allocation.tier_timing_sweep

Outputs:
  data/tier_timing_sweep/results.csv
  data/tier_timing_sweep/sharpe_vs_start.png
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .multi_tier_backtest_lse_lag import (
    bars_by_end_time,
    build_tier_returns,
    load_hourly_index,
    load_lse_cutoff,
    net_of_switch_cost,
    strategy_returns,
    summarise,
    switch_flags,
    tier_series,
)
from .tier_filters import apply_asymmetric_hysteresis

_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPLICED_DIR = _REPO_ROOT / "data_synthetic" / "hourly_spliced"
_HOURLY_DIR = _REPO_ROOT / "data_synthetic" / "hourly"
_OUT_DIR = _REPO_ROOT / "data" / "tier_timing_sweep"
_COST_BPS = 13.0
_ASYM_DAYS = 10
_MIN_YEARS = 2.0


def _build_full_net_returns(end_date: str) -> tuple[pd.Series, pd.Series]:
    """Compute full-period asym10d net returns and switch flags once."""
    tz, cutoff = load_lse_cutoff()
    vxn = load_hourly_index(_SPLICED_DIR / "VXN.csv")
    vix = load_hourly_index(_SPLICED_DIR / "VIX.csv")
    vxn_end = bars_by_end_time(vxn)
    vix_end = bars_by_end_time(vix)

    tier_returns = build_tier_returns(_HOURLY_DIR, "1999-01-01", end_date)
    dates = tier_returns.index
    _log.info("Tier returns: %s .. %s (%d days)", dates[0].date(), dates[-1].date(), len(dates))

    cutoff_tiers = tier_series(dates, vxn_end, vix_end, tz, cutoff)
    asym_tiers = apply_asymmetric_hysteresis(cutoff_tiers, _ASYM_DAYS)

    b_asym = strategy_returns(asym_tiers, tier_returns, lag=1)
    switches = switch_flags(asym_tiers, b_asym.index)
    net = net_of_switch_cost(b_asym, switches, _COST_BPS)
    return net, switches


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start-from", default="2000-01-01", help="First start date in sweep")
    p.add_argument("--start-to", default="2024-01-01", help="Last start date in sweep")
    p.add_argument("--end-date", default="2026-09-15", help="Fixed end date for all runs")
    p.add_argument("--out-dir", default=str(_OUT_DIR))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    net_full, switches_full = _build_full_net_returns(args.end_date)

    sweep_starts = pd.date_range(args.start_from, args.start_to, freq="MS")
    end_dt = pd.Timestamp(args.end_date)
    min_days = int(_MIN_YEARS * 252)

    rows = []
    for start in sweep_starts:
        r = net_full.loc[start:end_dt]
        if len(r) < min_days:
            continue
        sw = switches_full.loc[start:end_dt]
        years = len(r) / 252
        s = summarise(r, 100_000.0)
        rows.append({
            "start_date": start.date(),
            "n_days": len(r),
            "years": round(years, 1),
            "sharpe": round(s["sharpe"], 3),
            "sortino": round(s["sortino"], 3),
            "return_pct": round(s["total_return_pct"], 1),
            "max_dd_pct": round(s["max_drawdown_pct"], 2),
            "n_switches": int(sw.sum()),
            "switches_per_year": round(sw.sum() / years, 1),
        })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    _log.info("Saved %d rows -> %s", len(df), csv_path)

    _plot(df, args.end_date, out_dir)
    print(df.to_string(index=False))


def _plot(df: pd.DataFrame, end_date: str, out_dir: Path) -> None:
    starts = pd.to_datetime(df["start_date"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(starts, df["sharpe"], color="steelblue", lw=1.5)
    axes[0].axhline(0, color="red", lw=0.8, ls="--")
    axes[0].set_ylabel("Sharpe (annualised)")
    axes[0].set_title(
        f"Deployed 4-tier asym{_ASYM_DAYS}d, {_COST_BPS:.0f}bps/sw — "
        f"timing sensitivity (end fixed {end_date})"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(starts, df["return_pct"], color="forestgreen", lw=1.5)
    axes[1].axhline(0, color="red", lw=0.8, ls="--")
    axes[1].set_ylabel("Total return %")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(starts, df["max_dd_pct"], color="firebrick", lw=1.5)
    axes[2].set_ylabel("Max drawdown %")
    axes[2].set_xlabel("Start date")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = out_dir / "sharpe_vs_start.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    _log.info("Saved chart -> %s", chart_path)


if __name__ == "__main__":
    main()
