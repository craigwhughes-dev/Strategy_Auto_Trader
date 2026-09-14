"""Overlay cash parking returns on a live_sim position_summary.

Reads an existing live_sim position_summary CSV, reconstructs daily equity,
simulates cash parking on idle cash using the HMM + VIX signal, then reports
combined metrics: main strategy alone vs main + parking.

Usage:
    python scripts/combined_backtest_analysis.py \
        --position-summary data/journals/combined_fresh_summary_20260914.csv \
        [--pot-size 100000] [--start 2023-01-01]
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_BARS_PER_YEAR = 252
_TIER_RET_COL = {"equity": "isf_ret", "gilts": "igls_ret", "cash": "xstr_ret"}

from Strategy_Auto_Trader.cash_parking.signals import fetch_parking_signals as _fetch_parking_signals_impl


def _sharpe(rets: np.ndarray) -> float:
    if len(rets) < 2 or np.std(rets) == 0:
        return 0.0
    return float(np.mean(rets) / np.std(rets) * math.sqrt(_BARS_PER_YEAR))


def _sortino(rets: np.ndarray) -> float:
    down = rets[rets < 0]
    if len(down) < 2 or np.std(down) == 0:
        return 0.0
    return float(np.mean(rets) / np.std(down) * math.sqrt(_BARS_PER_YEAR))


def _max_drawdown(equity: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    return float(dd.min())


def load_position_summary(path: str, pot_size: float) -> pd.DataFrame:
    """Load position_summary, drop SUMMARY row, filter pot, sort by date."""
    df = pd.read_csv(path)
    df = df[df["date"] != "SUMMARY"].copy()
    df = df[df["pot_size"] == pot_size].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_daily_equity(ps: pd.DataFrame, pot_size: float) -> pd.Series:
    """Resample sparse position_summary equity curve to daily."""
    ps_indexed = ps.set_index("date")["portfolio_value"]
    # Extend to today
    end = ps_indexed.index.max() + timedelta(days=1)
    daily_idx = pd.date_range(ps_indexed.index.min(), end, freq="D")
    daily = ps_indexed.reindex(daily_idx).ffill().fillna(pot_size)
    return daily


def build_daily_cash(ps: pd.DataFrame, pot_size: float) -> pd.Series:
    """Resample sparse cash column to daily (forward-fill between trade events)."""
    ps_indexed = ps.set_index("date")["cash"]
    end = ps_indexed.index.max() + timedelta(days=1)
    daily_idx = pd.date_range(ps_indexed.index.min(), end, freq="D")
    daily = ps_indexed.reindex(daily_idx).ffill().fillna(pot_size)
    return daily


def fetch_parking_signals(start: str, end: str) -> pd.DataFrame:
    return _fetch_parking_signals_impl(start, end)


def tier_for(pbull: float, vix: float) -> str:
    from Strategy_Auto_Trader.cash_parking.strategy import tier_for as _tier_for
    return _tier_for(pbull, vix)


def simulate_parking_overlay(
    daily_cash: pd.Series,
    signals: pd.DataFrame,
    liquid_floor_pct: float = 0.10,
    rebalance_threshold_pct: float = 0.05,
    pot_size: float = 100_000.0,
    t2_days: int = 2,
) -> pd.Series:
    """Simulate cash parking returns on idle cash. Returns daily parking P&L series."""
    common_idx = daily_cash.index.intersection(signals.index)
    cash_s = daily_cash.reindex(common_idx)
    sig_s = signals.reindex(common_idx)

    parking_equity = 0.0    # current value of parked position
    current_tier = "cash"
    settling_days_left = 0
    parking_daily_pnl = []
    idx_out = []

    for dt in common_idx:
        row = sig_s.loc[dt]
        parkable = float(cash_s.loc[dt]) * (1 - liquid_floor_pct)

        # Accrue return on current parking position
        if parking_equity > 0:
            col = _TIER_RET_COL.get(current_tier, "xstr_ret")
            ret = float(row.get(col, 0.0))
            daily_gain = parking_equity * ret if pd.notna(ret) else 0.0
            parking_equity *= (1 + ret) if pd.notna(ret) else 1.0
        else:
            daily_gain = 0.0

        # Accrue SONIA on liquid floor (cash × floor × xstr_ret)
        xstr_ret = float(row.get("xstr_ret", 0.0))
        if pd.notna(xstr_ret):
            floor_cash = float(cash_s.loc[dt]) * liquid_floor_pct
            daily_gain += floor_cash * xstr_ret

        parking_daily_pnl.append(daily_gain)
        idx_out.append(dt)

        # Tier decision
        if settling_days_left > 0:
            settling_days_left -= 1
            continue

        pbull = float(row.get("p_bull_smooth", 0.5))
        vix = float(row.get("vix", 20.0))
        if pd.isna(pbull) or pd.isna(vix):
            continue

        desired_tier = tier_for(pbull, vix)

        if desired_tier != current_tier:
            delta = abs(parkable - parking_equity)
            if delta < rebalance_threshold_pct * pot_size and current_tier != "cash":
                continue
            parking_equity = parkable
            current_tier = desired_tier
            settling_days_left = t2_days if desired_tier != "cash" else 0

    return pd.Series(parking_daily_pnl, index=pd.DatetimeIndex(idx_out))


def print_combined_report(
    main_equity: pd.Series,
    parking_pnl: pd.Series,
    pot_size: float,
    label: str = "optimised_new",
) -> None:
    common = main_equity.index.intersection(parking_pnl.index)
    main_e = main_equity.reindex(common)
    parking_cum = parking_pnl.reindex(common).fillna(0).cumsum()
    combined_e = main_e + parking_cum

    main_rets = main_e.pct_change().dropna().values
    combined_rets = combined_e.pct_change().dropna().values

    total_days = (common[-1] - common[0]).days
    total_years = total_days / 365.25

    def stats(equity_arr, rets_arr, name):
        total_ret = float(equity_arr[-1] / equity_arr[0] - 1)
        ann_ret = (1 + total_ret) ** (1 / total_years) - 1 if total_years > 0 else 0
        sh = _sharpe(rets_arr)
        so = _sortino(rets_arr)
        dd = _max_drawdown(equity_arr)
        pnl = float(equity_arr[-1] - equity_arr[0])
        print(f"  {name:<28} Sharpe={sh:.3f}  Sortino={so:.3f}  "
              f"TotRet={total_ret*100:.1f}%  AnnRet={ann_ret*100:.1f}%  "
              f"MaxDD={dd*100:.1f}%  P&L=GBP{pnl:,.0f}")

    print(f"\n=== Combined Backtest: {label} + Cash Parking ===")
    print(f"Period: {common[0].date()} to {common[-1].date()} ({total_years:.1f} yrs)  Pot: GBP{pot_size:,.0f}")
    print()
    stats(main_e.values, main_rets, "Main strategy only")
    stats(combined_e.values, combined_rets, "Main + cash parking")
    parking_total = float(parking_cum.iloc[-1])
    print(f"\n  Cash parking contribution:  +GBP{parking_total:,.0f}  "
          f"({parking_total/pot_size*100:.1f}% of initial pot)")

    # Save chart
    _save_chart(main_e, combined_e, parking_cum, label)


def _save_chart(main_e, combined_e, parking_cum, label):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        ax1 = axes[0]
        ax1.plot(main_e.index, main_e.values / main_e.iloc[0] * 100 - 100,
                 label="Main strategy", color="steelblue", linewidth=1.5)
        ax1.plot(combined_e.index, combined_e.values / combined_e.iloc[0] * 100 - 100,
                 label="Main + parking", color="darkorange", linewidth=1.5, linestyle="--")
        ax1.axhline(0, color="gray", linewidth=0.5)
        ax1.set_ylabel("Total return %")
        ax1.legend()
        ax1.set_title(f"{label} + Cash Parking overlay")
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.fill_between(parking_cum.index, 0, parking_cum.values,
                         color="green", alpha=0.4, label="Parking cumulative P&L")
        ax2.set_ylabel("Parking P&L GBP")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        reports = ROOT / "reports"
        reports.mkdir(exist_ok=True)
        out = reports / f"{label}_combined_parking_chart.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\n  Chart: {out}")
    except Exception as exc:
        logger.warning(f"Chart save failed: {exc}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--position-summary", required=True)
    p.add_argument("--pot-size", type=float, default=100_000.0)
    p.add_argument("--liquid-floor", type=float, default=0.10)
    args = p.parse_args(argv)

    logger.info(f"Loading position summary: {args.position_summary}")
    ps = load_position_summary(args.position_summary, args.pot_size)
    if ps.empty:
        print(f"No rows for pot_size={args.pot_size}")
        return 1

    start = ps["date"].min().strftime("%Y-%m-%d")
    end = ps["date"].max().strftime("%Y-%m-%d")
    logger.info(f"Data range: {start} → {end}")

    strategy = str(ps["strategy"].iloc[0])
    daily_equity = build_daily_equity(ps, args.pot_size)
    daily_cash = build_daily_cash(ps, args.pot_size)

    signals = fetch_parking_signals(start, end)
    if signals.empty:
        print("Could not fetch parking signals")
        return 1

    logger.info("Simulating cash parking overlay...")
    parking_pnl = simulate_parking_overlay(
        daily_cash, signals,
        liquid_floor_pct=args.liquid_floor,
        pot_size=args.pot_size,
    )

    print_combined_report(daily_equity, parking_pnl, args.pot_size, label=strategy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
