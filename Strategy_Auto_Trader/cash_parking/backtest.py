"""Cash parking parameter sweep — tunes tier boundary constants in strategy.py.

Fetches parking signals via fetch_parking_signals (IBKR daily HMM + ETF returns),
then sweeps all boundary combinations as pure pandas ops (fast after the HMM run).

Usage:
    python -m Strategy_Auto_Trader.cash_parking.backtest --start 2009-01-01 --synthetic-data-dir data_synthetic/hourly

Output: ranked summary table (stdout).
Top result's parameters → replace GUESS constants in strategy.py.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BARS_PER_YEAR = 252  # daily bars


@dataclass
class TierBoundaries:
    equity_pbull_min: float = 0.75
    equity_vix_max: float = 12.0
    hy_pbull_min: float = 0.60
    hy_vix_max: float = 18.0
    gilts_pbull_min: float = 0.55
    gilts_vix_max: float = 22.0


def _tier_for(
    pbull: float,
    vix: float,
    b: TierBoundaries,
    liquid_floor_pct: float,
    parkable_frac: float,
) -> str:
    """Determine tier given boundaries and available parkable fraction."""
    if parkable_frac <= 0:
        return "cash"
    if pbull >= b.equity_pbull_min and vix < b.equity_vix_max:
        return "equity"
    if pbull >= b.hy_pbull_min and vix < b.hy_vix_max:
        return "hy_bonds"
    if pbull >= b.gilts_pbull_min and vix < b.gilts_vix_max:
        return "gilts"
    return "cash"


def _simulate_combo(
    daily_df: pd.DataFrame,
    boundaries: TierBoundaries,
    liquid_floor_pct: float,
    initial_cash: float = 100_000.0,
    t2_days: int = 2,
) -> dict:
    """Simulate cash parking for one parameter combination.

    daily_df columns: p_bull_smooth, vix, xstr_ret, igls_ret, isxf_ret, isf_ret
    All return columns are daily pct change (decimal, not percent).

    Returns summary statistics dict.
    """
    _TIER_RET_COL = {"equity": "isf_ret", "hy_bonds": "isxf_ret", "gilts": "igls_ret", "cash": "xstr_ret"}

    cash = initial_cash
    parked = 0.0
    tier = "cash"
    settling_days_left = 0
    equity_series = []
    parkable_frac = 1.0 - liquid_floor_pct

    for _, row in daily_df.iterrows():
        # Accrue return on parked position first
        if parked > 0:
            ret = row.get(_TIER_RET_COL.get(tier, "xstr_ret"), 0.0)
            if pd.notna(ret):
                parked *= (1 + ret)

        # Always accrue SONIA on liquid floor
        xstr_ret = row.get("xstr_ret", 0.0)
        if pd.notna(xstr_ret):
            cash_floor = initial_cash * liquid_floor_pct
            cash += cash_floor * xstr_ret

        # T+2 cooldown
        if settling_days_left > 0:
            settling_days_left -= 1
            equity_series.append(cash + parked)
            continue

        # Desired tier
        available = cash + parked
        desired_tier = _tier_for(
            float(row.get("p_bull_smooth", 0.5)),
            float(row.get("vix", 20.0)),
            boundaries,
            liquid_floor_pct,
            parkable_frac,
        )

        if desired_tier != tier:
            # Liquidate current parking
            cash += parked
            parked = 0.0
            tier = "cash"
            settling_days_left = t2_days

            if desired_tier != "cash":
                # Re-invest parkable fraction in new tier
                parkable = available * parkable_frac
                parked = parkable
                cash -= parkable
                tier = desired_tier
                settling_days_left = 0  # buy in same step after sell settles

        equity_series.append(cash + parked)

    if not equity_series:
        return {}

    eq = np.array(equity_series, dtype=float)
    daily_rets = np.diff(eq) / eq[:-1]
    total_ret = float(eq[-1] / eq[0] - 1)
    sharpe = (
        float(np.mean(daily_rets) / np.std(daily_rets) * math.sqrt(_BARS_PER_YEAR))
        if np.std(daily_rets) > 0 else 0.0
    )
    downside = daily_rets[daily_rets < 0]
    sortino = (
        float(np.mean(daily_rets) / np.std(downside) * math.sqrt(_BARS_PER_YEAR))
        if len(downside) and np.std(downside) > 0 else 0.0
    )
    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / running_max
    max_dd = float(drawdowns.min())

    return {
        "total_return": total_ret,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "final_equity": float(eq[-1]),
    }


def run_cash_parking_backtest(
    start_date: str = "2022-01-01",
    end_date: str | None = None,
    liquid_floor_pcts: tuple = (0.0, 0.10, 0.20, 0.30),
    equity_pbull_mins: tuple = (0.65, 0.70, 0.75, 0.80),
    equity_vix_maxes: tuple = (12.0, 15.0, 18.0),
    hy_pbull_mins: tuple = (0.55, 0.60, 0.65, 0.70),
    hy_vix_maxes: tuple = (15.0, 18.0, 22.0),
    gilts_pbull_mins: tuple = (0.45, 0.50, 0.55, 0.60),
    gilts_vix_maxes: tuple = (18.0, 20.0, 22.0, 25.0),
    initial_cash: float = 100_000.0,
    synthetic_data_dir: str | None = None,
) -> pd.DataFrame:
    """Sweep all parameter combinations and return ranked DataFrame.

    Steps:
    1. Fetch parking signals (p_bull_smooth, vix, xstr_ret, igls_ret, isxf_ret, isf_ret)
       via fetch_parking_signals — IBKR daily cache for ISF.L HMM, IBKR hourly
       (or synthetic hourly from synthetic_data_dir) for ETF returns.
    2. For each combo: simulate allocation, compute Sharpe/Sortino/drawdown.
    3. Also compute vs_xstr_alpha and vs_isf_alpha benchmarks.
    4. Return DataFrame sorted by Sharpe descending.

    Ordering constraint enforced: gilts_pbull < hy_pbull < equity_pbull
    (tighter pbull = higher bar to enter riskier tiers).
    """
    from .signals import fetch_parking_signals

    end_str = end_date or date.today().isoformat()

    logger.info("Fetching parking signals (IBKR daily HMM + ETF returns)...")
    daily = fetch_parking_signals(start_date, end_str, synthetic_data_dir=synthetic_data_dir)
    daily = daily.loc[start_date:end_str].dropna(subset=["p_bull_smooth"])

    if daily.empty:
        raise RuntimeError("No aligned daily data after join — check date range")

    logger.info(f"Aligned daily data: {len(daily)} days from {daily.index[0].date()} to {daily.index[-1].date()}")

    # --- Benchmarks ---
    xstr_only = _simulate_combo(
        daily, TierBoundaries(equity_pbull_min=99.0, hy_pbull_min=99.0, gilts_pbull_min=99.0),
        liquid_floor_pct=0.0, initial_cash=initial_cash,
    )
    isf_only = _simulate_combo(
        daily, TierBoundaries(equity_pbull_min=0.0, equity_vix_max=99.0,
                              hy_pbull_min=99.0, gilts_pbull_min=99.0),
        liquid_floor_pct=0.0, initial_cash=initial_cash,
    )
    xstr_total = xstr_only.get("total_return", 0.0)
    isf_total = isf_only.get("total_return", 0.0)

    # --- Sweep ---
    param_grid = list(product(
        liquid_floor_pcts,
        equity_pbull_mins, equity_vix_maxes,
        hy_pbull_mins, hy_vix_maxes,
        gilts_pbull_mins, gilts_vix_maxes,
    ))
    logger.info(f"Sweeping {len(param_grid)} parameter combinations...")

    rows = []
    for lf, eq_pb, eq_vx, hy_pb, hy_vx, gi_pb, gi_vx in param_grid:
        # Enforce ascending risk order: gilts < hy < equity
        if gi_pb >= hy_pb or hy_pb >= eq_pb:
            continue
        b = TierBoundaries(
            equity_pbull_min=eq_pb, equity_vix_max=eq_vx,
            hy_pbull_min=hy_pb, hy_vix_max=hy_vx,
            gilts_pbull_min=gi_pb, gilts_vix_max=gi_vx,
        )
        stats = _simulate_combo(daily, b, liquid_floor_pct=lf, initial_cash=initial_cash)
        if not stats:
            continue
        rows.append({
            "liquid_floor_pct": lf,
            "equity_pbull_min": eq_pb,
            "equity_vix_max": eq_vx,
            "hy_pbull_min": hy_pb,
            "hy_vix_max": hy_vx,
            "gilts_pbull_min": gi_pb,
            "gilts_vix_max": gi_vx,
            "sharpe": round(stats["sharpe"], 3),
            "sortino": round(stats["sortino"], 3),
            "total_return_pct": round(stats["total_return"] * 100, 2),
            "max_drawdown_pct": round(stats["max_drawdown"] * 100, 2),
            "vs_xstr_alpha_pct": round((stats["total_return"] - xstr_total) * 100, 2),
            "vs_isf_alpha_pct": round((stats["total_return"] - isf_total) * 100, 2),
        })

    if not rows:
        logger.warning("No valid combinations found")
        return pd.DataFrame()

    result_df = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    return result_df


def _print_ranked_table(df: pd.DataFrame, top_n: int = 20) -> None:
    if df.empty:
        print("No results.")
        return

    print("\n=== Cash Parking Parameter Sweep — Top Results ===")
    print(f"{'Rank':>4} {'LiqFloor':>8} {'EqPbull':>7} {'EqVix':>6} "
          f"{'HYPbull':>7} {'HYVix':>6} {'GiPbull':>7} {'GiVix':>6} "
          f"{'Sharpe':>7} {'Sortino':>7} {'TotRet%':>7} {'MaxDD%':>7} {'vsXSTR%':>7} {'vsISF%':>7}")
    print("-" * 115)
    for rank, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
        print(
            f"{rank:>4} {row.liquid_floor_pct:>8.2f} {row.equity_pbull_min:>7.2f} "
            f"{row.equity_vix_max:>6.1f} {row.hy_pbull_min:>7.2f} {row.hy_vix_max:>6.1f} "
            f"{row.gilts_pbull_min:>7.2f} {row.gilts_vix_max:>6.1f} "
            f"{row.sharpe:>7.3f} {row.sortino:>7.3f} "
            f"{row.total_return_pct:>7.2f} {row.max_drawdown_pct:>7.2f} "
            f"{row.vs_xstr_alpha_pct:>7.2f} {row.vs_isf_alpha_pct:>7.2f}"
        )

    best = df.iloc[0]
    print(f"\n--- Top result (bake into strategy.py) ---")
    print(f"LIQUID_FLOOR_PCT     = {best.liquid_floor_pct}")
    print(f"equity_pbull_min     = {best.equity_pbull_min}")
    print(f"equity_vix_max       = {best.equity_vix_max}")
    print(f"hy_pbull_min         = {best.hy_pbull_min}")
    print(f"hy_vix_max           = {best.hy_vix_max}")
    print(f"gilts_pbull_min      = {best.gilts_pbull_min}")
    print(f"gilts_vix_max        = {best.gilts_vix_max}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser(
        description="Cash parking parameter sweep — run after adding cash_parking module."
    )
    p.add_argument("--start", default="2022-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD, default today)")
    p.add_argument("--top", type=int, default=20, help="Rows to print in ranked table")
    p.add_argument("--initial-cash", type=float, default=100_000.0,
                   help="Notional starting cash for simulation (default 100000)")
    p.add_argument("--synthetic-data-dir", default=None,
                   help="Path to synthetic hourly CSVs (e.g. data_synthetic/hourly). "
                        "When set, ETF returns use Brownian-bridge synthetic data instead of IBKR hourly cache.")
    args = p.parse_args(argv)

    df = run_cash_parking_backtest(
        start_date=args.start,
        end_date=args.end,
        initial_cash=args.initial_cash,
        synthetic_data_dir=args.synthetic_data_dir,
    )
    _print_ranked_table(df, top_n=args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
