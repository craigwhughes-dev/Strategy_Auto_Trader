#!/usr/bin/env python
"""VXN threshold optimization for EQGB.L (GBP-hedged Nasdaq-100).

Test 6 VXN thresholds: hold EQGB.L only when daily VXN < threshold, else sit in cash.
Measure Sharpe, Sortino, max drawdown, total return for each threshold.
Report which VXN level gives best risk-adjusted return.
"""

from pathlib import Path
import pandas as pd
import numpy as np

CACHE_DIR = Path("data/cache/ibkr_hourly")

def load_data():
    """Load EQGB.L and VXN hourly data from cache."""
    print("Loading cached data...")

    eqgb = pd.read_csv(CACHE_DIR / "EQGB.L.csv", index_col=0, parse_dates=True)
    vxn = pd.read_csv(CACHE_DIR / "INDEX_VXN.csv", index_col=0, parse_dates=True)

    print(f"EQGB.L: {len(eqgb)} bars, {eqgb.index.min()} to {eqgb.index.max()}")
    print(f"VXN: {len(vxn)} bars, {vxn.index.min()} to {vxn.index.max()}")

    return eqgb, vxn


def resample_to_daily(eqgb, vxn):
    """Resample hourly data to daily (use Close)."""
    eqgb_daily = eqgb["Close"].resample("D").last()
    vxn_daily = vxn["Close"].resample("D").last()

    # Align on common dates
    common_dates = eqgb_daily.index.intersection(vxn_daily.index)
    eqgb_daily = eqgb_daily[common_dates]
    vxn_daily = vxn_daily[common_dates]

    print(f"\nAligned daily: {len(eqgb_daily)} days")
    return eqgb_daily, vxn_daily


def compute_metrics(returns_series):
    """Compute Sharpe, Sortino, max drawdown."""
    if len(returns_series) < 2 or returns_series.std() == 0:
        return {"Sharpe": np.nan, "Sortino": np.nan, "MaxDD": 0.0, "Return": 0.0}

    # Annualized (252 trading days)
    sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else np.nan

    # Sortino (downside only)
    downside = returns_series[returns_series < 0].std()
    sortino = returns_series.mean() / downside * np.sqrt(252) if downside > 0 else np.nan

    # Max drawdown
    cum_returns = (1 + returns_series).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()

    total_return = (cum_returns.iloc[-1] - 1) * 100  # %

    return {
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": max_dd * 100,  # %
        "Return": total_return
    }


def backtest_threshold(eqgb_daily, vxn_daily, vxn_threshold):
    """Backtest: hold EQGB.L only when VXN < threshold."""

    # Mark days when VXN < threshold
    in_position = vxn_daily < vxn_threshold

    # Calculate daily returns
    eqgb_returns = eqgb_daily.pct_change()

    # Apply mask: zero out returns on days we're not in position
    strategy_returns = eqgb_returns.copy()
    strategy_returns[~in_position] = 0.0

    # Metrics
    metrics = compute_metrics(strategy_returns)
    metrics["Days_In"] = in_position.sum()
    metrics["Days_Total"] = len(in_position)
    metrics["Allocation_%"] = (metrics["Days_In"] / metrics["Days_Total"]) * 100

    return metrics


def main():
    eqgb, vxn = load_data()
    eqgb_daily, vxn_daily = resample_to_daily(eqgb, vxn)

    thresholds = [12, 15, 18, 20, 23, 25]
    results = []

    print("\n" + "="*80)
    print("VXN THRESHOLD OPTIMIZATION (EQGB.L)")
    print("="*80)

    # Baseline: buy-and-hold
    baseline_returns = eqgb_daily.pct_change()
    baseline = compute_metrics(baseline_returns)
    baseline["Threshold"] = "BASELINE"
    baseline["Days_In"] = len(eqgb_daily)
    baseline["Days_Total"] = len(eqgb_daily)
    baseline["Allocation_%"] = 100.0
    results.append(baseline)

    print(f"\nBASELINE (Buy-Hold):")
    print(f"  Sharpe: {baseline['Sharpe']:7.3f}  Sortino: {baseline['Sortino']:7.3f}  "
          f"Return: {baseline['Return']:7.2f}%  MaxDD: {baseline['MaxDD']:7.2f}%")

    # Test each threshold
    print("\nVXN THRESHOLDS:")
    for thresh in thresholds:
        metrics = backtest_threshold(eqgb_daily, vxn_daily, thresh)
        metrics["Threshold"] = thresh
        results.append(metrics)

        print(f"  VXN<{thresh:2d}: Sharpe: {metrics['Sharpe']:7.3f}  Sortino: {metrics['Sortino']:7.3f}  "
              f"Return: {metrics['Return']:7.2f}%  MaxDD: {metrics['MaxDD']:7.2f}%  "
              f"Alloc: {metrics['Allocation_%']:5.1f}%")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    df_results = pd.DataFrame(results)
    df_results = df_results[["Threshold", "Sharpe", "Sortino", "Return", "MaxDD", "Allocation_%"]]
    print(df_results.to_string(index=False))

    # Best threshold
    best_sharpe = df_results[df_results["Threshold"] != "BASELINE"].loc[df_results["Sharpe"].idxmax()]
    print(f"\n>>> BEST SHARPE: VXN < {best_sharpe['Threshold']} (Sharpe={best_sharpe['Sharpe']:.3f})")

    # Save results
    df_results.to_csv("data/vxn_threshold_results.csv", index=False)
    print(f"\nResults saved to data/vxn_threshold_results.csv")


if __name__ == "__main__":
    main()
