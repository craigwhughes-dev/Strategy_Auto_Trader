"""Walk-forward validation: train on 2015-2023, test on 2024 holdout.

Finds optimal VIX threshold on in-sample period, measures validation Sharpe.
If out-of-sample Sharpe within 20% of in-sample, signal generalizes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .rotator import AllocationRotator

_log = logging.getLogger(__name__)


def run_walk_forward(
    market_df: pd.DataFrame,
    defensive_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    train_end_date: str = "2023-12-31",
    test_start_date: str = "2024-01-01",
    test_end_date: str = "2024-12-31",
    vix_thresholds: list[float] | None = None,
    initial_cash: float = 100_000.0,
    market_ticker: str = "SPY",
    mode: str = "binary",
) -> dict:
    """Run walk-forward test: optimize on train, validate on test.

    Args:
        market_df: Full historical OHLCV
        defensive_df: Full historical OHLCV
        vix_df: Full historical VIX
        train_end_date: Last day of training period
        test_start_date: First day of test period
        test_end_date: Last day of test period
        vix_thresholds: Thresholds to sweep on train
        initial_cash: Starting capital
        market_ticker: Market asset name
        mode: "binary" or "graduated"

    Returns:
        Dict with:
            - best_threshold: VIX threshold with highest train Sharpe
            - train_results: dict with Sharpe, Sortino, DD, return
            - test_results: dict with Sharpe, Sortino, DD, return
            - degradation: (train_sharpe - test_sharpe) / train_sharpe * 100
    """
    if vix_thresholds is None:
        vix_thresholds = [12.5, 15.0, 17.5, 20.0, 22.5, 25.0]

    # Split into train/test
    train_mask = market_df.index <= train_end_date
    test_mask = (market_df.index >= test_start_date) & (market_df.index <= test_end_date)

    market_train = market_df[train_mask]
    market_test = market_df[test_mask]
    defensive_train = defensive_df[train_mask]
    defensive_test = defensive_df[test_mask]
    vix_train = vix_df[train_mask]
    vix_test = vix_df[test_mask]

    _log.info(f"Train period: {market_train.index[0]} to {market_train.index[-1]} ({len(market_train)} bars)")
    _log.info(f"Test period: {market_test.index[0]} to {market_test.index[-1]} ({len(market_test)} bars)")

    # Normalize columns
    for df in [market_train, market_test, defensive_train, defensive_test, vix_train, vix_test]:
        df.columns = [c.upper() for c in df.columns]

    # Train: sweep thresholds, pick best Sharpe
    train_results = {}
    best_threshold = None
    best_sharpe = -np.inf

    _log.info("Training phase: sweeping VIX thresholds...")
    for vix_th in vix_thresholds:
        rotator = AllocationRotator(
            market_ticker=market_ticker,
            vix_threshold=vix_th,
            mode=mode,
        )
        result = rotator.backtest(
            market_df=market_train,
            defensive_df=defensive_train,
            vix_df=vix_train,
            initial_cash=initial_cash,
        )
        summary = result["summary"]
        train_results[vix_th] = summary

        _log.info(
            f"  VIX={vix_th:5.1f}: Sharpe={summary['sharpe']:7.2f} "
            f"Sortino={summary['sortino']:7.2f} DD={summary['max_drawdown_pct']:7.2f}% "
            f"Return={summary['total_return_pct']:7.2f}%"
        )

        if summary["sharpe"] > best_sharpe:
            best_sharpe = summary["sharpe"]
            best_threshold = vix_th

    _log.info(f"Best threshold: VIX {best_threshold} (Sharpe {best_sharpe:.2f})")

    # Test: apply best threshold to holdout
    _log.info("Validation phase: applying best threshold to holdout...")
    rotator = AllocationRotator(
        market_ticker=market_ticker,
        vix_threshold=best_threshold,
        mode=mode,
    )
    test_result = rotator.backtest(
        market_df=market_test,
        defensive_df=defensive_test,
        vix_df=vix_test,
        initial_cash=initial_cash,
    )
    test_summary = test_result["summary"]

    _log.info(
        f"Test results: Sharpe={test_summary['sharpe']:.2f} "
        f"Sortino={test_summary['sortino']:.2f} DD={test_summary['max_drawdown_pct']:.2f}% "
        f"Return={test_summary['total_return_pct']:.2f}%"
    )

    # Compute degradation
    degradation = (best_sharpe - test_summary["sharpe"]) / best_sharpe * 100 if best_sharpe > 0 else 0

    return {
        "best_threshold": best_threshold,
        "train_results": train_results,
        "train_summary": train_results[best_threshold],
        "test_summary": test_summary,
        "degradation_pct": degradation,
    }


def main():
    import argparse

    try:
        import yfinance as yf

        parser = argparse.ArgumentParser(description="Walk-forward validation")
        parser.add_argument("--market-ticker", default="SPY", help="Market asset")
        parser.add_argument(
            "--vix-thresholds",
            type=float,
            nargs="+",
            default=[12.5, 15.0, 17.5, 20.0, 22.5, 25.0],
            help="VIX thresholds to sweep",
        )
        parser.add_argument("--initial-cash", type=float, default=100_000, help="Starting capital")
        parser.add_argument("--mode", default="binary", choices=["binary", "graduated"])
        args = parser.parse_args()

        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

        # Fetch data
        print("\nFetching data...")
        market = yf.download(args.market_ticker, start="2015-01-01", end="2024-12-31", progress=False)
        defensive = yf.download("SHV", start="2015-01-01", end="2024-12-31", progress=False)
        vix = yf.download("^VIX", start="2015-01-01", end="2024-12-31", progress=False)

        # Flatten columns
        for df in [market, defensive, vix]:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df.columns = [col.upper() for col in df.columns]

        # Run walk-forward
        print()
        result = run_walk_forward(
            market,
            defensive,
            vix,
            vix_thresholds=args.vix_thresholds,
            initial_cash=args.initial_cash,
            market_ticker=args.market_ticker,
            mode=args.mode,
        )

        # Report
        train_sum = result["train_summary"]
        test_sum = result["test_summary"]
        degradation = result["degradation_pct"]

        print("\n" + "=" * 100)
        print("WALK-FORWARD VALIDATION RESULTS")
        print("=" * 100)
        print(f"\nBest threshold (on 2015-2023): VIX {result['best_threshold']}")
        print(f"\nTRAIN (2015-2023):")
        print(f"  Sharpe:    {train_sum['sharpe']:7.2f}")
        print(f"  Sortino:   {train_sum['sortino']:7.2f}")
        print(f"  Max DD:    {train_sum['max_drawdown_pct']:7.2f}%")
        print(f"  Return:    {train_sum['total_return_pct']:7.2f}%")

        print(f"\nTEST (2024 HOLDOUT):")
        print(f"  Sharpe:    {test_sum['sharpe']:7.2f}")
        print(f"  Sortino:   {test_sum['sortino']:7.2f}")
        print(f"  Max DD:    {test_sum['max_drawdown_pct']:7.2f}%")
        print(f"  Return:    {test_sum['total_return_pct']:7.2f}%")

        print(f"\nDEGRADATION: {degradation:.1f}%")
        if degradation < 0:
            print("[+] TEST OUTPERFORMED: 2024 was an unusually good year for signal")
            print("    Risk: Performance may not repeat; not overfitting but not conservative estimate")
        elif degradation < 20:
            print("[PASS] Signal generalizes (<20% degradation)")
        elif degradation < 50:
            print("[CAUTION] Moderate overfitting (20-50% degradation)")
        else:
            print("[FAIL] Severe overfitting (>50% degradation)")

        print("=" * 100 + "\n")

    except ImportError:
        print("yfinance required: pip install yfinance")


if __name__ == "__main__":
    main()
