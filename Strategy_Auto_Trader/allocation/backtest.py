"""Daily allocation backtest harness with parameter sweep.

Runs backtest over 2015-2024 for all (vix_threshold, pbull_threshold, defensive_asset)
combinations. Outputs summary.csv and plots.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..broker.ibkr_data import IBKRDataClient
from .rotator import AllocationRotator

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

_log = logging.getLogger(__name__)


def fetch_data(
    start_date: str,
    end_date: str,
    market_ticker: str = "SPY",
    ibkr_client: IBKRDataClient | None = None,
    use_cache: bool = True,
    prefer_yfinance: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV for all assets and VIX.

    Args:
        start_date: "YYYY-MM-DD"
        end_date: "YYYY-MM-DD"
        market_ticker: Market asset to test (SPY, ^FTSE, etc.)
        ibkr_client: IBKR client (created if None)
        use_cache: Use on-disk cache if available
        prefer_yfinance: Use yfinance instead of IBKR (for testing)

    Returns:
        Dict with market_ticker, GLD, TLT, SHV/CSH2, VIX keys.
        CSH2 replaced with SHV (short-term treasury) for yfinance compatibility.
    """
    if ibkr_client is None:
        ibkr_client = IBKRDataClient()

    # yfinance doesn't have CSH2 (UK fund), use SHV as proxy
    defensive = ["SHV", "GLD", "TLT"] if prefer_yfinance else ["CSH2", "GLD", "TLT"]
    tickers = [market_ticker] + defensive + ["VIX"]
    data = {}

    for ticker in tickers:
        df = None

        if prefer_yfinance and HAS_YFINANCE:
            yf_ticker = ticker if ticker != "VIX" else "^VIX"
            try:
                df = yf.download(yf_ticker, start=start_date, end=end_date, progress=False)
                if df is not None and not df.empty:
                    # yfinance returns MultiIndex columns (price_type, ticker)
                    # Extract price_type (first element) and uppercase
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[0] for col in df.columns]
                    df.columns = [col.upper() for col in df.columns]
                    df.index.name = None
                    _log.info(f"Fetched {ticker} (yfinance): {len(df)} bars, columns={df.columns.tolist()}")
            except Exception as e:
                _log.warning(f"yfinance fetch {ticker} failed: {e}")
                df = None
        else:
            if ticker == "VIX":
                # Fetch VIX as index, not stock
                df = ibkr_client.fetch_index_daily("VIX", "CBOE", currency="USD",
                                                    historical_only=False)
            else:
                df = ibkr_client.fetch_daily(ticker, period="max", use_cache=use_cache,
                                              historical_only=False)

            if df is not None and not df.empty:
                # Trim to requested date range
                df = df.loc[start_date:end_date]
                _log.info(f"Fetched {ticker} (IBKR): {len(df)} bars")

        if df is not None and not df.empty:
            data[ticker] = df
        else:
            _log.warning(f"Failed to fetch {ticker}")
            data[ticker] = pd.DataFrame()

    return data


def run_allocation_backtest(
    market_df: pd.DataFrame,
    defensive_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    vix_threshold: float,
    pbull_threshold: float = 0.5,
    initial_cash: float = 100_000.0,
    market_ticker: str = "SPY",
    mode: str = "binary",
    vix_min: float = 10.0,
    vix_max: float = 30.0,
) -> dict:
    """Run single backtest variant."""
    # Normalize column names to uppercase
    market_df.columns = [c.upper() for c in market_df.columns]
    defensive_df.columns = [c.upper() for c in defensive_df.columns]
    vix_df.columns = [c.upper() for c in vix_df.columns]

    rotator = AllocationRotator(
        market_ticker=market_ticker,
        vix_threshold=vix_threshold,
        pbull_threshold=pbull_threshold,
        mode=mode,
        vix_min=vix_min,
        vix_max=vix_max,
    )

    result = rotator.backtest(
        market_df=market_df,
        defensive_df=defensive_df,
        vix_df=vix_df,
        pbull_series=None,  # HMM integration deferred
        initial_cash=initial_cash,
    )

    return result


def sweep_backtests(
    data: dict[str, pd.DataFrame],
    vix_thresholds: list[float],
    pbull_thresholds: list[float],
    defensive_assets: list[str],
    initial_cash: float = 100_000.0,
    market_ticker: str = "SPY",
    mode: str = "binary",
    vix_min: float = 10.0,
    vix_max: float = 30.0,
) -> pd.DataFrame:
    """Sweep all combinations and return summary table."""
    rows = []

    if market_ticker not in data or data[market_ticker].empty:
        _log.error(f"Market ticker {market_ticker} has no data")
        return pd.DataFrame()

    for defensive in defensive_assets:
        if data[defensive].empty:
            _log.warning(f"Skipping {defensive}: no data")
            continue

        for vix_th in vix_thresholds:
            for pbull_th in pbull_thresholds:
                try:
                    result = run_allocation_backtest(
                        market_df=data[market_ticker],
                        defensive_df=data[defensive],
                        vix_df=data["VIX"],
                        vix_threshold=vix_th,
                        pbull_threshold=pbull_th,
                        initial_cash=initial_cash,
                        market_ticker=market_ticker,
                        mode=mode,
                        vix_min=vix_min,
                        vix_max=vix_max,
                    )

                    summary = result["summary"]
                    rows.append({
                        "defensive_asset": defensive,
                        "vix_threshold": vix_th,
                        "pbull_threshold": pbull_th,
                        "n_days": summary["n_days"],
                        "total_return_pct": summary["total_return_pct"],
                        "sharpe": summary["sharpe"],
                        "sortino": summary["sortino"],
                        "max_drawdown_pct": summary["max_drawdown_pct"],
                        "pct_time_in_market": summary["pct_time_in_market"],
                    })

                    _log.info(
                        f"{defensive:6s} VIX={vix_th:5.1f} P(B)={pbull_th:4.2f}: "
                        f"Sharpe={summary['sharpe']:6.2f} Sortino={summary['sortino']:6.2f} "
                        f"DD={summary['max_drawdown_pct']:6.2f}% Return={summary['total_return_pct']:6.2f}%"
                    )

                except Exception as e:
                    _log.error(
                        f"Backtest failed {defensive} VIX={vix_th} P(B)={pbull_th}: {e}",
                        exc_info=True,
                    )

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Allocation backtest sweep")
    parser.add_argument(
        "--start-date",
        default="2015-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        default="2024-12-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100_000.0,
        help="Starting capital",
    )
    parser.add_argument(
        "--market-ticker",
        default="SPY",
        help="Market asset (SPY, FTSE100, etc.)",
    )
    parser.add_argument(
        "--mode",
        choices=["binary", "graduated"],
        default="binary",
        help="Allocation mode: binary (cliff) or graduated (smooth)",
    )
    parser.add_argument(
        "--vix-min",
        type=float,
        default=10.0,
        help="VIX value for 100% market (graduated mode)",
    )
    parser.add_argument(
        "--vix-max",
        type=float,
        default=30.0,
        help="VIX value for 0% market (graduated mode)",
    )
    parser.add_argument(
        "--vix-thresholds",
        type=float,
        nargs="+",
        default=[15.0, 20.0, 25.0],
        help="VIX threshold(s) to test (binary mode only)",
    )
    parser.add_argument(
        "--pbull-thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
        help="P(Bull) threshold(s) to test",
    )
    parser.add_argument(
        "--defensive-assets",
        nargs="+",
        default=None,
        help="Defensive assets to test (default: CSH2/GLD/TLT for IBKR, SHV/GLD/TLT for yfinance)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/allocation_backtest",
        help="Output directory",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use IBKR cache",
    )
    parser.add_argument(
        "--use-yfinance",
        action="store_true",
        help="Use yfinance instead of IBKR (testing only)",
    )

    args = parser.parse_args()

    # Set defensive asset defaults based on data source
    if args.defensive_assets is None:
        args.defensive_assets = ["SHV", "GLD", "TLT"] if args.use_yfinance else ["CSH2", "GLD", "TLT"]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir) / f"allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    _log.info(f"Output: {output_dir}")
    _log.info(f"Period: {args.start_date} to {args.end_date}")
    _log.info(f"Data source: {'yfinance (testing)' if args.use_yfinance else 'IBKR'}")
    _log.info(f"Market asset: {args.market_ticker}")
    _log.info(f"Allocation mode: {args.mode}")
    if args.mode == "graduated":
        _log.info(f"VIX range: {args.vix_min}-{args.vix_max}")
    else:
        _log.info(f"VIX thresholds: {args.vix_thresholds}")
    _log.info(f"P(Bull) thresholds: {args.pbull_thresholds}")
    _log.info(f"Defensive assets: {args.defensive_assets}")

    # Fetch data
    _log.info("Fetching data...")
    ibkr_client = IBKRDataClient()
    data = fetch_data(
        args.start_date,
        args.end_date,
        market_ticker=args.market_ticker,
        ibkr_client=ibkr_client,
        use_cache=not args.no_cache,
        prefer_yfinance=args.use_yfinance,
    )

    # Run sweep
    _log.info("Running backtest sweep...")
    summary_df = sweep_backtests(
        data,
        vix_thresholds=args.vix_thresholds,
        pbull_thresholds=args.pbull_thresholds,
        defensive_assets=args.defensive_assets,
        initial_cash=args.initial_cash,
        market_ticker=args.market_ticker,
        mode=args.mode,
        vix_min=args.vix_min,
        vix_max=args.vix_max,
    )

    if summary_df.empty:
        _log.error("No results; backtest failed")
        return

    # Sort by Sharpe descending
    summary_df = summary_df.sort_values("sharpe", ascending=False)

    # Save summary
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    _log.info(f"Summary saved: {summary_path}")

    # Print top 10
    print("\n" + "=" * 120)
    print("TOP 10 CONFIGURATIONS (by Sharpe)")
    print("=" * 120)
    print(summary_df.head(10).to_string(index=False))

    print("\n" + "=" * 120)
    print(f"Full results: {summary_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()
