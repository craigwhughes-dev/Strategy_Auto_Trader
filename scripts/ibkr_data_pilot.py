"""One-off validation: compare IBKR historical bars against yfinance for a
small mixed basket of tickers, before deciding whether to wire IBKR in as a
`--source` option anywhere. Prints bar count, date range, and max abs price
diff on overlapping timestamps for each ticker/source pair.

Requires TWS or IB Gateway running and logged in (paper account is fine);
connects with client_id=2 so it never collides with a live daemon on
client_id=1.

Usage:
    uv run python scripts/ibkr_data_pilot.py [--tickers SPY AAPL HSBA.L BT-A.L]
                                             [--period 730d] [--port 7497]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Strategy_Auto_Trader.broker.ibkr_data import IBKRDataClient  # noqa: E402
from Strategy_Auto_Trader.quant_hmm.quant_engine import fetch_hourly  # noqa: E402

DEFAULT_TICKERS = ["SPY", "AAPL", "HSBA.L", "BT-A.L"]


def _describe(df: pd.DataFrame | None) -> dict:
    if df is None or df.empty:
        return {"bars": 0, "first": None, "last": None}
    return {
        "bars": len(df),
        "first": str(df.index[0]),
        "last": str(df.index[-1]),
        "days": (df.index[-1] - df.index[0]).days,
    }


def _max_price_diff(a: pd.DataFrame | None, b: pd.DataFrame | None) -> float | None:
    if a is None or b is None or a.empty or b.empty:
        return None
    a_close = a["Close"]
    b_close = b["Close"]
    a_close.index = a_close.index.tz_convert("UTC") if a_close.index.tz else a_close.index.tz_localize("UTC")
    b_close.index = b_close.index.tz_convert("UTC") if b_close.index.tz else b_close.index.tz_localize("UTC")
    joined = pd.concat([a_close.rename("a"), b_close.rename("b")], axis=1, join="inner")
    if joined.empty:
        return None
    return float((joined["a"] - joined["b"]).abs().max())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--period", default="730d")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=2)
    args = parser.parse_args(argv)

    client = IBKRDataClient(host=args.host, port=args.port, client_id=args.client_id)
    if not client.connect():
        print(f"Could not connect to TWS/Gateway at {args.host}:{args.port} "
              f"(client_id={args.client_id}). Is it running and logged in?")
        return 1

    print(f"{'ticker':<10} {'source':<9} {'bars':>6} {'days':>6}  {'first':<26} {'last':<26}")
    results = {}
    try:
        for ticker in args.tickers:
            yf_df = fetch_hourly(ticker, period=args.period)
            ib_df = client.fetch_hourly(ticker, period=args.period, use_cache=False)
            results[ticker] = (yf_df, ib_df)
            for name, df in (("yfinance", yf_df), ("ibkr", ib_df)):
                d = _describe(df)
                print(f"{ticker:<10} {name:<9} {d['bars']:>6} {d.get('days', 0):>6}  "
                      f"{str(d['first']):<26} {str(d['last']):<26}")
    finally:
        client.disconnect()

    print("\nPrice alignment (max abs close diff on overlapping timestamps):")
    for ticker, (yf_df, ib_df) in results.items():
        diff = _max_price_diff(yf_df, ib_df)
        print(f"  {ticker:<10} {diff if diff is not None else 'n/a (no overlap or missing data)'}")

    print("\nDone. Compare 'days' columns above against yfinance's ~730-day cap, "
          "and check the price-alignment diffs are small before switching any "
          "real backtest to --source ibkr.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
