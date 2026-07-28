#!/usr/bin/env python
"""Test Stooq fetcher with sample data."""
from Strategy_Auto_Trader.quant_hmm.data_cache import fetch_hourly_stooq

tickers = ["AAPL", "SPY", "GSK.L"]
for ticker in tickers:
    try:
        df = fetch_hourly_stooq(ticker)
        print(f"OK {ticker}: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Close range: {df['Close'].min():.2f} - {df['Close'].max():.2f}")
    except Exception as e:
        print(f"FAIL {ticker}: {e}")
