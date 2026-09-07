"""Test TQ window sizes (1m/2m/3m/6m/1yr/2yr) for regime stability.

Goal: find the window where vol-gate survival rate is most consistent across
market regimes (2000-02 dot-com, 2008 crisis, 2020 COVID, 2022 rate hike,
2023-24 recovery). Pick on stability grounds, not P&L maximisation.

Usage:
    uv run python scripts/calibrate_tq_window.py
    uv run python scripts/calibrate_tq_window.py --tickers AAPL MSFT NFLX
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Strategy_Auto_Trader.quant_hmm.vol_screen import rolling_trend_quality
from Strategy_Auto_Trader.synthetic_backtest_data.generate import load_synthetic_hourly

_HOURLY_DIR = Path("data_synthetic/hourly")
_UNIVERSE_FILE = Path("config/universe_sp_ftse.json")

_WINDOWS = {
    "1m":  21,
    "2m":  42,
    "3m":  63,
    "6m":  126,
    "1yr": 252,
    "2yr": 504,
}

# Regime slices: (label, start, end)
_REGIMES = [
    ("dot-com 2000-02", "2000-01-01", "2002-12-31"),
    ("pre-GFC 2005-07", "2005-01-01", "2007-12-31"),
    ("GFC 2008-09",     "2008-01-01", "2009-12-31"),
    ("bull 2012-19",    "2012-01-01", "2019-12-31"),
    ("COVID 2020",      "2020-01-01", "2020-12-31"),
    ("rate-hike 2022",  "2022-01-01", "2022-12-31"),
    ("recovery 23-24",  "2023-01-01", "2024-12-31"),
    ("2025-26",         "2025-01-01", "2026-09-01"),
]

_TQ_THRESHOLD = 0.0


def _daily_ohlc_from_synthetic(ticker: str) -> pd.DataFrame | None:
    """Load synthetic hourly CSV and resample to daily OHLC exactly as
    daily_ohlc_from_hourly() does in the live system."""
    df = load_synthetic_hourly(ticker, hourly_dir=_HOURLY_DIR)
    if df is None:
        return None
    daily = df.resample("1D").agg({"High": "max", "Low": "min", "Close": "last"})
    return daily.dropna(subset=["Close"])


def _survival_rate(tq: pd.Series, start: str, end: str) -> float | None:
    """Fraction of trading days in [start, end] where tq >= threshold (non-NaN)."""
    window = tq.loc[start:end].dropna()
    if len(window) < 20:
        return None
    return (window >= _TQ_THRESHOLD).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Ticker subset (default: top-50 from universe)")
    parser.add_argument("--n", type=int, default=50,
                        help="How many tickers to sample from universe (default 50)")
    args = parser.parse_args()

    if args.tickers:
        tickers = args.tickers
    else:
        data = json.loads(_UNIVERSE_FILE.read_text(encoding="utf-8"))
        tickers = data["tickers"][:args.n]

    # Accumulate per-regime survival rates across all tickers
    # results[window_label][regime_label] = list of per-ticker survival rates
    results: dict[str, dict[str, list[float]]] = {
        w: {r[0]: [] for r in _REGIMES} for w in _WINDOWS
    }

    loaded = 0
    for ticker in tickers:
        daily = _daily_ohlc_from_synthetic(ticker)
        if daily is None or len(daily) < 200:
            continue
        loaded += 1
        for wlabel, wsize in _WINDOWS.items():
            tq = rolling_trend_quality(daily, window=wsize, min_periods=min(100, wsize))
            for regime_label, start, end in _REGIMES:
                rate = _survival_rate(tq, start, end)
                if rate is not None:
                    results[wlabel][regime_label].append(rate)

    print(f"\nTQ window calibration — {loaded} tickers, threshold={_TQ_THRESHOLD}")
    print(f"Survival rate = fraction of days where TQ >= {_TQ_THRESHOLD}\n")

    # Header
    regime_labels = [r[0] for r in _REGIMES]
    col_w = 16
    print(f"{'Window':<8}", end="")
    for r in regime_labels:
        print(f"{r[:col_w]:>{col_w}}", end="")
    print(f"{'std(regimes)':>{col_w}}")
    print("-" * (8 + col_w * (len(regime_labels) + 1)))

    for wlabel in _WINDOWS:
        row_rates = []
        print(f"{wlabel:<8}", end="")
        for r in regime_labels:
            vals = results[wlabel][r]
            if vals:
                mean = np.mean(vals)
                row_rates.append(mean)
                print(f"{mean:>{col_w}.1%}", end="")
            else:
                print(f"{'n/a':>{col_w}}", end="")
        if row_rates:
            print(f"{np.std(row_rates):>{col_w}.1%}")
        else:
            print()

    print("\nLowest std(regimes) = most stable across market conditions.")


if __name__ == "__main__":
    main()
