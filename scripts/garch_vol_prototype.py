"""Prototype: does GARCH(1,1) forecast next-period volatility better than a naive
trailing rolling-std baseline (which is what vol_screen.volatility_profile's
ann_vol already approximates, just as a single static snapshot)?

Walk-forward evaluation, refitting GARCH periodically (not every bar - too slow):
  1. For each ticker, pull daily returns.
  2. Every REFIT_EVERY days, refit GARCH(1,1) on trailing WINDOW returns and hold the
     fitted params constant until next refit, forecasting 1-day-ahead conditional vol
     each day in between (this is how a live system would actually use it, not an
     oracle refit-every-day best case).
  3. Baseline: rolling REALIZED_WINDOW-day std of returns as the naive forecast.
  4. Score both against realized next-day squared return (proxy for realized vol)
     using MSE and rank correlation.

Usage:
    uv run python scripts/garch_vol_prototype.py --tickers WEIR.L,SGE.L,REL.L,EXPN.L,HSBA.L
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from scipy.stats import spearmanr

WINDOW = 252  # trailing days used to fit GARCH at each refit
REFIT_EVERY = 5  # trading days between refits (weekly-ish, matches realistic live cadence)
REALIZED_WINDOW = 20  # naive baseline rolling window
TARGET_HORIZON = 5  # forward realized-vol target window (days), less noisy than 1-day |return|


def fetch_returns(ticker: str, period: str = "3y") -> pd.Series:
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"no data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"].dropna()
    return close.pct_change().dropna() * 100  # arch wants returns in % scale


def garch_walkforward_vol(returns: pd.Series) -> pd.Series:
    """1-day-ahead conditional vol forecast, refit every REFIT_EVERY days."""
    forecasts = pd.Series(index=returns.index, dtype=float)
    n = len(returns)
    fitted_res = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(WINDOW, n):
            if fitted_res is None or (i - WINDOW) % REFIT_EVERY == 0:
                window_data = returns.iloc[i - WINDOW:i]
                model = arch_model(window_data, vol="Garch", p=1, q=1, dist="normal")
                fitted_res = model.fit(disp="off")
            fcast = fitted_res.forecast(horizon=1, reindex=False)
            forecasts.iloc[i] = np.sqrt(fcast.variance.values[-1, 0])
    return forecasts


def naive_walkforward_vol(returns: pd.Series) -> pd.Series:
    return returns.rolling(REALIZED_WINDOW).std().shift(1)


def evaluate_ticker(ticker: str) -> dict | None:
    try:
        returns = fetch_returns(ticker)
    except Exception as exc:
        print(f"  {ticker}: fetch failed ({exc})")
        return None
    if len(returns) < WINDOW + 60:
        print(f"  {ticker}: not enough history ({len(returns)} days)")
        return None

    garch_fc = garch_walkforward_vol(returns)
    naive_fc = naive_walkforward_vol(returns)

    # forward realized vol: std of the next TARGET_HORIZON days' returns, less noisy
    # than a single day's |return|. rolling(H).std()[j] = std(returns[j-H+1..j]),
    # so shifting by -H aligns it to std(returns[i+1..i+H]) at position i.
    realized_next = returns.rolling(TARGET_HORIZON).std().shift(-TARGET_HORIZON)

    valid = garch_fc.notna() & naive_fc.notna() & realized_next.notna()
    g, nv, target = garch_fc[valid], naive_fc[valid], realized_next[valid]

    garch_mse = float(((g - target) ** 2).mean())
    naive_mse = float(((nv - target) ** 2).mean())
    garch_corr, _ = spearmanr(g, target)
    naive_corr, _ = spearmanr(nv, target)

    return {
        "ticker": ticker,
        "n_obs": int(valid.sum()),
        "garch_mse": garch_mse,
        "naive_mse": naive_mse,
        "garch_wins_mse": garch_mse < naive_mse,
        "garch_corr": round(float(garch_corr), 4),
        "naive_corr": round(float(naive_corr), 4),
        "garch_wins_corr": garch_corr > naive_corr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", required=True, help="Comma-separated ticker list")
    args = parser.parse_args()
    tickers = [t.strip() for t in args.tickers.split(",")]

    results = []
    for ticker in tickers:
        print(f"evaluating {ticker}...")
        r = evaluate_ticker(ticker)
        if r:
            results.append(r)

    if not results:
        print("no results")
        return 1

    df = pd.DataFrame(results)
    print(f"\n{'Ticker':10s} {'N':>5s} {'GARCH_MSE':>10s} {'Naive_MSE':>10s} {'MSE_win':>8s} "
          f"{'GARCH_rho':>10s} {'Naive_rho':>10s} {'Rho_win':>8s}")
    for _, row in df.iterrows():
        print(f"{row['ticker']:10s} {row['n_obs']:>5d} {row['garch_mse']:>10.4f} {row['naive_mse']:>10.4f} "
              f"{'GARCH' if row['garch_wins_mse'] else 'naive':>8s} "
              f"{row['garch_corr']:>10.4f} {row['naive_corr']:>10.4f} "
              f"{'GARCH' if row['garch_wins_corr'] else 'naive':>8s}")

    print(f"\nGARCH beats naive on MSE: {df['garch_wins_mse'].sum()}/{len(df)}")
    print(f"GARCH beats naive on rank-corr: {df['garch_wins_corr'].sum()}/{len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
