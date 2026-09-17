#!/usr/bin/env python
"""Compare XSTR.L (distributing) vs CSH2.L (accumulating) as tier-4 cash assets.

XSTR pays semi-annual dividends, so a price-only series understates its return
(price drifts down ~0.2%/yr as distributions leave the fund). Total return here
reinvests each dividend at the ex-date close. Yahoo's "Adj Close" is NOT
dividend-adjusted for XSTR.L, so it is not used.

Inputs:
  data/cache/ibkr_daily/XSTR.L.csv        real IBKR daily closes (pence)
  data/cache/XSTR.L_dividends.csv          ex_date, dividend_pence (from yfinance)
  data/cache/ibkr_hourly/CSH2.L.csv        real IBKR hourly, resampled to daily
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"


def load_xstr_total_return() -> tuple[pd.Series, pd.Series]:
    """Return (price index, total-return index), both normalised to 1.0 at start."""
    px = pd.read_csv(CACHE / "ibkr_daily" / "XSTR.L.csv", index_col=0, parse_dates=True)["Close"]
    px.index = pd.to_datetime(px.index, utc=True).tz_convert(None).normalize()
    px = px[~px.index.duplicated()]
    div = pd.read_csv(CACHE / "XSTR.L_dividends.csv", index_col=0, parse_dates=True)["dividend_pence"]
    # Roll each ex-date forward to the first price bar on/after it (ex-dates can land on non-trading days).
    div_on_bars = pd.Series(0.0, index=px.index)
    for ex_date, amount in div.items():
        pos = px.index.searchsorted(ex_date)
        if pos < len(px):
            div_on_bars.iloc[pos] += amount
    daily_tr = px.pct_change() + div_on_bars / px.shift(1)
    tr = (1 + daily_tr.fillna(0)).cumprod()
    return px / px.iloc[0], tr


def load_csh2_real() -> pd.Series:
    c = pd.read_csv(CACHE / "ibkr_hourly" / "CSH2.L.csv", index_col=0, parse_dates=True)["Close"]
    c.index = pd.to_datetime(c.index, utc=True).tz_convert(None)
    return c.resample("D").last().dropna()


def metrics(s: pd.Series, start: str, end: str) -> dict:
    s = s.loc[start:end].dropna()
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    total = s.iloc[-1] / s.iloc[0] - 1
    r = s.pct_change().dropna()
    cum = (1 + r).cumprod()
    return {
        "total": total,
        "cagr": (1 + total) ** (1 / yrs) - 1,
        "vol": r.std() * np.sqrt(252),
        "max_dd": (cum / cum.cummax() - 1).min(),
    }


def calendar_year_returns(s: pd.Series) -> pd.Series:
    return s.resample("YE").last().pct_change().dropna()


def main() -> None:
    xstr_px, xstr_tr = load_xstr_total_return()
    csh2 = load_csh2_real()
    start = max(xstr_tr.index[0], csh2.index[0]).strftime("%Y-%m-%d")
    end = min(xstr_tr.index[-1], csh2.index[-1]).strftime("%Y-%m-%d")

    print(f"Real-vs-real window (IBKR): {start} -> {end}\n")
    print(f"{'series':<28} {'total':>8} {'CAGR':>8} {'vol':>7} {'max DD':>8}")
    for name, s in [("XSTR total return", xstr_tr), ("XSTR price only", xstr_px), ("CSH2 (acc)", csh2)]:
        m = metrics(s, start, end)
        print(f"{name:<28} {m['total']*100:7.2f}% {m['cagr']*100:7.2f}% {m['vol']*100:6.2f}% {m['max_dd']*100:7.2f}%")

    print("\nCalendar-year returns (%)")
    cy = pd.DataFrame({
        "XSTR TR": calendar_year_returns(xstr_tr),
        "XSTR px": calendar_year_returns(xstr_px),
        "CSH2": calendar_year_returns(csh2),
    })
    cy.index = cy.index.year
    print((cy * 100).round(2).to_string(na_rep="-"))

    gap = metrics(csh2, start, end)["cagr"] - metrics(xstr_tr, start, end)["cagr"]
    print(f"\nCSH2 minus XSTR total-return CAGR: {gap*100:+.2f}%/yr "
          "(TER 0.05% vs 0.15%, plus CSH2 swap targets SONIA + spread)")


if __name__ == "__main__":
    main()
