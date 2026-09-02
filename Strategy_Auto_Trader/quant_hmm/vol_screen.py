"""Screen tickers by volatility *character*, not just magnitude.

The HMM quant engine (HMM regime probabilities) works best on stocks with
trending, momentum-driven price action and struggles on choppy, mean-reverting
names — the HMM regime keeps whipsawing, leading to repeated stop-losses.

This was validated empirically: across 23 FTSE tickers traded 2026-01-12 to
2026-06-29, Kaufman Efficiency Ratio correlated +0.29 with total P&L and
autocorrelation correlated +0.34 with win rate, while annualised volatility
and daily sign-change frequency correlated negatively with both. The worst
performers (WEIR.L, SGE.L, REL.L, EXPN.L, -£19 to -£41) all had efficiency
ratio < 0.07 — classic choppy/mean-reverting tape.

Usage:
    uv run python -m Strategy_Auto_Trader.vol_screen --tickers HSBA.L,WEIR.L,PCT.L
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from ..core.cli_logging import setup_cli_logger

logger = logging.getLogger(__name__)



def _trend_quality_score(
    efficiency_ratio: float | pd.Series,
    autocorr: float | pd.Series,
    ann_vol: float | pd.Series,
    sign_change_freq: float | pd.Series,
):
    """Composite trend-quality score, weighted by empirical correlation strength
    (see module docstring). Normalised so 0 = typical FTSE name, positive = more
    trend-friendly. Shared by volatility_profile() (single snapshot) and
    rolling_trend_quality() (time series) so the two can't drift apart —
    operates element-wise on either plain floats or aligned pandas Series.
    """
    return (
        1.5 * (efficiency_ratio - 0.07) / 0.05
        + 1.5 * (autocorr - 0.0) / 0.04
        - 1.0 * (ann_vol - 0.25) / 0.05
        - 1.0 * (sign_change_freq - 0.52) / 0.03
    )


def volatility_profile(ticker: str, period: str = "2y", source: str = "ibkr") -> dict | None:
    """Compute volatility-character metrics for a ticker from daily data.

    source="ibkr" derives daily OHLC by resampling quant_engine.fetch_hourly's
    IBKR-backed cache instead of a second, separate IBKR daily fetch/cache —
    no need for a parallel daily paging path. Resampling by UTC calendar day
    is safe here: both NYSE and LSE trading hours fall entirely within one
    UTC day (neither session crosses UTC midnight), so no bar ever gets
    split across two resampled days.

    The ibkr branch always requests the full growing on-disk cache ("max"),
    ignoring `period` — this project only ever runs source="ibkr" live, so
    there's no yfinance window to stay in parity with (unlike the old
    2y-matching behaviour this replaced). `period` still governs the
    yfinance branch below, which does have a real ~730d-ish practical
    ceiling on intraday-derived history.

    Returns dict with:
      - ann_vol: annualised total volatility (both up and down moves)
      - downside_vol: annualised downside volatility only (RMS of negative returns)
      - efficiency_ratio: Kaufman ER, net move / total path length (0-1, higher=more trending)
      - autocorr: lag-1 autocorrelation of daily returns (positive=momentum, negative=mean-reverting)
      - choppiness_idx: 14-day Choppiness Index, averaged (0-100, higher=choppier)
      - sign_change_freq: fraction of days where return sign flips vs prior day
      - trend_quality: composite score (higher = better fit for the HMM quant engine)
    """
    if source == "ibkr":
        from .quant_engine import fetch_hourly

        try:
            hourly = fetch_hourly(ticker, period="max", source="ibkr")
        except Exception:
            return None
        if hourly is None or hourly.empty:
            return None
        df = hourly.resample("1D").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna(subset=["Close"])
    else:
        import yfinance as yf
        from ..core.net_retry import call_with_timeout_retry

        df = call_with_timeout_retry(
            lambda: yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    close = df["Close"].dropna()
    high = df["High"].dropna()
    low = df["Low"].dropna()
    if len(close) < 100:
        return None

    returns = close.pct_change().dropna()
    ann_vol = float(returns.std() * np.sqrt(252))
    downside_vol = float(np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)) * np.sqrt(252))

    net_change = abs(float(close.iloc[-1]) - float(close.iloc[0]))
    path_length = float(close.diff().abs().sum())
    efficiency_ratio = net_change / path_length if path_length > 0 else 0.0

    n = 14
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_sum = tr.rolling(n).sum()
    rng = high.rolling(n).max() - low.rolling(n).min()
    with np.errstate(divide="ignore", invalid="ignore"):
        ci = 100 * np.log10(atr_sum / rng) / np.log10(n)
    ci_clean = ci.replace([np.inf, -np.inf], np.nan).dropna()
    choppiness_idx = float(ci_clean.mean()) if not ci_clean.empty else float("nan")

    autocorr = float(returns.autocorr(lag=1))
    if not np.isfinite(autocorr):
        autocorr = 0.0

    signs = np.sign(returns)
    sign_change_freq = float((signs != signs.shift()).sum() / len(signs))

    trend_quality = _trend_quality_score(efficiency_ratio, autocorr, ann_vol, sign_change_freq)

    return {
        "ticker": ticker,
        "ann_vol": round(ann_vol, 4),
        "downside_vol": round(downside_vol, 4),
        "efficiency_ratio": round(efficiency_ratio, 4),
        "autocorr": round(autocorr, 4),
        "choppiness_idx": round(choppiness_idx, 2) if np.isfinite(choppiness_idx) else None,
        "sign_change_freq": round(sign_change_freq, 4),
        "trend_quality": round(float(trend_quality), 3),
    }


def rolling_trend_quality(
    daily_ohlc: pd.DataFrame, window: int = 504, min_periods: int = 100
) -> pd.Series:
    """Trend-quality as a daily time series, computed with a trailing rolling
    window ending at each date — for matching live trading's daily overnight
    rescreen (overnight_scope.py) inside a historical backtest, where a single
    "as of today" volatility_profile() snapshot would apply hindsight (a
    ticker choppy today isn't necessarily choppy two years ago).

    Every component is a vectorized pandas rolling op, not .apply() — cheap
    even for hundreds of tickers x years of daily bars. window=504 trading
    days approximates volatility_profile()'s default period="2y". Below
    min_periods (matching volatility_profile()'s own len(close) < 100 cutoff)
    there isn't enough trailing history for a lookahead-free score; those
    entries are NaN — callers should treat NaN as "no verdict" (permissive),
    matching resolve_strategy()'s documented default when no ticker context
    is available.

    The whole series is shifted 1 day: real screening runs overnight using
    data through yesterday's close and trades on that today, so day-t's score
    must not depend on day-t's own bar.

    daily_ohlc must have High/Low/Close columns, ascending date index.
    """
    high = daily_ohlc["High"]
    low = daily_ohlc["Low"]
    close = daily_ohlc["Close"]

    returns = close.pct_change()
    ann_vol = returns.rolling(window, min_periods=min_periods).std() * np.sqrt(252)
    downside_sq = returns.clip(upper=0.0) ** 2

    net_change = (close - close.shift(window)).abs()
    path_length = close.diff().abs().rolling(window, min_periods=min_periods).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        efficiency_ratio = (net_change / path_length).where(path_length > 0, 0.0)

    autocorr = returns.rolling(window, min_periods=min_periods).corr(returns.shift(1))

    signs = np.sign(returns)
    sign_change_freq = (signs != signs.shift()).rolling(window, min_periods=min_periods).mean()

    trend_quality = _trend_quality_score(efficiency_ratio, autocorr, ann_vol, sign_change_freq)

    # Enough total history at all (min_periods bars), independent of the
    # per-component rolling windows above, which can each be non-NaN slightly
    # differently at the boundary — align on ann_vol's mask as the anchor
    # since std() is the strictest (needs >= 2 non-null returns per window).
    trend_quality = trend_quality.where(ann_vol.notna())

    return trend_quality.shift(1)


def screen_tickers(
    tickers: list[str],
    *,
    min_trend_quality: float = 0.0,
    max_downside_vol: float | None = None,
    period: str = "2y",
    source: str = "ibkr",
    verbose: bool = True,
) -> tuple[list[str], list[dict]]:
    """Filter a ticker list down to those with acceptable trend quality and downside vol.

    If max_downside_vol is None, only trend_quality is checked.
    Returns (kept_tickers, all_profiles).
    """
    kept = []
    profiles = []

    for i, ticker in enumerate(tickers, 1):
        if verbose and (i == 1 or i % 20 == 0):
            logger.info(f"  [{i}/{len(tickers)}] screening {ticker}...")

        prof = volatility_profile(ticker, period=period, source=source)
        if prof is None:
            continue
        profiles.append(prof)
        if prof["trend_quality"] >= min_trend_quality:
            if max_downside_vol is None or prof["downside_vol"] <= max_downside_vol:
                kept.append(ticker)

    return kept, profiles


def main() -> int:
    setup_cli_logger("vol_screen")

    parser = argparse.ArgumentParser(prog="vol-screen")
    parser.add_argument("--tickers", help="Comma-separated ticker list")
    parser.add_argument("--watchlist", help="Path to watchlist JSON (uses 'tickers' key)")
    parser.add_argument("--min-trend-quality", type=float, default=0.0,
                        help="Minimum trend-quality score to keep a ticker (default: 0.0)")
    parser.add_argument("--period", default="2y")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
    elif args.watchlist:
        import json
        with open(args.watchlist, encoding="utf-8") as f:
            wl = json.load(f)
        tickers = [t["ticker"] if isinstance(t, dict) else t for t in wl.get("tickers", [])]
    else:
        logger.info("Provide --tickers or --watchlist")
        return 1

    kept, profiles = screen_tickers(tickers, min_trend_quality=args.min_trend_quality, period=args.period)

    df = pd.DataFrame(profiles).sort_values("trend_quality", ascending=False)
    logger.info(f"\n{'Ticker':10s} {'TrendQ':>8s} {'EffRatio':>9s} {'Autocorr':>9s} {'AnnVol':>8s} {'DownVol':>8s} {'SignChg':>8s}  Verdict")
    for _, row in df.iterrows():
        verdict = "KEEP" if row["trend_quality"] >= args.min_trend_quality else "exclude (choppy)"
        logger.info(f"{row['ticker']:10s} {row['trend_quality']:>8.2f} {row['efficiency_ratio']:>9.3f} "
              f"{row['autocorr']:>9.3f} {row['ann_vol']:>8.3f} {row['downside_vol']:>8.3f} {row['sign_change_freq']:>8.3f}  {verdict}")

    logger.info(f"\n  Kept: {len(kept)}/{len(tickers)} tickers (trend_quality >= {args.min_trend_quality})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
