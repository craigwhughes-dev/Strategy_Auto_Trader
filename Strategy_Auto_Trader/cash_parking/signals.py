"""Fetch daily parking signals for the cash parking module.

Returns a tz-naive daily DataFrame indexed by date with columns:
  vix, xstr_ret, igls_ret, isxf_ret

All data sourced from IBKR (no yfinance):
  - ETF daily returns (XSTR.L, IGLS.L, ISXF.L): IBKR hourly cache resampled to daily,
    OR synthetic hourly from data_synthetic/hourly/ when synthetic_data_dir is supplied
    (synthetic hourly generated via Brownian bridge on IBKR daily closes — keeps
    synthetic and real caches completely separate).
  - VIX: IBKR daily cache via fetch_vix_ibkr (always real).

Tier decisions based on VIX level only (no market regime model needed).

Used by both the standalone backtest overlay (scripts/combined_backtest_analysis.py)
and the live_sim concurrent parking simulation (--cash-parking flag).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_STRIP_TZ = lambda idx: idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx


def _fetch_etf_daily(ticker: str, synthetic_data_dir: Path | None = None) -> pd.Series | None:
    """Fetch daily closes for an ETF (pence→GBP).

    synthetic_data_dir: when set, loads from synthetic hourly CSVs (Brownian-bridge
    data in data_synthetic/hourly/) instead of the IBKR hourly cache. Keeps
    synthetic and real data completely separate.
    """
    if synthetic_data_dir is not None:
        from ..synthetic_backtest_data.generate import load_synthetic_hourly
        df = load_synthetic_hourly(ticker, hourly_dir=synthetic_data_dir)
        if df is None or df.empty:
            logger.warning(f"  {ticker}: not in synthetic_data_dir {synthetic_data_dir} — skipping")
            return None
    else:
        from ..quant_hmm.quant_engine import fetch_hourly
        df = fetch_hourly(ticker, source="ibkr", historical_only=True)
        if df is None or df.empty:
            return None
    close = df["Close"].resample("D").last().dropna()
    close.index = _STRIP_TZ(close.index)
    if str(ticker).upper().endswith(".L"):
        close = close / 100.0
    return close


def fetch_parking_signals(
    start: str,
    end: str,
    synthetic_data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch daily VIX and ETF returns via IBKR cache.

    synthetic_data_dir: when set (synthetic backtest mode), ETF daily returns are
    sourced from Brownian-bridge synthetic hourly CSVs in that directory instead of
    the IBKR hourly cache. VIX always uses real IBKR data.
    Pass live_sim's --synthetic-data-dir value here to keep synthetic and real
    data completely separate.

    Returns DataFrame indexed by tz-naive date with columns:
      vix, xstr_ret, igls_ret, isxf_ret
    (forward-filled; NaN gaps bridged).
    Returns empty DataFrame if IBKR data is unavailable.
    """
    _synth_dir = Path(synthetic_data_dir) if synthetic_data_dir is not None else None
    from ..quant_hmm.quant_engine import fetch_vix_ibkr

    _src = f"synthetic ({_synth_dir})" if _synth_dir else "IBKR hourly cache"
    logger.info(f"Parking signals: fetching ETF daily prices via {_src} (XSTR.L, IGLS.L, ISXF.L)...")
    etf_closes: dict[str, pd.Series] = {}
    for ticker, col in [
        ("XSTR.L", "xstr_ret"), ("IGLS.L", "igls_ret"),
        ("ISXF.L", "isxf_ret"),
    ]:
        s = _fetch_etf_daily(ticker, synthetic_data_dir=_synth_dir)
        if s is not None and not s.empty:
            etf_closes[col] = s.pct_change()
        else:
            logger.warning(f"  {ticker}: unavailable — {col} will be NaN")

    etf_ret = pd.DataFrame(etf_closes)
    if not etf_ret.empty:
        etf_ret.index = _STRIP_TZ(etf_ret.index)

    logger.info("Parking signals: fetching VIX...")
    vix_df = fetch_vix_ibkr(historical_only=True)
    if vix_df is None or vix_df.empty:
        logger.warning("  VIX: IBKR cache unavailable — returning empty signals")
        return pd.DataFrame()
    vix_series = vix_df["Close"].dropna().rename("vix")
    vix_series.index = _STRIP_TZ(vix_series.index)

    signals = pd.DataFrame(vix_series)
    signals = signals.join(etf_ret, how="left")
    signals = signals.ffill()

    # Fallback chains: when a tier's primary ETF has no data (pre-launch), use
    # the next available instrument rather than earning 0%.
    #   hy_bonds tier (isxf_ret): ISXF.L → IGLS.L → XSTR.L → 0%
    #   gilts tier    (igls_ret): IGLS.L → XSTR.L → 0%  (both fixed income)
    #   cash tier     (xstr_ret): XSTR.L → 0%
    if "isxf_ret" in signals.columns:
        if "igls_ret" in signals.columns:
            signals["isxf_ret"] = signals["isxf_ret"].where(
                signals["isxf_ret"].notna(), signals["igls_ret"]
            )
        if "xstr_ret" in signals.columns:
            signals["isxf_ret"] = signals["isxf_ret"].where(
                signals["isxf_ret"].notna(), signals["xstr_ret"]
            )
    if "igls_ret" in signals.columns and "xstr_ret" in signals.columns:
        signals["igls_ret"] = signals["igls_ret"].where(
            signals["igls_ret"].notna(), signals["xstr_ret"]
        )

    return signals.loc[start:]
