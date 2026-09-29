"""Daily closes for world equity indices, cached locally — P4a of PLAN_CRASH_PHASES.md.

Source is Yahoo's chart endpoint. **This is a deliberate, scoped exception to the project's
"always use IBKR, never yfinance" rule**, and it does not touch the trading pipeline:

  - What is needed here is *daily closes on foreign indices, for dating historical crash
    episodes*. The standing objection to yfinance concerns its 730-day intraday cap and
    stale/adjusted hourly bars feeding backtests — neither applies to daily index closes
    used only to locate peaks and troughs.
  - IBKR cannot supply these: the account has no index data subscriptions (see
    project_ibkr_gateway_constraints_20260917), and the free local Stooq dump holds only US
    and UK single names plus ^VIX — no index series at all.
  - Nothing fetched here is ever used to price a trade, size a position, or drive a signal.

Every cached file records the as-of date it was fetched, so a rerun months later is
reproducible and a silently-revised history is visible.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "yahoo_index"
_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1=0&period2=9999999999&interval=1d"
_UA = "Mozilla/5.0"

# Local currency of each index, because episode depth measured in a foreign currency is an FX
# artefact as much as a market one — the lesson from COVID reading -21.6% on the GBP Nasdaq leg.
MARKETS = {
    "^GSPC": ("S&P 500", "USD"),
    "^IXIC": ("Nasdaq Composite", "USD"),
    "^FTSE": ("FTSE 100", "GBP"),
    "^GDAXI": ("DAX", "EUR"),
    "^N225": ("Nikkei 225", "JPY"),
    "^HSI": ("Hang Seng", "HKD"),
    "^AORD": ("ASX All Ordinaries", "AUD"),
    "^BVSP": ("Bovespa", "BRL"),
    "^KS11": ("KOSPI", "KRW"),
    "^TWII": ("TAIEX", "TWD"),
    "^BSESN": ("BSE Sensex", "INR"),
    "^MXX": ("IPC Mexico", "MXN"),
}


def cache_path(symbol: str, cache_dir: Path | None = None) -> Path:
    return (CACHE_DIR if cache_dir is None else cache_dir) / f"{symbol.lstrip('^').lower()}.csv"


def fetch_daily_closes(symbol: str, timeout: int = 40) -> pd.Series:
    """Full available daily close history for one index. Raises on a malformed response."""
    url = _CHART.format(sym=urllib.parse.quote(symbol))
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    result = payload["chart"]["result"][0]
    stamps = result.get("timestamp") or []
    closes = result["indicators"]["quote"][0]["close"]
    rows = [(t, c) for t, c in zip(stamps, closes) if c is not None]
    if not rows:
        raise ValueError(f"no close data returned for {symbol}")
    idx = pd.to_datetime([r[0] for r in rows], unit="s", utc=True).normalize()
    out = pd.Series([float(r[1]) for r in rows], index=idx, name=symbol)
    return out.groupby(level=0).last().sort_index()


def load_daily_closes(
    symbol: str,
    cache_dir: Path | None = None,
    refresh: bool = False,
    sleep: float = 0.6,
) -> pd.Series:
    """Cached daily closes. Fetches and writes the cache on a miss or when `refresh` is set."""
    path = cache_path(symbol, cache_dir)
    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col=0, comment="#")
        return pd.Series(df["close"].to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(df.index, utc=True)), name=symbol)
    series = fetch_daily_closes(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    name, ccy = MARKETS.get(symbol, (symbol, "?"))
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(f"# {symbol} {name} ({ccy}) daily closes, Yahoo chart endpoint, fetched {date.today().isoformat()}\n")
        series.rename("close").to_frame().to_csv(fh, index_label="date")
    time.sleep(sleep)
    return series
