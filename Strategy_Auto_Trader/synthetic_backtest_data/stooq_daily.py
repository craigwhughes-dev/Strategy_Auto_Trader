"""Local Stooq bulk EOD dump as a daily-close source.

Stooq's free bulk daily-data archive (one flat text file per ticker,
`<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>`)
copied to data/cache/stooq_daily/{uk,us}/ — a snapshot of *currently listed*
LSE/NYSE/NASDAQ/NYSE MKT tickers (no delisted-ticker history; same
survivorship-gap limitation as everywhere else in this project, see
markov_cli/full_scan.py's build_sp_ftse_universe docstring).

Verified against IBKR's fetch_daily on HSBA.L, VOD.L, AAPL (2yr window):
no unit/scale mismatch, closes agree to within ~0.1-0.3% (different
closing-auction convention, not a data-quality issue), zero missing
trading days. Not stress-tested against a ticker with a recent
split/rights issue.

Chosen over IBKR for this project's daily-close needs because it is
instant (no reqHistoricalData paging/throttling) and goes back further
for UK names — confirmed live: IBKR's UK daily history starts 1998-07 for
both HSBA.L and VOD.L, while Stooq's HSBA file starts 1992-06.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "stooq_daily"

_COLUMNS = ["<TICKER>", "<PER>", "<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>", "<OPENINT>"]


def _stooq_path(ticker: str, cache_dir: Path | None = None) -> Path:
    cache_dir = CACHE_DIR if cache_dir is None else cache_dir
    if ticker.endswith(".L"):
        return cache_dir / "uk" / f"{ticker[:-2].lower()}.uk.txt"
    return cache_dir / "us" / f"{ticker.lower()}.us.txt"


def load_stooq_daily(ticker: str, cache_dir: Path | None = None) -> pd.DataFrame | None:
    """Load a ticker's daily OHLCV from the local Stooq dump. Returns None
    if no file exists for this ticker (not in the current LSE/US snapshot,
    or a symbol-convention mismatch) — same failure contract as
    IBKRDataClient.fetch_daily, so callers can fall back the same way."""
    path = _stooq_path(ticker, cache_dir)
    if not path.exists():
        return None

    df = pd.read_csv(path, names=_COLUMNS, header=0)
    if df.empty:
        return None

    df.index = pd.to_datetime(df["<DATE>"], format="%Y%m%d", utc=True)
    df.index.name = None
    out = df.rename(columns={
        "<OPEN>": "Open", "<HIGH>": "High", "<LOW>": "Low",
        "<CLOSE>": "Close", "<VOL>": "Volume",
    })[["Open", "High", "Low", "Close", "Volume"]]
    return out.sort_index()
