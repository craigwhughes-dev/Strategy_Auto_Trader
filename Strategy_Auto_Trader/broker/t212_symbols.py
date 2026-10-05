"""Ticker symbol mapping between yfinance conventions and Trading212 instrument codes.

Unlike ibkr symbols.py's static tables (each entry documents a *live, verified*
reqContractDetails lookup against the paper Gateway), there is no verified T212
account available to probe at the time this module was written — Trading212's
public instrument-code convention for non-US listings (LSE equities/ETFs) is
not confirmed from their docs (only "AAPL_US_EQ"-style US codes are documented
with an example). Guessing a suffix convention here would risk silently
resolving a UK ticker to the wrong instrument or a 404 at order time.

Instead this module requires an EXPLICIT mapping, loaded from
config/t212_symbol_map.json, keyed by the same yfinance ticker the rest of the
pipeline already uses ("HSBA.L", "SPY", ...). Populate it by calling
fetch_instrument_catalog() once (rate-limited 1 req/50s — do not call it from
a hot path) against a real T212 API key and matching tickers by name/ISIN, then
hand-confirm each entry before trading it live.

A ticker with no mapping entry raises — fail closed, never guess.
"""

from __future__ import annotations

import json
from pathlib import Path

_DEFAULT_MAP_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "t212_symbol_map.json"


def load_symbol_map(path: Path | None = None) -> dict[str, str]:
    """Load the yfinance-ticker -> T212-ticker mapping from JSON.

    Returns an empty dict if the file doesn't exist yet (adapter then raises
    a clear per-ticker error rather than failing at import time, so the rest
    of the package — including tests — works before anyone has populated it).
    """
    map_path = path or _DEFAULT_MAP_PATH
    if not map_path.exists():
        return {}
    with map_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def t212_ticker(ticker: str, symbol_map: dict[str, str]) -> str:
    """Map a yfinance ticker to its T212 instrument code. Raises if unmapped."""
    code = symbol_map.get(ticker)
    if code is None:
        raise KeyError(
            f"{ticker}: no T212 instrument code in config/t212_symbol_map.json — "
            f"confirm the code via fetch_instrument_catalog() and add it explicitly "
            f"before trading this ticker through T212"
        )
    return code


def yfinance_ticker_from_t212(code: str, symbol_map: dict[str, str]) -> str:
    """Inverse of t212_ticker — used to label positions/fills read back from T212."""
    for yf_ticker, t212_code in symbol_map.items():
        if t212_code == code:
            return yf_ticker
    raise KeyError(
        f"{code}: T212 instrument code not found in config/t212_symbol_map.json "
        f"reverse lookup — a position exists on T212 that this pipeline doesn't "
        f"recognise; add it to the map before relying on get_open_positions()"
    )


def fetch_instrument_catalog(request_fn) -> list[dict]:
    """One-off helper: fetch the full instrument list to source codes from.

    `request_fn` is T212Adapter._request (or an equivalent callable taking
    (method, path)) — kept as a plain function here, not a method on the
    adapter, since building the symbol map is a manual setup step done once
    via a script, not part of the live trading path. Rate-limited by T212 to
    1 request per 50 seconds; callers must not loop this per-ticker.
    """
    return request_fn("GET", "/api/v0/equity/metadata/instruments")
