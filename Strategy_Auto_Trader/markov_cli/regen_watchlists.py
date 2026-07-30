"""One-off: widen the live daemon's FTSE/S&P500 watchlists to the full
validated universe (config/universe_sp_ftse.json), splitting by ticker suffix
(".L" = FTSE, no suffix = US). Run manually, review the diff, before enabling
top_k_screen — not wired into the daemon's automatic cycle.

Preserves any existing per-ticker overrides (e.g. a "strategy" override) by
ticker, and leaves each watchlist's own "defaults" block untouched.

Usage:
    uv run python -m Strategy_Auto_Trader.markov_cli.regen_watchlists
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .overnight_scope import CONFIG_DIR, OVERRIDE_KEYS

UNIVERSE_PATH = CONFIG_DIR / "universe_sp_ftse.json"


def _existing_overrides(watchlist_path: Path) -> dict[str, dict]:
    """{ticker: full ticker-entry dict} for entries that carry an override key,
    keyed so a regenerated entry can be swapped back in unchanged."""
    if not watchlist_path.exists():
        return {}
    data = json.loads(watchlist_path.read_text(encoding="utf-8"))
    overrides = {}
    for entry in data.get("tickers", []):
        if isinstance(entry, dict) and any(k in OVERRIDE_KEYS for k in entry if k != "ticker"):
            overrides[entry["ticker"]] = entry
    return overrides


def regen_watchlist(watchlist_path: Path, universe_tickers: list[str]) -> tuple[int, int]:
    """Overwrite watchlist_path's "tickers" list with universe_tickers,
    preserving the file's own "defaults" block and any existing per-ticker
    overrides. Returns (old_count, new_count)."""
    existing = json.loads(watchlist_path.read_text(encoding="utf-8")) if watchlist_path.exists() else {"defaults": {}, "tickers": []}
    old_count = len(existing.get("tickers", []))
    overrides = _existing_overrides(watchlist_path)

    new_tickers = [overrides.get(t, {"ticker": t}) for t in universe_tickers]
    out = {"defaults": existing.get("defaults", {}), "tickers": new_tickers}
    watchlist_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return old_count, len(new_tickers)


def main() -> int:
    universe = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))["tickers"]
    ftse_tickers = sorted(t for t in universe if t.endswith(".L"))
    sp500_tickers = sorted(t for t in universe if not t.endswith(".L"))

    print(f"Universe: {len(universe)} tickers ({len(ftse_tickers)} FTSE, {len(sp500_tickers)} US)")

    ftse_old, ftse_new = regen_watchlist(CONFIG_DIR / "watchlist_ftse.json", ftse_tickers)
    print(f"watchlist_ftse.json: {ftse_old} -> {ftse_new} tickers")

    sp500_old, sp500_new = regen_watchlist(CONFIG_DIR / "watchlist_sp500.json", sp500_tickers)
    print(f"watchlist_sp500.json: {sp500_old} -> {sp500_new} tickers")

    print("\nReview the diff before running the daemon's overnight_scope with top_k_screen enabled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
