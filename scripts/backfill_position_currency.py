#!/usr/bin/env python
"""One-off backfill script: add market/currency fields to existing open positions.

Run once with the daemon stopped. Reads execution_state.json, adds market/currency
to any position missing them (assumes FTSE=GBP, others=USD), and writes back atomically.

Usage:
    uv run python scripts/backfill_position_currency.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"


def get_market_currency(ticker: str) -> str:
    """Infer currency from ticker suffix (L=FTSE/GBP, others=USD)."""
    if ticker.endswith(".L"):
        return "GBP"
    return "USD"


def get_market_from_ticker(ticker: str) -> str:
    """Infer market from ticker suffix (L=FTSE, others=SP500)."""
    if ticker.endswith(".L"):
        return "ftse"
    return "sp500"


def backfill_currencies() -> None:
    """Add market/currency to positions that lack them."""
    state_path = STATE_DIR / "execution_state.json"

    if not state_path.exists():
        print("No execution_state.json found. Nothing to backfill.")
        return

    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Failed to load execution_state.json: {e}")
        return

    positions = data.get("positions", {})
    if not positions:
        print("No positions found. Nothing to backfill.")
        return

    updated_count = 0
    for ticker, pos in positions.items():
        if "market" not in pos or not pos["market"]:
            pos["market"] = get_market_from_ticker(ticker)
            updated_count += 1
        if "currency" not in pos or not pos["currency"]:
            pos["currency"] = get_market_currency(ticker)
            updated_count += 1
        if "cost_value" not in pos or pos["cost_value"] == 0:
            pos["cost_value"] = pos.get("fill_price", 0.0) * pos.get("quantity", 0)
            updated_count += 1

    if updated_count == 0:
        print("All positions already have market/currency. No changes needed.")
        return

    # Write back atomically
    from Strategy_Auto_Trader.core.atomic_io import atomic_write_json

    atomic_write_json(state_path, data)
    print(f"Backfill complete: updated {updated_count} position field(s)")
    print(f"Positions: {len(positions)}")
    for ticker, pos in positions.items():
        print(f"  {ticker}: market={pos.get('market')}, currency={pos.get('currency')}, "
              f"qty={pos.get('quantity')}, cost_value={pos.get('cost_value', 0):.2f}")


if __name__ == "__main__":
    backfill_currencies()
