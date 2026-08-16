"""Nightly IBKR trade-correction reconciliation.

fetch_hourly()'s incremental cache only ever pages the gap since the last
cached bar (see broker/ibkr_data.py) — it never re-checks a bar once it has
been written to disk. If IBKR revises an already-published historical TRADES
bar (a trade correction/bust, or a rare vendor-side data-quality fix), that
revision would be silently missed forever. This script re-fetches the last
`--lookback-days` of raw bars fresh for every ticker and diffs them against
the stored cache, overwriting any bar that changed — see
broker.ibkr_data.reconcile_recent_bars for the diff/merge logic.

Corrected closes then flow through PersistentHMMRegimeModel's own rtol
tolerance + relabel-warning logging on the next daemon cycle automatically;
this script only needs to keep the on-disk bar cache honest.

Usage:
    uv run python -m Strategy_Auto_Trader.markov_cli.ibkr_reconcile
        [--lookback-days 14] [--universe config/universe_sp_ftse.json]
        [--port 4002] [--client-id 4] [--limit 10]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from ..broker import ibkr_data
from ..broker.ibkr_data import IBKRDataClient, reconcile_recent_bars
from ..core.cli_logging import setup_cli_logger

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_UNIVERSE = ROOT / "config" / "universe_sp_ftse.json"


def _load_universe(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["tickers"]


def main(argv: list[str] | None = None) -> int:
    setup_cli_logger("ibkr_reconcile")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lookback-days", type=int, default=14,
                        help="How far back to re-check for corrections (default: 14)")
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=4,
                        help="Distinct from execution (1), routine data-fetch (2), "
                             "and backfill (3) client ids (default: 4)")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N tickers (0 = all)")
    args = parser.parse_args(argv)

    tickers = _load_universe(args.universe)
    if args.limit:
        tickers = tickers[:args.limit]

    client = IBKRDataClient(host=args.host, port=args.port, client_id=args.client_id)
    if not client.connect():
        logger.error(f"Could not connect to TWS/Gateway at {args.host}:{args.port} "
                     f"(client_id={args.client_id}). Is it running and logged in?")
        return 1

    logger.info(f"Reconciling {len(tickers)} tickers, last {args.lookback_days}d "
               f"against stored IBKR cache...")

    total_checked = total_corrected = 0
    affected_tickers: set[str] = set()
    try:
        for i, ticker in enumerate(tickers, 1):
            try:
                result = reconcile_recent_bars(ticker, client, lookback_days=args.lookback_days)
            except Exception as exc:
                logger.warning(f"[{i}/{len(tickers)}] {ticker}: reconcile failed ({exc})")
                continue

            total_checked += result["checked"]
            if result["corrected"]:
                total_corrected += result["corrected"]
                affected_tickers.add(ticker)
                for d in result["diffs"]:
                    logger.warning(
                        f"[{i}/{len(tickers)}] {ticker}: {d['field']} at {d['date']} "
                        f"revised {d['old']} -> {d['new']}"
                    )
            time.sleep(ibkr_data._PAGE_SLEEP_S)
    finally:
        client.disconnect()

    logger.info(f"Reconciliation complete: {total_checked} bars checked, "
               f"{total_corrected} correction(s) found across {len(tickers)} tickers.")
    if affected_tickers:
        logger.warning(f"{len(affected_tickers)} ticker(s) had corrections applied: "
                       f"{sorted(affected_tickers)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
