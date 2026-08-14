"""One-time (resumable) throttled backfill of IBKR hourly history for the
full S&P500+FTSE100 universe, so every ticker the live daemon might touch
already has deep local history before its own day-to-day incremental
gap-fill takes over (see broker/ibkr_data.py's module docstring).

Respects IBKR's ~60-requests-per-10-minutes pacing cap
(https://interactivebrokers.github.io/tws-api/historical_limitations.html)
by overriding ibkr_data._PAGE_SLEEP_S for this process only — the live
daemon's own gap-fill calls (routinely 1-2 requests per ticker per day) keep
their normal, faster 2.0s spacing; only a bulk multi-ticker bootstrap like
this one needs the wider global-rate-limit-safe spacing.

Resumable: a ticker whose cache already spans the target period is skipped
without any network call, so an interrupted run can just be re-launched —
each ticker's cache is saved (atomically) as soon as it completes, so there
is no partial-run state to clean up.

Usage:
    uv run python scripts/ibkr_backfill_universe.py [--period 1095d]
                                                      [--port 4002] [--client-id 3]
                                                      [--universe config/universe_sp_ftse.json]
                                                      [--limit 10]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Strategy_Auto_Trader.broker import ibkr_data  # noqa: E402
from Strategy_Auto_Trader.broker.ibkr_data import IBKRDataClient  # noqa: E402

DEFAULT_UNIVERSE = REPO_ROOT / "config" / "universe_sp_ftse.json"

# 60 requests / 10 min is IBKR's global historical-data pacing cap; 10.5s
# gives headroom for jitter. This overrides the module constant for this
# process's lifetime only — it never touches the live daemon's own spacing.
_BACKFILL_PAGE_SLEEP_S = 10.5


def _load_universe(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["tickers"]


def _ensure_connected(client: IBKRDataClient) -> bool:
    """Reconnect if Gateway dropped the socket mid-run.

    A ~9-hour run crosses Gateway's own nightly restart/logoff window, which
    silently kills the connection — without this check, every subsequent
    ticker's fetch_hourly() call fails on the dead socket and is swallowed by
    its own broad except (returns cached/None), so the run just burns through
    the remaining tickers producing "no data" with no indication anything
    went wrong until the summary line."""
    if client._ib is not None and client._ib.isConnected():
        return True
    print("    connection lost — reconnecting...")
    client.disconnect()
    return client.connect()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--period", default="1095d",
                        help="Target history depth, yfinance-style period string "
                             "(default: 1095d / 3y)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=3,
                        help="Distinct from the live daemon's execution (1) and routine "
                             "data-fetch (2) client ids, so a long-running backfill never "
                             "collides with either (default: 3)")
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--limit", type=int, default=0, help="Stop after N tickers (0 = all)")
    args = parser.parse_args(argv)

    ibkr_data._PAGE_SLEEP_S = _BACKFILL_PAGE_SLEEP_S

    tickers = _load_universe(args.universe)
    if args.limit:
        tickers = tickers[:args.limit]
    target_days = ibkr_data._period_to_days(args.period)

    client = IBKRDataClient(host=args.host, port=args.port, client_id=args.client_id)
    if not client.connect():
        print(f"Could not connect to TWS/Gateway at {args.host}:{args.port} "
              f"(client_id={args.client_id}). Is it running and logged in?")
        return 1

    print(f"Backfilling {len(tickers)} tickers to {args.period} "
          f"(~{_BACKFILL_PAGE_SLEEP_S}s/request pacing)...")

    skipped = done = failed = 0
    consecutive_reconnect_failures = 0
    try:
        for i, ticker in enumerate(tickers, 1):
            cached = ibkr_data._load_cache(ticker)
            if cached is not None:
                span_days = (cached.index[-1] - cached.index[0]).days
                if span_days >= target_days:
                    skipped += 1
                    print(f"[{i}/{len(tickers)}] {ticker}: already {span_days}d cached, skipping")
                    continue

            if not _ensure_connected(client):
                consecutive_reconnect_failures += 1
                print(f"    reconnect failed ({consecutive_reconnect_failures} in a row)")
                if consecutive_reconnect_failures >= 3:
                    print("\nGateway unreachable after 3 reconnect attempts — aborting. "
                          "Re-run once it's back up; already-bootstrapped tickers are skipped.")
                    return 1
                time.sleep(_BACKFILL_PAGE_SLEEP_S)
                continue
            consecutive_reconnect_failures = 0

            print(f"[{i}/{len(tickers)}] {ticker}: bootstrapping to {args.period}...", flush=True)
            try:
                df = client.fetch_hourly(ticker, period=args.period)
            except Exception as exc:
                print(f"    ERROR: {type(exc).__name__}: {exc}")
                df = None

            if df is None or df.empty:
                print("    no data")
                failed += 1
            else:
                print(f"    {len(df)} bars, {df.index[0]} -> {df.index[-1]}")
                done += 1

            # Inter-ticker spacing: fetch_hourly's own paging already sleeps
            # _PAGE_SLEEP_S *between* pages of one ticker, but not after the
            # last page — without this, back-to-back tickers could fire
            # requests with no gap at all.
            time.sleep(_BACKFILL_PAGE_SLEEP_S)
    finally:
        client.disconnect()

    print(f"\nBackfill finished: {done} bootstrapped, {skipped} already done, {failed} failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
