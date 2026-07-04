"""One-shot IBKR connectivity check.

Verifies that TWS / IB Gateway is reachable on the paper port, the API is
enabled, and the account can be queried. Places no orders.

Usage:
    uv run python check_ibkr.py            # TWS paper (7497)
    uv run python check_ibkr.py --port 4002  # IB Gateway paper
"""

from __future__ import annotations

import argparse
import sys

from Strategy_Auto_Trader.broker.ibkr_adapter import IBKRAdapter


def main() -> int:
    parser = argparse.ArgumentParser(description="Check IBKR API connectivity.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497,
                        help="7497=TWS paper, 4002=IB Gateway paper")
    parser.add_argument("--client-id", type=int, default=99,
                        help="Use an id the daemon does not use (daemon uses 1)")
    args = parser.parse_args()

    adapter = IBKRAdapter(host=args.host, port=args.port, client_id=args.client_id)
    try:
        adapter.connect()
    except Exception as exc:
        print(f"FAILED to connect to {args.host}:{args.port} — {exc}")
        print("Check: TWS/Gateway running? Logged into PAPER account? "
              "API enabled with this port in Global Configuration -> API -> Settings?")
        return 1

    try:
        ib = adapter._ib
        accounts = ib.managedAccounts()
        print(f"Connected to {args.host}:{args.port}")
        print(f"Accounts: {accounts}")
        if accounts and not any(a.startswith("DU") for a in accounts):
            print("WARNING: no account id starting with 'DU' — this may not "
                  "be a paper account. Do not flip dry_run off until it is.")
        summary = {v.tag: v.value for v in ib.accountSummary()
                   if v.tag in ("NetLiquidation", "AvailableFunds", "BuyingPower")}
        for tag, value in summary.items():
            print(f"  {tag}: {value}")
        positions = adapter.get_open_positions()
        print(f"Open positions: {positions or 'none'}")
        print("OK — API connection is working.")
        return 0
    finally:
        adapter.disconnect()


if __name__ == "__main__":
    sys.exit(main())
