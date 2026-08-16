"""Phase 1 diagnostic: does a GBP-quoted LSE contract exist for the 3
currency-mismatch tickers (CPG.L, IHG.L, MTLN.L)?

Run with TWS paper gateway on port 7497:
    uv run python scratchpad/diagnose_lse_currency.py

For each ticker, dumps every candidate contract IBKR returns when we ask
with NO currency constraint — so if a GBP line exists alongside USD/EUR,
we'll see it.  Result determines Phase 2 decision per PLAN_LSE_CURRENCY_MISMATCH.md:
  - GBP line present  -> mapping fix (Phase 3a), remove from IBKR_UNRESOLVABLE
  - GBP line absent   -> permanently excluded, IBKR_UNRESOLVABLE entry stands
"""

from __future__ import annotations

import asyncio
from ib_async import IB, Contract

TICKERS = {
    "CPG.L":  {"symbol": "CPG",  "candidates": [("LSE", None), ("SMART", None)]},
    "IHG.L":  {"symbol": "IHGL", "candidates": [("LSE", None), ("SMART", None)]},
    "MTLN.L": {"symbol": "MTLN", "candidates": [("LSE", None), ("SMART", None)]},
}

PORT = 4002
CLIENT_ID = 99


async def main() -> None:
    ib = IB()
    await ib.connectAsync("127.0.0.1", PORT, clientId=CLIENT_ID)
    print(f"Connected to TWS on port {PORT}\n")

    for yf_ticker, info in TICKERS.items():
        print(f"{'='*60}")
        print(f"yfinance: {yf_ticker}  (IBKR base symbol tried: {info['symbol']})")
        print(f"{'='*60}")

        for exchange, currency in info["candidates"]:
            c = Contract(
                symbol=info["symbol"],
                secType="STK",
                exchange=exchange,
            )
            if currency:
                c.currency = currency

            details = await ib.reqContractDetailsAsync(c)
            label = f"exchange={exchange}" + (f" currency={currency}" if currency else " (no currency filter)")
            if not details:
                print(f"  [{label}] -> no results")
                continue

            print(f"  [{label}] -> {len(details)} contract(s):")
            for d in details:
                ct = d.contract
                print(
                    f"    conId={ct.conId:<12} symbol={ct.symbol:<12} "
                    f"currency={ct.currency:<6} exchange={ct.exchange:<8} "
                    f"primaryExch={ct.primaryExchange:<8} "
                    f"tradingClass={ct.tradingClass:<10} "
                    f"longName={d.longName!r}"
                )
        print()

    ib.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
