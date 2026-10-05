"""Read-only probe: what market data type and bars each IB gateway serves for CBOE indices.

Connects to each port in turn (default live 4001 and paper 4002), reports the account
(DU/DF prefix = paper), then per index:
  - a streaming quote requested as market data type 1 (live); IBKR's answer in
    ticker.marketDataType shows whether it fell back to delayed (3) or frozen (2/4)
  - the last two completed-looking hourly and daily bars from reqHistoricalData

Never touches data/cache or any state file: it calls ib_async directly, not IBKRDataClient,
so the shared INDEX_* cache is neither read nor written. Use distinct client ids from the
daemon (1) and data client (2).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

DEFAULT_PORTS = (4001, 4002)
DEFAULT_SYMBOLS = ("VIX", "VXN", "VVIX")
MARKET_DATA_TYPE_NAMES = {1: "live", 2: "frozen", 3: "delayed", 4: "delayed-frozen"}


def _connect(ib, host: str, port: int, client_id: int) -> None:
    ib.connect(host, port, clientId=client_id, timeout=15, readonly=True)


def _errors_collector(ib):
    errors: list[str] = []

    def _on_error(reqId, errorCode, errorString, contract=None, *_):
        errors.append(f"{errorCode}: {errorString}")

    ib.errorEvent += _on_error
    return errors, _on_error


def _quote(ib, contract, requested_type: int) -> str:
    ib.reqMarketDataType(requested_type)
    ticker = ib.reqMktData(contract, "", False, False)
    ib.sleep(4)
    now = datetime.now(timezone.utc)
    ticker_time = getattr(ticker, "time", None)
    age = f"{(now - ticker_time).total_seconds():.0f}s old" if ticker_time else "no tick time"
    served = getattr(ticker, "marketDataType", None)
    served_name = MARKET_DATA_TYPE_NAMES.get(served, "none served")
    fields = {name: getattr(ticker, name, None) for name in ("last", "close", "bid", "ask", "delayedLastTimestamp")}
    ib.cancelMktData(contract)
    raw = ", ".join(f"{name}={value}" for name, value in fields.items())
    return f"quote requested type {requested_type}: served={served} ({served_name}), {raw}, {age}"


def _bars(ib, contract, duration: str, bar_size: str) -> list[str]:
    bars = ib.reqHistoricalData(
        contract, endDateTime="", durationStr=duration,
        barSizeSetting=bar_size, whatToShow="TRADES", useRTH=True,
    )
    if not bars:
        return [f"{bar_size} bars: none returned"]
    return [f"{bar_size} bar {b.date}: close={b.close}" for b in bars[-2:]]


def probe_port(host: str, port: int, client_id: int, symbols: tuple[str, ...]) -> None:
    from ib_async import IB, Index

    ib = IB()
    print(f"\n=== {host}:{port} (clientId={client_id}) ===")
    try:
        _connect(ib, host, port, client_id)
    except Exception as e:
        print(f"connect failed: {e}")
        return

    try:
        print(f"accounts: {ib.managedAccounts()}")
        for symbol in symbols:
            contract = Index(symbol, "CBOE", "USD")
            qualified = ib.qualifyContracts(contract)
            if not qualified or qualified[0] is None:
                print(f"{symbol}: could not qualify")
                continue
            errors, handler = _errors_collector(ib)
            try:
                print(f"{symbol}:")
                for requested_type in (1, 3):
                    print(f"  {_quote(ib, contract, requested_type)}")
                for line in _bars(ib, contract, "2 D", "1 hour") + _bars(ib, contract, "5 D", "1 day"):
                    print(f"  {line}")
            finally:
                ib.errorEvent -= handler
            if errors:
                print(f"  IBKR errors: {'; '.join(errors)}")
    finally:
        ib.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--ports", type=int, nargs="+", default=list(DEFAULT_PORTS))
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--client-id", type=int, default=11, help="base id; each port gets base + index")
    args = parser.parse_args()

    for offset, port in enumerate(args.ports):
        probe_port(args.host, port, args.client_id + offset, tuple(args.symbols))


if __name__ == "__main__":
    main()
