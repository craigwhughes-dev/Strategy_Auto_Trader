"""Compare Yahoo chart and CBOE delayed-quote feeds for VXN/VVIX to measure data lag.

Run during US market hours. Each poll prints Yahoo's last trade time (lag vs now)
and last price next to CBOE's quote, so a persistent ~15 min gap shows delay.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1h&range=1d"
CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/quotes/_{index}.json"
# Yahoo rejects requests without a browser-style User-Agent
USER_AGENT = "Mozilla/5.0"
SYMBOLS = ("VXN", "VVIX")


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def parse_yahoo_chart(payload: dict) -> dict:
    result = payload["chart"]["result"][0]
    meta = result["meta"]
    closes = result["indicators"]["quote"][0]["close"]
    last_close = next((c for c in reversed(closes) if c is not None), None)
    return {
        "last_trade_epoch": meta["regularMarketTime"],
        "last_price": last_close,
    }


def parse_cboe_quote(payload: dict) -> dict:
    data = payload["data"]
    return {
        "timestamp": payload["timestamp"],
        "last_trade_time": data["last_trade_time"],
        "price": data["current_price"],
    }


def fetch_yahoo(symbol: str) -> dict:
    return parse_yahoo_chart(_get_json(YAHOO_URL.format(symbol=f"%5E{symbol}")))


def fetch_cboe(symbol: str) -> dict:
    return parse_cboe_quote(_get_json(CBOE_URL.format(index=symbol)))


def snapshot(symbol: str, now: datetime) -> str:
    yahoo = fetch_yahoo(symbol)
    cboe = fetch_cboe(symbol)
    lag_min = (now.timestamp() - yahoo["last_trade_epoch"]) / 60
    yahoo_ts = datetime.fromtimestamp(yahoo["last_trade_epoch"], timezone.utc).strftime("%H:%M:%S")
    return (
        f"{now:%H:%M:%S}Z {symbol:<5} "
        f"yahoo {yahoo_ts}Z {yahoo['last_price']:.2f} lag {lag_min:5.1f}m | "
        f"cboe {cboe['timestamp']} {cboe['price']:.2f} (last_trade {cboe['last_trade_time']})"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--polls", type=int, default=1, help="number of snapshots to take")
    parser.add_argument("--every", type=int, default=300, help="seconds between polls")
    args = parser.parse_args(argv)

    for poll in range(args.polls):
        now = datetime.now(timezone.utc)
        for symbol in SYMBOLS:
            print(snapshot(symbol, now))
        if poll < args.polls - 1:
            time.sleep(args.every)


if __name__ == "__main__":
    main()
