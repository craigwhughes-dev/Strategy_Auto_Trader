"""Trading212 execution adapter via the T212 Public API (REST, beta).

*** VERIFICATION STATUS: UNVERIFIED AGAINST A LIVE T212 ACCOUNT ***
This adapter was written from docs.trading212.com without access to a real
API key — field names (order status enum, response shapes) are best-effort
from the published docs, not confirmed by a live call the way every comment
in ibkr_adapter.py / symbols.py is dated against a verified paper-Gateway
lookup. Before `real_money: true` (or any live order) is enabled with this
adapter:
  1. Run against https://demo.trading212.com against a demo API key first.
  2. Confirm order status enum values in _TERMINAL_NON_FILL_STATUSES below
     actually match what GET /api/v0/equity/orders/{id} returns.
  3. Confirm whether LSE-instrument prices/stop prices are quoted in pence
     or pounds (ibkr_adapter.py's pence handling was verified live; this
     adapter deliberately does NOT replicate that conversion unverified).
  4. Populate config/t212_symbol_map.json (see t212_symbols.py) and confirm
     each mapped ticker against a real position/order on the demo account.

No price-feed endpoint exists in the T212 public API (confirmed from docs:
no last-price/quote/historical-bar route). get_last_price() reads from an
in-memory cache that must be fed externally via set_prices() — same pattern
as NullBroker — typically from the same hourly bars the daemon already
fetches for signal computation, not from T212 itself.

Auth: HTTP Basic, base64("API_KEY:API_SECRET"). Env split by base URL, not by
account id (T212 has no DU/DF-style paper-account prefix) — demo and live use
different hosts and different, separately-generated API keys, so pointing
this adapter at the live host is the only way to touch real money, mirroring
ibkr_adapter.py's allow_live_account gate.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from .t212_symbols import load_symbol_map, t212_ticker, yfinance_ticker_from_t212
from .types import (
    AmbiguousOrderError,
    FillResult,
    InsufficientFundsError,
    OpenOrderInfo,
    OrderRequest,
    PendingCancelEvent,
    StopOrderRequest,
    StopOrderResult,
)

logger = logging.getLogger(__name__)

DEMO_BASE_URL = "https://demo.trading212.com"
LIVE_BASE_URL = "https://live.trading212.com"

# Confirmed from docs.trading212.com/api/orders/orders: FILLED, CANCELLED,
# PARTIALLY_FILLED are real enum values. REJECTED/EXPIRED are assumed by
# analogy with IBKR's terminal-non-fill set, NOT confirmed in T212 docs —
# verify against a demo order before relying on this set (see module banner).
_FILLED_STATUS = "FILLED"

# Held-position prices are re-read at most this often; the daemon asks for prices every cycle (~60s)
POSITION_PRICE_TTL_SECONDS = 30.0

# T212's error type for a buy that exceeds free cash, as it appears in the RuntimeError text
_INSUFFICIENT_FREE_CASH = "insufficient-free-for-stocks-buy"
_TERMINAL_NON_FILL_STATUSES = ("CANCELLED", "REJECTED", "EXPIRED")

# How long a cancel can stay unconfirmed before check_pending_cancels()
# raises one alert about it — mirrors ibkr_adapter.py's same constant.
PENDING_CANCEL_ALERT_SECONDS = 30 * 60


class T212AccountMismatchError(RuntimeError):
    """Refusing to trade — live host requested without allow_live_account."""


class T212Adapter:
    """Satisfies BrokerAdapterProtocol via T212's REST API.

    No persistent connection (REST, not a socket) — connect()/disconnect()
    exist to satisfy the protocol and to run the one-time account-identity
    check, not to hold a session open.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        environment: str = "demo",
        allow_live_account: bool = False,
        timeout: float = 30.0,
        symbol_map: dict[str, str] | None = None,
    ) -> None:
        if environment not in ("demo", "live"):
            raise ValueError(f"environment must be 'demo' or 'live', got {environment!r}")
        if environment == "live" and not allow_live_account:
            raise T212AccountMismatchError(
                "environment='live' requested without allow_live_account — "
                "refusing to construct an adapter that could place real-money orders"
            )
        self._base_url = LIVE_BASE_URL if environment == "live" else DEMO_BASE_URL
        self._environment = environment
        self._auth_header = "Basic " + base64.b64encode(
            f"{api_key}:{api_secret}".encode("utf-8")
        ).decode("ascii")
        self._timeout = timeout
        self._symbol_map = symbol_map if symbol_map is not None else load_symbol_map()
        self._connected = False
        self._prices: dict[str, float] = {}
        self._positions_cache: dict[str, dict] | None = None
        self._positions_cache_at = 0.0
        self._pending_cancels: list[dict] = []

    def set_prices(self, prices: dict[str, float]) -> None:
        """Feed last-close prices from the daemon's own data fetch.

        T212's API has no quote endpoint — see module docstring. Called
        each cycle the same way NullBroker.set_prices() is.
        """
        self._prices.update(prices)

    def _request(self, method: str, path: str, body: dict | None = None):
        """Issue one REST call, raising RuntimeError with T212's own message on failure."""
        url = self._base_url + path
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", self._auth_header)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"T212 {method} {path} failed: {e.code} {detail}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"T212 {method} {path} unreachable: {e.reason}") from e

    def connect(self) -> None:
        """Verify credentials against GET /api/v0/equity/account/info."""
        self._request("GET", "/api/v0/equity/account/info")
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_last_price(self, ticker: str) -> float:
        if ticker in self._prices:
            return self._prices[ticker]
        held = self._held_prices()
        if ticker in held:
            return held[ticker]
        raise ValueError(
            f"{ticker}: no price available — T212 only prices held positions and there is no quote "
            f"endpoint; call set_prices() with this cycle's latest closes or buy the instrument first"
        )

    def _held_prices(self) -> dict[str, float]:
        """currentPrice of each held position. Units are the quote currency (pence for GBX lines), as sizing_price expects."""
        return {
            ticker: float(pos["currentPrice"])
            for ticker, pos in self._cached_positions().items()
            if pos.get("currentPrice")
        }

    def _cached_positions(self) -> dict[str, dict]:
        """Positions from one shared read, refreshed at most every POSITION_PRICE_TTL_SECONDS.

        T212 rate-limits the positions endpoint (429 when the daemon reads it every cycle), so the price
        fallback and the position reconciliation share one read. place_order clears it after each order.
        """
        now = time.monotonic()
        if self._positions_cache is None or now - self._positions_cache_at >= POSITION_PRICE_TTL_SECONDS:
            self._positions_cache = self._fetch_positions()
            self._positions_cache_at = now
        return self._positions_cache

    def _fetch_positions(self) -> dict[str, dict]:
        """Raw position records keyed by yfinance ticker; instruments outside the symbol map are skipped."""
        result: dict[str, dict] = {}
        for pos in self._request("GET", "/api/v0/equity/positions") or []:
            code = (pos.get("instrument") or {}).get("ticker")
            if code is None:
                continue
            try:
                ticker = yfinance_ticker_from_t212(code, self._symbol_map)
            except KeyError as e:
                logger.warning(f"Skipping unmapped T212 position: {e}")
                continue
            result[ticker] = pos
        return result

    def _get_order(self, order_id: int) -> dict:
        """One order read. T212 can 404 an order for a moment after placing it; that reads as not visible yet."""
        try:
            return self._request("GET", f"/api/v0/equity/orders/{order_id}")
        except RuntimeError as e:
            if " failed: 404 " in str(e):
                return {}
            raise

    def _poll_order(self, order_id: int, deadline: float) -> dict:
        """Poll GET .../orders/{id} until a terminal status or the deadline."""
        order = self._get_order(order_id)
        status = order.get("status")
        while (
            status != _FILLED_STATUS
            and status not in _TERMINAL_NON_FILL_STATUSES
            and time.monotonic() < deadline
        ):
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
            order = self._get_order(order_id)
            status = order.get("status")
        return order

    def place_order(self, req: OrderRequest) -> FillResult | None:
        self._positions_cache = None  # the fill changes holdings, so the next read must go to the API
        t212_ticker(req.ticker, self._symbol_map)  # unmapped tickers fail here, before anything is sent
        try:
            return self._place_market_order(req)
        except RuntimeError as e:
            if " failed: 400 " in str(e) or isinstance(e, InsufficientFundsError):
                raise  # T212 refused the order outright, so nothing was placed
            self._reconcile_unconfirmed(req, e)
            raise AmbiguousOrderError(f"{req.action} {req.ticker} state unknown after: {e}") from e
        except Exception as e:  # timeouts and dropped connections: the order may have been placed
            self._reconcile_unconfirmed(req, e)
            raise AmbiguousOrderError(f"{req.action} {req.ticker} state unknown after: {e}") from e

    def _reconcile_unconfirmed(self, req: OrderRequest, cause: Exception) -> None:
        """Cancel any open order matching an ambiguous one, so it cannot fill unseen.

        An order that already filled is not open and nothing is cancelled. The next position read shows which.
        """
        self._positions_cache = None
        code = t212_ticker(req.ticker, self._symbol_map)
        try:
            open_orders = self._request("GET", "/api/v0/equity/orders") or []
        except Exception as e:
            logger.error(f"Order state unknown for {req.action} {req.ticker} ({cause}); could not list open orders ({e}) "
                         f"— check T212 before trading this ticker")
            return
        for order in open_orders:
            if order.get("ticker") != code or order.get("side") != req.action:
                continue
            try:
                self._request("DELETE", f"/api/v0/equity/orders/{order['id']}")
                logger.warning(f"Cancelled open {req.action} {req.ticker} order {order['id']} left by an ambiguous error ({cause})")
            except Exception as e:
                logger.error(f"Could not cancel open {req.action} {req.ticker} order {order['id']}: {e}")

    def _place_market_order(self, req: OrderRequest) -> FillResult | None:
        code = t212_ticker(req.ticker, self._symbol_map)
        signed_qty = req.quantity if req.action == "BUY" else -req.quantity
        try:
            resp = self._request(
                "POST", "/api/v0/equity/orders/market",
                {"ticker": code, "quantity": signed_qty},
            )
        except RuntimeError as e:
            if _INSUFFICIENT_FREE_CASH in str(e):
                raise InsufficientFundsError(str(e)) from e
            raise
        order_id = resp.get("id")
        if order_id is None:
            logger.warning(f"T212 market order for {req.ticker} accepted but no id returned: {resp}")
            return None

        deadline = time.monotonic() + self._timeout
        order = self._poll_order(order_id, deadline)
        status = order.get("status")

        if status != _FILLED_STATUS:
            if status not in _TERMINAL_NON_FILL_STATUSES:
                # Still working at the deadline — cancel it ourselves rather
                # than leave it resting unattended (mirrors ibkr_adapter.py).
                try:
                    self._request("DELETE", f"/api/v0/equity/orders/{order_id}")
                except Exception as e:
                    logger.warning(f"Cancel request failed for order {order_id}: {e}")
                self._pending_cancels.append({
                    "order_id": order_id,
                    "ticker": req.ticker,
                    "action": req.action,
                    "quantity": req.quantity,
                    "requested_at": time.monotonic(),
                    "alerted": False,
                })
                logger.warning(
                    f"Order not filled for {req.ticker}: cancel requested but "
                    f"unconfirmed (status={status}) — will keep checking each cycle"
                )
                return None
            logger.warning(f"Order not filled for {req.ticker}: status={status}, requested_qty={req.quantity}")
            return None

        fill_price = float(order.get("fillPrice") or order.get("averagePrice") or 0.0)
        return FillResult(
            ticker=req.ticker,
            action=req.action,
            fill_price=fill_price,
            quantity=req.quantity,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_available_cash(self) -> float:
        """Cash the account can trade with now, from the account summary (GBP)."""
        summary = self._request("GET", "/api/v0/equity/account/summary") or {}
        return float(summary["cash"]["availableToTrade"])

    def get_open_positions(self) -> dict[str, float]:
        return {
            ticker: float(pos["quantity"])
            for ticker, pos in self._cached_positions().items()
            if pos.get("quantity")
        }

    def get_open_orders(self) -> list[dict]:
        orders = self._request("GET", "/api/v0/equity/orders") or []
        result = []
        for order in orders:
            status = order.get("status")
            if status in (_FILLED_STATUS, *_TERMINAL_NON_FILL_STATUSES):
                continue
            code = order.get("ticker")
            try:
                ticker = yfinance_ticker_from_t212(code, self._symbol_map)
            except KeyError as e:
                logger.warning(f"Skipping unmapped open order: {e}")
                continue
            result.append({"ticker": ticker, "action": order.get("side"), "status": status})
        return result

    def place_stop_order(self, req: StopOrderRequest) -> StopOrderResult | None:
        code = t212_ticker(req.ticker, self._symbol_map)
        resp = self._request(
            "POST", "/api/v0/equity/orders/stop",
            {"ticker": code, "quantity": -req.quantity, "stopPrice": req.stop_price},
        )
        order_id = resp.get("id")
        if order_id is None:
            logger.warning(f"T212 stop order for {req.ticker} accepted but no id returned: {resp}")
            return None
        return StopOrderResult(
            perm_id=order_id,
            stop_price=req.stop_price,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_open_stop_orders(self) -> dict[int, OpenOrderInfo]:
        orders = self._request("GET", "/api/v0/equity/orders") or []
        result = {}
        for order in orders:
            if (
                order.get("type") == "STOP"
                and order.get("side") == "SELL"
                and order.get("status") not in (_FILLED_STATUS, *_TERMINAL_NON_FILL_STATUSES)
            ):
                order_id = order.get("id")
                code = order.get("ticker")
                try:
                    ticker = yfinance_ticker_from_t212(code, self._symbol_map)
                except KeyError as e:
                    logger.warning(f"Skipping unmapped open stop order: {e}")
                    continue
                result[order_id] = OpenOrderInfo(
                    ticker=ticker,
                    quantity=abs(float(order.get("quantity", 0))),
                    stop_price=float(order.get("stopPrice", 0.0)),
                    perm_id=order_id,
                )
        return result

    def cancel_stop_order(self, perm_id: int) -> str:
        try:
            order = self._request("GET", f"/api/v0/equity/orders/{perm_id}")
        except Exception as e:
            logger.warning(f"Error looking up stop order {perm_id}: {e}")
            return "Error"
        if order is None:
            return "NotFound"
        if order.get("status") == _FILLED_STATUS:
            return "Filled"
        try:
            self._request("DELETE", f"/api/v0/equity/orders/{perm_id}")
        except Exception as e:
            logger.warning(f"Error cancelling stop order {perm_id}: {e}")
            return "Error"
        return "Cancelled"

    def get_stop_fill(self, perm_id: int) -> FillResult | None:
        try:
            order = self._request("GET", f"/api/v0/equity/orders/{perm_id}")
        except Exception as e:
            logger.warning(f"Error retrieving stop fill for {perm_id}: {e}")
            return None
        if order is None or order.get("status") != _FILLED_STATUS:
            return None
        code = order.get("ticker")
        try:
            ticker = yfinance_ticker_from_t212(code, self._symbol_map)
        except KeyError:
            return None
        return FillResult(
            ticker=ticker,
            action="SELL",
            fill_price=float(order.get("fillPrice") or order.get("averagePrice") or 0.0),
            quantity=abs(float(order.get("filledQuantity") or order.get("quantity") or 0)),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def check_pending_cancels(self) -> list[PendingCancelEvent]:
        if not self._pending_cancels:
            return []
        events: list[PendingCancelEvent] = []
        survivors: list[dict] = []
        for record in self._pending_cancels:
            try:
                order = self._request("GET", f"/api/v0/equity/orders/{record['order_id']}")
            except Exception as e:
                logger.warning(f"check_pending_cancels: lookup failed for {record['order_id']}: {e}")
                survivors.append(record)
                continue
            status = order.get("status") if order else None
            if status == _FILLED_STATUS:
                events.append(PendingCancelEvent(
                    ticker=record["ticker"], action=record["action"],
                    quantity=record["quantity"], outcome="filled",
                    fill=FillResult(
                        ticker=record["ticker"], action=record["action"],
                        fill_price=float(order.get("fillPrice") or order.get("averagePrice") or 0.0),
                        quantity=record["quantity"],
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                ))
                continue
            if status in _TERMINAL_NON_FILL_STATUSES:
                events.append(PendingCancelEvent(
                    ticker=record["ticker"], action=record["action"],
                    quantity=record["quantity"], outcome="cancelled",
                ))
                continue
            elapsed = time.monotonic() - record["requested_at"]
            if elapsed >= PENDING_CANCEL_ALERT_SECONDS and not record["alerted"]:
                record["alerted"] = True
                events.append(PendingCancelEvent(
                    ticker=record["ticker"], action=record["action"],
                    quantity=record["quantity"], outcome="timeout_alert",
                    elapsed_minutes=elapsed / 60.0,
                ))
            survivors.append(record)
        self._pending_cancels = survivors
        return events
