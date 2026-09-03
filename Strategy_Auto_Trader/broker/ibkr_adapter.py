"""IBKR execution adapter via ib_async.

Connects to Interactive Brokers TWS or IB Gateway.
Default port 7497 = paper trading account in TWS.
Default port 4002 = paper trading account in IB Gateway.

TWS setup: Preferences -> API -> Enable ActiveX and Socket Clients,
set Trusted IP to 127.0.0.1, port 7497 for paper.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from .symbols import PENCE_PER_POUND, ibkr_contract_params, yfinance_ticker
from .types import FillResult, OrderRequest, PendingCancelEvent

logger = logging.getLogger(__name__)

# Statuses IBKR reports for an order still resting/working (not yet a
# terminal fill/cancel outcome).
_WORKING_STATUSES = ("PendingSubmit", "PreSubmitted", "Submitted", "ApiPending", "Acknowledged")
_TERMINAL_NON_FILL_STATUSES = ("Cancelled", "ApiCancelled", "Inactive")

# How long a cancel can stay unconfirmed before we send one alert about it.
# Non-blocking: checked once per daemon cycle via check_pending_cancels(),
# never waited-for synchronously inside place_order().
PENDING_CANCEL_ALERT_SECONDS = 30 * 60


class IBKRAdapter:
    """Wraps ib_async for order placement and position queries.

    ib_async is imported lazily so the rest of the package works even if
    it is not installed (NullBroker / --dry-run does not need it).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        timeout: float = 30.0,
        connect_timeout: float = 30.0,
    ) -> None:
        self._host = host
        self._port = port
        self._client_id = client_id
        self._timeout = timeout
        # ib_async's default connect timeout (4s) is too tight for a busy
        # TWS; its handshake is also known to time out transiently.
        self._connect_timeout = connect_timeout
        self._ib = None
        # Orders where our cancel request wasn't confirmed within the short
        # in-call poll — resolved later, once per cycle, by check_pending_cancels().
        self._pending_cancels: list[dict] = []

    def connect(self) -> None:
        """Connect to TWS / IB Gateway. Raises RuntimeError if ib_async is missing."""
        try:
            from ib_async import IB
        except ImportError as exc:
            raise RuntimeError(
                "ib_async is not installed. Run: uv add ib_async"
            ) from exc
        # ib_async logs its own ERROR for transient socket events (e.g. weekend
        # TWS restart) before raising — our adapter already catches and re-logs
        # these as WARNING, so suppress the library's own noise.
        logging.getLogger("ib_async").setLevel(logging.CRITICAL)
        logging.getLogger("ibapi").setLevel(logging.CRITICAL)
        self._ib = IB()
        self._ib.connect(self._host, self._port, clientId=self._client_id,
                         timeout=self._connect_timeout)

    def managed_accounts(self) -> list[str]:
        """Return the account ids the session is authorised for."""
        return list(self._ib.managedAccounts())

    def disconnect(self) -> None:
        """Disconnect cleanly (safe to call even if not connected)."""
        if self._ib is not None:
            try:
                self._ib.disconnect()
            except Exception as e:
                logger.debug("disconnect() suppressed error: %s", e)
            self._ib = None

    def is_connected(self) -> bool:
        """Check if broker is currently connected."""
        if self._ib is None:
            return False
        try:
            return self._ib.isConnected()
        except Exception as e:
            logger.debug("is_connected() check failed (suppressed): %s", e)
            return False

    def get_last_price(self, ticker: str) -> float:
        """Return last traded / midpoint price (pence for LSE tickers)."""
        from ib_async import Stock
        contract = Stock(*ibkr_contract_params(ticker))
        self._ib.qualifyContracts(contract)
        tdata = self._ib.reqMktData(contract, "", True, False)
        self._ib.sleep(2)
        mid = tdata.midpoint()
        if mid and mid > 0:
            return float(mid)
        if tdata.last and tdata.last > 0:
            return float(tdata.last)
        return float(tdata.close or 0.0)

    def place_order(self, req: OrderRequest) -> FillResult | None:
        """Submit a market order and wait for fill (up to self._timeout seconds).

        Returns FillResult if order is fully filled, None if cancelled/partially filled/inactive.
        Raises ConnectionError if socket is disconnected, other exceptions for order failures.
        """
        if not self.is_connected():
            raise ConnectionError(
                f"Socket disconnect: not connected to {self._host}:{self._port}"
            )

        try:
            from ib_async import Stock, MarketOrder
            contract = Stock(*ibkr_contract_params(req.ticker))
            self._ib.qualifyContracts(contract)
            order = MarketOrder(req.action, req.quantity, tif="GTC")
            logger.info(
                "About to place order: %s %s×%s @ market",
                req.action, req.quantity, req.ticker,
            )
            trade = self._ib.placeOrder(contract, order)
            logger.info(
                "Order placed (awaiting fill): %s %s×%s orderId=%s",
                req.action, req.quantity, req.ticker,
                getattr(trade.order, "orderId", "?"),
            )
            # waitOnUpdate() returns on the *first* incoming update event
            # (typically just the PendingSubmit/Submitted ack), not after
            # self._timeout elapses — so a single call badly undershoots the
            # intended wait-for-fill window. Loop until Filled, a terminal
            # non-fill status, or the cumulative deadline is actually spent.
            deadline = time.monotonic() + self._timeout
            order_status = trade.orderStatus.status
            while order_status in _WORKING_STATUSES:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._ib.waitOnUpdate(timeout=remaining)
                order_status = trade.orderStatus.status

            # Check order status — only return FillResult if fully filled
            if order_status != "Filled":
                # Account TIF preset is GTC, so an unfilled order rests on
                # IBKR's books indefinitely instead of auto-expiring at end
                # of day — cancel it ourselves so it can't fill unattended
                # after this call returns and the in-flight marker clears.
                if order_status in _WORKING_STATUSES:
                    self._ib.cancelOrder(trade.order)
                    for _ in range(10):
                        self._ib.waitOnUpdate(timeout=0.5)
                        if trade.orderStatus.status in (*_TERMINAL_NON_FILL_STATUSES, "Filled"):
                            break
                    order_status = trade.orderStatus.status
                    if order_status == "Filled":
                        # Filled while the cancel request was in flight — fall
                        # through to the normal fill-price handling below.
                        pass
                    elif order_status in _TERMINAL_NON_FILL_STATUSES:
                        logger.warning(
                            f"Order not filled for {req.ticker}: cancelled resting "
                            f"order (was {order_status}), requested_qty={req.quantity}"
                        )
                        return None
                    else:
                        # Cancel not confirmed within the short poll. Don't block
                        # the daemon cycle waiting further — hand it off to
                        # check_pending_cancels(), which resolves it (or alerts
                        # once, non-blockingly) on subsequent cycles.
                        logger.warning(
                            f"Order not filled for {req.ticker}: cancel requested but "
                            f"unconfirmed (status={order_status}), requested_qty={req.quantity} "
                            f"— will keep checking each cycle"
                        )
                        self._pending_cancels.append({
                            "trade": trade,
                            "ticker": req.ticker,
                            "action": req.action,
                            "quantity": req.quantity,
                            "requested_at": time.monotonic(),
                            "alerted": False,
                        })
                        return None
                else:
                    logger.warning(
                        f"Order not filled for {req.ticker}: status={order_status}, "
                        f"requested_qty={req.quantity}"
                    )
                    return None

            fill_price = float(trade.orderStatus.avgFillPrice or 0.0)
            if fill_price <= 0:
                # orderStatus.status can flip to "Filled" on an earlier event
                # tick than avgFillPrice/fills populate — poll a few more
                # ticks before giving up.
                for _ in range(5):
                    if trade.fills:
                        exec_price = (trade.fills[-1].execution.avgPrice
                                       or trade.fills[-1].execution.price)
                        if exec_price:
                            fill_price = float(exec_price)
                            break
                    self._ib.waitOnUpdate(timeout=1.0)
                    fill_price = float(trade.orderStatus.avgFillPrice or fill_price)

            logger.info(
                "Order filled: %s %s×%s @ %.4f",
                req.action, req.quantity, req.ticker, fill_price,
            )
            return FillResult(
                ticker=req.ticker,
                action=req.action,
                fill_price=fill_price,
                quantity=req.quantity,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as e:
            if not self.is_connected():
                raise ConnectionError(
                    f"Socket disconnect during order placement: {e}"
                ) from e
            raise

    def get_open_positions(self) -> dict[str, int]:
        """Return {ticker: quantity} for all open positions, keyed by yfinance ticker."""
        return {
            yfinance_ticker(pos.contract.symbol, pos.contract.currency): int(pos.position)
            for pos in self._ib.positions()
            if hasattr(pos.contract, "symbol") and pos.position != 0
        }

    def _min_tick_for(self, contract, price: float) -> float:
        """Tick size applicable at *price* (native contract currency).

        Uses the exchange's price-band rule table (reqMarketRule), not the
        flat ContractDetails.minTick — LSE stocks trade under the MiFID II
        tick-size regime where the increment widens at higher price bands, so
        a single global minTick is wrong for them (US SMART equities happen
        to have a single flat band, so this also covers them correctly).
        Falls back to 0.01 (cent/penny) if the lookup fails for any reason.
        """
        try:
            details = self._ib.reqContractDetails(contract)
            if details:
                rule_ids = [r for r in details[0].marketRuleIds.split(",") if r]
                if rule_ids:
                    rules = self._ib.reqMarketRule(int(rule_ids[0]))
                    if rules:
                        increment = rules[0].increment
                        for r in sorted(rules, key=lambda r: r.lowEdge):
                            if price >= r.lowEdge:
                                increment = r.increment
                            else:
                                break
                        if increment > 0:
                            return increment
        except Exception as e:
            logger.debug("_min_tick_for lookup failed, using 0.01 fallback: %s", e)
        return 0.01

    def place_stop_order(self, req):
        """Place a resting GTC stop-sell order. Returns StopOrderResult with permId on acceptance, None if rejected."""
        if not self.is_connected():
            raise ConnectionError(
                f"Socket disconnect: not connected to {self._host}:{self._port}"
            )

        try:
            from ib_async import Stock, StopOrder
            contract = Stock(*ibkr_contract_params(req.ticker))
            self._ib.qualifyContracts(contract)
            # Stop price math (stop_level * (1 - buffer_pct)) produces floats
            # with far more precision than the exchange accepts — IBKR rejects
            # with Error 110 "does not conform to the minimum price variation"
            # if not rounded to the contract's tick size. ContractDetails.minTick
            # is a flat, instrument-wide value and is WRONG for LSE stocks: those
            # trade under the MiFID II price-band tick regime (a table of
            # increments that widens at higher prices), only obtainable via
            # marketRuleIds -> reqMarketRule. That table is denominated in the
            # contract's native currency (pounds for LSE, not pence), so find
            # the tick and round in req.stop_price's own units (pot currency)
            # before converting to pence below.
            min_tick = self._min_tick_for(contract, req.stop_price)
            native_stop = round(round(req.stop_price / min_tick) * min_tick, 8)
            # req.stop_price is pot currency (pounds); LSE orders quote in pence.
            exchange_stop = native_stop
            if req.ticker.upper().endswith(".L"):
                exchange_stop = native_stop * PENCE_PER_POUND
            order = StopOrder("SELL", req.quantity, exchange_stop, tif="GTC")
            trade = self._ib.placeOrder(contract, order)
            # waitOnUpdate() returns on the *first* incoming update event
            # (typically just the PendingSubmit ack), not once the order has
            # actually settled onto the book — same undershoot as place_order
            # (see comment there). Poll until it leaves the transient
            # pending states or the deadline is spent.
            deadline = time.monotonic() + self._timeout
            order_status = trade.orderStatus.status
            while order_status in ("PendingSubmit", "ApiPending"):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._ib.waitOnUpdate(timeout=remaining)
                order_status = trade.orderStatus.status

            if order_status not in ("PreSubmitted", "Submitted", "Acknowledged"):
                logger.warning(
                    f"Stop order not accepted for {req.ticker}: status={order_status}"
                )
                return None

            perm_id = trade.order.permId
            if not perm_id:
                for _ in range(5):
                    self._ib.waitOnUpdate(timeout=1.0)
                    perm_id = trade.order.permId
                    if perm_id:
                        break

            if not perm_id:
                logger.warning(
                    f"Stop order for {req.ticker} accepted but permId never populated"
                )
                return None

            from .types import StopOrderResult
            return StopOrderResult(
                perm_id=perm_id,
                stop_price=req.stop_price,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as e:
            if not self.is_connected():
                raise ConnectionError(
                    f"Socket disconnect during stop order placement: {e}"
                ) from e
            raise

    def get_open_orders(self) -> list[dict]:
        """Return still-working orders of any type/action (not yet filled/cancelled).

        Position-only reconciliation can miss an order that was accepted by
        IBKR right before the client's socket dropped: the position comparison
        looks clean (order hasn't filled yet) but the order is still live and
        could fill any time after startup, creating an untracked position.
        Startup reconciliation checks this for the in-flight marker's ticker
        before clearing the marker.
        """
        if not self.is_connected():
            raise ConnectionError(
                f"Socket disconnect: not connected to {self._host}:{self._port}"
            )

        try:
            self._ib.reqAllOpenOrders()
            self._ib.waitOnUpdate(timeout=2.0)

            orders = []
            for trade in self._ib.trades():
                if (trade.contract and hasattr(trade.contract, "symbol") and
                    trade.orderStatus.status in
                        ("PendingSubmit", "PreSubmitted", "Submitted", "Acknowledged")):
                    try:
                        orders.append({
                            "ticker": yfinance_ticker(
                                trade.contract.symbol, trade.contract.currency
                            ),
                            "action": trade.order.action,
                            "status": trade.orderStatus.status,
                        })
                    except Exception as e:
                        logger.warning(
                            f"Skipping malformed open order for "
                            f"{getattr(trade.contract, 'symbol', '?')}: {e}"
                        )
            return orders
        except ConnectionError:
            raise
        except Exception as e:
            logger.warning(f"Error retrieving open orders: {e}")
            raise

    def get_open_stop_orders(self) -> dict:
        """Return {permId: OpenOrderInfo} for open SELL STP orders using reqAllOpenOrders."""
        if not self.is_connected():
            raise ConnectionError(
                f"Socket disconnect: not connected to {self._host}:{self._port}"
            )

        try:
            from .types import OpenOrderInfo
            self._ib.reqAllOpenOrders()
            self._ib.waitOnUpdate(timeout=2.0)

            result = {}
            for trade in self._ib.trades():
                if (trade.contract and hasattr(trade.order, 'action') and
                    trade.order.action == "SELL" and
                    hasattr(trade.order, 'orderType') and
                    trade.order.orderType == "STP" and
                    trade.order.permId and
                    trade.orderStatus.status in ("PreSubmitted", "Submitted", "Acknowledged")):
                    try:
                        ticker_key = yfinance_ticker(
                            trade.contract.symbol, trade.contract.currency
                        )
                        stop_price = float(trade.order.auxPrice or 0.0)
                        # auxPrice is exchange units (pence for LSE) — report
                        # pot currency to match internal state.
                        if ticker_key.upper().endswith(".L"):
                            stop_price /= PENCE_PER_POUND
                        result[trade.order.permId] = OpenOrderInfo(
                            ticker=ticker_key,
                            quantity=int(trade.order.totalQuantity),
                            stop_price=stop_price,
                            perm_id=trade.order.permId,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Skipping malformed open stop order for "
                            f"{getattr(trade.contract, 'symbol', '?')}: {e}"
                        )
            return result
        except ConnectionError:
            raise
        except Exception as e:
            logger.warning(f"Error retrieving open stop orders: {e}")
            raise

    def check_pending_cancels(self) -> list[PendingCancelEvent]:
        """Resolve entry orders whose cancel request wasn't confirmed in-call.

        Called once per daemon cycle (non-blocking) rather than waiting
        synchronously inside place_order() — a stuck cancel can take a long
        time to confirm and must never stall the trading loop. Each pending
        order is checked against its live Trade object (already updating in
        the background as long as the connection stays open); resolved
        entries are dropped, still-working ones are alerted once after
        PENDING_CANCEL_ALERT_SECONDS and then left for the next cycle.
        """
        if not self._pending_cancels:
            return []
        if self.is_connected():
            try:
                self._ib.waitOnUpdate(timeout=0.5)
            except Exception as e:
                logger.debug(f"check_pending_cancels: waitOnUpdate failed (suppressed): {e}")

        events: list[PendingCancelEvent] = []
        survivors: list[dict] = []
        for record in self._pending_cancels:
            status = record["trade"].orderStatus.status
            if status == "Filled":
                fill_price = float(record["trade"].orderStatus.avgFillPrice or 0.0)
                events.append(PendingCancelEvent(
                    ticker=record["ticker"], action=record["action"],
                    quantity=record["quantity"], outcome="filled",
                    fill=FillResult(
                        ticker=record["ticker"], action=record["action"],
                        fill_price=fill_price, quantity=record["quantity"],
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

    def cancel_stop_order(self, perm_id: int) -> str:
        """Cancel a stop order by permId. Returns 'Cancelled' | 'Filled' | 'NotFound' | 'Error'."""
        if not self.is_connected():
            # Disconnected is not "not found" — callers treat NotFound as
            # safe-to-proceed with a market sell, which is unsafe here.
            return "Error"

        try:
            for trade in self._ib.trades():
                if trade.order.permId == perm_id:
                    if trade.orderStatus.status == "Filled":
                        return "Filled"
                    self._ib.cancelOrder(trade.order)
                    for _ in range(10):
                        self._ib.waitOnUpdate(timeout=0.5)
                        if trade.orderStatus.status in ("Cancelled", "Filled"):
                            return trade.orderStatus.status
                    return "Cancelled"
            return "NotFound"
        except Exception as e:
            logger.warning(f"Error cancelling stop order {perm_id}: {e}")
            return "Error"

    def get_stop_fill(self, perm_id: int):
        """Look up execution for a stop order by permId. Returns FillResult or None."""
        if not self.is_connected():
            return None

        try:
            fills_list = self._ib.fills()
            for fill in fills_list:
                if fill.execution.permId == perm_id:
                    from .types import FillResult
                    return FillResult(
                        ticker=yfinance_ticker(
                            fill.contract.symbol, fill.contract.currency
                        ),
                        action="SELL",
                        fill_price=float(fill.execution.price or 0.0),
                        quantity=int(fill.execution.shares or 0),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
            return None
        except Exception as e:
            logger.warning(f"Error retrieving stop fill for {perm_id}: {e}")
            return None
