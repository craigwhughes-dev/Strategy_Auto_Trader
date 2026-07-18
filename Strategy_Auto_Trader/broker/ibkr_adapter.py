"""IBKR execution adapter via ib_insync.

Connects to Interactive Brokers TWS or IB Gateway.
Default port 7497 = paper trading account in TWS.
Default port 4002 = paper trading account in IB Gateway.

TWS setup: Preferences -> API -> Enable ActiveX and Socket Clients,
set Trusted IP to 127.0.0.1, port 7497 for paper.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .symbols import PENCE_PER_POUND, ibkr_contract_params, yfinance_ticker
from .types import FillResult, OrderRequest

logger = logging.getLogger(__name__)


class IBKRAdapter:
    """Wraps ib_insync for order placement and position queries.

    ib_insync is imported lazily so the rest of the package works even if
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
        # ib_insync's default connect timeout (4s) is too tight for a busy
        # TWS; its handshake is also known to time out transiently.
        self._connect_timeout = connect_timeout
        self._ib = None

    def connect(self) -> None:
        """Connect to TWS / IB Gateway. Raises RuntimeError if ib_insync is missing."""
        try:
            from ib_insync import IB
        except ImportError as exc:
            raise RuntimeError(
                "ib_insync is not installed. Run: uv add ib_insync"
            ) from exc
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
            except Exception:
                pass
            self._ib = None

    def is_connected(self) -> bool:
        """Check if broker is currently connected."""
        if self._ib is None:
            return False
        try:
            return self._ib.isConnected()
        except Exception:
            return False

    def get_last_price(self, ticker: str) -> float:
        """Return last traded / midpoint price (pence for LSE tickers)."""
        from ib_insync import Stock
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
            from ib_insync import Stock, MarketOrder
            contract = Stock(*ibkr_contract_params(req.ticker))
            self._ib.qualifyContracts(contract)
            order = MarketOrder(req.action, req.quantity)
            trade = self._ib.placeOrder(contract, order)
            self._ib.waitOnUpdate(timeout=self._timeout)

            # Check order status — only return FillResult if fully filled
            order_status = trade.orderStatus.status
            if order_status != "Filled":
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

    def place_stop_order(self, req):
        """Place a resting GTC stop-sell order. Returns StopOrderResult with permId on acceptance, None if rejected."""
        if not self.is_connected():
            raise ConnectionError(
                f"Socket disconnect: not connected to {self._host}:{self._port}"
            )

        try:
            from ib_insync import Stock, StopOrder
            contract = Stock(*ibkr_contract_params(req.ticker))
            self._ib.qualifyContracts(contract)
            # req.stop_price is pot currency (pounds); LSE orders quote in pence.
            exchange_stop = req.stop_price
            if req.ticker.upper().endswith(".L"):
                exchange_stop = req.stop_price * PENCE_PER_POUND
            order = StopOrder("SELL", req.quantity, exchange_stop, tif="GTC")
            trade = self._ib.placeOrder(contract, order)
            self._ib.waitOnUpdate(timeout=self._timeout)

            order_status = trade.orderStatus.status
            if order_status not in ("PreSubmitted", "Submitted"):
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
                    except Exception:
                        pass
            return result
        except ConnectionError:
            raise
        except Exception as e:
            logger.warning(f"Error retrieving open stop orders: {e}")
            raise

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
