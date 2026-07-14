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

from .symbols import ibkr_contract_params, yfinance_ticker
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
