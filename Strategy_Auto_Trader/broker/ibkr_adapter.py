"""IBKR execution adapter via ib_insync.

Connects to Interactive Brokers TWS or IB Gateway.
Default port 7497 = paper trading account in TWS.
Default port 4002 = paper trading account in IB Gateway.

TWS setup: Preferences -> API -> Enable ActiveX and Socket Clients,
set Trusted IP to 127.0.0.1, port 7497 for paper.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .symbols import ibkr_contract_params
from .types import FillResult, OrderRequest


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
            self._ib.disconnect()
            self._ib = None

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

    def place_order(self, req: OrderRequest) -> FillResult:
        """Submit a market order and wait for fill (up to self._timeout seconds)."""
        from ib_insync import Stock, MarketOrder
        contract = Stock(*ibkr_contract_params(req.ticker))
        self._ib.qualifyContracts(contract)
        order = MarketOrder(req.action, req.quantity)
        trade = self._ib.placeOrder(contract, order)
        self._ib.waitOnUpdate(timeout=self._timeout)
        fill_price = float(trade.orderStatus.avgFillPrice or 0.0)
        return FillResult(
            ticker=req.ticker,
            action=req.action,
            fill_price=fill_price,
            quantity=req.quantity,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_open_positions(self) -> dict[str, int]:
        """Return {ticker: quantity} for all open positions in the account."""
        return {
            pos.contract.symbol: int(pos.position)
            for pos in self._ib.positions()
            if hasattr(pos.contract, "symbol") and pos.position != 0
        }
