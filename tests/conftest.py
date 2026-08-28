from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _mock_ibkr_no_connection(request, monkeypatch):
    """Prevent non-broker tests from making real IBKR TCP connections.

    With source="ibkr" as the default, fetch_hourly attempts IBKRDataClient.connect()
    on every call. If TWS isn't running, ib_async can hang for its full connect_timeout
    (30s). Patching connect() to return False makes it fall through to yfinance
    immediately — identical behaviour to the old source="yfinance" default.

    Broker tests (tests/broker/) mock at the ib_async.IB level and manage their
    own IBKR fixtures, so we skip this patch for them.
    """
    if "broker" in request.node.nodeid:
        return
    monkeypatch.setattr(
        "Strategy_Auto_Trader.broker.ibkr_data.IBKRDataClient.connect",
        lambda self: False,
    )


def _dates(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _rising_prices(n: int = 300, start: float = 100.0, daily_pct: float = 0.005) -> pd.Series:
    """Steadily rising price series (should label Bull)."""
    idx = _dates(n)
    prices = start * (1 + daily_pct) ** np.arange(n)
    return pd.Series(prices, index=idx, name="close")


def _falling_prices(n: int = 300, start: float = 100.0, daily_pct: float = 0.005) -> pd.Series:
    """Steadily falling price series (should label Bear)."""
    idx = _dates(n)
    prices = start * (1 - daily_pct) ** np.arange(n)
    return pd.Series(prices, index=idx, name="close")


def _flat_prices(n: int = 300, price: float = 100.0) -> pd.Series:
    """Flat price series (should label Sideways)."""
    idx = _dates(n)
    return pd.Series(np.full(n, price), index=idx, name="close")


def _long_rising_prices(n: int = 600, start: float = 100.0, daily_pct: float = 0.003) -> pd.Series:
    """Long rising series for backtest warmup requirements."""
    idx = _dates(n)
    prices = start * (1 + daily_pct) ** np.arange(n)
    return pd.Series(prices, index=idx, name="close")
