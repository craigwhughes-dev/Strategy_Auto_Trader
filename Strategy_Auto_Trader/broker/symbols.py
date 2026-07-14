"""Ticker symbol mapping between yfinance conventions and IBKR contracts.

Watchlists use yfinance symbols ("HSBA.L", "BT-A.L", "SPY"). IBKR wants the
bare symbol plus an exchange and currency, and LSE share classes use "." where
yfinance uses "-".
"""

from __future__ import annotations

PENCE_PER_POUND = 100.0


def ibkr_contract_params(ticker: str) -> tuple[str, str, str]:
    """Map a yfinance ticker to (symbol, exchange, currency) for an IBKR Stock.

    ".L" suffix → LSE/GBP with the suffix stripped; everything else is treated
    as a US equity on SMART/USD.
    """
    if ticker.upper().endswith(".L"):
        return ticker[:-2].replace("-", "."), "LSE", "GBP"
    return ticker, "SMART", "USD"


def yfinance_ticker(symbol: str, currency: str) -> str:
    """Map an IBKR contract's (symbol, currency) back to a yfinance ticker.

    Inverse of ibkr_contract_params: GBP contracts are LSE-listed, so append
    ".L" and turn share-class dots back into yfinance's hyphen.
    """
    if currency.upper() == "GBP":
        return symbol.replace(".", "-") + ".L"
    return symbol


def sizing_price(ticker: str, price: float) -> float:
    """Convert a quoted price into pot-currency units for position sizing.

    LSE prices (yfinance and IBKR alike) are quoted in pence; the capital pot
    is in pounds, so divide by 100. Other prices pass through unchanged.
    """
    if ticker.upper().endswith(".L"):
        return price / PENCE_PER_POUND
    return price
