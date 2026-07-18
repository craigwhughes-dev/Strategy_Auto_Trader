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


def normalize_fill_price(ticker: str, fill_price: float, reference_pence: float) -> float:
    """Normalize a broker fill price to pot currency (pounds for LSE).

    IBKR returns LSE execution prices inconsistently — sometimes pence
    (HSBA.L filled as 1462.0), sometimes pounds (VOD.L filled as 1.139) —
    so a fixed conversion mis-prices one case or the other. Disambiguate
    against a pence reference price from local context (signal close, stop
    level x 100): a pence-scale fill sits near the reference, a pounds-scale
    fill sits ~100x below it. Cut at 20% of reference — 5x margin against
    both intraday moves and the 100x unit gap.

    Non-LSE tickers and non-positive inputs pass through unchanged. With no
    usable reference, assume exchange units (pence) and convert.
    """
    if not ticker.upper().endswith(".L") or fill_price <= 0:
        return fill_price
    if reference_pence and reference_pence > 0:
        if fill_price / reference_pence > 0.2:
            return fill_price / PENCE_PER_POUND   # pence-scale fill
        return fill_price                          # already pounds
    return fill_price / PENCE_PER_POUND
