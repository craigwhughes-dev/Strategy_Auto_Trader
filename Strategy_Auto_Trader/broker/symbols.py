"""Ticker symbol mapping between yfinance conventions and IBKR contracts.

Watchlists use yfinance symbols ("HSBA.L", "BT-A.L", "SPY"). IBKR wants the
bare symbol plus an exchange and currency, and LSE share classes use "." where
yfinance uses "-".
"""

from __future__ import annotations

PENCE_PER_POUND = 100.0

# Tickers confirmed unresolvable by IBKR in the GBP/pence currency the sizing
# pipeline requires.  Each entry explains what IBKR actually returns and the
# date it was verified live against the paper TWS gateway.
#
# build_sp_ftse_universe() filters this set out so a nightly Wikipedia refresh
# cannot silently reintroduce a ticker known to be broken on the IBKR side.
#
# If a Phase-1 reqContractDetails diagnostic later confirms a GBP line exists
# for one of these (e.g. a tradingClass or primaryExchange tweak missed by the
# generic lookup), remove the entry here and add an explicit override to
# _IBKR_EXPLICIT_OVERRIDES instead (see ibkr_contract_params below).
IBKR_UNRESOLVABLE: frozenset[str] = frozenset({
    "CPG.L",   # Compass Group: IBKR resolves as CPG/LSE/USD only — confirmed 2026-08-16
    "IHG.L",   # InterContinental Hotels: IBKR resolves as IHGL/LSE/USD only — confirmed 2026-08-16
    "MTLN.L",  # Metlen Energy (ex-Mytilineos): IBKR resolves as MTLN/LSE/EUR only — confirmed 2026-08-16
    "AVB",     # AvalonBay Communities (REIT): IBKR returns Error 200 "No security definition" — confirmed 2026-08-19
    "EQR",     # Equity Residential (REIT): IBKR returns Error 200 "No security definition" — confirmed 2026-08-19
})

# A handful of short LSE codes are registered on IBKR with a literal trailing
# "." as part of the symbol itself (disambiguation against common English
# abbreviations: AV./BA./BP./JD./NG./RR./SN./UU.) — yfinance's own ".L"
# country suffix hides this, so the generic strip-and-pass-through below
# resolves to a bare symbol IBKR has no listing for ("No security definition
# has been found", verified live 2026-08-16). Confirmed via reqContractDetails
# against the live paper Gateway; all eight still GBP as expected.
_LSE_DOT_SYMBOLS = {"AV", "BA", "BP", "JD", "NG", "RR", "SN", "UU"}

# LSE-listed ETFs use exchange "LSEETF" on IBKR, not "LSE" (which is for equities).
# Stock("ISF","LSE","GBP") returns Error 200; Stock("ISF","LSEETF","GBP") resolves correctly.
# Verified 2026-09-14 on paper Gateway port 4002: conIds 13444223/104145585/68490081.
_LSEETF_SYMBOLS: frozenset[str] = frozenset({"ISF", "XSTR", "IGLS", "ISXF", "EQGB"})  # EQGB = Invesco Nasdaq-100 GBP-hedged UCITS

# CSH2 (Amundi Smart Overnight Return UCITS ETF, GBP-hedged share class) is
# registered on IBKR as Stock("CSH2", "LSE", "GBP") — conId 196610715,
# tradingClass ETFS — not on LSEETF like the other LSE ETFs above. The
# IBIS/EUR line (conId 185829298) qualifies but this account has no market
# data permission for IBIS at all (Error 354 live/delayed, Error 162
# historical), so quotes come back NaN. Verified 2026-09-17 on paper Gateway
# port 4002: LSE/GBP returns delayed quotes and 1-min historical bars.
_LSE_EXPLICIT_EXCHANGE: dict[str, str] = {"CSH2": "LSE"}


def ibkr_contract_params(ticker: str) -> tuple[str, str, str]:
    """Map a yfinance ticker to (symbol, exchange, currency) for an IBKR Stock.

    ".L" suffix → LSE/GBP with the suffix stripped and share-class hyphen
    turned into IBKR's dot (plus the _LSE_DOT_SYMBOLS trailing-dot fixups
    above). LSE ETFs in _LSEETF_SYMBOLS use exchange "LSEETF" instead of "LSE".
    _LSE_EXPLICIT_EXCHANGE pins the exchange for LSE ETFs that IBKR lists
    outside LSEETF (CSH2 lives on plain LSE with tradingClass ETFS).
    Everything else is treated as a US equity on SMART/USD; US dual-class
    tickers use yfinance's hyphen (e.g. "BRK-B", "BF-B") where IBKR wants
    a space ("BRK B", "BF B").
    """
    if ticker.upper().endswith(".L"):
        base = ticker[:-2].replace("-", ".")
        if base.upper() in _LSE_DOT_SYMBOLS:
            base = base + "."
        if base.upper() in _LSE_EXPLICIT_EXCHANGE:
            exch = _LSE_EXPLICIT_EXCHANGE[base.upper()]
        else:
            exch = "LSEETF" if base.upper() in _LSEETF_SYMBOLS else "LSE"
        return base, exch, "GBP"
    return ticker.replace("-", " "), "SMART", "USD"


def ibkr_order_contract_kwargs(ticker: str) -> dict[str, str]:
    """Keyword args for the ib_async Stock used to PLACE ORDERS for ticker.

    Orders are always SMART-routed, with the listing venue carried as
    primaryExchange for disambiguation. The Gateway's API precautionary
    settings reject every direct-routed (non-SMART) order with Error 10311
    "This order will be directly routed to <exchange>" followed by Error 201
    "Order was discarded" — observed since 2026-09-10 for LSE and LSEETF alike
    (last direct-routed fill 2026-08-28). Verified 2026-09-17 on paper
    Gateway: SMART+primaryExchange fills for HSBA, ISF, EQGB and CSH2.

    Market/historical data keeps using ibkr_contract_params() (direct venue):
    reqHistoricalData on a SMART contract returns "HMDS query returned no
    data" for LSE ETFs such as CSH2.
    """
    symbol, exchange, currency = ibkr_contract_params(ticker)
    kwargs = {"symbol": symbol, "exchange": "SMART", "currency": currency}
    if exchange != "SMART":
        kwargs["primaryExchange"] = exchange
    return kwargs


def yfinance_ticker(symbol: str, currency: str) -> str:
    """Map an IBKR contract's (symbol, currency) back to a yfinance ticker.

    Inverse of ibkr_contract_params: GBP contracts are LSE-listed, so strip
    the _LSE_DOT_SYMBOLS trailing dot (if present), append ".L", and turn
    share-class dots back into yfinance's hyphen. USD contracts turn IBKR's
    dual-class space back into yfinance's hyphen.
    """
    if currency.upper() == "GBP":
        base = symbol[:-1] if symbol.upper().rstrip(".") in _LSE_DOT_SYMBOLS and symbol.endswith(".") else symbol
        return base.replace(".", "-") + ".L"
    return symbol.replace(" ", "-")


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
