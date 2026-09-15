"""LSE ETF instruments used for cash parking.

All tickers use yfinance convention (.L suffix = LSE/GBP).
"""

from __future__ import annotations

# Tier-to-ticker mapping. All listed on LSE, settled in GBP.
PARKING_TICKERS: dict[str, str] = {
    "cash":     "XSTR.L",   # Xtrackers Sterling Cash (SONIA tracker ~5.2%)
    "gilts":    "IGLS.L",   # iShares UK Gilts 0-5yr
    "hy_bonds": "ISXF.L",   # iShares $ HY Corp Bond UCITS ETF GBP Hedged (~USD HY, FX-hedged)
}

EXCHANGE: str = "LSE"
CURRENCY: str = "GBP"
