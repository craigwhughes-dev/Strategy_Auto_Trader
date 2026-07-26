"""IBKR cash interest rates — tiered by balance and currency.

Interest accrues daily on uninvested cash (capital_pot - deployed_value).
Rates are annual percentages applied to average daily balance.

IBKR Tiered plan rates (from 2026-07 pricing):
  GBP: Tier 1 (0-10k): 4.0%, Tier 2 (10k-100k): 4.5%, Tier 3 (100k+): 5.0%
  USD: Tier 1 (0-25k): 4.2%, Tier 2 (25k-100k): 4.7%, Tier 3 (100k+): 5.2%

Usage:
    interest_model = IbkrTieredInterest("GBP")
    annual_rate = interest_model.annual_rate(balance=5000)  # 4.0%
    daily_accrual = interest_model.daily_accrual(balance=5000)  # ~0.0005479%
"""

from __future__ import annotations


_GBP_TIERS = [
    (10_001, 0.040),      # 0-10k @ 4.0%
    (100_001, 0.045),     # 10k-100k @ 4.5%
    (float("inf"), 0.050),  # 100k+ @ 5.0%
]

_USD_TIERS = [
    (25_001, 0.042),      # 0-25k @ 4.2%
    (100_001, 0.047),     # 25k-100k @ 4.7%
    (float("inf"), 0.052),  # 100k+ @ 5.2%
]

_DAYS_PER_YEAR = 365


class IbkrTieredInterest:
    """IBKR tiered interest rate lookup by balance and currency."""

    def __init__(self, currency: str = "GBP") -> None:
        """Initialize with currency. Supports 'GBP' or 'USD'. Defaults to GBP if empty."""
        c = (currency or "GBP").upper()
        if c == "GBP":
            self._tiers = _GBP_TIERS
        elif c == "USD":
            self._tiers = _USD_TIERS
        else:
            raise ValueError(f"Unknown currency '{c}'. Choices: GBP, USD")
        self._currency = c

    def annual_rate(self, balance: float) -> float:
        """Annual interest rate (0.04 = 4%) for a given cash balance."""
        balance = max(0.0, float(balance))
        for threshold, rate in self._tiers:
            if balance < threshold:
                return rate
        return self._tiers[-1][1]

    def daily_accrual(self, balance: float) -> float:
        """Daily interest accrual in currency units.

        balance: cash available (GBP or USD)
        returns: interest earned in one day at current annual rate
        """
        balance = max(0.0, float(balance))
        rate = self.annual_rate(balance)
        return balance * rate / _DAYS_PER_YEAR

    def daily_interest_pct(self, balance: float) -> float:
        """Daily interest rate as a fraction (not percentage).

        Useful for compounding or daily P&L calculations.
        """
        rate = self.annual_rate(balance)
        return rate / _DAYS_PER_YEAR
