"""Tests for IBKR tiered interest rates."""

import pytest

from Strategy_Auto_Trader.plugins.interest import IbkrTieredInterest


class TestIbkrTieredInterest:
    """IBKR tiered interest rate lookups."""

    def test_gbp_tier1_rate(self):
        """GBP 0-10k tier: 4.0% annual."""
        interest = IbkrTieredInterest("GBP")
        assert interest.annual_rate(5000) == 0.04
        assert interest.annual_rate(9999) == 0.04

    def test_gbp_tier2_rate(self):
        """GBP 10k-100k tier: 4.5% annual."""
        interest = IbkrTieredInterest("GBP")
        assert interest.annual_rate(10001) == 0.045
        assert interest.annual_rate(50000) == 0.045
        assert interest.annual_rate(100000) == 0.045

    def test_gbp_tier3_rate(self):
        """GBP 100k+ tier: 5.0% annual."""
        interest = IbkrTieredInterest("GBP")
        assert interest.annual_rate(100001) == 0.05
        assert interest.annual_rate(1000000) == 0.05

    def test_usd_tier1_rate(self):
        """USD 0-25k tier: 4.2% annual."""
        interest = IbkrTieredInterest("USD")
        assert interest.annual_rate(10000) == 0.042
        assert interest.annual_rate(24999) == 0.042

    def test_usd_tier2_rate(self):
        """USD 25k-100k tier: 4.7% annual."""
        interest = IbkrTieredInterest("USD")
        assert interest.annual_rate(25001) == 0.047
        assert interest.annual_rate(50000) == 0.047
        assert interest.annual_rate(100000) == 0.047

    def test_usd_tier3_rate(self):
        """USD 100k+ tier: 5.2% annual."""
        interest = IbkrTieredInterest("USD")
        assert interest.annual_rate(100001) == 0.052
        assert interest.annual_rate(1000000) == 0.052

    def test_daily_accrual_gbp(self):
        """Daily accrual on £10k at 4.0% (tier 1)."""
        interest = IbkrTieredInterest("GBP")
        daily = interest.daily_accrual(10000)
        # 10000 * 0.04 / 365 ≈ 1.096
        assert 1.09 < daily < 1.11

    def test_daily_accrual_usd(self):
        """Daily accrual on $10k at 4.2%."""
        interest = IbkrTieredInterest("USD")
        daily = interest.daily_accrual(10000)
        # 10000 * 0.042 / 365 ≈ 1.151
        assert 1.14 < daily < 1.16

    def test_daily_interest_pct(self):
        """Daily interest as fraction (not percentage)."""
        interest = IbkrTieredInterest("GBP")
        daily_pct = interest.daily_interest_pct(10000)
        # 0.04 / 365 ≈ 0.0001096
        assert 0.0001 < daily_pct < 0.00011

    def test_zero_balance_no_accrual(self):
        """Zero balance = zero accrual."""
        interest = IbkrTieredInterest("GBP")
        assert interest.daily_accrual(0) == 0
        assert interest.annual_rate(0) == 0.04  # Still returns tier 1 rate

    def test_negative_balance_treated_as_zero(self):
        """Negative balance is floored at zero."""
        interest = IbkrTieredInterest("GBP")
        assert interest.daily_accrual(-1000) == 0

    def test_currency_case_insensitive(self):
        """Currency codes are case-insensitive."""
        gbp_upper = IbkrTieredInterest("GBP")
        gbp_lower = IbkrTieredInterest("gbp")
        gbp_mixed = IbkrTieredInterest("GbP")

        assert gbp_upper.annual_rate(5000) == gbp_lower.annual_rate(5000)
        assert gbp_lower.annual_rate(5000) == gbp_mixed.annual_rate(5000)

    def test_empty_currency_defaults_to_gbp(self):
        """Empty currency string defaults to GBP."""
        interest = IbkrTieredInterest("")
        assert interest.annual_rate(5000) == 0.04  # GBP tier 1

    def test_unknown_currency_raises(self):
        """Unknown currency raises ValueError."""
        with pytest.raises(ValueError, match="Unknown currency"):
            IbkrTieredInterest("JPY")
