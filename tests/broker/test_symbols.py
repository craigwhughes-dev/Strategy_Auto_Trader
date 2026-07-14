"""Tests for broker/symbols.py — yfinance → IBKR mapping and pence conversion."""

from __future__ import annotations

import pytest

from Strategy_Auto_Trader.broker.symbols import (
    ibkr_contract_params,
    yfinance_ticker,
    sizing_price,
)


class TestIbkrContractParams:
    def test_lse_ticker_maps_to_lse_gbp(self):
        assert ibkr_contract_params("HSBA.L") == ("HSBA", "LSE", "GBP")

    def test_lse_share_class_hyphen_becomes_dot(self):
        assert ibkr_contract_params("BT-A.L") == ("BT.A", "LSE", "GBP")

    def test_lowercase_suffix_recognised(self):
        assert ibkr_contract_params("hsba.l") == ("hsba", "LSE", "GBP")

    def test_us_ticker_maps_to_smart_usd(self):
        assert ibkr_contract_params("SPY") == ("SPY", "SMART", "USD")

    def test_us_ticker_with_dot_class_not_treated_as_lse(self):
        assert ibkr_contract_params("BRK.B") == ("BRK.B", "SMART", "USD")

    def test_single_letter_lse_symbol(self):
        assert ibkr_contract_params("R.L") == ("R", "LSE", "GBP")


class TestYfinanceTicker:
    def test_lse_gbp_contract_maps_back_to_lse_ticker(self):
        assert yfinance_ticker("HSBA", "GBP") == "HSBA.L"

    def test_lse_share_class_dot_becomes_hyphen(self):
        assert yfinance_ticker("BT.A", "GBP") == "BT-A.L"

    def test_us_usd_contract_unchanged(self):
        assert yfinance_ticker("SPY", "USD") == "SPY"

    def test_us_usd_with_dot_class_preserved(self):
        assert yfinance_ticker("BRK.B", "USD") == "BRK.B"

    def test_single_letter_lse_symbol(self):
        assert yfinance_ticker("R", "GBP") == "R.L"

    def test_round_trip_hsba(self):
        symbol, exchange, currency = ibkr_contract_params("HSBA.L")
        assert yfinance_ticker(symbol, currency) == "HSBA.L"

    def test_round_trip_bt_a(self):
        symbol, exchange, currency = ibkr_contract_params("BT-A.L")
        assert yfinance_ticker(symbol, currency) == "BT-A.L"

    def test_round_trip_spy(self):
        symbol, exchange, currency = ibkr_contract_params("SPY")
        assert yfinance_ticker(symbol, currency) == "SPY"

    def test_round_trip_brk_b(self):
        symbol, exchange, currency = ibkr_contract_params("BRK.B")
        assert yfinance_ticker(symbol, currency) == "BRK.B"

    def test_cross_listed_collision_prevention(self):
        us_bp = yfinance_ticker("BP", "USD")
        uk_bp = yfinance_ticker("BP", "GBP")
        assert us_bp == "BP"
        assert uk_bp == "BP.L"
        assert us_bp != uk_bp


class TestSizingPrice:
    def test_lse_price_converted_pence_to_pounds(self):
        assert sizing_price("HSBA.L", 700.0) == pytest.approx(7.0)

    def test_us_price_unchanged(self):
        assert sizing_price("SPY", 500.0) == 500.0

    def test_zero_price(self):
        assert sizing_price("HSBA.L", 0.0) == 0.0

    def test_sub_pound_price(self):
        assert sizing_price("LLOY.L", 54.32) == pytest.approx(0.5432)

    def test_quantity_flow_lse(self):
        """£250 slot × 0.1 Kelly on a 700p (£7) stock buys 3 shares, not 1."""
        from pathlib import Path

        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager

        pm = PortfolioManager(10_000, 40, Path("nonexistent_state.json"))
        qty = pm.compute_quantity(0.1, sizing_price("HSBA.L", 700.0))
        assert qty == 3
