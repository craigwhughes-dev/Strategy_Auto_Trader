"""Tests for the IBKR account identity guard.

The guard is what stands between an edited port and unintended real-money
orders, so every rejection path is asserted explicitly.
"""

from __future__ import annotations

import sys
import types

import pytest

from Strategy_Auto_Trader.broker.ibkr_adapter import (
    AccountMismatchError,
    is_paper_account,
    verify_account,
)


class TestIsPaperAccount:
    @pytest.mark.parametrize("account", ["DU123456", "DUR166977", "du123456", " DU9 "])
    def test_du_prefix_is_paper(self, account):
        assert is_paper_account(account) is True

    def test_df_demo_prefix_is_paper(self):
        assert is_paper_account("DF123456") is True

    @pytest.mark.parametrize("account", ["U1234567", "F1234567", "I1234567", ""])
    def test_everything_else_is_treated_as_real_money(self, account):
        assert is_paper_account(account) is False


class TestVerifyAccount:
    def test_expected_account_present_is_selected(self):
        got = verify_account(["DU111", "DU222"], "DU222", allow_live_account=False)
        assert got == "DU222"

    def test_expected_account_absent_raises(self):
        with pytest.raises(AccountMismatchError, match="not in this session"):
            verify_account(["DU111"], "DU999", allow_live_account=False)

    def test_no_accounts_raises(self):
        with pytest.raises(AccountMismatchError, match="no managed accounts"):
            verify_account([], "DU111", allow_live_account=False)

    def test_single_account_inferred_when_none_expected(self):
        assert verify_account(["DU111"], None, allow_live_account=False) == "DU111"

    def test_multiple_accounts_without_expected_raises(self):
        with pytest.raises(AccountMismatchError, match="refusing to guess"):
            verify_account(["DU111", "DU222"], None, allow_live_account=False)

    def test_live_account_rejected_without_real_money(self):
        with pytest.raises(AccountMismatchError, match="refusing to trade real money"):
            verify_account(["U1234567"], "U1234567", allow_live_account=False)

    def test_live_account_allowed_with_real_money(self):
        assert verify_account(["U1234567"], "U1234567", allow_live_account=True) == "U1234567"

    def test_paper_account_allowed_when_real_money_enabled(self):
        # real_money is permission, not a requirement — paper must still work.
        assert verify_account(["DU111"], "DU111", allow_live_account=True) == "DU111"

    def test_live_account_rejected_even_when_sole_account(self):
        # The inference path must not become a way around the real-money gate.
        with pytest.raises(AccountMismatchError, match="refusing to trade real money"):
            verify_account(["U1234567"], None, allow_live_account=False)


class TestAdapterWiring:
    def test_adapter_stores_guard_settings(self):
        from Strategy_Auto_Trader.broker.ibkr_adapter import IBKRAdapter
        adapter = IBKRAdapter(expected_account="DU111", allow_live_account=True)
        assert adapter._expected_account == "DU111"
        assert adapter._allow_live_account is True

    def test_adapter_defaults_reject_live_accounts(self):
        from Strategy_Auto_Trader.broker.ibkr_adapter import IBKRAdapter
        adapter = IBKRAdapter()
        assert adapter._expected_account is None
        assert adapter._allow_live_account is False

    def test_connect_disconnects_on_mismatch(self, monkeypatch):
        """A rejected session must be dropped, not left open and unused."""
        from Strategy_Auto_Trader.broker import ibkr_adapter as mod

        disconnected = []

        class FakeIB:
            def connect(self, *a, **kw):
                return None

            def managedAccounts(self):
                return ["U7654321"]

            def disconnect(self):
                disconnected.append(True)

        monkeypatch.setitem(sys.modules, "ib_async", types.SimpleNamespace(IB=FakeIB))
        adapter = mod.IBKRAdapter(expected_account="DU111")
        with pytest.raises(AccountMismatchError):
            adapter.connect()
        assert disconnected == [True]
        assert adapter._ib is None
