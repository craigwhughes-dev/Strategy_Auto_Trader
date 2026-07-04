"""Tests for core/self_check.py — startup self-checks must catch broken
environments (the silent-hmmlearn-breakage class of failure) loudly."""

from __future__ import annotations

from unittest import mock

import pytest

from Strategy_Auto_Trader.core import self_check
from Strategy_Auto_Trader.core.self_check import (
    SelfCheckError,
    check_broker_module,
    check_data_stack,
    check_hmm_fit,
    run_startup_checks,
)


class TestIndividualChecks:

    def test_data_stack_passes(self):
        msg = check_data_stack()
        assert "numpy" in msg and "pandas" in msg and "yfinance" in msg

    def test_hmm_fit_passes(self):
        pytest.importorskip("hmmlearn")
        msg = check_hmm_fit()
        assert "hmmlearn" in msg and "OK" in msg

    def test_hmm_fit_raises_when_hmmlearn_broken(self):
        # None in sys.modules makes `import hmmlearn` raise ImportError,
        # mimicking the gutted-install failure mode.
        with mock.patch.dict(
                "sys.modules", {"hmmlearn": None, "hmmlearn.hmm": None}):
            with pytest.raises(ImportError):
                check_hmm_fit()

    def test_hmm_fit_raises_when_fit_produces_no_model(self):
        pytest.importorskip("hmmlearn")
        from Strategy_Auto_Trader.quant_hmm import quant_engine
        with mock.patch.object(quant_engine, "fit_hmm_expanding",
                               return_value=None):
            with pytest.raises(SelfCheckError, match="no model"):
                check_hmm_fit()

    def test_broker_module_passes(self):
        pytest.importorskip("ib_insync")
        msg = check_broker_module()
        assert "ib_insync" in msg

    def test_broker_module_raises_when_missing(self):
        with mock.patch.dict("sys.modules", {"ib_insync": None}):
            with pytest.raises(ImportError):
                check_broker_module()


class TestBrokerConnectivity:

    def _adapter_mock(self):
        adapter = mock.Mock()
        adapter.managed_accounts.return_value = ["DU123456"]
        return adapter

    def test_connect_and_query_succeeds(self):
        adapter = self._adapter_mock()
        with mock.patch("Strategy_Auto_Trader.broker.ibkr_adapter.IBKRAdapter",
                        return_value=adapter):
            msg = self_check.check_broker_connectivity(port=7497)
        assert "7497" in msg and "DU123456" in msg
        adapter.connect.assert_called_once()
        adapter.disconnect.assert_called_once()

    def test_transient_connect_failure_is_retried(self):
        adapter = self._adapter_mock()
        adapter.connect.side_effect = [TimeoutError("handshake"), None]
        with mock.patch("Strategy_Auto_Trader.broker.ibkr_adapter.IBKRAdapter",
                        return_value=adapter):
            msg = self_check.check_broker_connectivity()
        assert "DU123456" in msg
        assert adapter.connect.call_count == 2

    def test_persistent_connect_failure_raises(self):
        adapter = self._adapter_mock()
        adapter.connect.side_effect = ConnectionRefusedError("no TWS")
        with mock.patch("Strategy_Auto_Trader.broker.ibkr_adapter.IBKRAdapter",
                        return_value=adapter):
            with pytest.raises(SelfCheckError, match="cannot connect"):
                self_check.check_broker_connectivity(port=4002)
        assert adapter.connect.call_count == 2   # original + one retry

    def test_no_accounts_raises_and_disconnects(self):
        adapter = self._adapter_mock()
        adapter.managed_accounts.return_value = []
        with mock.patch("Strategy_Auto_Trader.broker.ibkr_adapter.IBKRAdapter",
                        return_value=adapter):
            with pytest.raises(SelfCheckError, match="no accounts"):
                self_check.check_broker_connectivity()
        adapter.disconnect.assert_called()


class TestRunStartupChecks:

    def test_all_pass_returns_messages(self):
        pytest.importorskip("hmmlearn")
        pytest.importorskip("ib_insync")
        with mock.patch.object(self_check, "check_broker_connectivity",
                               return_value="connected (stub)"):
            passed = run_startup_checks(require_broker=True)
        assert len(passed) == 4
        assert any(p.startswith("data stack:") for p in passed)
        assert any(p.startswith("HMM fit:") for p in passed)
        assert any(p.startswith("broker module:") for p in passed)
        assert any(p.startswith("broker connectivity:") for p in passed)

    def test_broker_connectivity_failure_fails_startup(self):
        pytest.importorskip("hmmlearn")
        pytest.importorskip("ib_insync")
        with mock.patch.object(
                self_check, "check_broker_connectivity",
                side_effect=SelfCheckError("cannot connect to TWS")):
            with pytest.raises(SelfCheckError, match="broker connectivity"):
                run_startup_checks(require_broker=True)

    def test_hmm_not_required_skips_hmm_check(self):
        with mock.patch.object(self_check, "check_hmm_fit",
                               side_effect=AssertionError("must not run")):
            passed = run_startup_checks(require_hmm=False)
        assert len(passed) == 1

    def test_broker_not_required_by_default(self):
        pytest.importorskip("hmmlearn")
        with mock.patch.dict("sys.modules", {"ib_insync": None}):
            passed = run_startup_checks()   # must not touch ib_insync
        assert len(passed) == 2

    def test_failure_raises_selfcheckerror_naming_the_check(self):
        pytest.importorskip("hmmlearn")
        with mock.patch.dict(
                "sys.modules", {"hmmlearn": None, "hmmlearn.hmm": None}):
            with pytest.raises(SelfCheckError, match="HMM fit"):
                run_startup_checks()

    def test_all_failures_collected_not_just_first(self):
        with mock.patch.object(self_check, "check_data_stack",
                               side_effect=RuntimeError("boom-data")), \
             mock.patch.object(self_check, "check_hmm_fit",
                               side_effect=RuntimeError("boom-hmm")):
            with pytest.raises(SelfCheckError) as exc_info:
                run_startup_checks()
        assert "boom-data" in str(exc_info.value)
        assert "boom-hmm" in str(exc_info.value)

    def test_logger_receives_pass_and_fail_lines(self):
        pytest.importorskip("hmmlearn")
        logger = mock.Mock()
        with mock.patch.object(self_check, "check_hmm_fit",
                               side_effect=RuntimeError("boom")):
            with pytest.raises(SelfCheckError):
                run_startup_checks(logger=logger)
        assert logger.info.called       # data stack passed
        assert logger.error.called      # hmm failed
