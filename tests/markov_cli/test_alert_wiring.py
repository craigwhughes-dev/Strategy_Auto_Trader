"""End-to-end wiring of failure alerts into the SMTP sender.

Existing reconciliation tests inject their own `send_alert`, which proves the
call site fires but not that the DEFAULT path reaches SMTP. A silently broken
default is the failure mode that matters: the alert is the only way a halt
becomes visible when nobody is watching the logs.
"""

from __future__ import annotations

from unittest import mock

import pytest

from Strategy_Auto_Trader.markov_cli import live_daemon
from Strategy_Auto_Trader.output import emailer


class _Broker:
    def __init__(self, positions):
        self._positions = positions

    def is_connected(self):
        return True

    def get_open_positions(self):
        return self._positions


class _Portfolio:
    def __init__(self, positions):
        self.positions = positions

    def save(self):
        pass


@pytest.fixture
def sent(monkeypatch):
    """Capture (subject, html) at the SMTP boundary without sending anything."""
    calls = []
    monkeypatch.setattr(emailer, "_send",
                        lambda subject, html, to=None: calls.append((subject, html)))
    return calls


class TestReconciliationAlertReachesSmtp:
    def _run(self, state=None):
        portfolio = _Portfolio({"SHEL.L": {"quantity": 10}})
        broker = _Broker({})  # broker holds nothing -> mismatch
        return live_daemon.run_reconciliation(
            portfolio, broker, state if state is not None else {}, mock.Mock(),
            save_state=lambda s: None,
        )

    def test_mismatch_reaches_the_smtp_sender(self, sent):
        assert self._run() == "mismatch"
        assert len(sent) == 1

    def test_subject_names_the_halt(self, sent):
        self._run()
        subject = sent[0][0]
        assert "RECONCILIATION MISMATCH" in subject
        assert "halted" in subject

    def test_body_lists_the_discrepancy(self, sent):
        self._run()
        assert "SHEL.L" in sent[0][1]

    def test_halt_is_set_even_if_the_email_raises(self, monkeypatch):
        """A dead mail server must not cost us the trading halt."""
        monkeypatch.setattr(emailer, "_send",
                            mock.Mock(side_effect=RuntimeError("smtp down")))
        state = {}
        assert self._run(state) == "mismatch"
        assert state["halt_new_entries"] is True

    def test_repeat_identical_mismatch_is_suppressed(self, sent):
        state = {}
        self._run(state)
        self._run(state)
        assert len(sent) == 1, "unchanged mismatch should not re-alert"


class TestSmtpCredentialHandling:
    def test_missing_credentials_raise_a_descriptive_error(self, monkeypatch):
        monkeypatch.delenv("SMTP_USER", raising=False)
        monkeypatch.delenv("SMTP_PASSWORD", raising=False)
        with pytest.raises(RuntimeError, match="SMTP_USER and SMTP_PASSWORD"):
            emailer._get_smtp_creds()

    def test_blank_password_is_treated_as_missing(self, monkeypatch):
        monkeypatch.setenv("SMTP_USER", "a@b.c")
        monkeypatch.setenv("SMTP_PASSWORD", "")
        with pytest.raises(RuntimeError):
            emailer._get_smtp_creds()

    def test_reconciliation_alert_builds_a_message_per_discrepancy(self, sent):
        emailer.send_reconciliation_alert(["A: 1 vs 0", "B: 2 vs 3"])
        subject, html = sent[0]
        assert "2 discrepancies" in subject
        assert "A: 1 vs 0" in html and "B: 2 vs 3" in html

    def test_single_discrepancy_subject_is_singular(self, sent):
        emailer.send_reconciliation_alert(["A: 1 vs 0"])
        assert "1 discrepancy" in sent[0][0]
