"""Tests for check_pending_order_cancels() — the daemon-side resolution of
entry-order cancel requests that IBKR hasn't confirmed within the in-call poll.

Account TIF preset is GTC, so an unconfirmed cancel won't auto-expire on its
own; this is the non-blocking, once-per-cycle follow-up (see ibkr_adapter.py
check_pending_cancels() for the broker-side half)."""

from __future__ import annotations

import logging
from unittest import mock

import pytest


class TestCheckPendingOrderCancels:
    def test_broker_without_check_pending_cancels_is_a_noop(self):
        from Strategy_Auto_Trader.markov_cli.live_daemon import check_pending_order_cancels

        broker = object()  # no check_pending_cancels attribute at all
        run_recon = mock.Mock()

        check_pending_order_cancels(
            mock.Mock(), {}, broker, logging.getLogger(__name__), run_recon=run_recon,
        )

        run_recon.assert_not_called()

    def test_check_fn_exception_is_suppressed(self):
        from Strategy_Auto_Trader.markov_cli.live_daemon import check_pending_order_cancels

        broker = mock.Mock()
        broker.check_pending_cancels.side_effect = RuntimeError("boom")
        run_recon = mock.Mock()

        check_pending_order_cancels(
            mock.Mock(), {}, broker, logging.getLogger(__name__), run_recon=run_recon,
        )

        run_recon.assert_not_called()

    def test_cancelled_event_does_not_trigger_reconciliation(self):
        from Strategy_Auto_Trader.markov_cli.live_daemon import check_pending_order_cancels
        from Strategy_Auto_Trader.broker.types import PendingCancelEvent

        broker = mock.Mock()
        broker.check_pending_cancels.return_value = [
            PendingCancelEvent(ticker="AAPL", action="BUY", quantity=10, outcome="cancelled"),
        ]
        run_recon = mock.Mock()

        check_pending_order_cancels(
            mock.Mock(), {}, broker, logging.getLogger(__name__), run_recon=run_recon,
        )

        run_recon.assert_not_called()

    def test_filled_event_triggers_immediate_reconciliation(self):
        """A cancel-race fill is a real, previously-untracked position — must
        reconcile right away, not wait for the next nightly pass."""
        from Strategy_Auto_Trader.markov_cli.live_daemon import check_pending_order_cancels
        from Strategy_Auto_Trader.broker.types import PendingCancelEvent, FillResult

        portfolio = mock.Mock()
        daemon_state = {"halt_new_entries": False}
        broker = mock.Mock()
        fill = FillResult("AAPL", "BUY", 195.5, 10, "2026-08-17T13:04:30+00:00")
        broker.check_pending_cancels.return_value = [
            PendingCancelEvent(ticker="AAPL", action="BUY", quantity=10, outcome="filled", fill=fill),
        ]
        run_recon = mock.Mock()
        save_state = mock.Mock()

        check_pending_order_cancels(
            portfolio, daemon_state, broker, logging.getLogger(__name__),
            run_recon=run_recon, save_state=save_state,
        )

        run_recon.assert_called_once_with(
            portfolio, broker, daemon_state, mock.ANY, save_state=save_state,
        )

    def test_timeout_alert_sends_email_not_reconciliation(self, monkeypatch):
        from Strategy_Auto_Trader.markov_cli.live_daemon import check_pending_order_cancels
        from Strategy_Auto_Trader.broker.types import PendingCancelEvent

        broker = mock.Mock()
        broker.check_pending_cancels.return_value = [
            PendingCancelEvent(
                ticker="AAPL", action="BUY", quantity=10,
                outcome="timeout_alert", elapsed_minutes=31.0,
            ),
        ]
        run_recon = mock.Mock()
        alert = mock.Mock()
        monkeypatch.setattr(
            "Strategy_Auto_Trader.output.emailer.send_pending_cancel_timeout_alert", alert
        )

        check_pending_order_cancels(
            mock.Mock(), {}, broker, logging.getLogger(__name__), run_recon=run_recon,
        )

        run_recon.assert_not_called()
        alert.assert_called_once_with("AAPL", "BUY", 10, 31.0)
