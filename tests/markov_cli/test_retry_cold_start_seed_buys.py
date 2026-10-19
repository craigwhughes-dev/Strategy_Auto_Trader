"""retry_cold_start_seed_buys: retries an unheld tier asset's seed buy every few minutes
instead of waiting for the next hourly market cycle, throttled so it doesn't repeat the
API burst that can trigger a T212 429, and stops entirely once resolved.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from unittest import mock

from Strategy_Auto_Trader.markov_cli import live_daemon
from Strategy_Auto_Trader.markov_cli.live_daemon import retry_cold_start_seed_buys

LOG = logging.getLogger("test_retry_cold_start_seed_buys")


def _allocation_mgr(enabled):
    mgr = mock.Mock()
    mgr.enabled_assets.return_value = enabled
    return mgr


def test_skips_when_allocation_mgr_is_none(monkeypatch):
    save_calls = []
    monkeypatch.setattr(live_daemon, "save_daemon_state", lambda s: save_calls.append(s))
    broker = mock.Mock()

    retry_cold_start_seed_buys(None, broker, mock.Mock(), {}, {}, LOG)

    broker.get_open_positions.assert_not_called()
    assert save_calls == []


def test_skips_when_already_resolved(monkeypatch):
    monkeypatch.setattr(live_daemon, "save_daemon_state", lambda s: None)
    broker = mock.Mock()
    daemon_state = {"cold_start_seed_buys_resolved": True}

    retry_cold_start_seed_buys(_allocation_mgr(["EQGB.L"]), broker, mock.Mock(), {}, daemon_state, LOG)

    broker.get_open_positions.assert_not_called()


def test_skips_when_halted_or_paused(monkeypatch):
    monkeypatch.setattr(live_daemon, "save_daemon_state", lambda s: None)
    broker = mock.Mock()

    retry_cold_start_seed_buys(_allocation_mgr(["EQGB.L"]), broker, mock.Mock(), {},
                                {"halt_new_entries": True}, LOG)
    retry_cold_start_seed_buys(_allocation_mgr(["EQGB.L"]), broker, mock.Mock(), {},
                                {"paused_by_user": True}, LOG)

    broker.get_open_positions.assert_not_called()


def test_skips_within_cooldown_window(monkeypatch):
    monkeypatch.setattr(live_daemon, "save_daemon_state", lambda s: None)
    broker = mock.Mock()
    future = (datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat()
    daemon_state = {"next_seed_buy_retry_at": future}

    retry_cold_start_seed_buys(_allocation_mgr(["EQGB.L"]), broker, mock.Mock(), {}, daemon_state, LOG)

    broker.get_open_positions.assert_not_called()


def test_marks_resolved_and_stops_once_all_enabled_assets_are_held(monkeypatch):
    saved = {}
    monkeypatch.setattr(live_daemon, "save_daemon_state", lambda s: saved.update(s))
    broker = mock.Mock()
    broker.get_open_positions.return_value = {"EQGB.L": 0.01, "CSH2.L": 0.02}
    daemon_state = {}

    retry_cold_start_seed_buys(_allocation_mgr(["EQGB.L", "CSH2.L"]), broker, mock.Mock(), {}, daemon_state, LOG)

    assert daemon_state["cold_start_seed_buys_resolved"] is True
    assert "next_seed_buy_retry_at" not in daemon_state


def test_seeds_only_the_missing_tickers_and_sets_cooldown(monkeypatch):
    monkeypatch.setattr(live_daemon, "save_daemon_state", lambda s: None)
    seeded = []
    monkeypatch.setattr(live_daemon, "_seed_tier_price",
                         lambda ticker, *a, **k: seeded.append(ticker))
    broker = mock.Mock()
    broker.get_open_positions.return_value = {"EQGB.L": 0.01}  # CSH2.L missing
    daemon_state = {}

    retry_cold_start_seed_buys(_allocation_mgr(["EQGB.L", "CSH2.L"]), broker, mock.Mock(), {}, daemon_state, LOG)

    assert seeded == ["CSH2.L"]
    assert "next_seed_buy_retry_at" in daemon_state
    assert daemon_state.get("cold_start_seed_buys_resolved") is None


def test_get_open_positions_failure_does_not_raise(monkeypatch):
    monkeypatch.setattr(live_daemon, "save_daemon_state", lambda s: None)
    broker = mock.Mock()
    broker.get_open_positions.side_effect = RuntimeError("network error")
    daemon_state = {}

    retry_cold_start_seed_buys(_allocation_mgr(["EQGB.L"]), broker, mock.Mock(), {}, daemon_state, LOG)

    assert "next_seed_buy_retry_at" not in daemon_state
