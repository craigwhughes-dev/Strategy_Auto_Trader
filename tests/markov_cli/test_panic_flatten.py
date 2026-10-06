"""panic_flatten: broker selection follows broker.provider, and a flatten pauses buying in daemon state."""

from __future__ import annotations

import json

import pytest

from Strategy_Auto_Trader.broker.t212_adapter import T212Adapter
from Strategy_Auto_Trader.markov_cli import panic_flatten


class TestBuildBroker:
    def test_t212_provider_builds_t212_adapter(self, monkeypatch):
        monkeypatch.setenv("T212_API_KEY", "k")
        monkeypatch.setenv("T212_API_SECRET", "s")
        broker = panic_flatten.build_broker({"broker": {"provider": "t212", "t212": {"environment": "demo"}}})
        assert isinstance(broker, T212Adapter)

    def test_t212_without_credentials_refuses(self, monkeypatch):
        monkeypatch.delenv("T212_API_KEY", raising=False)
        monkeypatch.delenv("T212_API_SECRET", raising=False)
        with pytest.raises(SystemExit, match="T212_API_KEY"):
            panic_flatten.build_broker({"broker": {"provider": "t212"}})

    def test_t212_live_requires_real_money_flag(self, monkeypatch):
        monkeypatch.setenv("T212_API_KEY", "k")
        monkeypatch.setenv("T212_API_SECRET", "s")
        config = {"broker": {"provider": "t212", "t212": {"environment": "live"}}, "execution": {"real_money": False}}
        with pytest.raises(SystemExit, match="allow_live_account"):
            panic_flatten.build_broker(config)

    def test_ibkr_provider_still_requires_expected_account(self):
        with pytest.raises(SystemExit, match="expected_account"):
            panic_flatten.build_broker({"broker": {"provider": "ibkr"}})

    def test_missing_provider_defaults_to_ibkr(self):
        with pytest.raises(SystemExit, match="expected_account"):
            panic_flatten.build_broker({"broker": {}})


class TestPauseBuyingInDaemonState:
    def test_sets_paused_flag_and_keeps_other_state(self, tmp_path, monkeypatch):
        import Strategy_Auto_Trader.markov_cli.live_daemon as live_daemon
        monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)
        (tmp_path / "daemon_state.json").write_text(json.dumps({"last_overnight_date": "2026-10-05"}), encoding="utf-8")

        panic_flatten.pause_buying_in_daemon_state()

        state = json.loads((tmp_path / "daemon_state.json").read_text(encoding="utf-8"))
        assert state["paused_by_user"] is True
        assert state["last_overnight_date"] == "2026-10-05"

    def test_creates_state_when_none_exists(self, tmp_path, monkeypatch):
        import Strategy_Auto_Trader.markov_cli.live_daemon as live_daemon
        monkeypatch.setattr(live_daemon, "STATE_DIR", tmp_path)

        panic_flatten.pause_buying_in_daemon_state()

        assert json.loads((tmp_path / "daemon_state.json").read_text(encoding="utf-8"))["paused_by_user"] is True
