"""Tests for cross-profile broker collision detection.

Copying a profile config and forgetting to change `broker.client_id` or
`expected_account` is the expected mistake; both silently break the other
daemon (shared client id disconnects it, shared account double-trades it), so
they are caught before any order is placed.
"""

from __future__ import annotations

from unittest import mock

import pytest

from Strategy_Auto_Trader.markov_cli import live_daemon


def _write_pid(state_dir, pid, client_id=None, account=None, started="2026-09-29T09:00:00"):
    state_dir.mkdir(parents=True, exist_ok=True)
    fields = [str(pid), started]
    if client_id is not None or account is not None:
        fields.append("" if client_id is None else str(client_id))
        fields.append(account or "")
    (state_dir / "daemon.pid").write_text("|".join(fields) + "\n", encoding="utf-8")


class TestReadPidRecord:
    def test_missing_file_returns_none(self, tmp_path):
        assert live_daemon._read_pid_record(tmp_path / "daemon.pid") is None

    def test_legacy_two_field_file_parses(self, tmp_path):
        _write_pid(tmp_path, 4242)
        rec = live_daemon._read_pid_record(tmp_path / "daemon.pid")
        assert rec["pid"] == 4242
        assert rec["client_id"] is None
        assert rec["account"] is None

    def test_full_record_parses(self, tmp_path):
        _write_pid(tmp_path, 7, client_id=3, account="DU111")
        rec = live_daemon._read_pid_record(tmp_path / "daemon.pid")
        assert (rec["pid"], rec["client_id"], rec["account"]) == (7, 3, "DU111")

    def test_blank_optional_fields_become_none(self, tmp_path):
        (tmp_path / "daemon.pid").write_text("9|t||\n", encoding="utf-8")
        rec = live_daemon._read_pid_record(tmp_path / "daemon.pid")
        assert rec["client_id"] is None and rec["account"] is None

    def test_garbage_pid_returns_none(self, tmp_path):
        (tmp_path / "daemon.pid").write_text("not-a-pid|t\n", encoding="utf-8")
        assert live_daemon._read_pid_record(tmp_path / "daemon.pid") is None

    def test_non_numeric_client_id_is_ignored_not_fatal(self, tmp_path):
        (tmp_path / "daemon.pid").write_text("9|t|abc|DU111\n", encoding="utf-8")
        rec = live_daemon._read_pid_record(tmp_path / "daemon.pid")
        assert rec["client_id"] is None
        assert rec["account"] == "DU111"


class TestCheckProfileCollisions:
    @pytest.fixture(autouse=True)
    def _alive(self, monkeypatch):
        """Treat every recorded pid as a running daemon unless a test says otherwise."""
        monkeypatch.setattr(live_daemon, "_pid_is_live_daemon", lambda pid: True)

    def test_no_siblings_is_clear(self, tmp_path):
        assert live_daemon.check_profile_collisions(
            1, "DU111", mock.Mock(), state_dirs={}) == []

    def test_own_profile_is_never_a_conflict(self, tmp_path):
        _write_pid(tmp_path, 10, client_id=1, account="DU111")
        conflicts = live_daemon.check_profile_collisions(
            1, "DU111", mock.Mock(),
            state_dirs={live_daemon.PROFILE.name: tmp_path})
        assert conflicts == []

    def test_shared_client_id_is_reported(self, tmp_path):
        _write_pid(tmp_path, 10, client_id=1, account="DU999")
        conflicts = live_daemon.check_profile_collisions(
            1, "DU111", mock.Mock(), state_dirs={"other": tmp_path})
        assert len(conflicts) == 1
        assert "client_id 1" in conflicts[0]

    def test_shared_account_is_reported(self, tmp_path):
        _write_pid(tmp_path, 10, client_id=5, account="DU111")
        conflicts = live_daemon.check_profile_collisions(
            1, "DU111", mock.Mock(), state_dirs={"other": tmp_path})
        assert len(conflicts) == 1
        assert "double-trade" in conflicts[0]

    def test_both_collisions_reported_together(self, tmp_path):
        _write_pid(tmp_path, 10, client_id=1, account="DU111")
        conflicts = live_daemon.check_profile_collisions(
            1, "DU111", mock.Mock(), state_dirs={"other": tmp_path})
        assert len(conflicts) == 2

    def test_distinct_settings_are_clear(self, tmp_path):
        _write_pid(tmp_path, 10, client_id=5, account="U7654321")
        assert live_daemon.check_profile_collisions(
            1, "DU111", mock.Mock(), state_dirs={"other": tmp_path}) == []

    def test_dead_sibling_is_not_a_conflict(self, tmp_path, monkeypatch):
        """A stale pid file must not block a restart."""
        monkeypatch.setattr(live_daemon, "_pid_is_live_daemon", lambda pid: False)
        _write_pid(tmp_path, 10, client_id=1, account="DU111")
        assert live_daemon.check_profile_collisions(
            1, "DU111", mock.Mock(), state_dirs={"other": tmp_path}) == []

    def test_legacy_sibling_without_identity_is_not_a_conflict(self, tmp_path):
        _write_pid(tmp_path, 10)
        assert live_daemon.check_profile_collisions(
            1, "DU111", mock.Mock(), state_dirs={"other": tmp_path}) == []

    def test_no_account_configured_only_checks_client_id(self, tmp_path):
        _write_pid(tmp_path, 10, client_id=5, account="DU111")
        assert live_daemon.check_profile_collisions(
            1, None, mock.Mock(), state_dirs={"other": tmp_path}) == []

    def test_conflicts_are_logged_critical(self, tmp_path):
        _write_pid(tmp_path, 10, client_id=1, account="DU999")
        logger = mock.Mock()
        live_daemon.check_profile_collisions(
            1, "DU111", logger, state_dirs={"other": tmp_path})
        assert logger.critical.called


class TestPidIsLiveDaemon:
    def test_missing_psutil_reports_not_live(self, monkeypatch):
        monkeypatch.setattr(live_daemon, "psutil", None)
        assert live_daemon._pid_is_live_daemon(1) is False

    def test_dead_pid_reports_not_live(self, monkeypatch):
        import types

        class NoSuchProcess(Exception):
            pass

        def _raise(pid):
            raise NoSuchProcess()

        monkeypatch.setattr(live_daemon, "psutil",
                            types.SimpleNamespace(Process=_raise,
                                                  NoSuchProcess=NoSuchProcess))
        assert live_daemon._pid_is_live_daemon(999999) is False

    def test_daemon_in_this_checkout_reports_live(self, monkeypatch):
        import types
        proc = mock.Mock()
        proc.cmdline.return_value = ["python", "-m",
                                     "Strategy_Auto_Trader.markov_cli.live_daemon"]
        proc.exe.return_value = None
        proc.cwd.return_value = str(live_daemon.ROOT)
        monkeypatch.setattr(live_daemon, "psutil",
                            types.SimpleNamespace(Process=lambda pid: proc))
        assert live_daemon._pid_is_live_daemon(123) is True

    def test_non_daemon_process_reports_not_live(self, monkeypatch):
        import types
        proc = mock.Mock()
        proc.cmdline.return_value = ["python", "-m", "pytest"]
        proc.exe.return_value = None
        proc.cwd.return_value = str(live_daemon.ROOT)
        monkeypatch.setattr(live_daemon, "psutil",
                            types.SimpleNamespace(Process=lambda pid: proc))
        assert live_daemon._pid_is_live_daemon(123) is False
