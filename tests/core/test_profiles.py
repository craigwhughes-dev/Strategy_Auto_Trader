"""Tests for per-profile filesystem isolation.

The default profile MUST keep the historical paths: an existing deployment's
ledger, lock and logs live there, and silently moving them would orphan a
running daemon's state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Strategy_Auto_Trader.core.profiles import (
    DEFAULT_PROFILE,
    PROFILE_ENV_VAR,
    ProfileError,
    active_profile_name,
    list_profile_state_dirs,
    resolve_profile,
    validate_profile_name,
)


class TestValidateProfileName:
    @pytest.mark.parametrize("name", ["live", "paper2", "a", "A_b-c", "x" * 32])
    def test_accepts_safe_names(self, name):
        assert validate_profile_name(name) == name

    @pytest.mark.parametrize("name", ["", "x" * 33, "has space", "dot.name",
                                      "../escape", "slash/name", "semi;colon"])
    def test_rejects_unsafe_names(self, name):
        with pytest.raises(ProfileError):
            validate_profile_name(name)

    def test_path_traversal_is_rejected(self):
        """Names become directory names, so traversal must never resolve."""
        with pytest.raises(ProfileError):
            validate_profile_name("..")


class TestActiveProfileName:
    def test_defaults_when_unset(self, monkeypatch):
        monkeypatch.delenv(PROFILE_ENV_VAR, raising=False)
        assert active_profile_name() == DEFAULT_PROFILE

    def test_reads_environment(self, monkeypatch):
        monkeypatch.setenv(PROFILE_ENV_VAR, "paper2")
        assert active_profile_name() == "paper2"

    def test_explicit_argument_wins_over_environment(self, monkeypatch):
        monkeypatch.setenv(PROFILE_ENV_VAR, "paper2")
        assert active_profile_name("live") == "live"

    def test_blank_environment_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(PROFILE_ENV_VAR, "   ")
        assert active_profile_name() == DEFAULT_PROFILE

    def test_invalid_environment_value_raises(self, monkeypatch):
        monkeypatch.setenv(PROFILE_ENV_VAR, "../evil")
        with pytest.raises(ProfileError):
            active_profile_name()


class TestResolveProfileDefault:
    def test_default_keeps_historical_paths(self, tmp_path, monkeypatch):
        monkeypatch.delenv(PROFILE_ENV_VAR, raising=False)
        p = resolve_profile(root=tmp_path)
        assert p.name == DEFAULT_PROFILE
        assert p.is_default is True
        assert p.config_dir == tmp_path / "config"
        assert p.state_dir == tmp_path / "state"
        assert p.logs_dir == tmp_path / "logs"
        assert p.data_dir == tmp_path / "data"
        assert p.config_path == tmp_path / "config" / "overnight_strategy.json"
        assert p.lock_path == tmp_path / "state" / "daemon.lock"
        assert p.pid_path == tmp_path / "state" / "daemon.pid"
        assert p.commands_dir == tmp_path / "state" / "commands"
        assert p.execution_state_path == tmp_path / "state" / "execution_state.json"


class TestResolveProfileNamed:
    def test_named_profile_nests_under_profiles(self, tmp_path):
        p = resolve_profile("live", root=tmp_path)
        assert p.is_default is False
        assert p.config_dir == tmp_path / "config" / "profiles" / "live"
        assert p.state_dir == tmp_path / "state" / "profiles" / "live"
        assert p.logs_dir == tmp_path / "logs" / "profiles" / "live"

    def test_named_profile_shares_the_data_cache(self, tmp_path):
        """Market data is expensive and profile-independent."""
        assert resolve_profile("live", root=tmp_path).data_dir == tmp_path / "data"

    def test_two_profiles_share_no_state_path(self, tmp_path):
        a = resolve_profile("live", root=tmp_path)
        b = resolve_profile("paper", root=tmp_path)
        for attr in ("state_dir", "logs_dir", "config_dir", "lock_path",
                     "pid_path", "commands_dir", "execution_state_path"):
            assert getattr(a, attr) != getattr(b, attr), attr

    def test_named_profile_differs_from_default_state(self, tmp_path):
        assert (resolve_profile("live", root=tmp_path).state_dir
                != resolve_profile(root=tmp_path).state_dir)

    def test_profile_paths_are_frozen(self, tmp_path):
        p = resolve_profile("live", root=tmp_path)
        with pytest.raises(Exception):
            p.name = "other"

    def test_invalid_name_raises(self, tmp_path):
        with pytest.raises(ProfileError):
            resolve_profile("../escape", root=tmp_path)


class TestListProfileStateDirs:
    def test_empty_when_nothing_on_disk(self, tmp_path):
        assert list_profile_state_dirs(tmp_path) == {}

    def test_finds_default_state_dir(self, tmp_path):
        (tmp_path / "state").mkdir()
        assert list_profile_state_dirs(tmp_path) == {DEFAULT_PROFILE: tmp_path / "state"}

    def test_finds_named_profiles(self, tmp_path):
        (tmp_path / "state" / "profiles" / "live").mkdir(parents=True)
        (tmp_path / "state" / "profiles" / "paper").mkdir(parents=True)
        found = list_profile_state_dirs(tmp_path)
        assert set(found) == {DEFAULT_PROFILE, "live", "paper"}
        assert found["live"] == tmp_path / "state" / "profiles" / "live"

    def test_ignores_files_and_illegal_names(self, tmp_path):
        profiles = tmp_path / "state" / "profiles"
        profiles.mkdir(parents=True)
        (profiles / "live").mkdir()
        (profiles / "not a profile").mkdir()
        (profiles / "stray.txt").write_text("x", encoding="utf-8")
        assert set(list_profile_state_dirs(tmp_path)) == {DEFAULT_PROFILE, "live"}


class TestDaemonUsesProfilePaths:
    def test_daemon_constants_come_from_the_profile(self):
        from Strategy_Auto_Trader.markov_cli import live_daemon
        assert live_daemon.STATE_DIR == live_daemon.PROFILE.state_dir
        assert live_daemon.CONFIG_DIR == live_daemon.PROFILE.config_dir
        assert live_daemon.LOGS_DIR == live_daemon.PROFILE.logs_dir

    def test_manual_control_targets_the_same_commands_dir(self):
        from Strategy_Auto_Trader.markov_cli import live_daemon, manual_control
        assert manual_control.COMMANDS_DIR == live_daemon.PROFILE.commands_dir
