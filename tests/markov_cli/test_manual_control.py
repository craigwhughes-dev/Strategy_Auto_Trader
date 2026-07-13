"""Tests for manual_control.py — Windows CLI for pause/resume."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from Strategy_Auto_Trader.markov_cli.manual_control import _write_command, main


@pytest.fixture
def commands_dir(tmp_path):
    """Temporary commands directory."""
    cmds = tmp_path / "commands"
    (cmds / "pending").mkdir(parents=True, exist_ok=True)
    return cmds


def test_write_command_pause(commands_dir):
    """Write a PAUSE_BUYING command to pending directory."""
    cmd_id = _write_command("PAUSE_BUYING", commands_dir)

    assert cmd_id
    cmd_file = commands_dir / "pending" / f"{cmd_id}.json"
    assert cmd_file.exists()

    cmd_data = json.loads(cmd_file.read_text(encoding="utf-8"))
    assert cmd_data["Id"] == cmd_id
    assert cmd_data["Action"] == "PAUSE_BUYING"
    assert cmd_data["Ticker"] is None
    assert cmd_data["Status"] == "pending"
    assert cmd_data["Source"] == "windows-cli"
    assert "RequestedAtUtc" in cmd_data
    assert "ExpiresAtUtc" in cmd_data
    assert cmd_data["ExpiresAtUtc"].endswith("Z")


def test_write_command_resume(commands_dir):
    """Write a RESUME_BUYING command to pending directory."""
    cmd_id = _write_command("RESUME_BUYING", commands_dir)

    assert cmd_id
    cmd_file = commands_dir / "pending" / f"{cmd_id}.json"
    assert cmd_file.exists()

    cmd_data = json.loads(cmd_file.read_text(encoding="utf-8"))
    assert cmd_data["Id"] == cmd_id
    assert cmd_data["Action"] == "RESUME_BUYING"
    assert cmd_data["Ticker"] is None
    assert cmd_data["Status"] == "pending"


def test_write_command_creates_pending_dir(tmp_path):
    """_write_command creates pending directory if it doesn't exist."""
    commands_dir = tmp_path / "commands"
    # Don't create the directory
    cmd_id = _write_command("PAUSE_BUYING", commands_dir)

    assert (commands_dir / "pending").exists()
    assert (commands_dir / "pending" / f"{cmd_id}.json").exists()


def test_write_command_expiry_hours(commands_dir):
    """PAUSE_BUYING command expires in 4 hours."""
    from datetime import datetime, timedelta, timezone

    cmd_id = _write_command("PAUSE_BUYING", commands_dir)
    cmd_file = commands_dir / "pending" / f"{cmd_id}.json"
    cmd_data = json.loads(cmd_file.read_text(encoding="utf-8"))

    requested = datetime.fromisoformat(cmd_data["RequestedAtUtc"].replace("Z", "+00:00"))
    expires = datetime.fromisoformat(cmd_data["ExpiresAtUtc"].replace("Z", "+00:00"))
    delta = expires - requested

    # Should be approximately 4 hours (allow 1 second leeway)
    assert 4 * 3600 - 1 <= delta.total_seconds() <= 4 * 3600 + 1


def test_main_pause_command(commands_dir):
    """CLI pause command creates a PAUSE_BUYING command."""
    with mock.patch("Strategy_Auto_Trader.markov_cli.manual_control.COMMANDS_DIR", commands_dir):
        exit_code = main(["pause"])

    assert exit_code == 0

    # Check that a command was created
    pending_files = list((commands_dir / "pending").glob("*.json"))
    assert len(pending_files) == 1

    cmd_data = json.loads(pending_files[0].read_text(encoding="utf-8"))
    assert cmd_data["Action"] == "PAUSE_BUYING"


def test_main_unpause_command(commands_dir):
    """CLI unpause command creates a RESUME_BUYING command."""
    with mock.patch("Strategy_Auto_Trader.markov_cli.manual_control.COMMANDS_DIR", commands_dir):
        exit_code = main(["unpause"])

    assert exit_code == 0

    pending_files = list((commands_dir / "pending").glob("*.json"))
    assert len(pending_files) == 1

    cmd_data = json.loads(pending_files[0].read_text(encoding="utf-8"))
    assert cmd_data["Action"] == "RESUME_BUYING"


def test_main_requires_subcommand():
    """CLI requires a subcommand (pause or unpause)."""
    with pytest.raises(SystemExit):
        main([])


def test_main_invalid_subcommand():
    """CLI rejects invalid subcommand."""
    with pytest.raises(SystemExit):
        main(["invalid"])


def test_main_pause_prints_message(commands_dir, capsys):
    """CLI pause prints confirmation message."""
    with mock.patch("Strategy_Auto_Trader.markov_cli.manual_control.COMMANDS_DIR", commands_dir):
        exit_code = main(["pause"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "pause command queued" in captured.out
    assert "Takes effect within ~60s" in captured.out


def test_main_unpause_prints_message(commands_dir, capsys):
    """CLI unpause prints confirmation message."""
    with mock.patch("Strategy_Auto_Trader.markov_cli.manual_control.COMMANDS_DIR", commands_dir):
        exit_code = main(["unpause"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "unpause command queued" in captured.out
