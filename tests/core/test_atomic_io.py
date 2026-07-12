"""Tests for atomic_io.py — atomic JSON writes."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from Strategy_Auto_Trader.core.atomic_io import atomic_write_json


def test_atomic_write_json_creates_file(tmp_path):
    """atomic_write_json creates the file if it doesn't exist."""
    path = tmp_path / "test.json"
    data = {"key": "value", "number": 42}

    atomic_write_json(path, data)

    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == data


def test_atomic_write_json_replaces_existing(tmp_path):
    """atomic_write_json overwrites an existing file."""
    path = tmp_path / "test.json"
    old_data = {"old": True}
    new_data = {"new": True, "number": 99}

    path.write_text(json.dumps(old_data), encoding="utf-8")
    atomic_write_json(path, new_data)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == new_data
    assert "old" not in loaded


def test_atomic_write_json_creates_parent_dirs(tmp_path):
    """atomic_write_json creates parent directories if needed."""
    path = tmp_path / "deep" / "nested" / "dir" / "test.json"
    data = {"nested": "structure"}

    atomic_write_json(path, data)

    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == data


def test_atomic_write_json_pretty_prints(tmp_path):
    """atomic_write_json writes with 2-space indentation."""
    path = tmp_path / "test.json"
    data = {"a": 1, "b": {"c": 2}}

    atomic_write_json(path, data)

    text = path.read_text(encoding="utf-8")
    assert "  " in text  # Has indentation
    # Verify it's valid JSON still
    loaded = json.loads(text)
    assert loaded == data


def test_atomic_write_json_handles_complex_types(tmp_path):
    """atomic_write_json preserves various JSON-serializable types."""
    path = tmp_path / "test.json"
    data = {
        "string": "value",
        "number": 42,
        "float": 3.14,
        "bool": True,
        "null": None,
        "array": [1, 2, 3],
        "nested": {"inner": "value"},
    }

    atomic_write_json(path, data)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == data


def test_atomic_write_json_cleans_temp_on_failure(tmp_path, monkeypatch):
    """atomic_write_json cleans up temp file on write failure."""
    import os as os_module

    path = tmp_path / "test.json"
    original_replace = os_module.replace

    call_count = [0]

    def failing_replace(src, dst):
        call_count[0] += 1
        raise OSError("Simulated replace failure")

    monkeypatch.setattr("os.replace", failing_replace)

    with pytest.raises(OSError, match="Simulated replace failure"):
        atomic_write_json(path, {"data": "value"})

    # After failure, temp files should be cleaned up
    temp_files = list(tmp_path.glob(".tmp_*"))
    assert len(temp_files) == 0


def test_atomic_write_json_encoding_utf8(tmp_path):
    """atomic_write_json writes UTF-8 encoded JSON."""
    path = tmp_path / "test.json"
    data = {"unicode": "café", "emoji": "🚀"}

    atomic_write_json(path, data)

    # Read as UTF-8 and verify encoding is preserved through round-trip
    text = path.read_text(encoding="utf-8")
    loaded = json.loads(text)
    assert loaded["unicode"] == "café"
    assert loaded["emoji"] == "🚀"
