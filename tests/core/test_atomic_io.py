"""Tests for atomic_io.py — atomic JSON and CSV writes."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from Strategy_Auto_Trader.core.atomic_io import atomic_write_csv, atomic_write_json


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


def test_atomic_write_json_retries_transient_permission_error(tmp_path, monkeypatch):
    """A transient PermissionError on os.replace (e.g. AV scanner or a reader
    briefly holding the file without FILE_SHARE_DELETE on Windows) is retried
    rather than immediately failing the write."""
    import os as os_module

    path = tmp_path / "test.json"
    original_replace = os_module.replace
    call_count = [0]

    def flaky_replace(src, dst):
        call_count[0] += 1
        if call_count[0] < 3:
            raise PermissionError("Simulated transient lock")
        return original_replace(src, dst)

    monkeypatch.setattr("os.replace", flaky_replace)
    monkeypatch.setattr("time.sleep", lambda _: None)

    atomic_write_json(path, {"data": "value"})

    assert call_count[0] == 3
    assert json.loads(path.read_text(encoding="utf-8")) == {"data": "value"}


def test_atomic_write_json_raises_after_exhausting_retries(tmp_path, monkeypatch):
    """A persistent PermissionError on os.replace still raises (and cleans up
    the temp file) after exhausting retries, rather than retrying forever."""

    def always_fails(src, dst):
        raise PermissionError("Simulated persistent lock")

    monkeypatch.setattr("os.replace", always_fails)
    monkeypatch.setattr("time.sleep", lambda _: None)

    path = tmp_path / "test.json"
    with pytest.raises(PermissionError, match="Simulated persistent lock"):
        atomic_write_json(path, {"data": "value"})

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


def _sample_df() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    return pd.DataFrame(
        {"Open": [1.0, 2.0, 3.0], "Close": [1.5, 2.5, 3.5]}, index=idx)


def test_atomic_write_csv_creates_file(tmp_path):
    path = tmp_path / "test.csv"
    df = _sample_df()

    atomic_write_csv(path, df)

    assert path.exists()
    loaded = pd.read_csv(path, index_col=0, parse_dates=True)
    assert list(loaded["Close"]) == [1.5, 2.5, 3.5]


def test_atomic_write_csv_replaces_existing(tmp_path):
    path = tmp_path / "test.csv"
    path.write_text("garbage,not,a,real,csv\n", encoding="utf-8")

    atomic_write_csv(path, _sample_df())

    loaded = pd.read_csv(path, index_col=0, parse_dates=True)
    assert list(loaded["Close"]) == [1.5, 2.5, 3.5]


def test_atomic_write_csv_creates_parent_dirs(tmp_path):
    path = tmp_path / "deep" / "nested" / "dir" / "test.csv"

    atomic_write_csv(path, _sample_df())

    assert path.exists()


def test_atomic_write_csv_target_untouched_on_replace_failure(tmp_path, monkeypatch):
    """A crash/failure during the atomic replace must never leave a torn or
    corrupted target file — callers like the IBKR hourly cache trust the
    last row's timestamp to compute the next gap-fill, so a partial write
    would either force a full re-fetch or leave a permanent hole."""
    path = tmp_path / "test.csv"
    original_df = _sample_df()
    atomic_write_csv(path, original_df)
    original_bytes = path.read_bytes()

    def always_fails(src, dst):
        raise OSError("Simulated replace failure")

    monkeypatch.setattr("os.replace", always_fails)

    new_df = pd.DataFrame(
        {"Open": [99.0], "Close": [99.0]},
        index=pd.date_range("2026-06-01", periods=1, tz="UTC"))
    with pytest.raises(OSError, match="Simulated replace failure"):
        atomic_write_csv(path, new_df)

    assert path.read_bytes() == original_bytes
    temp_files = list(tmp_path.glob(".tmp_*"))
    assert len(temp_files) == 0
