"""Tests for broker/in_flight_marker.py."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from Strategy_Auto_Trader.broker.in_flight_marker import (
    write_marker,
    read_marker,
    clear_marker,
)


class TestInFlightMarker:
    def test_write_and_read_marker(self, tmp_path):
        marker_path = tmp_path / "marker.json"
        write_marker(marker_path, "SPY", "BUY", 10)
        result = read_marker(marker_path)

        assert result is not None
        assert result["ticker"] == "SPY"
        assert result["action"] == "BUY"
        assert result["quantity"] == 10
        assert "timestamp" in result

    def test_marker_timestamp_is_iso(self, tmp_path):
        marker_path = tmp_path / "marker.json"
        before = datetime.now(timezone.utc).isoformat()
        write_marker(marker_path, "AAPL", "SELL", 5)
        after = datetime.now(timezone.utc).isoformat()

        result = read_marker(marker_path)
        assert before <= result["timestamp"] <= after

    def test_read_marker_missing_file(self, tmp_path):
        marker_path = tmp_path / "nonexistent.json"
        result = read_marker(marker_path)
        assert result is None

    def test_read_marker_corrupt_json(self, tmp_path):
        marker_path = tmp_path / "corrupt.json"
        marker_path.write_text("{ invalid json")
        result = read_marker(marker_path)
        assert result is None

    def test_clear_marker(self, tmp_path):
        marker_path = tmp_path / "marker.json"
        write_marker(marker_path, "TSLA", "BUY", 3)
        assert marker_path.exists()

        clear_marker(marker_path)
        assert not marker_path.exists()

    def test_clear_marker_missing_file(self, tmp_path):
        marker_path = tmp_path / "nonexistent.json"
        clear_marker(marker_path)
        assert not marker_path.exists()

    def test_overwrite_marker(self, tmp_path):
        marker_path = tmp_path / "marker.json"
        write_marker(marker_path, "SPY", "BUY", 10)
        first_read = read_marker(marker_path)

        write_marker(marker_path, "QQQ", "SELL", 5)
        second_read = read_marker(marker_path)

        assert first_read["ticker"] == "SPY"
        assert second_read["ticker"] == "QQQ"
        assert first_read["timestamp"] != second_read["timestamp"]

    def test_marker_creates_parent_directory(self, tmp_path):
        nested_path = tmp_path / "deep" / "nested" / "marker.json"
        write_marker(nested_path, "XYZ", "BUY", 1)
        assert nested_path.exists()
        result = read_marker(nested_path)
        assert result["ticker"] == "XYZ"
