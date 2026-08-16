"""Tests for markov_cli.refresh_universe — before/after diff wrapper around
full_scan.build_sp_ftse_universe()."""

from __future__ import annotations

import json

from Strategy_Auto_Trader.markov_cli import full_scan, refresh_universe


class TestMain:
    def test_reports_added_and_removed(self, tmp_path, monkeypatch, capsys):
        out_path = tmp_path / "universe_sp_ftse.json"
        out_path.write_text(json.dumps({
            "built": "2026-07-16T00:00:00",
            "sources": {"ftse100": 1, "sp500": 2},
            "tickers": ["AAPL", "EA", "SHEL.L"],
        }), encoding="utf-8")
        monkeypatch.setattr(full_scan, "SP_FTSE_UNIVERSE_FILE", out_path)
        monkeypatch.setattr(full_scan, "_sp500_tickers", lambda: ["AAPL", "FERG"])
        monkeypatch.setattr(full_scan, "_ftse100_tickers", lambda: ["SHEL.L"])

        assert refresh_universe.main() == 0

        out = capsys.readouterr().out
        assert "Removed (1): ['EA']" in out
        assert "Added (1): ['FERG']" in out
        assert "regen_watchlists.py" in out

    def test_no_prior_file_reports_all_added(self, tmp_path, monkeypatch, capsys):
        out_path = tmp_path / "universe_sp_ftse.json"
        monkeypatch.setattr(full_scan, "SP_FTSE_UNIVERSE_FILE", out_path)
        monkeypatch.setattr(full_scan, "_sp500_tickers", lambda: ["AAPL"])
        monkeypatch.setattr(full_scan, "_ftse100_tickers", lambda: ["SHEL.L"])

        assert refresh_universe.main() == 0

        out = capsys.readouterr().out
        assert "Universe: 0 -> 2 tickers" in out
        assert "Removed: none" in out

    def test_no_change_reports_none(self, tmp_path, monkeypatch, capsys):
        out_path = tmp_path / "universe_sp_ftse.json"
        out_path.write_text(json.dumps({
            "built": "2026-07-16T00:00:00",
            "sources": {"ftse100": 1, "sp500": 1},
            "tickers": ["AAPL", "SHEL.L"],
        }), encoding="utf-8")
        monkeypatch.setattr(full_scan, "SP_FTSE_UNIVERSE_FILE", out_path)
        monkeypatch.setattr(full_scan, "_sp500_tickers", lambda: ["AAPL"])
        monkeypatch.setattr(full_scan, "_ftse100_tickers", lambda: ["SHEL.L"])

        assert refresh_universe.main() == 0

        out = capsys.readouterr().out
        assert "Removed: none" in out
        assert "Added: none" in out
        assert "regen_watchlists.py" not in out
