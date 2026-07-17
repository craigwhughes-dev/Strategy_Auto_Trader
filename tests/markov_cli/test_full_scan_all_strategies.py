"""Tests for markov_cli.full_scan_all_strategies — the strategy-loop
orchestrator. Never calls the real full_scan.main (no network/backtest),
just asserts loop behaviour and argv forwarding."""

from __future__ import annotations

import pytest

from Strategy_Auto_Trader.markov_cli import full_scan_all_strategies as fsa
from Strategy_Auto_Trader.strategy.base.registry import STRATEGY_REGISTRY


@pytest.fixture(autouse=True)
def _fake_universe(monkeypatch):
    monkeypatch.setattr(fsa.full_scan, "SP_FTSE_UNIVERSE_FILE", _AlwaysExists())
    monkeypatch.setattr(fsa.full_scan, "load_sp_ftse_universe", lambda: ["AAA", "BBB"])


class _AlwaysExists:
    def exists(self):
        return True


class TestStrategiesConstant:
    def test_covers_full_registry(self):
        assert fsa.STRATEGIES == sorted(STRATEGY_REGISTRY)


class TestMainLoop:
    def test_calls_full_scan_once_per_strategy(self, monkeypatch):
        calls = []
        monkeypatch.setattr(fsa.full_scan, "main", lambda argv: calls.append(argv))
        rc = fsa.main([])
        assert rc == 0
        assert len(calls) == len(fsa.STRATEGIES)
        called_strategies = [argv[argv.index("--strategy") + 1] for argv in calls]
        assert called_strategies == fsa.STRATEGIES

    def test_no_sentiment_always_forwarded(self, monkeypatch):
        calls = []
        monkeypatch.setattr(fsa.full_scan, "main", lambda argv: calls.append(argv))
        fsa.main([])
        assert all("--no-sentiment" in argv for argv in calls)

    def test_force_and_limit_forwarded(self, monkeypatch):
        calls = []
        monkeypatch.setattr(fsa.full_scan, "main", lambda argv: calls.append(argv))
        fsa.main(["--force", "--limit", "5"])
        for argv in calls:
            assert "--force" in argv
            assert argv[argv.index("--limit") + 1] == "5"

    def test_strategies_subset_selectable(self, monkeypatch):
        calls = []
        monkeypatch.setattr(fsa.full_scan, "main", lambda argv: calls.append(argv))
        rc = fsa.main(["--strategies", "default", "conservative"])
        assert rc == 0
        assert len(calls) == 2

    def test_unknown_strategy_rejected(self):
        rc = fsa.main(["--strategies", "not_a_real_strategy"])
        assert rc == 1

    def test_one_strategy_crashing_does_not_stop_others(self, monkeypatch):
        calls = []

        def fake_main(argv):
            calls.append(argv)
            if argv[argv.index("--strategy") + 1] == fsa.STRATEGIES[0]:
                raise RuntimeError("boom")

        monkeypatch.setattr(fsa.full_scan, "main", fake_main)
        rc = fsa.main([])
        assert len(calls) == len(fsa.STRATEGIES)
        assert rc == 1

    def test_all_clean_returns_zero(self, monkeypatch):
        monkeypatch.setattr(fsa.full_scan, "main", lambda argv: None)
        assert fsa.main([]) == 0

    def test_builds_universe_when_missing(self, monkeypatch):
        class _NeverExists:
            def exists(self):
                return False

        monkeypatch.setattr(fsa.full_scan, "SP_FTSE_UNIVERSE_FILE", _NeverExists())
        built = []
        monkeypatch.setattr(fsa.full_scan, "build_sp_ftse_universe", lambda: built.append(True))
        monkeypatch.setattr(fsa.full_scan, "main", lambda argv: None)
        fsa.main([])
        assert built == [True]

    def test_workers_flag_passed_through(self, monkeypatch):
        calls = []
        monkeypatch.setattr(fsa.full_scan, "main", lambda argv: calls.append(argv))
        fsa.main(["--workers", "4"])
        for argv in calls:
            assert "--workers" in argv
            assert argv[argv.index("--workers") + 1] == "4"

    def test_workers_default_passed_through(self, monkeypatch):
        calls = []
        monkeypatch.setattr(fsa.full_scan, "main", lambda argv: calls.append(argv))
        fsa.main([])
        for argv in calls:
            assert "--workers" in argv
            # Default should be 2
            assert argv[argv.index("--workers") + 1] == "2"


class TestDataCutoffForwarding:
    """Uses fsa.main's scan_main DI seam — no monkeypatching."""

    def test_default_cutoff_is_today_same_for_every_strategy(self):
        from datetime import date
        calls = []
        fsa.main([], scan_main=calls.append)
        cutoffs = {argv[argv.index("--data-cutoff") + 1] for argv in calls}
        assert cutoffs == {date.today().isoformat()}

    def test_explicit_date_forwarded(self):
        calls = []
        fsa.main(["--data-cutoff", "2026-07-01"], scan_main=calls.append)
        for argv in calls:
            assert argv[argv.index("--data-cutoff") + 1] == "2026-07-01"

    def test_none_disables_cutoff(self):
        calls = []
        fsa.main(["--data-cutoff", "none"], scan_main=calls.append)
        assert all("--data-cutoff" not in argv for argv in calls)

    def test_bad_cutoff_rejected(self):
        with pytest.raises(SystemExit):
            fsa.main(["--data-cutoff", "yesterday-ish"], scan_main=lambda argv: None)
