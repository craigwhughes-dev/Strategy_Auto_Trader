"""Tests for overnight_scope.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from Strategy_Auto_Trader.markov_cli import overnight_scope


@pytest.fixture
def config():
    """Sample overnight_strategy.json."""
    return {
        "markets": {
            "test_market": {
                "watchlist": "config/watchlist.json",
                "timezone": "Europe/London",
                "trading_start": "08:00",
                "trading_end": "16:30",
                "vol_screen": {"enabled": True, "min_trend_quality": 0.0, "period": "2y"},
                "exempt_if_open_position": True,
            }
        },
        "execution": {
            "capital_pot": 20000,
        },
    }


@pytest.fixture
def exec_state():
    """Sample execution_state.json with one open position."""
    return {
        "positions": {
            "OPEN_TICKER": {"quantity": 10, "fill_price": 100.0},
        },
        "trade_log": [],
        "trades_today": {"date": "2026-07-03", "buys": 0, "sells": 0},
    }


def test_load_watchlist_root_relative_path():
    """Config watchlist paths like "config/watchlist_ftse.json" resolve from repo root, not config/config/."""
    wl = overnight_scope.load_watchlist("config/watchlist_ftse.json")
    assert wl.get("tickers"), "expected tickers in config/watchlist_ftse.json"


def test_load_watchlist_bare_filename_falls_back_to_config_dir():
    wl = overnight_scope.load_watchlist("watchlist_ftse.json")
    assert wl.get("tickers")


def test_load_watchlist_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        overnight_scope.load_watchlist("config/does_not_exist.json")


def test_screen_market_vol_screen_excluded():
    """Ticker fails vol screen and has no open position — excluded."""
    market_cfg = {
        "watchlist": "config/watchlist.json",
        "vol_screen": {"enabled": True, "min_trend_quality": 0.0, "period": "2y"},
        "exempt_if_open_position": True,
    }

    with mock.patch("Strategy_Auto_Trader.quant_hmm.vol_screen.screen_tickers") as mock_vol:
        with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.load_watchlist") as mock_wl:
            mock_wl.return_value = {
                "tickers": [
                    {"ticker": "GOOD_TICKER"},
                    {"ticker": "BAD_TICKER"},
                ]
            }
            mock_vol.return_value = (["GOOD_TICKER"], [])

            result = overnight_scope.screen_market("test", market_cfg, {})

            assert "GOOD_TICKER" in result["kept"]
            assert "BAD_TICKER" not in result["kept"]
            assert any(e["ticker"] == "BAD_TICKER" for e in result["excluded"])


def test_screen_market_choppy_strategy_inverts_vol_screen():
    """mean_reversion/choppy_vol are designed to trade what vol_screen vetoes —
    scope screening must keep the low-trend-quality names, not exclude them."""
    market_cfg = {
        "watchlist": "config/watchlist.json",
        "defaults": {"strategy": "mean_reversion"},
        "vol_screen": {"enabled": True, "min_trend_quality": 0.0, "period": "2y"},
        "exempt_if_open_position": True,
    }

    with mock.patch("Strategy_Auto_Trader.quant_hmm.vol_screen.screen_tickers") as mock_vol:
        with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.load_watchlist") as mock_wl:
            mock_wl.return_value = {
                "tickers": [
                    {"ticker": "TRENDING"},
                    {"ticker": "CHOPPY"},
                ]
            }
            mock_vol.return_value = (
                ["TRENDING"],
                [
                    {"ticker": "TRENDING", "trend_quality": 1.5, "downside_vol": 0.1},
                    {"ticker": "CHOPPY", "trend_quality": -0.5, "downside_vol": 0.1},
                ],
            )

            result = overnight_scope.screen_market("test", market_cfg, {})

            assert "CHOPPY" in result["kept"]
            assert "TRENDING" not in result["kept"]
            assert any(e["ticker"] == "TRENDING" and e["reason"] == "vol_screen_inverted"
                       for e in result["excluded"])


def test_screen_market_choppy_strategy_still_caps_downside_vol():
    """Inverted screen still excludes low-trend-quality names that blow the
    downside-vol risk cap — mean_reversion wants choppy, not reckless."""
    market_cfg = {
        "watchlist": "config/watchlist.json",
        "defaults": {"strategy": "choppy_vol"},
        "vol_screen": {"enabled": True, "min_trend_quality": 0.0, "max_downside_vol": 0.25, "period": "2y"},
        "exempt_if_open_position": True,
    }

    with mock.patch("Strategy_Auto_Trader.quant_hmm.vol_screen.screen_tickers") as mock_vol:
        with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.load_watchlist") as mock_wl:
            mock_wl.return_value = {
                "tickers": [
                    {"ticker": "CHOPPY_SAFE"},
                    {"ticker": "CHOPPY_RECKLESS"},
                ]
            }
            mock_vol.return_value = (
                [],
                [
                    {"ticker": "CHOPPY_SAFE", "trend_quality": -0.5, "downside_vol": 0.1},
                    {"ticker": "CHOPPY_RECKLESS", "trend_quality": -0.5, "downside_vol": 0.9},
                ],
            )

            result = overnight_scope.screen_market("test", market_cfg, {})

            assert "CHOPPY_SAFE" in result["kept"]
            assert "CHOPPY_RECKLESS" not in result["kept"]


def test_screen_market_non_choppy_strategy_unaffected():
    """Regular (trend-following) strategy default keeps the original vol_screen behavior."""
    market_cfg = {
        "watchlist": "config/watchlist.json",
        "defaults": {"strategy": "default"},
        "vol_screen": {"enabled": True, "min_trend_quality": 0.0, "period": "2y"},
        "exempt_if_open_position": True,
    }

    with mock.patch("Strategy_Auto_Trader.quant_hmm.vol_screen.screen_tickers") as mock_vol:
        with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.load_watchlist") as mock_wl:
            mock_wl.return_value = {
                "tickers": [{"ticker": "TRENDING"}, {"ticker": "CHOPPY"}]
            }
            mock_vol.return_value = (
                ["TRENDING"],
                [
                    {"ticker": "TRENDING", "trend_quality": 1.5, "downside_vol": 0.1},
                    {"ticker": "CHOPPY", "trend_quality": -0.5, "downside_vol": 0.1},
                ],
            )

            result = overnight_scope.screen_market("test", market_cfg, {})

            assert "TRENDING" in result["kept"]
            assert "CHOPPY" not in result["kept"]


def test_screen_market_open_position_exempt():
    """Open position is always kept, even if vol screen would exclude it."""
    market_cfg = {
        "watchlist": "config/watchlist.json",
        "vol_screen": {"enabled": True, "min_trend_quality": 0.0, "period": "2y"},
        "exempt_if_open_position": True,
    }
    exec_state = {"positions": {"OPEN_TICKER": {"quantity": 10}}}

    with mock.patch("Strategy_Auto_Trader.quant_hmm.vol_screen.screen_tickers") as mock_vol:
        with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.load_watchlist") as mock_wl:
            mock_wl.return_value = {
                "tickers": [{"ticker": "OPEN_TICKER"}, {"ticker": "GOOD_TICKER"}]
            }
            mock_vol.return_value = (["GOOD_TICKER"], [])

            result = overnight_scope.screen_market("test", market_cfg, exec_state)

            assert "OPEN_TICKER" in result["kept"]
            assert "OPEN_TICKER" in result["open_positions"]


def test_screen_market_open_position_kept_even_if_dropped_from_watchlist():
    """An open position's ticker stays in kept (and is reported in
    orphaned_positions) even if it's no longer in the watchlist file at
    all — the actual gap being fixed (previously only worked if the ticker
    was still watchlist-listed)."""
    market_cfg = {
        "watchlist": "config/watchlist.json",
        "vol_screen": {"enabled": False},
        "exempt_if_open_position": True,
    }
    exec_state = {"positions": {"ORPHANED_TICKER": {"market": "test"}}}

    with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.load_watchlist") as mock_wl:
        mock_wl.return_value = {"tickers": [{"ticker": "GOOD_TICKER"}]}  # ORPHANED_TICKER absent

        result = overnight_scope.screen_market("test", market_cfg, exec_state)

        assert "ORPHANED_TICKER" in result["kept"]
        assert "ORPHANED_TICKER" in result["orphaned_positions"]
        assert "GOOD_TICKER" in result["kept"]


def test_screen_market_orphaned_positions_empty_when_no_drift():
    market_cfg = {
        "watchlist": "config/watchlist.json",
        "vol_screen": {"enabled": False},
        "exempt_if_open_position": True,
    }
    exec_state = {"positions": {"OPEN_TICKER": {"market": "test"}}}

    with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.load_watchlist") as mock_wl:
        mock_wl.return_value = {"tickers": [{"ticker": "OPEN_TICKER"}]}

        result = overnight_scope.screen_market("test", market_cfg, exec_state)

        assert result["orphaned_positions"] == []


def test_screen_market_position_scoped_by_market_field_not_leaked_across_markets():
    """A position tagged for a different market must not appear in this
    market's open_positions/kept — even if its ticker happens to also be in
    this market's watchlist (market attribution is now by the position's own
    recorded field, not by watchlist membership)."""
    market_cfg = {
        "watchlist": "config/watchlist.json",
        "vol_screen": {"enabled": False},
        "exempt_if_open_position": True,
    }
    exec_state = {"positions": {"CROSS_MARKET_TICKER": {"market": "other_market"}}}

    with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.load_watchlist") as mock_wl:
        mock_wl.return_value = {"tickers": [{"ticker": "CROSS_MARKET_TICKER"}]}

        result = overnight_scope.screen_market("test", market_cfg, exec_state)

        # Not in open_positions for "test" market, but stage1 (vol screen
        # disabled) still keeps it since it's a normal watchlist ticker.
        assert "CROSS_MARKET_TICKER" not in result["open_positions"]
        assert "CROSS_MARKET_TICKER" in result["kept"]


def test_screen_market_position_missing_market_field_defaults_to_current_market():
    """Legacy position data without a "market" key defaults to matching the
    current market rather than being silently excluded."""
    market_cfg = {
        "watchlist": "config/watchlist.json",
        "vol_screen": {"enabled": False},
        "exempt_if_open_position": True,
    }
    exec_state = {"positions": {"LEGACY_TICKER": {}}}  # no "market" key

    with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.load_watchlist") as mock_wl:
        mock_wl.return_value = {"tickers": []}  # not in watchlist at all

        result = overnight_scope.screen_market("test", market_cfg, exec_state)

        assert "LEGACY_TICKER" in result["kept"]
        assert "LEGACY_TICKER" in result["orphaned_positions"]


def test_generate_scoped_watchlist_merges_defaults(tmp_path):
    """Generated watchlist merges original defaults with execution config."""
    original_watchlist = {
        "defaults": {"strategy": "conservative", "initial_cash": 20000},
        "tickers": [{"ticker": "TICKER1"}, {"ticker": "TICKER2"}],
    }
    exec_cfg = {
        "capital_pot": 50000,
    }

    gen_dir = tmp_path / "generated"
    gen_dir.mkdir()

    with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.load_watchlist") as mock_wl:
        with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.CONFIG_DIR", tmp_path):
            mock_wl.return_value = original_watchlist

            overnight_scope.generate_scoped_watchlist("test", "config/watchlist.json", ["TICKER1"], exec_cfg)

            output_file = gen_dir / "watchlist_test_scoped.json"
            assert output_file.exists()

            with open(output_file, encoding="utf-8") as f:
                parsed = json.load(f)

            assert parsed["defaults"]["capital_pot"] == 50000
            assert parsed["defaults"]["strategy"] == "conservative"


def test_screen_market_overrides_populated_for_dict_tickers_with_override_keys():
    """screen_market() returns overrides populated only for tickers with OVERRIDE_KEYS."""
    market_cfg = {
        "watchlist": "config/watchlist.json",
        "vol_screen": {"enabled": False},
        "exempt_if_open_position": True,
    }

    with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.load_watchlist") as mock_wl:
        mock_wl.return_value = {
            "tickers": [
                "BARE_STRING_TICKER",
                {"ticker": "DICT_NO_OVERRIDE"},
                {"ticker": "DICT_WITH_STRATEGY", "strategy": "breakout_momentum"},
                {"ticker": "DICT_WITH_UNRELATED_KEY", "notes": "some metadata"},
            ]
        }

        result = overnight_scope.screen_market("test", market_cfg, {})

        assert "overrides" in result
        assert "BARE_STRING_TICKER" not in result["overrides"]
        assert "DICT_NO_OVERRIDE" not in result["overrides"]
        assert "DICT_WITH_UNRELATED_KEY" not in result["overrides"]
        assert "DICT_WITH_STRATEGY" in result["overrides"]
        assert result["overrides"]["DICT_WITH_STRATEGY"]["strategy"] == "breakout_momentum"


def test_screen_market_overrides_round_trips_through_write_scope_result(tmp_path):
    """overrides key survives write_scope_result and is readable from written JSON."""
    market_cfg = {
        "watchlist": "config/watchlist.json",
        "vol_screen": {"enabled": False},
        "exempt_if_open_position": True,
    }

    with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.load_watchlist") as mock_wl:
        with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.STATE_DIR", tmp_path):
            mock_wl.return_value = {
                "tickers": [
                    {"ticker": "T1", "strategy": "default"},
                    {"ticker": "T2", "strategy": "conservative"},
                ]
            }

            result = overnight_scope.screen_market("test", market_cfg, {})
            overnight_scope.write_scope_result("test", result)

            # Read back from written file
            output_file = tmp_path / "in_scope_test.json"
            assert output_file.exists()
            with open(output_file, encoding="utf-8") as f:
                persisted = json.load(f)

            assert "overrides" in persisted
            assert persisted["overrides"]["T1"]["strategy"] == "default"
            assert persisted["overrides"]["T2"]["strategy"] == "conservative"


def test_with_merged_defaults_merges_top_level_blocks_market_wins():
    """The config-merge bug fix: main() must merge top-level vol_screen/
    exempt_if_open_position into market_cfg before screen_market() reads
    them, with market-level keys taking precedence."""
    config = {
        "vol_screen": {"enabled": True, "max_downside_vol": 0.25},
        "exempt_if_open_position": True,
    }
    market_cfg = {"watchlist": "x.json"}  # no own overrides

    merged = overnight_scope._with_merged_defaults(market_cfg, config)

    assert merged["vol_screen"]["max_downside_vol"] == 0.25
    assert merged["exempt_if_open_position"] is True
    assert merged["watchlist"] == "x.json"


def test_with_merged_defaults_market_level_override_wins():
    config = {"vol_screen": {"enabled": True, "max_downside_vol": 0.25}}
    market_cfg = {"watchlist": "x.json", "vol_screen": {"enabled": False}}

    merged = overnight_scope._with_merged_defaults(market_cfg, config)

    assert merged["vol_screen"] == {"enabled": False}


class TestComputeGlobalTopK:

    def test_disabled_returns_none(self):
        config = {"top_k_screen": {"enabled": False}}
        assert overnight_scope.compute_global_top_k(config, {}) is None

    def test_missing_config_block_returns_none(self):
        assert overnight_scope.compute_global_top_k({}, {}) is None

    def test_success_writes_state_and_returns_top_k(self, tmp_path):
        config = {"top_k_screen": {"enabled": True, "k": 1}}

        def fake_run(cmd, cwd, timeout, capture_output, text):
            output_path = Path(cmd[cmd.index("--output") + 1])
            output_path.write_text(json.dumps({"A": 0.9, "B": 0.1}), encoding="utf-8")
            return mock.Mock(returncode=0, stderr="")

        with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.STATE_DIR", tmp_path):
            with mock.patch("subprocess.run", side_effect=fake_run):
                result = overnight_scope.compute_global_top_k(config, {})

        assert result == {"A"}
        state_file = tmp_path / "top_k_universe.json"
        assert state_file.exists()
        state = json.loads(state_file.read_text())
        assert state["status"] == "ok"
        assert state["tickers"] == ["A"]

    def test_open_position_exempt_even_outside_top_k(self, tmp_path):
        config = {"top_k_screen": {"enabled": True, "k": 1}}
        exec_state = {"positions": {"LOW_SCORE": {}}}

        def fake_run(cmd, cwd, timeout, capture_output, text):
            output_path = Path(cmd[cmd.index("--output") + 1])
            output_path.write_text(json.dumps({"A": 0.9, "LOW_SCORE": 0.1}), encoding="utf-8")
            return mock.Mock(returncode=0, stderr="")

        with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.STATE_DIR", tmp_path):
            with mock.patch("subprocess.run", side_effect=fake_run):
                result = overnight_scope.compute_global_top_k(config, exec_state)

        assert "LOW_SCORE" in result

    def test_timeout_falls_back_to_previous_state(self, tmp_path):
        import subprocess as subprocess_mod

        prior = {"date": "2026-07-29", "k": 70, "strategy": "optimised",
                 "tickers": ["PRIOR_A", "PRIOR_B"], "scores": {}, "status": "ok"}
        (tmp_path / "top_k_universe.json").write_text(json.dumps(prior), encoding="utf-8")

        config = {"top_k_screen": {"enabled": True, "k": 70}}

        with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.STATE_DIR", tmp_path):
            with mock.patch("subprocess.run", side_effect=subprocess_mod.TimeoutExpired(cmd="x", timeout=1)):
                result = overnight_scope.compute_global_top_k(config, {})

        assert result == {"PRIOR_A", "PRIOR_B"}

    def test_nonzero_exit_falls_back_to_previous_state(self, tmp_path):
        prior = {"date": "2026-07-29", "k": 70, "strategy": "optimised",
                 "tickers": ["PRIOR_A"], "scores": {}, "status": "ok"}
        (tmp_path / "top_k_universe.json").write_text(json.dumps(prior), encoding="utf-8")

        config = {"top_k_screen": {"enabled": True, "k": 70}}

        with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.STATE_DIR", tmp_path):
            with mock.patch("subprocess.run", return_value=mock.Mock(returncode=1, stderr="boom")):
                result = overnight_scope.compute_global_top_k(config, {})

        assert result == {"PRIOR_A"}

    def test_failure_with_no_prior_state_returns_none(self, tmp_path):
        config = {"top_k_screen": {"enabled": True, "k": 70}}

        with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.STATE_DIR", tmp_path):
            with mock.patch("subprocess.run", return_value=mock.Mock(returncode=1, stderr="boom")):
                result = overnight_scope.compute_global_top_k(config, {})

        assert result is None


def test_screen_market_strategy_opt_out_skips_vol_screen_stage1():
    """A strategy with skip_overnight_vol_screen=True bypasses stage-1 even
    when vol_screen.enabled=True in config — strategy opt-out always wins."""
    market_cfg = {
        "watchlist": "config/watchlist.json",
        "defaults": {"strategy": "optimised_new"},
        "vol_screen": {"enabled": True, "min_trend_quality": 0.5, "period": "2y"},
        "exempt_if_open_position": True,
    }

    with mock.patch("Strategy_Auto_Trader.quant_hmm.vol_screen.screen_tickers") as mock_vol:
        with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.load_watchlist") as mock_wl:
            mock_wl.return_value = {
                "tickers": [{"ticker": "TICKER_A"}, {"ticker": "TICKER_B"}]
            }

            result = overnight_scope.screen_market("test", market_cfg, {})

            mock_vol.assert_not_called()
            assert "TICKER_A" in result["kept"]
            assert "TICKER_B" in result["kept"]
            assert result["excluded"] == []


def test_main_calls_compute_global_top_k_once_not_per_market(tmp_path):
    """Ranking is global, not per-market — must be computed once and shared,
    not recomputed once per market (which would silently double the nightly
    compute budget)."""
    config = {
        "markets": {
            "ftse": {"watchlist": "config/watchlist_ftse.json", "defaults": {}},
            "sp500": {"watchlist": "config/watchlist_sp500.json", "defaults": {}},
        },
        "top_k_screen": {"enabled": True, "k": 70},
    }

    with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.load_config", return_value=config):
        with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.load_execution_state", return_value={}):
            with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.compute_global_top_k", return_value={"A"}) as mock_topk:
                with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.screen_market",
                                 return_value={"kept": [], "excluded": [], "open_positions": []}):
                    with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.write_scope_result"):
                        with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.generate_scoped_watchlist"):
                            overnight_scope.main()

    mock_topk.assert_called_once()


class TestRefreshUniverseAndWatchlists:
    def test_disabled_by_default_is_a_noop(self):
        with mock.patch("Strategy_Auto_Trader.markov_cli.full_scan.build_sp_ftse_universe") as mock_build:
            overnight_scope.refresh_universe_and_watchlists({})
        mock_build.assert_not_called()

    def test_enabled_refreshes_universe_and_regenerates_both_watchlists(self):
        config = {"universe_refresh": {"enabled": True}}
        with mock.patch("Strategy_Auto_Trader.markov_cli.full_scan.SP_FTSE_UNIVERSE_FILE") as mock_path:
            mock_path.exists.return_value = True
            with mock.patch("Strategy_Auto_Trader.markov_cli.full_scan.load_sp_ftse_universe",
                             return_value=["AAPL", "EA", "SHEL.L"]):
                with mock.patch("Strategy_Auto_Trader.markov_cli.full_scan.build_sp_ftse_universe",
                                 return_value=["AAPL", "FERG", "SHEL.L"]) as mock_build:
                    with mock.patch("Strategy_Auto_Trader.markov_cli.regen_watchlists.regen_watchlist",
                                     return_value=(1, 1)) as mock_regen:
                        overnight_scope.refresh_universe_and_watchlists(config)

        mock_build.assert_called_once()
        assert mock_regen.call_count == 2
        calls = {c.args[0].name: c.args[1] for c in mock_regen.call_args_list}
        assert calls["watchlist_ftse.json"] == ["SHEL.L"]
        assert calls["watchlist_sp500.json"] == ["AAPL", "FERG"]

    def test_no_prior_universe_file_treats_all_as_added(self):
        config = {"universe_refresh": {"enabled": True}}
        with mock.patch("Strategy_Auto_Trader.markov_cli.full_scan.SP_FTSE_UNIVERSE_FILE") as mock_path:
            mock_path.exists.return_value = False
            with mock.patch("Strategy_Auto_Trader.markov_cli.full_scan.build_sp_ftse_universe",
                             return_value=["AAPL"]):
                with mock.patch("Strategy_Auto_Trader.markov_cli.regen_watchlists.regen_watchlist",
                                 return_value=(0, 1)) as mock_regen:
                    overnight_scope.refresh_universe_and_watchlists(config)
        assert mock_regen.call_count == 2

    def test_fetch_failure_is_caught_and_watchlists_untouched(self):
        config = {"universe_refresh": {"enabled": True}}
        with mock.patch("Strategy_Auto_Trader.markov_cli.full_scan.SP_FTSE_UNIVERSE_FILE") as mock_path:
            mock_path.exists.return_value = True
            with mock.patch("Strategy_Auto_Trader.markov_cli.full_scan.load_sp_ftse_universe",
                             return_value=["AAPL"]):
                with mock.patch("Strategy_Auto_Trader.markov_cli.full_scan.build_sp_ftse_universe",
                                 side_effect=RuntimeError("Wikipedia unreachable")):
                    with mock.patch("Strategy_Auto_Trader.markov_cli.regen_watchlists.regen_watchlist") as mock_regen:
                        overnight_scope.refresh_universe_and_watchlists(config)
        mock_regen.assert_not_called()

    def test_main_calls_refresh_before_top_k(self):
        """Watchlists must be current before compute_global_top_k/screen_market
        read them — see refresh_universe_and_watchlists' docstring."""
        config = {
            "markets": {"ftse": {"watchlist": "config/watchlist_ftse.json", "defaults": {}}},
            "universe_refresh": {"enabled": True},
        }
        call_order = []
        with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.load_config", return_value=config):
            with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.load_execution_state", return_value={}):
                with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.refresh_universe_and_watchlists",
                                 side_effect=lambda c: call_order.append("refresh")):
                    with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.compute_global_top_k",
                                     side_effect=lambda *a, **k: call_order.append("top_k") or None):
                        with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.screen_market",
                                         return_value={"kept": [], "excluded": [], "open_positions": []}):
                            with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.write_scope_result"):
                                with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.generate_scoped_watchlist"):
                                    overnight_scope.main()

        assert call_order == ["refresh", "top_k"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
