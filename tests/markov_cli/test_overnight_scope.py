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
                "sentiment_screen": {"enabled": True, "min_sentiment_score": -0.3, "exclude_labels": ["bearish"]},
                "exempt_if_open_position": True,
            }
        },
        "execution": {
            "capital_pot": 20000,
            "max_positions": 5,
            "daily_buy_limit": 2,
            "daily_sell_limit": None,
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
        "sentiment_screen": {"enabled": False},
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


def test_screen_market_open_position_exempt():
    """Open position is always kept, even if vol screen would exclude it."""
    market_cfg = {
        "watchlist": "config/watchlist.json",
        "vol_screen": {"enabled": True, "min_trend_quality": 0.0, "period": "2y"},
        "sentiment_screen": {"enabled": False},
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


def test_screen_market_sentiment_bearish_excluded():
    """Bearish sentiment excluded (unless open position)."""
    market_cfg = {
        "watchlist": "config/watchlist.json",
        "vol_screen": {"enabled": False},
        "sentiment_screen": {
            "enabled": True,
            "min_sentiment_score": -0.3,
            "exclude_labels": ["bearish"],
        },
        "exempt_if_open_position": True,
    }

    with mock.patch("Strategy_Auto_Trader.markov_cli.overnight_scope.load_watchlist") as mock_wl:
        with mock.patch("Strategy_Auto_Trader.quant_hmm.sentiment.composite_sentiment") as mock_sent:
            mock_wl.return_value = {
                "tickers": [{"ticker": "BULLISH"}, {"ticker": "BEARISH"}]
            }

            def sentiment_side_effect(ticker):
                if ticker == "BEARISH":
                    return {"sentiment_label": "bearish", "sentiment_score": -0.5}
                return {"sentiment_label": "bullish", "sentiment_score": 0.5}

            mock_sent.side_effect = sentiment_side_effect

            result = overnight_scope.screen_market("test", market_cfg, {})

            assert "BULLISH" in result["kept"]
            assert "BEARISH" not in result["kept"]
            assert any(e["ticker"] == "BEARISH" for e in result["excluded"])


def test_generate_scoped_watchlist_merges_defaults(tmp_path):
    """Generated watchlist merges original defaults with execution config."""
    original_watchlist = {
        "defaults": {"strategy": "conservative", "initial_cash": 20000},
        "tickers": [{"ticker": "TICKER1"}, {"ticker": "TICKER2"}],
    }
    exec_cfg = {
        "capital_pot": 50000,
        "max_positions": 3,
        "daily_buy_limit": 5,
        "daily_sell_limit": 2,
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
            assert parsed["defaults"]["max_positions"] == 3
            assert parsed["defaults"]["daily_buy_limit"] == 5
            assert parsed["defaults"]["daily_sell_limit"] == 2
            assert parsed["defaults"]["strategy"] == "conservative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
