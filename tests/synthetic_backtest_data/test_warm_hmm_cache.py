from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from Strategy_Auto_Trader.synthetic_backtest_data import warm_hmm_cache


def _write_synthetic_csv(path, n=800, start="2007-06-01", seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=n)))
    df = pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1000},
        index=idx,
    )
    df.to_csv(path)
    return df


class TestLoadSyntheticHourly:
    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(warm_hmm_cache, "_SYNTHETIC_HOURLY_DIR", tmp_path)
        assert warm_hmm_cache._load_synthetic_hourly("NOPE") is None

    def test_parses_tz_aware(self, tmp_path, monkeypatch):
        monkeypatch.setattr(warm_hmm_cache, "_SYNTHETIC_HOURLY_DIR", tmp_path)
        _write_synthetic_csv(tmp_path / "AAPL.csv", n=10)
        out = warm_hmm_cache._load_synthetic_hourly("AAPL")
        assert out is not None
        assert out.index.tz is not None
        assert len(out) == 10


class TestWarmHmmCacheForTicker:
    def test_no_data_when_csv_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(warm_hmm_cache, "_SYNTHETIC_HOURLY_DIR", tmp_path)
        result = warm_hmm_cache.warm_hmm_cache_for_ticker("AAPL", "2008-01-01", "2009-07-31")
        assert result["status"] == "no_data"

    def test_no_data_when_window_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(warm_hmm_cache, "_SYNTHETIC_HOURLY_DIR", tmp_path)
        _write_synthetic_csv(tmp_path / "AAPL.csv", n=10, start="2020-01-01")
        result = warm_hmm_cache.warm_hmm_cache_for_ticker("AAPL", "2008-01-01", "2009-07-31")
        assert result["status"] == "no_data"

    def test_calls_run_ticker_backtest_with_synthetic_cache_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(warm_hmm_cache, "_SYNTHETIC_HOURLY_DIR", tmp_path)
        _write_synthetic_csv(tmp_path / "AAPL.csv", n=800, start="2008-01-01")

        with patch.object(
            warm_hmm_cache, "run_ticker_backtest",
            return_value=(pd.DataFrame({"x": [1]}), pd.DataFrame()),
        ) as m_backtest:
            result = warm_hmm_cache.warm_hmm_cache_for_ticker("AAPL", "2008-01-01", "2009-07-31")

        assert result["status"] == "ok"
        _, kwargs = m_backtest.call_args
        assert kwargs["use_persistent_cache"] is True
        assert kwargs["hmm_cache_dir"] == warm_hmm_cache.SYNTHETIC_HMM_CACHE_DIR

    def test_no_data_when_detail_is_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(warm_hmm_cache, "_SYNTHETIC_HOURLY_DIR", tmp_path)
        _write_synthetic_csv(tmp_path / "AAPL.csv", n=800, start="2008-01-01")

        with patch.object(warm_hmm_cache, "run_ticker_backtest", return_value=(None, None)):
            result = warm_hmm_cache.warm_hmm_cache_for_ticker("AAPL", "2008-01-01", "2009-07-31")

        assert result["status"] == "no_data"

    def test_error_is_caught_not_raised(self, tmp_path, monkeypatch):
        monkeypatch.setattr(warm_hmm_cache, "_SYNTHETIC_HOURLY_DIR", tmp_path)
        _write_synthetic_csv(tmp_path / "AAPL.csv", n=800, start="2008-01-01")

        with patch.object(warm_hmm_cache, "run_ticker_backtest", side_effect=RuntimeError("boom")):
            result = warm_hmm_cache.warm_hmm_cache_for_ticker("AAPL", "2008-01-01", "2009-07-31")

        assert result["status"] == "error"
        assert "boom" in result["note"]


class TestParseArgs:
    def test_requires_start_and_end_date(self):
        with pytest.raises(SystemExit):
            warm_hmm_cache._parse_args([])

    def test_workers_below_one_rejected(self):
        with pytest.raises(SystemExit):
            warm_hmm_cache._parse_args(
                ["--start-date", "2008-01-01", "--end-date", "2009-07-31", "--workers", "0"])

    def test_tickers_default_none(self):
        args = warm_hmm_cache._parse_args(["--start-date", "2008-01-01", "--end-date", "2009-07-31"])
        assert args.tickers is None


class TestMainCLI:
    def test_defaults_to_full_universe_file(self, tmp_path, monkeypatch, capsys):
        universe_path = tmp_path / "universe.json"
        universe_path.write_text(json.dumps({"tickers": ["A", "B"]}), encoding="utf-8")
        monkeypatch.setattr(warm_hmm_cache, "_UNIVERSE_FILE", universe_path)

        with patch.object(
            warm_hmm_cache, "warm_hmm_cache_for_ticker",
            return_value={"ticker": "X", "status": "ok", "n_bars": 5},
        ) as m_warm:
            warm_hmm_cache.main(["--start-date", "2008-01-01", "--end-date", "2009-07-31"])

        assert m_warm.call_count == 2
        called_tickers = {c.args[0] for c in m_warm.call_args_list}
        assert called_tickers == {"A", "B"}

    def test_explicit_tickers_override_universe_file(self, monkeypatch):
        with patch.object(
            warm_hmm_cache, "warm_hmm_cache_for_ticker",
            return_value={"ticker": "X", "status": "ok", "n_bars": 5},
        ) as m_warm:
            warm_hmm_cache.main(
                ["--tickers", "AAPL", "--start-date", "2008-01-01", "--end-date", "2009-07-31"])

        m_warm.assert_called_once()
        assert m_warm.call_args.args[0] == "AAPL"

    def test_parallel_path_uses_process_pool_executor(self, monkeypatch):
        monkeypatch.setattr(warm_hmm_cache, "ProcessPoolExecutor", ThreadPoolExecutor)
        with patch.object(
            warm_hmm_cache, "warm_hmm_cache_for_ticker",
            return_value={"ticker": "X", "status": "ok", "n_bars": 5},
        ) as m_warm:
            warm_hmm_cache.main(
                ["--tickers", "AAPL", "MSFT", "--start-date", "2008-01-01",
                 "--end-date", "2009-07-31", "--workers", "2"])

        assert m_warm.call_count == 2
