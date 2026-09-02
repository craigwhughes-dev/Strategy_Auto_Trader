from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from Strategy_Auto_Trader.synthetic_backtest_data import generate


def _make_daily(n_days: int, start_price: float = 100.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = start_price * np.exp(np.cumsum(rng.normal(0, 0.01, size=n_days)))
    idx = pd.date_range("2026-01-01", periods=n_days, freq="D", tz="UTC")
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1000},
        index=idx,
    )


class TestGenerateSyntheticHourly:
    """Stooq's real local cache may have a file for any given ticker, so
    every test here patches stooq_daily.load_stooq_daily explicitly (either
    to the fixture frame, to exercise the primary path, or to None, to
    force the IBKR fallback) rather than relying on what happens to be on
    disk."""

    def test_row_count_matches_days_times_bars_per_day(self):
        daily = _make_daily(30)
        with patch.object(generate.stooq_daily, "load_stooq_daily", return_value=daily):
            df = generate.generate_synthetic_hourly("AAPL", vol_window=21, bars_per_day=7, seed=1)

        # vol_window=21: rolling std on the log-return series (itself missing
        # its first value to the shift) first becomes valid at daily index 21
        # (see test_vol.py's warmup-length test) — transitions before that
        # are skipped, not fabricated with a NaN/garbage sigma. Valid
        # transitions are daily indices 21..29 inclusive: 9 days.
        n_days_with_vol = 30 - 21
        assert df is not None
        assert len(df) == n_days_with_vol * 7

    def test_columns_and_tz_aware_index(self):
        daily = _make_daily(30)
        with patch.object(generate.stooq_daily, "load_stooq_daily", return_value=daily):
            df = generate.generate_synthetic_hourly("AAPL", seed=1)

        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert df.index.tz is not None
        assert df.index.is_monotonic_increasing

    def test_last_bar_of_each_day_hits_real_daily_close(self):
        daily = _make_daily(30)
        with patch.object(generate.stooq_daily, "load_stooq_daily", return_value=daily):
            df = generate.generate_synthetic_hourly("AAPL", vol_window=21, bars_per_day=7, seed=1)

        bars_per_day = 7
        last_bar_closes = df["Close"].iloc[bars_per_day - 1::bars_per_day].to_numpy()
        expected_closes = daily["Close"].iloc[21:].to_numpy()  # first usable transition is day 21
        assert np.allclose(last_bar_closes, expected_closes)

    def test_returns_none_when_both_sources_fail(self):
        with patch.object(generate.stooq_daily, "load_stooq_daily", return_value=None), \
             patch.object(generate.IBKRDataClient, "fetch_daily", return_value=None):
            assert generate.generate_synthetic_hourly("AAPL") is None

    def test_deterministic_given_seed(self):
        daily = _make_daily(30)
        with patch.object(generate.stooq_daily, "load_stooq_daily", return_value=daily):
            df1 = generate.generate_synthetic_hourly("AAPL", seed=7)
            df2 = generate.generate_synthetic_hourly("AAPL", seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_falls_back_to_ibkr_when_stooq_has_no_file(self):
        daily = _make_daily(30)
        with patch.object(generate.stooq_daily, "load_stooq_daily", return_value=None), \
             patch.object(generate.IBKRDataClient, "fetch_daily", return_value=daily) as ibkr_fetch:
            df = generate.generate_synthetic_hourly("SOME.NEW.LISTING")

        ibkr_fetch.assert_called_once()
        assert df is not None

    def test_does_not_call_ibkr_when_stooq_has_data(self):
        daily = _make_daily(30)
        with patch.object(generate.stooq_daily, "load_stooq_daily", return_value=daily), \
             patch.object(generate.IBKRDataClient, "fetch_daily") as ibkr_fetch:
            df = generate.generate_synthetic_hourly("AAPL")

        ibkr_fetch.assert_not_called()
        assert df is not None


class TestGenerateAndWriteWorker:
    def test_ok_writes_file_and_reports_status(self, tmp_path):
        daily = _make_daily(30)
        with patch.object(generate, "generate_synthetic_hourly", return_value=daily):
            result = generate._generate_and_write_worker("AAPL", 21, 7, 1, tmp_path)

        assert result["status"] == "ok"
        assert result["n_bars"] == len(daily)
        assert (tmp_path / "AAPL.csv").exists()

    def test_no_data_when_generation_returns_none(self, tmp_path):
        with patch.object(generate, "generate_synthetic_hourly", return_value=None):
            result = generate._generate_and_write_worker("AAPL", 21, 7, 1, tmp_path)

        assert result["status"] == "no_data"
        assert not list(tmp_path.iterdir())

    def test_error_is_caught_not_raised(self, tmp_path):
        with patch.object(generate, "generate_synthetic_hourly", side_effect=RuntimeError("boom")):
            result = generate._generate_and_write_worker("AAPL", 21, 7, 1, tmp_path)

        assert result["status"] == "error"
        assert "boom" in result["note"]

    def test_ticker_with_slash_sanitized_in_filename(self, tmp_path):
        daily = _make_daily(30)
        with patch.object(generate, "generate_synthetic_hourly", return_value=daily):
            result = generate._generate_and_write_worker("BRK/B", 21, 7, 1, tmp_path)

        assert (tmp_path / "BRK-B.csv").exists()
        assert result["status"] == "ok"


class TestParseArgs:
    def test_workers_defaults_to_one(self):
        args = generate._parse_args(["--tickers", "AAPL"])
        assert args.workers == 1

    def test_workers_below_one_rejected(self):
        with pytest.raises(SystemExit):
            generate._parse_args(["--tickers", "AAPL", "--workers", "0"])


class TestMainCLI:
    def test_sequential_writes_all_tickers(self, tmp_path, capsys):
        daily = _make_daily(30)
        with patch.object(generate, "generate_synthetic_hourly", return_value=daily):
            generate.main(["--tickers", "AAPL", "MSFT", "--output-dir", str(tmp_path)])

        assert (tmp_path / "AAPL.csv").exists()
        assert (tmp_path / "MSFT.csv").exists()
        out = capsys.readouterr().out
        assert "AAPL" in out and "MSFT" in out

    def test_parallel_path_uses_process_pool_executor_and_writes_all(self, tmp_path, monkeypatch, capsys):
        """ProcessPoolExecutor spawns fresh processes that re-import this
        module, so a unittest.mock patch on generate_synthetic_hourly made
        in the test process would be invisible to real worker processes.
        Swap in ThreadPoolExecutor instead — same futures/as_completed
        wiring, but same-process so the mock is visible; this validates
        main()'s pool wiring without real-multiprocess flakiness."""
        monkeypatch.setattr(generate, "ProcessPoolExecutor", ThreadPoolExecutor)
        daily = _make_daily(30)
        with patch.object(generate, "generate_synthetic_hourly", return_value=daily):
            generate.main(["--tickers", "AAPL", "MSFT", "--workers", "2", "--output-dir", str(tmp_path)])

        assert (tmp_path / "AAPL.csv").exists()
        assert (tmp_path / "MSFT.csv").exists()
        out = capsys.readouterr().out
        assert "AAPL" in out and "MSFT" in out

    def test_one_ticker_failure_does_not_block_others(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(generate, "ProcessPoolExecutor", ThreadPoolExecutor)
        daily = _make_daily(30)

        def _side_effect(ticker, **kwargs):
            if ticker == "BAD":
                raise RuntimeError("boom")
            return daily

        with patch.object(generate, "generate_synthetic_hourly", side_effect=_side_effect):
            generate.main(["--tickers", "AAPL", "BAD", "--workers", "2", "--output-dir", str(tmp_path)])

        assert (tmp_path / "AAPL.csv").exists()
        assert not (tmp_path / "BAD.csv").exists()
