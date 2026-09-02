from __future__ import annotations

import pandas as pd

from Strategy_Auto_Trader.synthetic_backtest_data import stooq_daily

_HEADER = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"


def _write_stooq_file(path, ticker_suffix: str, rows: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HEADER + "\n".join(rows) + "\n", encoding="utf-8")


class TestStooqPath:
    def test_uk_ticker_maps_to_uk_subdir(self, tmp_path):
        path = stooq_daily._stooq_path("HSBA.L", tmp_path)
        assert path == tmp_path / "uk" / "hsba.uk.txt"

    def test_us_ticker_maps_to_us_subdir(self, tmp_path):
        path = stooq_daily._stooq_path("AAPL", tmp_path)
        assert path == tmp_path / "us" / "aapl.us.txt"

    def test_dotted_us_ticker_lowercased_dash_preserved(self, tmp_path):
        path = stooq_daily._stooq_path("BRK-B", tmp_path)
        assert path == tmp_path / "us" / "brk-b.us.txt"


class TestLoadStooqDaily:
    def test_missing_file_returns_none(self, tmp_path):
        assert stooq_daily.load_stooq_daily("NOPE", tmp_path) is None

    def test_parses_uk_file(self, tmp_path):
        _write_stooq_file(
            tmp_path / "uk" / "hsba.uk.txt", "uk",
            ["HSBA.UK,D,19920611,000000,121.009,122.999,121.009,122.999,0,0",
             "HSBA.UK,D,19920710,000000,71.682,75.4457,71.2351,73.3954,106609197,0"],
        )
        out = stooq_daily.load_stooq_daily("HSBA.L", tmp_path)
        assert out is not None
        assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(out) == 2
        assert out.index.tz is not None
        assert out["Close"].iloc[0] == 122.999
        assert out.index[0] == pd.Timestamp("1992-06-11", tz="UTC")

    def test_parses_us_file(self, tmp_path):
        _write_stooq_file(
            tmp_path / "us" / "aapl.us.txt", "us",
            ["AAPL.US,D,20240903,000000,100,105,99,104,1000000,0"],
        )
        out = stooq_daily.load_stooq_daily("AAPL", tmp_path)
        assert out is not None
        assert out["Close"].iloc[0] == 104

    def test_output_sorted_by_date(self, tmp_path):
        _write_stooq_file(
            tmp_path / "us" / "aapl.us.txt", "us",
            ["AAPL.US,D,20240905,000000,1,1,1,3,1,0",
             "AAPL.US,D,20240903,000000,1,1,1,1,1,0",
             "AAPL.US,D,20240904,000000,1,1,1,2,1,0"],
        )
        out = stooq_daily.load_stooq_daily("AAPL", tmp_path)
        assert list(out["Close"]) == [1, 2, 3]
        assert out.index.is_monotonic_increasing
