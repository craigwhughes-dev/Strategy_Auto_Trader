"""Tests for the index-history cache. No network: the fetch is injected."""

from __future__ import annotations

import pandas as pd
import pytest

from Strategy_Auto_Trader.research import index_history as ih


@pytest.fixture
def closes() -> pd.Series:
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"], utc=True)
    return pd.Series([100.0, 101.5, 99.25], index=idx, name="^GSPC")


def test_cache_path_strips_the_caret_and_lowercases(tmp_path):
    assert ih.cache_path("^GDAXI", tmp_path).name == "gdaxi.csv"
    assert ih.cache_path("^N225", tmp_path).parent == tmp_path


def test_every_declared_market_has_a_name_and_a_currency():
    for sym, value in ih.MARKETS.items():
        assert sym.startswith("^")
        name, ccy = value
        assert name and len(ccy) == 3


def test_load_writes_a_cache_then_reads_it_back_without_refetching(tmp_path, monkeypatch, closes):
    calls = []

    def fake_fetch(symbol, timeout=40):
        calls.append(symbol)
        return closes

    monkeypatch.setattr(ih, "fetch_daily_closes", fake_fetch)
    first = ih.load_daily_closes("^GSPC", cache_dir=tmp_path, sleep=0.0)
    assert calls == ["^GSPC"]
    assert ih.cache_path("^GSPC", tmp_path).exists()

    second = ih.load_daily_closes("^GSPC", cache_dir=tmp_path, sleep=0.0)
    assert calls == ["^GSPC"]  # served from cache, not refetched
    pd.testing.assert_series_equal(first, second, check_names=False)


def test_refresh_refetches_even_when_the_cache_exists(tmp_path, monkeypatch, closes):
    calls = []
    monkeypatch.setattr(ih, "fetch_daily_closes", lambda symbol, timeout=40: calls.append(symbol) or closes)
    ih.load_daily_closes("^GSPC", cache_dir=tmp_path, sleep=0.0)
    ih.load_daily_closes("^GSPC", cache_dir=tmp_path, refresh=True, sleep=0.0)
    assert calls == ["^GSPC", "^GSPC"]


def test_cache_file_records_the_fetch_date_so_a_rerun_is_reproducible(tmp_path, monkeypatch, closes):
    monkeypatch.setattr(ih, "fetch_daily_closes", lambda symbol, timeout=40: closes)
    ih.load_daily_closes("^GSPC", cache_dir=tmp_path, sleep=0.0)
    header = ih.cache_path("^GSPC", tmp_path).read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("# ^GSPC S&P 500 (USD)")
    assert "fetched 20" in header


def test_cached_index_round_trips_as_utc_dates(tmp_path, monkeypatch, closes):
    monkeypatch.setattr(ih, "fetch_daily_closes", lambda symbol, timeout=40: closes)
    ih.load_daily_closes("^GSPC", cache_dir=tmp_path, sleep=0.0)
    back = ih.load_daily_closes("^GSPC", cache_dir=tmp_path, sleep=0.0)
    assert str(back.index.tz) == "UTC"
    assert list(back.index.strftime("%Y-%m-%d")) == ["2020-01-02", "2020-01-03", "2020-01-06"]
