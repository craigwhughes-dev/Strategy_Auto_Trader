"""Tests for markov_cli.full_scan — universe building, indicator
augmentation, daily snapshots, and summary-row alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Strategy_Auto_Trader.markov_cli import full_scan
from Strategy_Auto_Trader.output.journal import TradeRecord


def _hourly_index(n: int, start: str = "2026-01-05 09:00", freq: str = "h") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq=freq, tz="Europe/London")


def _price_df(n: int = 300) -> pd.DataFrame:
    idx = _hourly_index(n)
    rng = np.random.default_rng(42)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, n)), index=idx)
    return pd.DataFrame({
        "Open": close.shift(1).fillna(close.iloc[0]),
        "High": close + 0.5,
        "Low": close - 0.5,
        "Close": close,
        "Volume": rng.integers(1_000, 10_000, n).astype(float),
    })


def _detail_df(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal engine-shaped detail frame aligned to df's index."""
    n = len(df)
    return pd.DataFrame({
        "close": df["Close"].values,
        "trade_event": [""] * n,
        "sell_reason": [""] * n,
        "strategy_return": np.zeros(n),
        "signal_score": np.zeros(n),
    }, index=df.index)


class TestAugmentDetail:
    def test_adds_all_indicator_columns(self):
        df = _price_df()
        out = full_scan.augment_detail(_detail_df(df), df)
        for col in ["sma20", "sma50", "sma200", "rolling_vol_20",
                    "macd", "macd_signal", "macd_hist", "macd_bear_cross",
                    "bb_mid", "bb_upper", "bb_lower", "bb_width",
                    "atr14", "atr_ratio", "consolidation", "sar",
                    "dist_sma20_pct", "dist_sma50_pct", "dist_sma200_pct",
                    "ret_1d_pct", "ret_5d_pct", "ret_20d_pct",
                    "volume", "open", "high", "low"]:
            assert col in out.columns, col

    def test_preserves_engine_columns_and_length(self):
        df = _price_df()
        detail = _detail_df(df)
        out = full_scan.augment_detail(detail, df)
        assert len(out) == len(detail)
        assert "signal_score" in out.columns

    def test_sma_values_match_rolling_mean(self):
        df = _price_df()
        out = full_scan.augment_detail(_detail_df(df), df)
        expected = df["Close"].rolling(20).mean().iloc[-1]
        assert out["sma20"].iloc[-1] == pytest.approx(expected, rel=1e-6)

    def test_no_volume_column_ok(self):
        df = _price_df().drop(columns=["Volume", "Open", "High", "Low"])
        out = full_scan.augment_detail(_detail_df(df), df)
        assert "volume" not in out.columns
        assert "sma20" in out.columns

    def test_dist_pct_sign(self):
        df = _price_df()
        out = full_scan.augment_detail(_detail_df(df), df).dropna(subset=["sma20"])
        above = out[out["close"] > out["sma20"]]
        assert (above["dist_sma20_pct"] > 0).all()


class TestDailySnapshot:
    def test_one_row_per_calendar_day(self):
        df = _price_df(48)  # 2 calendar days of 24 hourly bars
        hourly = full_scan.augment_detail(_detail_df(df), df)
        daily = full_scan.daily_snapshot(hourly)
        expected_days = pd.DatetimeIndex(df.index).tz_localize(None).normalize().nunique()
        assert len(daily) == expected_days

    def test_midday_trade_event_not_lost(self):
        df = _price_df(24)
        detail = _detail_df(df)
        detail.iloc[3, detail.columns.get_loc("trade_event")] = "BUY"
        detail.iloc[10, detail.columns.get_loc("trade_event")] = "SELL"
        daily = full_scan.daily_snapshot(detail)
        assert daily["day_trade_events"].iloc[0] == "BUY+SELL"
        assert daily["trade_event"].iloc[0] == "BUY+SELL"

    def test_day_return_and_range(self):
        df = _price_df(24)
        daily = full_scan.daily_snapshot(_detail_df(df))
        close = df["Close"]
        day0 = close[close.index.normalize() == close.index[0].normalize()]
        assert daily["day_close_high"].iloc[0] == pytest.approx(day0.max())
        assert daily["day_close_low"].iloc[0] == pytest.approx(day0.min())
        expected_ret = (day0.iloc[-1] / day0.iloc[0] - 1) * 100
        assert daily["day_return_pct"].iloc[0] == pytest.approx(expected_ret, abs=1e-3)

    def test_single_bar_day(self):
        df = _price_df(1)
        daily = full_scan.daily_snapshot(_detail_df(df))
        assert len(daily) == 1
        assert daily["n_bars"].iloc[0] == 1
        assert daily["day_return_pct"].iloc[0] == pytest.approx(0.0)

    def test_day_strategy_return_compounds(self):
        df = _price_df(3)
        detail = _detail_df(df)
        detail["strategy_return"] = [0.01, 0.01, 0.0]
        daily = full_scan.daily_snapshot(detail)
        assert daily["day_strategy_return"].iloc[0] == pytest.approx(1.01 * 1.01 - 1, abs=1e-6)


class TestSummaryRow:
    def test_partial_and_full_rows_align(self, tmp_path, monkeypatch):
        monkeypatch.setattr(full_scan, "SCAN_DIR", tmp_path)
        full_scan._append_summary_row({"ticker": "AAA", "status": "no_data"})
        full_scan._append_summary_row(
            {"ticker": "BBB", "status": "ok", "total_pl": 12.5, "n_buys": 3})
        s = pd.read_csv(tmp_path / "summary.csv")
        assert len(s) == 2
        assert list(s.columns) == full_scan._SUMMARY_COLUMNS
        assert s["total_pl"].iloc[1] == pytest.approx(12.5)
        assert pd.isna(s["total_pl"].iloc[0])

    def test_summary_columns_cover_stat_and_vol_keys(self):
        for k in full_scan._SUMMARY_STAT_KEYS + full_scan._VOL_PROFILE_KEYS:
            assert k in full_scan._SUMMARY_COLUMNS


class TestScanPaths:
    def test_paths_namespaced_by_strategy(self, tmp_path, monkeypatch):
        monkeypatch.setattr(full_scan, "SCAN_DIR", tmp_path / "reports")
        monkeypatch.setattr(full_scan, "JOURNAL_DIR", tmp_path / "journals")
        hourly, daily, journal = full_scan._scan_paths("default", "AAA")
        assert hourly == tmp_path / "reports" / "default" / "hourly" / "AAA.csv"
        assert daily == tmp_path / "reports" / "default" / "daily" / "AAA.csv"
        assert journal == tmp_path / "journals" / "default" / "AAA.csv"

    def test_paths_differ_across_strategies(self):
        h1, d1, j1 = full_scan._scan_paths("default", "AAA")
        h2, d2, j2 = full_scan._scan_paths("conservative", "AAA")
        assert h1 != h2
        assert d1 != d2
        assert j1 != j2

    def test_ticker_slashes_sanitized(self):
        hourly, daily, journal = full_scan._scan_paths("default", "BRK/B")
        assert hourly.name == "BRK-B.csv"
        assert daily.name == "BRK-B.csv"
        assert journal.name == "BRK-B.csv"

    def test_uk_ticker_dot_l_preserved(self):
        hourly, _, _ = full_scan._scan_paths("default", "SHEL.L")
        assert hourly.name == "SHEL.L.csv"


class TestUniverse:
    def _fake_tables(self, col: str, symbols: list[str]) -> list[pd.DataFrame]:
        return [pd.DataFrame({col: symbols, "Name": ["x"] * len(symbols)})]

    def test_sp500_symbol_mapping(self, monkeypatch):
        syms = [f"T{i}" for i in range(500)] + ["BRK.B", "BF.B"]
        monkeypatch.setattr(full_scan, "_wiki_tables",
                            lambda url: self._fake_tables("Symbol", syms))
        out = full_scan._sp500_tickers()
        assert "BRK-B" in out and "BF-B" in out
        assert "BRK.B" not in out

    def test_ftse_symbol_mapping(self, monkeypatch):
        syms = [f"T{i}" for i in range(95)] + ["BT.A", "SHEL", "AV."]
        monkeypatch.setattr(full_scan, "_wiki_tables",
                            lambda url: self._fake_tables("Ticker", syms))
        out = full_scan._ftse100_tickers()
        assert "BT-A.L" in out and "SHEL.L" in out and "AV.L" in out

    def test_small_table_rejected(self, monkeypatch):
        monkeypatch.setattr(full_scan, "_wiki_tables",
                            lambda url: self._fake_tables("Symbol", ["ONLY", "TWO"]))
        with pytest.raises(RuntimeError):
            full_scan._sp500_tickers()

    def test_watchlist_union(self, tmp_path, monkeypatch):
        cfg = tmp_path / "config"
        cfg.mkdir()
        (cfg / "watchlist_a.json").write_text(
            '{"tickers": [{"ticker": "AAA"}, {"ticker": "BBB"}]}', encoding="utf-8")
        (cfg / "watchlist_b.json").write_text(
            '{"tickers": ["BBB", "CCC"]}', encoding="utf-8")
        monkeypatch.setattr(full_scan, "ROOT", tmp_path)
        assert full_scan._watchlist_tickers() == ["AAA", "BBB", "CCC"]

    def test_build_sp_ftse_universe_no_watchlist_union(self, tmp_path, monkeypatch):
        monkeypatch.setattr(full_scan, "_sp500_tickers", lambda: ["AAPL", "MSFT"])
        monkeypatch.setattr(full_scan, "_ftse100_tickers", lambda: ["SHEL.L"])
        out_path = tmp_path / "universe_sp_ftse.json"
        tickers = full_scan.build_sp_ftse_universe(out_path)
        assert tickers == ["SHEL.L", "AAPL", "MSFT"]
        assert out_path.exists()

    def test_load_sp_ftse_universe(self, tmp_path, monkeypatch):
        out_path = tmp_path / "universe_sp_ftse.json"
        monkeypatch.setattr(full_scan, "SP_FTSE_UNIVERSE_FILE", out_path)
        monkeypatch.setattr(full_scan, "_sp500_tickers", lambda: ["AAPL"])
        monkeypatch.setattr(full_scan, "_ftse100_tickers", lambda: ["SHEL.L"])
        full_scan.build_sp_ftse_universe(out_path)
        assert full_scan.load_sp_ftse_universe() == ["SHEL.L", "AAPL"]


def _trade(pnl_usd, return_pct, opened, closed) -> TradeRecord:
    return TradeRecord(
        date_opened=opened, ticker="T", date_closed=closed,
        pnl_usd=pnl_usd, return_pct=return_pct,
    )


class TestTradeAggregates:
    def test_empty_trades(self):
        agg = full_scan._trade_aggregates([], {"n_buys": 0, "n_sells": 0}, 10.0, 30)
        assert agg["n_trades_total"] == 0
        assert agg["win_rate_pct"] == 0.0
        assert agg["transaction_costs_total"] == 0.0

    def test_win_rate_and_profit_factor(self):
        trades = [
            _trade(100.0, 0.05, "2026-01-01 09:00", "2026-01-02 09:00"),
            _trade(-50.0, -0.02, "2026-01-03 09:00", "2026-01-03 15:00"),
            _trade(-25.0, -0.01, "2026-01-04 09:00", "2026-01-04 12:00"),
        ]
        agg = full_scan._trade_aggregates(trades, {"n_buys": 3, "n_sells": 3}, 10.0, 30)
        assert agg["n_trades_total"] == 3
        assert agg["win_rate_pct"] == pytest.approx(100 / 3, abs=0.01)
        assert agg["profit_factor"] == pytest.approx(100 / 75, abs=1e-3)
        assert agg["transaction_costs_total"] == pytest.approx(60.0)

    def test_all_losers_profit_factor_zero(self):
        trades = [_trade(-10.0, -0.01, "2026-01-01 09:00", "2026-01-01 12:00")]
        agg = full_scan._trade_aggregates(trades, {"n_buys": 1, "n_sells": 1}, 10.0, 30)
        assert agg["profit_factor"] == pytest.approx(0.0)

    def test_all_winners_profit_factor_nan(self):
        """No losing trades makes profit factor (gross_win / gross_loss) undefined."""
        trades = [_trade(10.0, 0.01, "2026-01-01 09:00", "2026-01-01 12:00")]
        agg = full_scan._trade_aggregates(trades, {"n_buys": 1, "n_sells": 1}, 10.0, 30)
        assert np.isnan(agg["profit_factor"])

    def test_avg_hours_held(self):
        trades = [_trade(1.0, 0.01, "2026-01-01 09:00", "2026-01-01 19:00")]  # 10h
        agg = full_scan._trade_aggregates(trades, {"n_buys": 1, "n_sells": 1}, 0.0, 1)
        assert agg["avg_hours_held"] == pytest.approx(10.0)
        assert agg["days_in_market_pct"] == pytest.approx(100 * (10 / 24) / 1, abs=0.1)

    def test_days_covered_zero_no_division_error(self):
        trades = [_trade(1.0, 0.01, "2026-01-01 09:00", "2026-01-01 19:00")]
        agg = full_scan._trade_aggregates(trades, {"n_buys": 1, "n_sells": 1}, 0.0, 0)
        assert agg["days_in_market_pct"] == 0.0

    def test_unparseable_dates_skipped(self):
        trades = [_trade(1.0, 0.01, "", "")]
        agg = full_scan._trade_aggregates(trades, {"n_buys": 1, "n_sells": 1}, 0.0, 10)
        assert agg["avg_hours_held"] == 0.0


class TestTickerSentiment:
    def test_disabled_returns_all_none(self):
        out = full_scan._ticker_sentiment("SHEL.L", enabled=False)
        assert set(out.keys()) == set(full_scan._SENTIMENT_KEYS)
        assert all(v is None for v in out.values())

    def test_enabled_flattens_groups(self, monkeypatch):
        fake = {
            "options": {"iv_rank": 50.0, "iv_current": 0.3, "iv_signal": 0,
                        "put_call_ratio": 1.0, "put_call_signal": 0, "skew": 0.01},
            "vix": {"vix_current": 15.0, "vix_sma20": 14.0, "vix_regime": "normal",
                    "vix_signal": 0, "vix_term_structure": "contango"},
            "insider": {"insider_buys_90d": 2, "insider_sells_90d": 0,
                       "insider_net": 2, "insider_signal": 1, "insider_total_value": 1000.0},
            "short_interest": {"short_pct_float": 5.0, "short_ratio": 1.5, "short_signal": 0},
            "sentiment_score": 0.4, "sentiment_label": "bullish", "confidence": 3,
        }
        monkeypatch.setattr(full_scan.sentiment_mod, "composite_sentiment",
                           lambda t, **kwargs: fake)
        out = full_scan._ticker_sentiment("AAA", enabled=True)
        assert out["vix_current"] == 15.0
        assert out["insider_net"] == 2
        assert out["sentiment_label"] == "bullish"
        assert set(out.keys()) == set(full_scan._SENTIMENT_KEYS)


def _hourly_for_journal() -> pd.DataFrame:
    """Small hourly frame covering the trade timestamps used in these tests,
    with a vol-profile scalar and a day-only column already set (mirroring
    what scan_ticker does before calling _write_ticker_journal)."""
    idx = pd.date_range("2026-01-01 08:00", periods=8, freq="h")
    hourly = pd.DataFrame({"close": np.linspace(100, 107, 8), "rsi": np.linspace(50, 57, 8)}, index=idx)
    hourly["trend_quality"] = 0.5          # a _VOL_PROFILE_KEYS member
    hourly["n_bars"] = np.nan              # a _DAILY_ONLY_KEYS member
    return hourly


class TestSnapshotColumns:
    def test_excludes_vol_profile_and_daily_only_keys(self):
        hourly = _hourly_for_journal()
        cols = full_scan.snapshot_columns(hourly)
        assert "close" in cols and "rsi" in cols
        assert "trend_quality" not in cols
        assert "n_bars" not in cols


class TestBarSnapshot:
    def test_prefixes_and_picks_nearest_bar(self):
        hourly = _hourly_for_journal()
        cols = full_scan.snapshot_columns(hourly)
        snap = full_scan.bar_snapshot(hourly, "2026-01-01 09:00", cols, "entry_")
        assert snap["entry_close"] == hourly.loc["2026-01-01 09:00", "close"]
        assert snap["entry_rsi"] == hourly.loc["2026-01-01 09:00", "rsi"]

    def test_bad_timestamp_returns_none_for_all_columns(self):
        hourly = _hourly_for_journal()
        cols = full_scan.snapshot_columns(hourly)
        snap = full_scan.bar_snapshot(hourly, "not-a-date", cols, "exit_")
        assert all(v is None for v in snap.values())


class TestWriteTickerJournal:
    def test_writes_repeated_scalar_columns(self, tmp_path):
        trades = [
            _trade(10.0, 0.01, "2026-01-01 09:00", "2026-01-01 12:00"),
            _trade(-5.0, -0.01, "2026-01-01 10:00", "2026-01-01 13:00"),
        ]
        path = tmp_path / "AAA.csv"
        hourly = _hourly_for_journal()
        full_scan._write_ticker_journal(
            path, trades, {"trend_quality": 0.5, "sharpe_strategy": 1.2}, hourly)
        out = pd.read_csv(path)
        assert len(out) == 2
        assert (out["trend_quality"] == 0.5).all()
        assert (out["sharpe_strategy"] == 1.2).all()
        assert "pnl_usd" in out.columns

    def test_adds_entry_and_exit_snapshot_columns(self, tmp_path):
        trades = [_trade(10.0, 0.01, "2026-01-01 09:00", "2026-01-01 12:00")]
        path = tmp_path / "AAA.csv"
        hourly = _hourly_for_journal()
        full_scan._write_ticker_journal(path, trades, {"trend_quality": 0.5}, hourly)
        out = pd.read_csv(path)
        assert out.loc[0, "entry_close"] == pytest.approx(hourly.loc["2026-01-01 09:00", "close"])
        assert out.loc[0, "exit_close"] == pytest.approx(hourly.loc["2026-01-01 12:00", "close"])
        assert "entry_n_bars" not in out.columns   # day-only key excluded
        assert "entry_trend_quality" not in out.columns  # vol-profile key excluded

    def test_empty_trades_writes_header_only_csv(self, tmp_path):
        path = tmp_path / "AAA.csv"
        hourly = _hourly_for_journal()
        full_scan._write_ticker_journal(path, [], {"trend_quality": 0.5}, hourly)
        out = pd.read_csv(path)
        assert out.empty
        assert "trend_quality" in out.columns
        assert "ticker" in out.columns  # TradeRecord field
        assert "entry_close" in out.columns
        assert "exit_close" in out.columns


class TestCombineJournals:
    def test_concatenates_per_ticker_files(self, tmp_path):
        jdir = tmp_path / "journals"
        jdir.mkdir()
        pd.DataFrame({"ticker": ["AAA"], "pnl_usd": [1.0]}).to_csv(jdir / "AAA.csv", index=False)
        pd.DataFrame({"ticker": ["BBB"], "pnl_usd": [2.0]}).to_csv(jdir / "BBB.csv", index=False)
        out_path = full_scan.combine_journals(jdir, tmp_path / "combined.csv")
        combined = pd.read_csv(out_path)
        assert len(combined) == 2
        assert set(combined["ticker"]) == {"AAA", "BBB"}

    def test_no_journals_writes_empty(self, tmp_path):
        jdir = tmp_path / "journals"
        jdir.mkdir()
        out_path = full_scan.combine_journals(jdir, tmp_path / "combined.csv")
        assert out_path.exists()


class TestSortSummary:
    def test_sorts_by_ticker_then_strategy(self, tmp_path, monkeypatch):
        monkeypatch.setattr(full_scan, "SCAN_DIR", tmp_path)
        full_scan._append_summary_row({"ticker": "BBB", "strategy": "default", "status": "ok"})
        full_scan._append_summary_row({"ticker": "AAA", "strategy": "conservative", "status": "ok"})
        full_scan._append_summary_row({"ticker": "AAA", "strategy": "default", "status": "ok"})
        sorted_df = full_scan.sort_summary(tmp_path / "summary.csv")
        assert len(sorted_df) == 3
        assert sorted_df.iloc[0]["ticker"] == "AAA" and sorted_df.iloc[0]["strategy"] == "conservative"
        assert sorted_df.iloc[1]["ticker"] == "AAA" and sorted_df.iloc[1]["strategy"] == "default"
        assert sorted_df.iloc[2]["ticker"] == "BBB" and sorted_df.iloc[2]["strategy"] == "default"

    def test_empty_summary_returns_empty_dataframe(self, tmp_path, monkeypatch):
        monkeypatch.setattr(full_scan, "SCAN_DIR", tmp_path)
        df = full_scan.sort_summary(tmp_path / "summary.csv")
        assert df.empty


class TestAtomicWrites:
    def test_hourly_csv_atomic_write_tmp_replaced(self, tmp_path, monkeypatch):
        monkeypatch.setattr(full_scan, "SCAN_DIR", tmp_path / "reports")
        monkeypatch.setattr(full_scan, "HMM_CACHE_DIR", tmp_path / "hmm")
        hourly_path = tmp_path / "reports" / "default" / "hourly" / "TEST.csv"
        hourly = _hourly_for_journal()
        hourly_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = hourly_path.with_suffix(hourly_path.suffix + ".tmp")
        hourly.to_csv(tmp)
        tmp.replace(hourly_path)
        assert hourly_path.exists()
        assert not tmp.exists()

    def test_journal_atomic_write_tmp_replaced(self, tmp_path):
        trades = [_trade(10.0, 0.01, "2026-01-01 09:00", "2026-01-01 12:00")]
        hourly = _hourly_for_journal()
        journal_path = tmp_path / "default" / "TEST.csv"
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        full_scan._write_ticker_journal(journal_path, trades, {"trend_quality": 0.5}, hourly)
        assert journal_path.exists()
        assert not journal_path.with_suffix(journal_path.suffix + ".tmp").exists()


class TestVixDataDependencyInjection:
    def test_ticker_sentiment_uses_passed_vix_data(self, monkeypatch):
        fake_vix = {"vix_current": 20.0, "vix_sma20": 18.0, "vix_regime": "elevated",
                    "vix_signal": -1, "vix_term_structure": "backwardation"}
        fake_composite = {
            "options": {}, "vix": fake_vix, "insider": {}, "short_interest": {},
            "sentiment_score": 0.0, "sentiment_label": "neutral", "confidence": 1,
        }
        monkeypatch.setattr(full_scan.sentiment_mod, "composite_sentiment",
                           lambda t, vix_data=None: fake_composite)
        out = full_scan._ticker_sentiment("TEST", enabled=True, vix_data=fake_vix)
        assert out["vix_current"] == 20.0
        assert out["vix_regime"] == "elevated"

    def test_ticker_sentiment_vix_data_none_falls_back(self, monkeypatch):
        called_with = []
        def fake_composite(t, vix_data=None):
            called_with.append(vix_data)
            return {
                "options": {}, "vix": {}, "insider": {}, "short_interest": {},
                "sentiment_score": 0.0, "sentiment_label": "neutral", "confidence": 0,
            }
        monkeypatch.setattr(full_scan.sentiment_mod, "composite_sentiment", fake_composite)
        full_scan._ticker_sentiment("TEST", enabled=True, vix_data=None)
        assert called_with[0] is None


class TestWorkersValidation:
    def test_workers_zero_rejected(self):
        with pytest.raises(SystemExit):
            full_scan.main(["--workers", "0", "--tickers", "AAA"])

    def test_workers_negative_rejected(self):
        with pytest.raises(SystemExit):
            full_scan.main(["--workers", "-1", "--tickers", "AAA"])


class TestDataCutoff:
    """scan_ticker's data_cutoff drops bars dated on or after the cutoff.

    Uses scan_ticker's fetch_fn/vol_profile_fn/backtest_fn DI seams — no
    monkeypatching."""

    @staticmethod
    def _di_kwargs(df, backtest_fn=None):
        return {
            "fetch_fn": lambda t, period="730d": df,
            "vol_profile_fn": lambda t: None,
            "backtest_fn": backtest_fn or (lambda frame, **kw: {"detail": pd.DataFrame()}),
        }

    def test_truncates_bars_at_cutoff(self):
        df = _price_df(72)  # 2026-01-05 09:00 .. 2026-01-08 08:00 (hourly)
        seen = {}

        def fake_backtest(frame, **kwargs):
            seen["last"] = frame.index[-1]
            return {"detail": pd.DataFrame()}

        cutoff = pd.Timestamp("2026-01-07").date()
        row = full_scan.scan_ticker("TEST", "default", sentiment=False, data_cutoff=cutoff,
                                    **self._di_kwargs(df, fake_backtest))
        assert seen["last"].date() < cutoff
        assert row["bars_fetched"] < 72
        assert row["last_bar"].startswith("2026-01-06")

    def test_no_cutoff_keeps_all_bars(self):
        df = _price_df(72)
        row = full_scan.scan_ticker("TEST", "default", sentiment=False, **self._di_kwargs(df))
        assert row["bars_fetched"] == 72

    def test_cutoff_before_history_gives_no_data(self):
        df = _price_df(24)
        cutoff = pd.Timestamp("2020-01-01").date()
        row = full_scan.scan_ticker("TEST", "default", sentiment=False, data_cutoff=cutoff,
                                    **self._di_kwargs(df))
        assert row["status"] == "no_data"
        assert "data_cutoff" in row["note"]

    def test_cli_rejects_bad_cutoff(self):
        with pytest.raises(SystemExit):
            full_scan.main(["--tickers", "AAA", "--data-cutoff", "not-a-date"])


class TestScanTickerWorkerError:
    """Tests for _scan_ticker_worker error handling and payload structure."""

    def test_error_payload_structure(self, monkeypatch):
        """Test that _scan_ticker_worker returns error payload with expected keys."""
        # Mock fetch to return valid data so resolve_strategy is reached.
        df = _price_df(100)
        monkeypatch.setattr(full_scan, "fetch_hourly_cached", lambda t, period: df)
        monkeypatch.setattr(full_scan, "volatility_profile_cached", lambda t: {"trend_quality": 0.5})
        # Mock resolve_strategy to raise for nonexistent strategy.
        monkeypatch.setattr(
            full_scan, "resolve_strategy",
            lambda name, vol_filter_ok: (_ for _ in ()).throw(ValueError(f"Unknown strategy: {name}"))
        )
        payload = full_scan._scan_ticker_worker("TEST", "nonexistent_xyz", 10.0, False)
        assert set(payload.keys()) == {"row", "traceback"}
        assert payload["row"]["status"] == "error"
        assert "ValueError" in payload["row"]["note"]
        assert payload["traceback"] is not None and len(payload["traceback"]) > 0
        # row should NOT have a "traceback" key; that's only in the payload
        assert "traceback" not in payload["row"]

    def test_error_row_aligns_with_summary_columns(self, monkeypatch):
        """Test that error row can be safely reindexed to _SUMMARY_COLUMNS."""
        df = _price_df(100)
        monkeypatch.setattr(full_scan, "fetch_hourly_cached", lambda t, period: df)
        monkeypatch.setattr(full_scan, "volatility_profile_cached", lambda t: {"trend_quality": 0.5})
        monkeypatch.setattr(
            full_scan, "resolve_strategy",
            lambda name, vol_filter_ok: (_ for _ in ()).throw(RuntimeError("test error"))
        )
        payload = full_scan._scan_ticker_worker("TEST", "bad_strat", 10.0, False)
        row = payload["row"]
        # This should not raise; all _SUMMARY_COLUMNS should be available (or filled with NaN).
        frame = pd.DataFrame([row]).reindex(columns=full_scan._SUMMARY_COLUMNS)
        assert len(frame) == 1
        assert frame["status"].iloc[0] == "error"
        assert pd.notna(frame["ticker"].iloc[0])


class TestOutOfOrderAppend:
    """Test that out-of-order rows are correctly sorted by sort_summary."""

    def test_sort_summary_reorders_scrambled_rows(self, tmp_path, monkeypatch):
        """Append rows in scrambled (ticker, strategy) order, then assert sort_summary returns sorted."""
        monkeypatch.setattr(full_scan, "SCAN_DIR", tmp_path)
        # Append in non-alphabetical order
        full_scan._append_summary_row({"ticker": "CCC", "strategy": "default", "status": "ok"})
        full_scan._append_summary_row({"ticker": "AAA", "strategy": "conservative", "status": "ok"})
        full_scan._append_summary_row({"ticker": "BBB", "strategy": "default", "status": "ok"})
        full_scan._append_summary_row({"ticker": "AAA", "strategy": "default", "status": "ok"})

        sorted_df = full_scan.sort_summary(tmp_path / "summary.csv")
        assert len(sorted_df) == 4
        # Verify sort order: (ticker, strategy) ascending
        assert sorted_df.iloc[0]["ticker"] == "AAA" and sorted_df.iloc[0]["strategy"] == "conservative"
        assert sorted_df.iloc[1]["ticker"] == "AAA" and sorted_df.iloc[1]["strategy"] == "default"
        assert sorted_df.iloc[2]["ticker"] == "BBB" and sorted_df.iloc[2]["strategy"] == "default"
        assert sorted_df.iloc[3]["ticker"] == "CCC" and sorted_df.iloc[3]["strategy"] == "default"
