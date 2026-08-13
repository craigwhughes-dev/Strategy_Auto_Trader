from __future__ import annotations

from concurrent.futures import Future
from unittest import mock

import pandas as pd
import pytest

from Strategy_Auto_Trader.output.journal import TradeRecord
from Strategy_Auto_Trader.quant_hmm.ticker_ranking import (
    Candidate,
    filter_candidates_by_top_tickers,
    generate_candidates,
    rank_universe,
    recent_win_rate,
    run_ticker_backtest,
    ticker_ranking_score,
    trend_quality_asof,
)


class TestRunTickerBacktestSource:
    def test_source_reaches_fetch_hourly_cached(self):
        """source="ibkr" must reach fetch_hourly_cached, not just be accepted
        and dropped — this is what lets a full-universe live_sim revalidation
        run against the same local IBKR-backed cache the live daemon uses."""
        with mock.patch("Strategy_Auto_Trader.quant_hmm.ticker_ranking.fetch_hourly_cached",
                        return_value=None) as mock_fetch:
            run_ticker_backtest("AAPL", "default", source="ibkr")

        mock_fetch.assert_called_once_with("AAPL", period="730d", source="ibkr")

    def test_default_source_is_yfinance(self):
        with mock.patch("Strategy_Auto_Trader.quant_hmm.ticker_ranking.fetch_hourly_cached",
                        return_value=None) as mock_fetch:
            run_ticker_backtest("AAPL", "default")

        mock_fetch.assert_called_once_with("AAPL", period="730d", source="yfinance")


class _ImmediateExecutor:
    """Stand-in for ProcessPoolExecutor that runs submitted work synchronously,
    in-process. Lets tests exercise the parallel dispatch/collection wiring in
    generate_candidates() without real multiprocessing (which on Windows uses
    spawn and can't pick up mock.patch effects in child processes)."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, fn, *args, **kwargs):
        fut = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - defensive
            fut.set_exception(exc)
        return fut


@pytest.fixture
def base_record():
    """Template TradeRecord for constructing candidates."""
    return TradeRecord(
        date_opened="2026-01-12",
        ticker="TEST",
        strategy="test",
        entry_score=1.0,
        kelly_fraction=0.1,
        return_pct=0.05,
    )


@pytest.fixture
def ts_base():
    """Base timestamp for relative date construction."""
    return pd.Timestamp("2026-01-12", tz="UTC")


def make_candidate(ticker, day_offset, entry_score, kelly_fraction, return_pct, record, ts_base):
    """Helper to construct a Candidate dataclass."""
    date_opened = ts_base + pd.Timedelta(days=day_offset)
    date_closed = date_opened + pd.Timedelta(days=5)

    return Candidate(
        ticker=ticker,
        date_opened=date_opened,
        date_closed=date_closed,
        entry_score=entry_score,
        kelly_fraction=kelly_fraction,
        return_pct=return_pct,
        record=record,
    )


class TestGenerateCandidates:

    def test_sequential_and_parallel_dispatch_produce_identical_candidates(self, base_record, ts_base):
        """generate_candidates(workers=1) and workers>1 must return the same
        candidates from the same underlying per-ticker function — regression
        guard for the parallel dispatch/collection wiring."""

        def fake_fetch(ticker, strategy_name, vol_filter_tag, vol_filter_ok=True,
                      use_seasonal_volume=False, source="yfinance"):
            rec = TradeRecord(date_opened="2026-01-12", ticker=ticker, strategy=strategy_name,
                               entry_score=1.0, kelly_fraction=0.1, return_pct=0.05)
            cand = Candidate(
                ticker=ticker, date_opened=ts_base, date_closed=ts_base + pd.Timedelta(days=1),
                entry_score=1.0, kelly_fraction=0.1, return_pct=0.05, record=rec,
            )
            close = pd.Series([100.0, 101.0], index=[ts_base, ts_base + pd.Timedelta(days=1)])
            trend_quality = pd.Series([0.5, 0.5], index=[ts_base, ts_base + pd.Timedelta(days=1)])
            return [cand], close, trend_quality

        tickers = ["A", "B", "C"]

        with mock.patch("Strategy_Auto_Trader.quant_hmm.ticker_ranking.fetch_extract_and_prices", side_effect=fake_fetch):
            seq_candidates, seq_prices, seq_tq = generate_candidates(tickers, "test", workers=1)

        with mock.patch("Strategy_Auto_Trader.quant_hmm.ticker_ranking.fetch_extract_and_prices", side_effect=fake_fetch):
            with mock.patch("Strategy_Auto_Trader.quant_hmm.ticker_ranking.ProcessPoolExecutor", _ImmediateExecutor):
                par_candidates, par_prices, par_tq = generate_candidates(tickers, "test", workers=4)

        assert {c.ticker for c in seq_candidates} == {c.ticker for c in par_candidates} == set(tickers)
        assert len(seq_candidates) == len(par_candidates) == 3
        assert set(seq_prices.keys()) == set(par_prices.keys()) == set(tickers)
        assert set(seq_tq.keys()) == set(par_tq.keys()) == set(tickers)

    def test_workers_one_does_not_use_process_pool(self):
        with mock.patch("Strategy_Auto_Trader.quant_hmm.ticker_ranking.fetch_extract_and_prices", return_value=([], None, None)):
            with mock.patch("Strategy_Auto_Trader.quant_hmm.ticker_ranking.ProcessPoolExecutor") as mock_pool:
                generate_candidates(["A"], "test", workers=1)

        mock_pool.assert_not_called()


class TestRecentWinRate:

    def test_no_candidates_for_ticker_defaults_half(self, base_record, ts_base):
        assert recent_win_rate([], "AAPL") == 0.5

    def test_win_rate_computed_over_lookback_window(self, base_record, ts_base):
        win = make_candidate("AAPL", 0, 1.0, 0.1, 0.05, base_record, ts_base)
        loss = make_candidate("AAPL", 1, 1.0, 0.1, -0.05, base_record, ts_base)
        assert recent_win_rate([win, loss], "AAPL", lookback_days=60) == 0.5

    def test_outside_lookback_window_excluded(self, base_record, ts_base):
        recent_win = make_candidate("AAPL", 0, 1.0, 0.1, 0.05, base_record, ts_base)
        old_loss = make_candidate("AAPL", -120, 1.0, 0.1, -0.05, base_record, ts_base)
        assert recent_win_rate([recent_win, old_loss], "AAPL", lookback_days=60) == 1.0


class TestFilterCandidatesByTopTickers:

    def test_top_k_zero_or_negative_is_noop(self, base_record, ts_base):
        cand = make_candidate("AAPL", 0, 1.0, 0.1, 0.05, base_record, ts_base)
        filtered, scores = filter_candidates_by_top_tickers([cand], {}, top_k=0)
        assert filtered == [cand]
        assert scores == {}

    def test_keeps_only_top_k_tickers_by_median_score(self, base_record, ts_base):
        strong = make_candidate("STRONG", 0, 1.0, 0.1, 0.10, base_record, ts_base)
        weak = make_candidate("WEAK", 0, 1.0, 0.1, -0.10, base_record, ts_base)
        trend_quality = {
            "STRONG": pd.Series([0.9], index=[ts_base]),
            "WEAK": pd.Series([0.1], index=[ts_base]),
        }
        filtered, scores = filter_candidates_by_top_tickers(
            [strong, weak], trend_quality, top_k=1, vol_weight=0.7, win_rate_weight=0.3,
        )
        assert [c.ticker for c in filtered] == ["STRONG"]
        assert scores["STRONG"] > scores["WEAK"]


class TestRankUniverse:

    def test_zero_candidate_tickers_excluded_from_scores(self, base_record, ts_base):
        """A ticker the strategy never trades has no demonstrated edge — it
        must be dropped from the ranking entirely, not backfilled with a
        trend_quality-only estimate (see rank_universe's docstring)."""
        traded = make_candidate("TRADED", 0, 1.0, 0.1, 0.10, base_record, ts_base)

        with mock.patch(
            "Strategy_Auto_Trader.quant_hmm.ticker_ranking.generate_candidates",
            return_value=([traded], {}, {"TRADED": pd.Series([0.5], index=[ts_base])}),
        ):
            scores = rank_universe(["TRADED", "NEVER_TRADED"], "test", workers=1)

        assert "TRADED" in scores
        assert "NEVER_TRADED" not in scores

    def test_scores_match_filter_candidates_by_top_tickers_exactly(self, base_record, ts_base):
        """rank_universe must reuse filter_candidates_by_top_tickers's own
        scoring path — no drift from what live_sim.py --top-k validates."""
        cand_a = make_candidate("A", 0, 1.0, 0.1, 0.10, base_record, ts_base)
        cand_b = make_candidate("B", 0, 1.0, 0.1, -0.10, base_record, ts_base)
        trend_quality = {
            "A": pd.Series([0.8], index=[ts_base]),
            "B": pd.Series([0.3], index=[ts_base]),
        }

        with mock.patch(
            "Strategy_Auto_Trader.quant_hmm.ticker_ranking.generate_candidates",
            return_value=([cand_a, cand_b], {}, trend_quality),
        ):
            rank_scores = rank_universe(["A", "B"], "test", workers=1)

        _, direct_scores = filter_candidates_by_top_tickers(
            [cand_a, cand_b], trend_quality, top_k=2,
        )

        assert rank_scores == direct_scores


class TestTrendQualityAsof:

    def test_no_series_returns_none(self):
        assert trend_quality_asof("AAPL", pd.Timestamp("2026-01-12"), {}) is None

    def test_looks_up_last_known_score_at_or_before(self):
        series = pd.Series([0.3, 0.6], index=[pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-10")])
        score = trend_quality_asof("AAPL", pd.Timestamp("2026-01-12"), {"AAPL": series})
        assert score == 0.6


class TestTickerRankingScore:

    def test_missing_trend_quality_defaults_to_half(self, base_record, ts_base):
        cand = make_candidate("AAPL", 0, 1.0, 0.1, 0.05, base_record, ts_base)
        score = ticker_ranking_score("AAPL", [cand], {}, ts_base, vol_weight=0.7, win_rate_weight=0.3)
        # tq defaults 0.5, win_rate for a single winning trade = 1.0
        assert score == pytest.approx(0.7 * 0.5 + 0.3 * 1.0)
