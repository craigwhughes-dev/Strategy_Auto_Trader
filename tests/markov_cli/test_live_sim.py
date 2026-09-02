from __future__ import annotations

import json
from unittest import mock

import pandas as pd
import pytest

from Strategy_Auto_Trader.output.journal import TradeRecord


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
        entry_price=1.0,
    )


@pytest.fixture
def ts_base():
    """Base timestamp for relative date construction."""
    return pd.Timestamp("2026-01-12", tz="UTC")


def make_candidate(ticker, day_offset, entry_score, kelly_fraction, return_pct, record, ts_base):
    """Helper to construct a Candidate dataclass."""
    from Strategy_Auto_Trader.markov_cli.live_sim import Candidate

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


class TestSimulateStrategy:

    def test_empty_candidates_returns_empty_executed(self, base_record, ts_base):
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.fetch_and_extract", return_value=[]):
            result = simulate_strategy(
                tickers=["TEST"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=1000.0,
                trade_cost=1.0,
            )
        assert result == []

    def test_zero_cash_admits_nothing(self, base_record, ts_base):
        """With zero starting capital, no trades are admitted."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy

        cand = make_candidate("TEST", 0, 1.0, 0.1, 0.05, base_record, ts_base)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.fetch_and_extract", return_value=[cand]):
            result = simulate_strategy(
                tickers=["TEST"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=0.0,
                trade_cost=1.0,
            )
        assert result == []

    def test_cash_equals_trade_cost_skips_candidate(self, base_record, ts_base):
        """When cash == trade_cost exactly, candidate is skipped (cash <= trade_cost check)."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy

        cand = make_candidate("TEST", 0, 1.0, 0.1, 0.05, base_record, ts_base)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.fetch_and_extract", return_value=[cand]):
            result = simulate_strategy(
                tickers=["TEST"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=1.0,  # exactly trade_cost
                trade_cost=1.0,
            )
        assert result == []

    def test_kelly_fraction_positive_sizes_position(self, base_record, ts_base):
        """Kelly fraction > 0 sizes position as kelly_fraction * cash."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy

        rec = TradeRecord(
            date_opened="2026-01-12",
            ticker="TEST",
            strategy="test",
            entry_score=1.0,
            kelly_fraction=0.25,  # 25% of cash
            return_pct=0.10,
            entry_price=1.0,
        )
        cand = make_candidate("TEST", 0, 1.0, 0.25, 0.10, rec, ts_base)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.fetch_and_extract", return_value=[cand]):
            result = simulate_strategy(
                tickers=["TEST"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=1000.0,
                trade_cost=1.0,
            )
        assert len(result) == 1
        # alloc = 0.25 * 1000 = 250
        # pnl_usd = alloc * return_pct - 2*trade_cost = 250 * 0.10 - 2 = 23
        assert result[0].pnl_usd == pytest.approx(23.0)

    def test_kelly_zero_or_negative_rejected_not_admitted(self, base_record, ts_base):
        """kelly_fraction <= 0 is rejected outright, not sized via a flat
        fallback — matches live's PortfolioManager.compute_quantity(), which
        returns 0 for kelly_fraction <= 0 and never places the order."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy

        rec_zero = TradeRecord(
            date_opened="2026-01-12", ticker="ZERO", strategy="test",
            entry_score=1.0, kelly_fraction=0.0, return_pct=0.05, entry_price=1.0,
        )
        cand_zero = make_candidate("ZERO", 0, 1.0, 0.0, 0.05, rec_zero, ts_base)
        rec_neg = TradeRecord(
            date_opened="2026-01-12", ticker="NEG", strategy="test",
            entry_score=1.0, kelly_fraction=-0.1, return_pct=0.05, entry_price=1.0,
        )
        cand_neg = make_candidate("NEG", 0, 1.0, -0.1, 0.05, rec_neg, ts_base)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.fetch_and_extract",
                        side_effect=[[cand_zero], [cand_neg]]):
            result = simulate_strategy(
                tickers=["ZERO", "NEG"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=500.0,
                trade_cost=1.0,
            )
        assert result == []

    def test_alloc_clamped_by_available_cash(self, base_record, ts_base):
        """Allocation is clamped by (cash - trade_cost)."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy

        rec = TradeRecord(
            date_opened="2026-01-12",
            ticker="TEST",
            strategy="test",
            entry_score=1.0,
            kelly_fraction=0.5,  # Would want 50% of cash
            return_pct=0.10,
            entry_price=1.0,
        )
        cand = make_candidate("TEST", 0, 1.0, 0.5, 0.10, rec, ts_base)

        # Initial cash 100: 0.5 * 100 = 50, clamped by (100 - 1) = 99, so 50 is used
        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.fetch_and_extract", return_value=[cand]):
            result = simulate_strategy(
                tickers=["TEST"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=100.0,
                trade_cost=1.0,
            )
        assert len(result) == 1
        # alloc = min(50, 100 - 1) = 50
        # pnl_usd = 50 * 0.10 - 2*1 = 5 - 2 = 3
        assert result[0].pnl_usd == pytest.approx(3.0)

    def test_same_day_candidates_sorted_by_entry_score(self, base_record, ts_base):
        """Same-day candidates are sorted by entry_score (descending) and admitted in order."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy

        rec1 = TradeRecord(
            date_opened="2026-01-12",
            ticker="A",
            strategy="test",
            entry_score=3.0,  # Higher score
            kelly_fraction=0.1,
            return_pct=0.05,
            entry_price=1.0,
        )
        rec2 = TradeRecord(
            date_opened="2026-01-12",
            ticker="B",
            strategy="test",
            entry_score=1.0,  # Lower score
            kelly_fraction=0.1,
            return_pct=0.05,
            entry_price=1.0,
        )
        cand1 = make_candidate("A", 0, 3.0, 0.1, 0.05, rec1, ts_base)
        cand2 = make_candidate("B", 0, 1.0, 0.1, 0.05, rec2, ts_base)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.fetch_and_extract", side_effect=[[cand1], [cand2]]):
            result = simulate_strategy(
                tickers=["A", "B"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=500.0,
                trade_cost=1.0,
            )
        assert len(result) == 2
        # A admitted first (score 3.0), then B (score 1.0)
        assert result[0].ticker == "A"
        assert result[1].ticker == "B"

    def test_two_same_day_candidates_tied_score_deterministic_order(self, base_record, ts_base):
        """Two candidates with same entry_score on same day are both admitted (cash allows)."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy

        rec1 = TradeRecord(
            date_opened="2026-01-12",
            ticker="A",
            strategy="test",
            entry_score=2.0,
            kelly_fraction=0.1,
            return_pct=0.05,
            entry_price=1.0,
        )
        rec2 = TradeRecord(
            date_opened="2026-01-12",
            ticker="B",
            strategy="test",
            entry_score=2.0,  # Tied
            kelly_fraction=0.1,
            return_pct=0.05,
            entry_price=1.0,
        )
        cand1 = make_candidate("A", 0, 2.0, 0.1, 0.05, rec1, ts_base)
        cand2 = make_candidate("B", 0, 2.0, 0.1, 0.05, rec2, ts_base)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.fetch_and_extract", side_effect=[[cand1], [cand2]]):
            result = simulate_strategy(
                tickers=["A", "B"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=1000.0,
                trade_cost=1.0,
            )
        assert len(result) == 2  # Both admitted, no daily cap

    def test_cash_release_on_position_close_same_day(self, base_record, ts_base):
        """A position closing on day D frees cash for a new entry on the same day."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy, Candidate

        # Day 0: candidate opens, closes on day 0 (immediate)
        rec_close_day0 = TradeRecord(
            date_opened="2026-01-12",
            ticker="CLOSE",
            strategy="test",
            entry_score=1.0,
            kelly_fraction=0.1,
            return_pct=0.10,
            entry_price=1.0,
        )
        cand_close_day0 = Candidate(
            ticker="CLOSE",
            date_opened=ts_base,
            date_closed=ts_base,  # Closes same day
            entry_score=1.0,
            kelly_fraction=0.1,
            return_pct=0.10,
            record=rec_close_day0,
        )

        # Day 0: second candidate also opens
        rec_entry = TradeRecord(
            date_opened="2026-01-12",
            ticker="ENTRY",
            strategy="test",
            entry_score=2.0,  # Higher score to be admitted second
            kelly_fraction=0.1,
            return_pct=0.05,
            entry_price=1.0,
        )
        cand_entry = make_candidate("ENTRY", 0, 2.0, 0.1, 0.05, rec_entry, ts_base)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.fetch_and_extract", side_effect=[[cand_close_day0], [cand_entry]]):
            result = simulate_strategy(
                tickers=["CLOSE", "ENTRY"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=300.0,
                trade_cost=1.0,
            )
        # Both should be admitted: first closes immediately, freeing cash for second
        assert len(result) == 2

    def test_cash_not_released_for_future_close(self, base_record, ts_base):
        """A position closing on day D does not free cash for entry on day D-1."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy, Candidate

        # Day 0: candidate opens, closes on day 1
        rec = TradeRecord(
            date_opened="2026-01-12",
            ticker="TEST",
            strategy="test",
            entry_score=1.0,
            kelly_fraction=0.5,  # 50% of cash
            return_pct=0.10,
            entry_price=1.0,
        )
        cand = Candidate(
            ticker="TEST",
            date_opened=ts_base,
            date_closed=ts_base + pd.Timedelta(days=1),  # Closes day 1
            entry_score=1.0,
            kelly_fraction=0.5,
            return_pct=0.10,
            record=rec,
        )

        # Day 0: second candidate tries to enter (not enough cash)
        rec2 = TradeRecord(
            date_opened="2026-01-12",
            ticker="TEST2",
            strategy="test",
            entry_score=0.5,
            kelly_fraction=0.5,
            return_pct=0.05,
            entry_price=1.0,
        )
        cand2 = make_candidate("TEST2", 0, 0.5, 0.5, 0.05, rec2, ts_base)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.fetch_and_extract", return_value=[cand, cand2]):
            result = simulate_strategy(
                tickers=["TEST"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=200.0,
                trade_cost=1.0,
            )
        # First candidate takes 50% = 100, leaves 99 cash (after trade cost)
        # Second candidate needs 50% of remaining, can't fit
        assert len(result) <= 2

    def test_pnl_calculation_includes_exit_proceeds(self, base_record, ts_base):
        """PnL = (alloc * (1 + return_pct) - trade_cost) - alloc - trade_cost."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy

        rec = TradeRecord(
            date_opened="2026-01-12",
            ticker="TEST",
            strategy="test",
            entry_score=1.0,
            kelly_fraction=0.1,
            return_pct=0.20,  # 20% return
            entry_price=1.0,
        )
        cand = make_candidate("TEST", 0, 1.0, 0.1, 0.20, rec, ts_base)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.fetch_and_extract", return_value=[cand]):
            result = simulate_strategy(
                tickers=["TEST"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=1000.0,
                trade_cost=10.0,
            )
        assert len(result) == 1
        # alloc = 0.1 * 1000 = 100
        # exit_proceeds = 100 * 1.20 - 10 = 120 - 10 = 110
        # pnl_usd = 110 - 100 - 10 = 0
        assert result[0].pnl_usd == pytest.approx(0.0)

    def test_multiple_days_sequential_admission(self, base_record, ts_base):
        """Candidates on different days are processed in day order."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy

        # Day 0
        rec0 = TradeRecord(
            date_opened="2026-01-12",
            ticker="TEST0",
            strategy="test",
            entry_score=1.0,
            kelly_fraction=0.2,
            return_pct=0.10,
            entry_price=1.0,
        )
        cand0 = make_candidate("TEST0", 0, 1.0, 0.2, 0.10, rec0, ts_base)

        # Day 1
        rec1 = TradeRecord(
            date_opened="2026-01-13",
            ticker="TEST1",
            strategy="test",
            entry_score=1.0,
            kelly_fraction=0.2,
            return_pct=0.10,
            entry_price=1.0,
        )
        cand1 = make_candidate("TEST1", 1, 1.0, 0.2, 0.10, rec1, ts_base)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.fetch_and_extract", side_effect=[[cand0], [cand1]]):
            result = simulate_strategy(
                tickers=["TEST0", "TEST1"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=1000.0,
                trade_cost=1.0,
            )
        assert len(result) == 2

    def test_start_date_filter_excludes_earlier_candidates(self, base_record, ts_base):
        """Candidates before start_date are excluded."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy, Candidate

        # Day -1 (before start_date)
        rec_before = TradeRecord(
            date_opened="2026-01-11",
            ticker="BEFORE",
            strategy="test",
            entry_score=1.0,
            kelly_fraction=0.1,
            return_pct=0.05,
            entry_price=1.0,
        )
        cand_before = Candidate(
            ticker="BEFORE",
            date_opened=ts_base - pd.Timedelta(days=1),
            date_closed=ts_base + pd.Timedelta(days=4),
            entry_score=1.0,
            kelly_fraction=0.1,
            return_pct=0.05,
            record=rec_before,
        )

        # Day 0 (on start_date)
        rec_after = TradeRecord(
            date_opened="2026-01-12",
            ticker="AFTER",
            strategy="test",
            entry_score=1.0,
            kelly_fraction=0.1,
            return_pct=0.05,
            entry_price=1.0,
        )
        cand_after = make_candidate("AFTER", 0, 1.0, 0.1, 0.05, rec_after, ts_base)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.fetch_and_extract", return_value=[cand_before, cand_after]):
            result = simulate_strategy(
                tickers=["TEST"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=1000.0,
                trade_cost=1.0,
            )
        # Only AFTER should be admitted
        assert len(result) == 1
        assert result[0].ticker == "AFTER"

    def test_interest_accrues_on_idle_cash_across_day_gap(self, ts_base):
        """A long idle gap between trades earns interest, growing the pool for the next entry."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy, Candidate

        rec1 = TradeRecord(
            date_opened="2026-01-12", ticker="A", strategy="test",
            entry_score=1.0, kelly_fraction=0.2, return_pct=0.0, entry_price=1.0,
        )
        cand1 = Candidate(
            ticker="A", date_opened=ts_base, date_closed=ts_base,  # opens and closes day 0
            entry_score=1.0, kelly_fraction=0.2, return_pct=0.0, record=rec1,
        )

        rec2 = TradeRecord(
            date_opened="2026-07-31", ticker="B", strategy="test",
            entry_score=1.0, kelly_fraction=0.1, return_pct=0.10, entry_price=1.0,
        )
        cand2 = Candidate(
            ticker="B", date_opened=ts_base + pd.Timedelta(days=200),
            date_closed=ts_base + pd.Timedelta(days=205),
            entry_score=1.0, kelly_fraction=0.1, return_pct=0.10, record=rec2,
        )

        with mock.patch(
            "Strategy_Auto_Trader.markov_cli.live_sim.fetch_and_extract",
            side_effect=[[cand1], [cand2]],
        ):
            result = simulate_strategy(
                tickers=["A", "B"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=5000.0,
                trade_cost=0.0,
            )

        assert len(result) == 2
        # Idle cash after trade A (4000) earns ~200 days of GBP tier-1 interest (4%/yr)
        # before trade B is sized: cash = 4000 + 4000*0.04/365*200 + 1000(A's release) = 5087.67
        # kelly-implied alloc = 0.1 * 5087.67 = 508.77 shares at $1/share, floored to
        # 508 whole shares (entry_price=1.0 here, so 1 share == $1, isolating the
        # floor's rounding effect cleanly); pnl_B = 508 * 0.10 = 50.80
        pnl_b = result[1].pnl_usd
        assert pnl_b == pytest.approx(50.8, abs=0.01)
        # Without interest, alloc_B would be 0.1 * 5000 = 500, pnl_B = 50.00
        assert pnl_b > 50.0


class TestFilterCandidatesByDailyTrendQuality:

    def test_no_series_for_ticker_is_permissive(self, ts_base):
        """A ticker with no trend_quality series at all (e.g. fetch failure)
        is kept, not dropped — matches resolve_strategy()'s documented
        'no ticker context -> permissive' default."""
        from Strategy_Auto_Trader.markov_cli.live_sim import (
            _filter_candidates_by_daily_trend_quality, Candidate,
        )

        rec = TradeRecord(date_opened="2026-01-12", ticker="NODATA", strategy="test")
        cand = Candidate(ticker="NODATA", date_opened=ts_base, date_closed=ts_base,
                           entry_score=1.0, kelly_fraction=0.1, return_pct=0.05, record=rec)

        kept = _filter_candidates_by_daily_trend_quality(
            [cand], trend_quality_by_ticker={}, min_trend_quality=0.0, wants_low=False,
        )
        assert kept == [cand]

    def test_nan_score_before_min_periods_is_permissive(self, ts_base):
        from Strategy_Auto_Trader.markov_cli.live_sim import (
            _filter_candidates_by_daily_trend_quality, Candidate,
        )

        rec = TradeRecord(date_opened="2026-01-12", ticker="EARLY", strategy="test")
        cand = Candidate(ticker="EARLY", date_opened=ts_base, date_closed=ts_base,
                           entry_score=1.0, kelly_fraction=0.1, return_pct=0.05, record=rec)
        series = pd.Series([float("nan")], index=[ts_base])

        kept = _filter_candidates_by_daily_trend_quality(
            [cand], trend_quality_by_ticker={"EARLY": series}, min_trend_quality=0.0, wants_low=False,
        )
        assert kept == [cand]

    def test_below_threshold_dropped_above_kept_standard_direction(self, ts_base):
        from Strategy_Auto_Trader.markov_cli.live_sim import (
            _filter_candidates_by_daily_trend_quality, Candidate,
        )

        rec_choppy = TradeRecord(date_opened="2026-01-12", ticker="CHOPPY", strategy="test")
        cand_choppy = Candidate(ticker="CHOPPY", date_opened=ts_base, date_closed=ts_base,
                                  entry_score=1.0, kelly_fraction=0.1, return_pct=0.05, record=rec_choppy)
        rec_trend = TradeRecord(date_opened="2026-01-12", ticker="TRENDY", strategy="test")
        cand_trend = Candidate(ticker="TRENDY", date_opened=ts_base, date_closed=ts_base,
                                 entry_score=1.0, kelly_fraction=0.1, return_pct=0.05, record=rec_trend)

        trend_quality_by_ticker = {
            "CHOPPY": pd.Series([-1.0], index=[ts_base]),
            "TRENDY": pd.Series([1.0], index=[ts_base]),
        }

        kept = _filter_candidates_by_daily_trend_quality(
            [cand_choppy, cand_trend], trend_quality_by_ticker, min_trend_quality=0.0, wants_low=False,
        )
        assert kept == [cand_trend]

    def test_wants_low_inverts_direction(self, ts_base):
        """A choppy_vol-style strategy (wants_low=True) keeps the low-scoring
        ticker and drops the high-scoring one — inverse of the standard case."""
        from Strategy_Auto_Trader.markov_cli.live_sim import (
            _filter_candidates_by_daily_trend_quality, Candidate,
        )

        rec_choppy = TradeRecord(date_opened="2026-01-12", ticker="CHOPPY", strategy="test")
        cand_choppy = Candidate(ticker="CHOPPY", date_opened=ts_base, date_closed=ts_base,
                                  entry_score=1.0, kelly_fraction=0.1, return_pct=0.05, record=rec_choppy)
        rec_trend = TradeRecord(date_opened="2026-01-12", ticker="TRENDY", strategy="test")
        cand_trend = Candidate(ticker="TRENDY", date_opened=ts_base, date_closed=ts_base,
                                 entry_score=1.0, kelly_fraction=0.1, return_pct=0.05, record=rec_trend)

        trend_quality_by_ticker = {
            "CHOPPY": pd.Series([-1.0], index=[ts_base]),
            "TRENDY": pd.Series([1.0], index=[ts_base]),
        }

        kept = _filter_candidates_by_daily_trend_quality(
            [cand_choppy, cand_trend], trend_quality_by_ticker, min_trend_quality=0.0, wants_low=True,
        )
        assert kept == [cand_choppy]

    def test_same_ticker_different_entry_days_evaluated_independently(self):
        """The whole point of the daily rescreen: the SAME ticker can be
        in-scope on one entry day and out-of-scope on another, using only
        the trend_quality known as of each candidate's own entry day."""
        from Strategy_Auto_Trader.markov_cli.live_sim import (
            _filter_candidates_by_daily_trend_quality, Candidate,
        )

        day1 = pd.Timestamp("2026-01-01", tz="UTC")
        day2 = pd.Timestamp("2026-06-01", tz="UTC")

        rec1 = TradeRecord(date_opened="2026-01-01", ticker="SHIFT", strategy="test")
        cand_early = Candidate(ticker="SHIFT", date_opened=day1, date_closed=day1,
                                 entry_score=1.0, kelly_fraction=0.1, return_pct=0.05, record=rec1)
        rec2 = TradeRecord(date_opened="2026-06-01", ticker="SHIFT", strategy="test")
        cand_late = Candidate(ticker="SHIFT", date_opened=day2, date_closed=day2,
                                entry_score=1.0, kelly_fraction=0.1, return_pct=0.05, record=rec2)

        # Trending early in the year, choppy by June.
        series = pd.Series([1.0, -1.0], index=[day1, day2])

        kept = _filter_candidates_by_daily_trend_quality(
            [cand_early, cand_late], {"SHIFT": series}, min_trend_quality=0.0, wants_low=False,
        )
        assert kept == [cand_early]


_EMPTY_ARBITRATE_RESULT = {
    "executed": [], "equity_curve": [], "total_interest": 0.0,
    "final_cash": 0.0, "n_candidates": 0, "n_admitted": 0,
    "n_rejected_cash": 0, "n_rejected_kelly": 0, "n_rejected_concentration": 0,
    "n_rejected_vix": 0,
}


class TestMainCLI:

    @pytest.fixture(autouse=True)
    def _no_real_position_summary_writes(self):
        """main() always writes a position_summary CSV to data/journals/ when
        summary_rows is non-empty (one SUMMARY row per strategy/pot-size, even
        with empty candidates) — mock the writer so these CLI-wiring tests
        never touch the real filesystem."""
        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._write_position_summary") as m:
            yield m

    def test_main_requires_exactly_one_of_tickers_or_universe(self):
        from Strategy_Auto_Trader.markov_cli.live_sim import main

        with pytest.raises(SystemExit):
            main(["--strategies", "default"])  # neither given

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.full_scan.load_sp_ftse_universe", return_value=["A"]):
            with pytest.raises(SystemExit):
                main(["--tickers", "TEST", "--universe", "--strategies", "default"])  # both given

    def test_main_universe_flag_loads_sp_ftse_universe(self):
        from Strategy_Auto_Trader.markov_cli.live_sim import main

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.full_scan.load_sp_ftse_universe", return_value=["U1", "U2"]) as mock_universe:
            with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.generate_candidates", return_value=([], {}, {})) as mock_gen:
                with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.append_trades", return_value=0):
                    main(["--universe", "--strategies", "default"])

        mock_universe.assert_called_once()
        assert mock_gen.call_args_list[0][1]["tickers"] == ["U1", "U2"]

    def test_main_no_vol_filter_skips_daily_filtering(self):
        from Strategy_Auto_Trader.markov_cli.live_sim import main

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.generate_candidates", return_value=([], {}, {})):
            with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._filter_candidates_by_daily_trend_quality") as mock_filter:
                with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.append_trades", return_value=0):
                    main([
                        "--tickers", "TEST1", "TEST2",
                        "--strategies", "default",
                        "--no-vol-filter",
                    ])

        mock_filter.assert_not_called()

    def test_main_vol_filter_exempt_strategy_all_get_full_ticker_list(self):
        """Both strategies always get the full ticker list now — the vol veto
        no longer decides which tickers get backtested (that was the static-
        snapshot bug being fixed); exempt only means "skip the daily filter
        step after candidates come back", not "get a different ticker list"."""
        from Strategy_Auto_Trader.markov_cli.live_sim import main

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.generate_candidates", return_value=([], {}, {})) as mock_gen:
            with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._filter_candidates_by_daily_trend_quality", return_value=[]) as mock_filter:
                with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.append_trades", return_value=0):
                    main([
                        "--tickers", "TEST1", "TEST2",
                        "--strategies", "default", "exempt_strat",
                        "--vol-filter-exempt", "exempt_strat",
                    ])

        calls = mock_gen.call_args_list
        assert len(calls) == 2
        tickers_by_call = {c[1]["strategy_name"]: c[1]["tickers"] for c in calls}
        assert sorted(tickers_by_call["exempt_strat"]) == ["TEST1", "TEST2"]
        assert sorted(tickers_by_call["default"]) == ["TEST1", "TEST2"]

        # Daily filter runs only for the non-exempt strategy.
        assert mock_filter.call_count == 1

    def test_main_default_arguments(self):
        from Strategy_Auto_Trader.markov_cli.live_sim import main

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.generate_candidates", return_value=([], {}, {})) as mock_gen:
            with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.arbitrate", return_value=dict(_EMPTY_ARBITRATE_RESULT)) as mock_arb:
                with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.append_trades", return_value=0):
                    main(["--tickers", "TEST"])

        gen_kwargs = mock_gen.call_args_list[0][1]
        assert gen_kwargs["workers"] == 2
        assert gen_kwargs["source"] == "ibkr"

        arb_kwargs = mock_arb.call_args_list[0][1]
        assert arb_kwargs["initial_cash"] == 10_000.0
        assert arb_kwargs["trade_cost"] == 1.0
        assert "kelly_fallback" not in arb_kwargs
        assert "max_trades_per_day" not in arb_kwargs

    def test_main_source_ibkr_flag_reaches_generate_candidates(self):
        """--source ibkr must reach generate_candidates so a full-universe
        revalidation run uses the same data the live daemon trades on."""
        from Strategy_Auto_Trader.markov_cli.live_sim import main

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.generate_candidates", return_value=([], {}, {})) as mock_gen:
            with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.arbitrate", return_value=dict(_EMPTY_ARBITRATE_RESULT)):
                with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.append_trades", return_value=0):
                    main(["--tickers", "TEST", "--source", "ibkr"])

        gen_kwargs = mock_gen.call_args_list[0][1]
        assert gen_kwargs["source"] == "ibkr"

    def test_main_pot_sizes_sweep_calls_arbitrate_once_per_pot_size(self):
        from Strategy_Auto_Trader.markov_cli.live_sim import main

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.generate_candidates", return_value=([], {}, {})) as mock_gen:
            with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.arbitrate", return_value=dict(_EMPTY_ARBITRATE_RESULT)) as mock_arb:
                with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.append_trades", return_value=0):
                    main([
                        "--tickers", "TEST", "--strategies", "default",
                        "--pot-sizes", "25000", "50000", "100000",
                    ])

        # generate_candidates runs once per strategy regardless of pot-size count
        assert mock_gen.call_count == 1
        # arbitrate runs once per (strategy, pot_size) pair
        assert mock_arb.call_count == 3
        pot_sizes_used = [c[1]["initial_cash"] for c in mock_arb.call_args_list]
        assert pot_sizes_used == [25000.0, 50000.0, 100000.0]

    def test_main_resolves_same_day_cap_from_strategy_registry(self):
        """same_day_deployment_cap_pct is not a CLI flag — main() resolves it
        from the strategy's registered Entry class attribute and threads it
        into arbitrate(). A strategy whose Entry class doesn't declare the
        attribute gets None."""
        from Strategy_Auto_Trader.markov_cli.live_sim import main

        class _CappedEntry:
            same_day_deployment_cap_pct = 0.3

        fake_registry = {
            "capped": {"entry": _CappedEntry, "exit": object},
            "default": {"entry": object, "exit": object},  # no attribute -> None
        }

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.STRATEGY_REGISTRY", fake_registry):
            with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.generate_candidates", return_value=([], {}, {})):
                with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.arbitrate", return_value=dict(_EMPTY_ARBITRATE_RESULT)) as mock_arb:
                    with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.append_trades", return_value=0):
                        # --no-vol-filter skips wants_low_trend_quality(), which
                        # reads the REAL (unpatched) registry module directly and
                        # would KeyError on our fake "capped" strategy name.
                        main(["--tickers", "TEST", "--strategies", "capped", "default", "--no-vol-filter"])

        caps_by_call = [c[1]["same_day_deployment_cap_pct"] for c in mock_arb.call_args_list]
        assert caps_by_call == [0.3, None]

    def test_dump_ticker_scores_writes_json(self, base_record, ts_base, tmp_path):
        from Strategy_Auto_Trader.markov_cli.live_sim import main

        cand = make_candidate("TEST", 0, 1.0, 0.1, 0.05, base_record, ts_base)
        dump_path = tmp_path / "scores.json"

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.generate_candidates",
                        return_value=([cand], {}, {})):
            with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.arbitrate",
                            return_value=dict(_EMPTY_ARBITRATE_RESULT)):
                with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.append_trades", return_value=0):
                    main([
                        "--tickers", "TEST", "--strategies", "default",
                        "--top-k", "1",
                        "--dump-ticker-scores", str(dump_path),
                    ])

        assert dump_path.exists()
        scores = json.loads(dump_path.read_text())
        assert "TEST" in scores

    def test_dump_ticker_scores_noop_without_top_k(self, base_record, ts_base, tmp_path):
        from Strategy_Auto_Trader.markov_cli.live_sim import main

        cand = make_candidate("TEST", 0, 1.0, 0.1, 0.05, base_record, ts_base)
        dump_path = tmp_path / "scores.json"

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.generate_candidates",
                        return_value=([cand], {}, {})):
            with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.arbitrate",
                            return_value=dict(_EMPTY_ARBITRATE_RESULT)):
                with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.append_trades", return_value=0):
                    main([
                        "--tickers", "TEST", "--strategies", "default",
                        "--dump-ticker-scores", str(dump_path),
                    ])

        assert not dump_path.exists()


def _fake_synthetic_df(start="2007-06-01", periods=20000):
    idx = pd.date_range(start, periods=periods, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1000.0},
        index=idx,
    )


class TestMainCLISyntheticMode:

    @pytest.fixture(autouse=True)
    def _no_real_position_summary_writes(self):
        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._write_position_summary") as m:
            yield m

    def test_requires_both_synthetic_flags_together(self):
        from Strategy_Auto_Trader.markov_cli.live_sim import main

        with pytest.raises(SystemExit):
            main(["--tickers", "TEST", "--synthetic-data-dir", "somedir"])
        with pytest.raises(SystemExit):
            main(["--tickers", "TEST", "--synthetic-end-date", "2009-07-31"])

    def test_wires_df_by_ticker_and_synthetic_hmm_cache_dir(self):
        from Strategy_Auto_Trader.markov_cli.live_sim import SYNTHETIC_HMM_CACHE_DIR, main

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.load_synthetic_hourly",
                        return_value=_fake_synthetic_df()):
            with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.generate_candidates",
                            return_value=([], {}, {})) as mock_gen:
                with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.append_trades", return_value=0):
                    main([
                        "--tickers", "TEST", "--strategies", "default",
                        "--start-date", "2008-01-01",
                        "--synthetic-data-dir", "some/dir",
                        "--synthetic-end-date", "2009-07-31",
                    ])

        gen_kwargs = mock_gen.call_args_list[0][1]
        assert gen_kwargs["hmm_cache_dir"] == SYNTHETIC_HMM_CACHE_DIR
        assert gen_kwargs["use_persistent_cache"] is True
        assert "TEST" in gen_kwargs["df_by_ticker"]

    def test_ticker_with_no_synthetic_file_dropped_not_passed_through(self):
        """Regression guard: a ticker missing from the synthetic dir must be
        excluded from the tickers list entirely, never reach
        generate_candidates with a df_by_ticker miss (which would silently
        fall through to a REAL fetch of REAL current data — the whole point
        of an isolated historical stress test defeated)."""
        from Strategy_Auto_Trader.markov_cli.live_sim import main

        def fake_load(ticker, hourly_dir=None):
            return _fake_synthetic_df() if ticker == "HAS_DATA" else None

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.load_synthetic_hourly",
                        side_effect=fake_load):
            with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.generate_candidates",
                            return_value=([], {}, {})) as mock_gen:
                with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.append_trades", return_value=0):
                    main([
                        "--tickers", "HAS_DATA", "NO_DATA", "--strategies", "default",
                        "--start-date", "2008-01-01",
                        "--synthetic-data-dir", "some/dir",
                        "--synthetic-end-date", "2009-07-31",
                    ])

        gen_kwargs = mock_gen.call_args_list[0][1]
        assert gen_kwargs["tickers"] == ["HAS_DATA"]
        assert "NO_DATA" not in gen_kwargs["df_by_ticker"]

    def test_ticker_with_empty_window_dropped(self):
        """A synthetic file that exists but has no bars inside
        [start_date, synthetic_end_date] must be dropped the same way a
        missing file is — not passed through with an empty frame."""
        from Strategy_Auto_Trader.markov_cli.live_sim import main

        out_of_range_df = _fake_synthetic_df(start="2015-01-01", periods=100)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.load_synthetic_hourly",
                        return_value=out_of_range_df):
            with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.generate_candidates",
                            return_value=([], {}, {})) as mock_gen:
                with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.append_trades", return_value=0):
                    main([
                        "--tickers", "TEST", "--strategies", "default",
                        "--start-date", "2008-01-01",
                        "--synthetic-data-dir", "some/dir",
                        "--synthetic-end-date", "2009-07-31",
                    ])

        gen_kwargs = mock_gen.call_args_list[0][1]
        assert gen_kwargs["tickers"] == []
        assert gen_kwargs["df_by_ticker"] == {}

    def test_date_slicing_applied_to_synthetic_frame(self):
        from Strategy_Auto_Trader.markov_cli.live_sim import main

        full_df = _fake_synthetic_df(start="2007-06-01", periods=15000)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.load_synthetic_hourly",
                        return_value=full_df):
            with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.generate_candidates",
                            return_value=([], {}, {})) as mock_gen:
                with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.append_trades", return_value=0):
                    main([
                        "--tickers", "TEST", "--strategies", "default",
                        "--start-date", "2008-01-01",
                        "--synthetic-data-dir", "some/dir",
                        "--synthetic-end-date", "2009-07-31",
                    ])

        sliced = mock_gen.call_args_list[0][1]["df_by_ticker"]["TEST"]
        assert sliced.index.min() >= pd.Timestamp("2008-01-01", tz="UTC")
        assert sliced.index.max() <= pd.Timestamp("2009-07-31 23:59:59", tz="UTC")
        assert len(sliced) < len(full_df)

    def test_synthetic_mode_defaults_journal_under_data_synthetic(self):
        from Strategy_Auto_Trader.markov_cli.live_sim import _SYNTHETIC_JOURNAL_DIR, main

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.load_synthetic_hourly",
                        return_value=_fake_synthetic_df()):
            with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.generate_candidates",
                            return_value=([], {}, {})):
                with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.append_trades",
                                return_value=0) as mock_append:
                    main([
                        "--tickers", "TEST", "--strategies", "default",
                        "--start-date", "2008-01-01",
                        "--synthetic-data-dir", "some/dir",
                        "--synthetic-end-date", "2009-07-31",
                    ])

        journal_path = mock_append.call_args_list[0][0][0]
        assert journal_path.parent == _SYNTHETIC_JOURNAL_DIR

    def test_explicit_journal_overrides_synthetic_default(self, tmp_path):
        from Strategy_Auto_Trader.markov_cli.live_sim import main

        explicit = tmp_path / "custom.csv"
        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.load_synthetic_hourly",
                        return_value=_fake_synthetic_df()):
            with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.generate_candidates",
                            return_value=([], {}, {})):
                with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.append_trades",
                                return_value=0) as mock_append:
                    main([
                        "--tickers", "TEST", "--strategies", "default",
                        "--start-date", "2008-01-01",
                        "--synthetic-data-dir", "some/dir",
                        "--synthetic-end-date", "2009-07-31",
                        "--journal", str(explicit),
                    ])

        assert mock_append.call_args_list[0][0][0] == explicit

    def test_non_synthetic_path_passes_none_for_new_kwargs(self):
        """Omitting --synthetic-data-dir must reach generate_candidates with
        df_by_ticker=None and hmm_cache_dir=None — today's exact behavior,
        unchanged."""
        from Strategy_Auto_Trader.markov_cli.live_sim import main

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.generate_candidates",
                        return_value=([], {}, {})) as mock_gen:
            with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.append_trades", return_value=0):
                main(["--tickers", "TEST", "--strategies", "default"])

        gen_kwargs = mock_gen.call_args_list[0][1]
        assert gen_kwargs["df_by_ticker"] is None
        assert gen_kwargs["hmm_cache_dir"] is None
        assert gen_kwargs["use_persistent_cache"] is True


class TestArbitrate:

    def test_no_daily_admission_cap_all_admitted_when_cash_allows(self, ts_base):
        """No daily-admission cap and no position-count cap — every same-day
        candidate cash allows is admitted, matching the live daemon (cash-gated
        only, see .claude/rules/cli.md)."""
        from Strategy_Auto_Trader.markov_cli.live_sim import arbitrate, Candidate

        candidates = []
        for i in range(5):
            rec = TradeRecord(date_opened="2026-01-12", ticker=f"T{i}", strategy="test",
                               entry_score=float(i), kelly_fraction=0.1, return_pct=0.05,
                               entry_price=10.0)
            candidates.append(Candidate(
                ticker=f"T{i}", date_opened=ts_base, date_closed=ts_base + pd.Timedelta(days=1),
                entry_score=float(i), kelly_fraction=0.1, return_pct=0.05, record=rec,
            ))

        result = arbitrate(candidates, initial_cash=10_000.0, trade_cost=1.0)

        assert len(result["executed"]) == 5
        assert result["n_admitted"] == 5

    def test_admission_diagnostics_count_candidates_and_rejections(self, ts_base):
        """n_candidates/n_admitted/n_rejected_cash support a 'capital wasn't the
        constraint' conclusion with evidence, not just peak-vs-pot eyeballing."""
        from Strategy_Auto_Trader.markov_cli.live_sim import arbitrate, Candidate

        candidates = []
        for i in range(3):
            rec = TradeRecord(date_opened="2026-01-12", ticker=f"T{i}", strategy="test",
                               entry_score=float(3 - i), kelly_fraction=0.5, return_pct=0.05,
                               entry_price=0.4)
            candidates.append(Candidate(
                ticker=f"T{i}", date_opened=ts_base, date_closed=ts_base + pd.Timedelta(days=5),
                entry_score=float(3 - i), kelly_fraction=0.5, return_pct=0.05, record=rec,
            ))

        # cash=1.5, price=0.4: first candidate affords 1 share (0.4) + $1 fee =
        # 1.4, leaving 0.1 — too little to clear the next $0.4 share price, so
        # candidates 2 and 3 are rejected on cash, not fee.
        result = arbitrate(candidates, initial_cash=1.5, trade_cost=1.0)

        assert result["n_candidates"] == 3
        assert result["n_admitted"] + result["n_rejected_cash"] >= result["n_candidates"] - 1
        assert result["n_admitted"] < result["n_candidates"]

    def test_empty_candidates_returns_zeroed_result(self):
        from Strategy_Auto_Trader.markov_cli.live_sim import arbitrate

        result = arbitrate([], initial_cash=5000.0, trade_cost=1.0)
        assert result["executed"] == []
        assert result["equity_curve"] == []
        assert result["final_cash"] == 5000.0
        assert result["n_candidates"] == 0

    def test_pot_size_sweep_does_not_alias_records_across_runs(self, ts_base):
        """A --pot-sizes sweep reuses the same candidate list (and therefore
        the same cand.record objects) across multiple arbitrate() calls, per
        the "no need to re-backtest per pot size" design (.claude/rules/cli.md).
        arbitrate() must not mutate cand.record in place — doing so lets a
        later pot size's pnl_usd/position_size_gbp silently overwrite an
        earlier pot size's already-appended executed record, corrupting the
        trade journal for every pot size but the last."""
        from Strategy_Auto_Trader.markov_cli.live_sim import arbitrate, Candidate

        rec = TradeRecord(date_opened="2026-01-12", ticker="TEST", strategy="test",
                           entry_score=1.0, kelly_fraction=0.5, return_pct=0.05,
                           entry_price=10.0)
        cand = Candidate(
            ticker="TEST", date_opened=ts_base, date_closed=ts_base + pd.Timedelta(days=5),
            entry_score=1.0, kelly_fraction=0.5, return_pct=0.05, record=rec,
        )
        candidates = [cand]

        small_result = arbitrate(candidates, initial_cash=1_000.0, trade_cost=1.0)
        large_result = arbitrate(candidates, initial_cash=100_000.0, trade_cost=1.0)

        small_exec = small_result["executed"][0]
        large_exec = large_result["executed"][0]

        assert small_exec is not large_exec
        assert small_exec.position_size_gbp < large_exec.position_size_gbp
        assert small_exec.pnl_usd != large_exec.pnl_usd
        # the shared candidate's own record must survive both runs unmutated
        assert cand.record.position_size_gbp == 0.0

    def _make_same_day_candidates(self, n, kelly, entry_price, ts_base, day_offset=0):
        from Strategy_Auto_Trader.markov_cli.live_sim import Candidate

        cands = []
        for i in range(n):
            rec = TradeRecord(date_opened="2026-01-12", ticker=f"T{i}", strategy="test",
                               entry_score=float(n - i), kelly_fraction=kelly,
                               return_pct=0.05, entry_price=entry_price)
            cands.append(Candidate(
                ticker=f"T{i}",
                date_opened=ts_base + pd.Timedelta(days=day_offset),
                date_closed=ts_base + pd.Timedelta(days=day_offset + 5),
                entry_score=float(n - i), kelly_fraction=kelly,
                return_pct=0.05, record=rec,
            ))
        return cands

    def test_deployment_cap_none_behaves_identically_to_no_cap(self, ts_base):
        """same_day_deployment_cap_pct=None (or omitted) must be a true no-op
        — this is what makes the default-off rollout safe for every strategy
        that doesn't declare the attribute."""
        from Strategy_Auto_Trader.markov_cli.live_sim import arbitrate

        candidates = self._make_same_day_candidates(5, kelly=0.5, entry_price=10.0, ts_base=ts_base)

        omitted = arbitrate(candidates, initial_cash=1_000.0, trade_cost=1.0)
        explicit_none = arbitrate(candidates, initial_cash=1_000.0, trade_cost=1.0,
                                   same_day_deployment_cap_pct=None)

        assert omitted["n_admitted"] == explicit_none["n_admitted"]
        assert omitted["n_rejected_cash"] == explicit_none["n_rejected_cash"]
        assert omitted["n_rejected_concentration"] == explicit_none["n_rejected_concentration"] == 0

    def test_deployment_cap_zero_treated_as_disabled(self, ts_base):
        """cap_pct=0.0 is falsy — treated as 'no cap', not 'admit nothing',
        consistent with None/absent meaning disabled."""
        from Strategy_Auto_Trader.markov_cli.live_sim import arbitrate

        candidates = self._make_same_day_candidates(3, kelly=0.5, entry_price=10.0, ts_base=ts_base)

        result = arbitrate(candidates, initial_cash=1_000.0, trade_cost=1.0,
                            same_day_deployment_cap_pct=0.0)

        assert result["n_rejected_concentration"] == 0
        assert result["n_admitted"] == 3

    def test_deployment_cap_rejects_lowest_priority_candidates_first(self, ts_base):
        """Cash alone would admit all 5 candidates; the concentration cap
        narrows the admitted set to the highest-entry_score prefix that fits
        under cap_pct * initial_cash, rejecting the rest under
        n_rejected_concentration (not n_rejected_cash).

        5 candidates, kelly=0.1, price=10, initial_cash=100_000: T0 wants
        qty=floor(100_000*0.1/10)=1000 (alloc=10_000), and each subsequent
        admission shrinks `cash` so T1's alloc=9_000. cap_pct=0.25 ->
        daily_cap=25_000, which fits T0+T1 (19_000) but not +T2 (27_100) —
        admits exactly the top 2 by entry_score, rejects the rest."""
        from Strategy_Auto_Trader.markov_cli.live_sim import arbitrate

        candidates = self._make_same_day_candidates(5, kelly=0.1, entry_price=10.0, ts_base=ts_base)

        result = arbitrate(candidates, initial_cash=100_000.0, trade_cost=0.0,
                            same_day_deployment_cap_pct=0.25)

        assert result["n_rejected_cash"] == 0
        assert result["n_admitted"] == 2
        assert result["n_rejected_concentration"] == 3
        # admitted tickers must be the highest-entry_score prefix (T0 has the
        # highest score by construction: entry_score = n - i)
        admitted_tickers = {r.ticker for r in result["executed"]}
        assert admitted_tickers == {"T0", "T1"}

    def test_deployment_cap_resets_each_day(self, ts_base):
        """deployed_today resets per calendar day. One candidate per day,
        cap_pct chosen so a single day's admission (alloc ~100) fits under
        daily_cap=150, but two days' admissions summed (~190) would not —
        proving the cap is evaluated fresh each day rather than carried
        forward (which would incorrectly reject day1's candidate)."""
        from Strategy_Auto_Trader.markov_cli.live_sim import arbitrate

        day0 = self._make_same_day_candidates(1, kelly=0.001, entry_price=10.0, ts_base=ts_base, day_offset=0)
        day1 = self._make_same_day_candidates(1, kelly=0.001, entry_price=10.0, ts_base=ts_base, day_offset=1)
        day1[0].ticker = "D1_T0"
        day1[0].record.ticker = "D1_T0"

        result = arbitrate(day0 + day1, initial_cash=100_000.0, trade_cost=0.0,
                            same_day_deployment_cap_pct=0.0015)

        assert result["n_rejected_concentration"] == 0
        assert result["n_admitted"] == 2

    def test_deployment_cap_uses_fixed_initial_cash_not_mark_to_market(self, ts_base):
        """The cap's denominator is the fixed initial_cash, not a mark-to-
        market portfolio value — even when price_by_ticker is supplied (which
        would let mark-to-market pricing exist), the admit/reject boundary is
        unaffected by it."""
        from Strategy_Auto_Trader.markov_cli.live_sim import arbitrate

        candidates = self._make_same_day_candidates(5, kelly=0.1, entry_price=10.0, ts_base=ts_base)
        # Wildly different mark-to-market prices for open positions — should
        # have no bearing on same-day admission, which happens before any
        # position is "open" long enough to be marked.
        price_by_ticker = {
            f"T{i}": pd.Series([1000.0], index=[ts_base]) for i in range(5)
        }

        without_prices = arbitrate(candidates, initial_cash=100_000.0, trade_cost=0.0,
                                    same_day_deployment_cap_pct=0.01)
        with_prices = arbitrate(candidates, initial_cash=100_000.0, trade_cost=0.0,
                                 same_day_deployment_cap_pct=0.01,
                                 price_by_ticker=price_by_ticker)

        assert without_prices["n_admitted"] == with_prices["n_admitted"]
        assert without_prices["n_rejected_concentration"] == with_prices["n_rejected_concentration"]


class TestMarkToMarket:

    def test_open_position_valued_at_current_price_not_cost_basis(self, ts_base):
        """A position still open at a later snapshot day is marked to its
        current price, not frozen at its entry cost basis — the panel-reviewed
        fix for the mark-to-cost bias that would otherwise misstate portfolio
        value/drawdown for a real funding decision."""
        from Strategy_Auto_Trader.markov_cli.live_sim import arbitrate, Candidate

        rec = TradeRecord(date_opened="2026-01-12", ticker="RISER", strategy="test",
                           entry_score=1.0, kelly_fraction=0.5, return_pct=0.20)
        cand = Candidate(
            ticker="RISER", date_opened=ts_base, date_closed=ts_base + pd.Timedelta(days=10),
            entry_score=1.0, kelly_fraction=0.5, return_pct=0.20, record=rec,
        )
        # entry_price defaults to 0.0 on the TradeRecord unless set explicitly —
        # mark-to-market needs a real entry price to compute a price ratio.
        rec.entry_price = 100.0

        # Price doubles by day 5, still open (closes day 10) — mark-to-market
        # deployed value should reflect that, not the frozen cost basis.
        price_series = pd.Series(
            [100.0, 200.0],
            index=[ts_base, ts_base + pd.Timedelta(days=5)],
        )

        result = arbitrate(
            [cand], initial_cash=1000.0, trade_cost=0.0,
            price_by_ticker={"RISER": price_series},
        )

        # cost basis alloc = 0.5 * 1000 = 500
        rows_by_date = {row["date"]: row for row in result["equity_curve"]}
        opening_row = rows_by_date[ts_base.tz_localize(None).normalize()]
        assert opening_row["deployed"] == pytest.approx(500.0)

        # No later snapshot day exists in this scenario (only open/close event
        # days are sampled) — but if price is looked up as-of the close day
        # using the last known price (200, held from day 5), deployed should
        # double relative to cost basis at that point too, given entry_price=100.
        close_row = rows_by_date[(ts_base + pd.Timedelta(days=10)).tz_localize(None).normalize()]
        # Position closes ON this day, so it's released before the deployed
        # figure is computed — deployed on the close day itself is 0.
        assert close_row["deployed"] == pytest.approx(0.0)

    def test_missing_price_data_falls_back_to_cost_basis(self, ts_base):
        """No price_by_ticker entry for a ticker (or price_by_ticker=None
        entirely) falls back to cost-basis valuation rather than crashing."""
        from Strategy_Auto_Trader.markov_cli.live_sim import arbitrate, Candidate

        rec = TradeRecord(date_opened="2026-01-12", ticker="NODATA", strategy="test",
                           entry_score=1.0, kelly_fraction=0.5, return_pct=0.10)
        rec.entry_price = 50.0
        cand = Candidate(
            ticker="NODATA", date_opened=ts_base, date_closed=ts_base + pd.Timedelta(days=10),
            entry_score=1.0, kelly_fraction=0.5, return_pct=0.10, record=rec,
        )

        result = arbitrate([cand], initial_cash=1000.0, trade_cost=0.0, price_by_ticker=None)

        opening_row = result["equity_curve"][0]
        assert opening_row["deployed"] == pytest.approx(500.0)  # cost basis, no crash
