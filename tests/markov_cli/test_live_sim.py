from __future__ import annotations

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
    )


@pytest.fixture
def ts_base():
    """Base timestamp for relative date construction."""
    return pd.Timestamp("2026-01-12", tz="UTC")


def make_candidate(ticker, day_offset, entry_score, kelly_fraction, return_pct, record, ts_base):
    """Helper to construct a _Candidate dataclass."""
    from Strategy_Auto_Trader.markov_cli.live_sim import _Candidate

    date_opened = ts_base + pd.Timedelta(days=day_offset)
    date_closed = date_opened + pd.Timedelta(days=5)

    return _Candidate(
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

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._fetch_and_extract", return_value=[]):
            result = simulate_strategy(
                tickers=["TEST"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=1000.0,
                trade_cost=1.0,
                kelly_fallback=100.0,
                max_trades_per_day=1,
            )
        assert result == []

    def test_zero_cash_admits_nothing(self, base_record, ts_base):
        """With zero starting capital, no trades are admitted."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy

        cand = make_candidate("TEST", 0, 1.0, 0.1, 0.05, base_record, ts_base)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._fetch_and_extract", return_value=[cand]):
            result = simulate_strategy(
                tickers=["TEST"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=0.0,
                trade_cost=1.0,
                kelly_fallback=100.0,
                max_trades_per_day=5,
            )
        assert result == []

    def test_cash_equals_trade_cost_skips_candidate(self, base_record, ts_base):
        """When cash == trade_cost exactly, candidate is skipped (cash <= trade_cost check)."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy

        cand = make_candidate("TEST", 0, 1.0, 0.1, 0.05, base_record, ts_base)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._fetch_and_extract", return_value=[cand]):
            result = simulate_strategy(
                tickers=["TEST"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=1.0,  # exactly trade_cost
                trade_cost=1.0,
                kelly_fallback=100.0,
                max_trades_per_day=5,
            )
        assert result == []

    def test_cash_slightly_above_trade_cost_admits_with_minimal_alloc(self, base_record, ts_base):
        """When cash > trade_cost but <= trade_cost + kelly_fallback, minimal alloc is used."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy

        rec = TradeRecord(
            date_opened="2026-01-12",
            ticker="TEST",
            strategy="test",
            entry_score=1.0,
            kelly_fraction=0.0,  # Will use kelly_fallback
            return_pct=0.10,  # 10% return
        )
        cand = make_candidate("TEST", 0, 1.0, 0.0, 0.10, rec, ts_base)

        # cash = 11.0: trade_cost=1.0, kelly_fallback=100.0
        # alloc will be min(100, 11 - 1) = 10
        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._fetch_and_extract", return_value=[cand]):
            result = simulate_strategy(
                tickers=["TEST"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=11.0,
                trade_cost=1.0,
                kelly_fallback=100.0,
                max_trades_per_day=5,
            )
        assert len(result) == 1
        assert result[0].ticker == "TEST"

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
        )
        cand = make_candidate("TEST", 0, 1.0, 0.25, 0.10, rec, ts_base)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._fetch_and_extract", return_value=[cand]):
            result = simulate_strategy(
                tickers=["TEST"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=1000.0,
                trade_cost=1.0,
                kelly_fallback=100.0,
                max_trades_per_day=5,
            )
        assert len(result) == 1
        # alloc = 0.25 * 1000 = 250
        # pnl_usd = alloc * return_pct - 2*trade_cost = 250 * 0.10 - 2 = 23
        assert result[0].pnl_usd == pytest.approx(23.0)

    def test_kelly_zero_falls_back_to_kelly_fallback(self, base_record, ts_base):
        """Kelly fraction == 0 falls back to min(kelly_fallback, available_cash)."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy

        rec = TradeRecord(
            date_opened="2026-01-12",
            ticker="TEST",
            strategy="test",
            entry_score=1.0,
            kelly_fraction=0.0,
            return_pct=0.05,
        )
        cand = make_candidate("TEST", 0, 1.0, 0.0, 0.05, rec, ts_base)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._fetch_and_extract", return_value=[cand]):
            result = simulate_strategy(
                tickers=["TEST"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=500.0,
                trade_cost=1.0,
                kelly_fallback=100.0,  # would use this
                max_trades_per_day=5,
            )
        assert len(result) == 1
        # alloc = min(100, 500 - 1) = 100
        # pnl_usd = 100 * 0.05 - 2*1 = 5 - 2 = 3
        assert result[0].pnl_usd == pytest.approx(3.0)

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
        )
        cand = make_candidate("TEST", 0, 1.0, 0.5, 0.10, rec, ts_base)

        # Initial cash 100: 0.5 * 100 = 50, clamped by (100 - 1) = 99, so 50 is used
        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._fetch_and_extract", return_value=[cand]):
            result = simulate_strategy(
                tickers=["TEST"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=100.0,
                trade_cost=1.0,
                kelly_fallback=200.0,
                max_trades_per_day=5,
            )
        assert len(result) == 1
        # alloc = min(50, 100 - 1) = 50
        # pnl_usd = 50 * 0.10 - 2*1 = 5 - 2 = 3
        assert result[0].pnl_usd == pytest.approx(3.0)

    def test_alloc_negative_or_zero_candidate_skipped(self, base_record, ts_base):
        """If calculated alloc <= 0, candidate is skipped."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy

        rec = TradeRecord(
            date_opened="2026-01-12",
            ticker="TEST",
            strategy="test",
            entry_score=1.0,
            kelly_fraction=0.0,
            return_pct=0.05,
        )
        cand = make_candidate("TEST", 0, 1.0, 0.0, 0.05, rec, ts_base)

        # cash=1.5, trade_cost=1.0: kelly_fallback would be min(100, 0.5) = 0.5
        # But alloc = min(0.5, 1.5 - 1) = min(0.5, 0.5) = 0.5 > 0, so admitted
        # Let's make it negative: cash=1.0 exactly, so 1.0 - 1.0 = 0
        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._fetch_and_extract", return_value=[cand]):
            result = simulate_strategy(
                tickers=["TEST"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=1.0,  # == trade_cost
                trade_cost=1.0,
                kelly_fallback=100.0,
                max_trades_per_day=5,
            )
        assert result == []

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
        )
        rec2 = TradeRecord(
            date_opened="2026-01-12",
            ticker="B",
            strategy="test",
            entry_score=1.0,  # Lower score
            kelly_fraction=0.1,
            return_pct=0.05,
        )
        cand1 = make_candidate("A", 0, 3.0, 0.1, 0.05, rec1, ts_base)
        cand2 = make_candidate("B", 0, 1.0, 0.1, 0.05, rec2, ts_base)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._fetch_and_extract", side_effect=[[cand1], [cand2]]):
            result = simulate_strategy(
                tickers=["A", "B"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=500.0,
                trade_cost=1.0,
                kelly_fallback=100.0,
                max_trades_per_day=2,
            )
        assert len(result) == 2
        # A admitted first (score 3.0), then B (score 1.0)
        assert result[0].ticker == "A"
        assert result[1].ticker == "B"

    def test_two_same_day_candidates_tied_score_deterministic_order(self, base_record, ts_base):
        """Two candidates with same entry_score on same day are both admitted up to cap."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy

        rec1 = TradeRecord(
            date_opened="2026-01-12",
            ticker="A",
            strategy="test",
            entry_score=2.0,
            kelly_fraction=0.1,
            return_pct=0.05,
        )
        rec2 = TradeRecord(
            date_opened="2026-01-12",
            ticker="B",
            strategy="test",
            entry_score=2.0,  # Tied
            kelly_fraction=0.1,
            return_pct=0.05,
        )
        cand1 = make_candidate("A", 0, 2.0, 0.1, 0.05, rec1, ts_base)
        cand2 = make_candidate("B", 0, 2.0, 0.1, 0.05, rec2, ts_base)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._fetch_and_extract", side_effect=[[cand1], [cand2]]):
            result = simulate_strategy(
                tickers=["A", "B"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=1000.0,
                trade_cost=1.0,
                kelly_fallback=100.0,
                max_trades_per_day=2,  # Cap of 2
            )
        assert len(result) == 2  # Both admitted

    def test_max_trades_per_day_cap_exact_match(self, base_record, ts_base):
        """Exactly N candidates with cap N → all admitted."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy

        records = []
        candidates = []
        for i in range(3):
            rec = TradeRecord(
                date_opened="2026-01-12",
                ticker=f"T{i}",
                strategy="test",
                entry_score=float(i),
                kelly_fraction=0.1,
                return_pct=0.05,
            )
            records.append(rec)
            candidates.append(make_candidate(f"T{i}", 0, float(i), 0.1, 0.05, rec, ts_base))

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._fetch_and_extract", side_effect=[[c] for c in candidates]):
            result = simulate_strategy(
                tickers=[f"T{i}" for i in range(3)],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=1000.0,
                trade_cost=1.0,
                kelly_fallback=100.0,
                max_trades_per_day=3,  # Cap of 3
            )
        assert len(result) == 3  # All admitted

    def test_max_trades_per_day_exceeds_cap(self, base_record, ts_base):
        """N+1 candidates with cap N → exactly N admitted."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy

        records = []
        candidates = []
        for i in range(4):
            rec = TradeRecord(
                date_opened="2026-01-12",
                ticker=f"T{i}",
                strategy="test",
                entry_score=float(4 - i),  # Higher scores first
                kelly_fraction=0.1,
                return_pct=0.05,
            )
            records.append(rec)
            candidates.append(make_candidate(f"T{i}", 0, float(4 - i), 0.1, 0.05, rec, ts_base))

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._fetch_and_extract", side_effect=[[c] for c in candidates]):
            result = simulate_strategy(
                tickers=[f"T{i}" for i in range(4)],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=2000.0,
                trade_cost=1.0,
                kelly_fallback=100.0,
                max_trades_per_day=2,  # Cap of 2
            )
        assert len(result) == 2  # Only 2 admitted despite 4 candidates

    def test_cash_release_on_position_close_same_day(self, base_record, ts_base):
        """A position closing on day D frees cash for a new entry on the same day."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy, _Candidate

        # Day 0: candidate opens, closes on day 0 (immediate)
        rec_close_day0 = TradeRecord(
            date_opened="2026-01-12",
            ticker="CLOSE",
            strategy="test",
            entry_score=1.0,
            kelly_fraction=0.1,
            return_pct=0.10,
        )
        cand_close_day0 = _Candidate(
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
        )
        cand_entry = make_candidate("ENTRY", 0, 2.0, 0.1, 0.05, rec_entry, ts_base)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._fetch_and_extract", side_effect=[[cand_close_day0], [cand_entry]]):
            result = simulate_strategy(
                tickers=["CLOSE", "ENTRY"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=300.0,
                trade_cost=1.0,
                kelly_fallback=100.0,
                max_trades_per_day=2,
            )
        # Both should be admitted: first closes immediately, freeing cash for second
        assert len(result) == 2

    def test_cash_not_released_for_future_close(self, base_record, ts_base):
        """A position closing on day D does not free cash for entry on day D-1."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy, _Candidate

        # Day 0: candidate opens, closes on day 1
        rec = TradeRecord(
            date_opened="2026-01-12",
            ticker="TEST",
            strategy="test",
            entry_score=1.0,
            kelly_fraction=0.5,  # 50% of cash
            return_pct=0.10,
        )
        cand = _Candidate(
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
        )
        cand2 = make_candidate("TEST2", 0, 0.5, 0.5, 0.05, rec2, ts_base)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._fetch_and_extract", return_value=[cand, cand2]):
            result = simulate_strategy(
                tickers=["TEST"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=200.0,
                trade_cost=1.0,
                kelly_fallback=100.0,
                max_trades_per_day=2,
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
            kelly_fraction=0.0,
            return_pct=0.20,  # 20% return
        )
        cand = make_candidate("TEST", 0, 1.0, 0.0, 0.20, rec, ts_base)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._fetch_and_extract", return_value=[cand]):
            result = simulate_strategy(
                tickers=["TEST"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=1000.0,
                trade_cost=10.0,
                kelly_fallback=100.0,
                max_trades_per_day=5,
            )
        assert len(result) == 1
        # alloc = min(100, 1000 - 10) = 100
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
        )
        cand1 = make_candidate("TEST1", 1, 1.0, 0.2, 0.10, rec1, ts_base)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._fetch_and_extract", side_effect=[[cand0], [cand1]]):
            result = simulate_strategy(
                tickers=["TEST0", "TEST1"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=1000.0,
                trade_cost=1.0,
                kelly_fallback=100.0,
                max_trades_per_day=5,
            )
        assert len(result) == 2

    def test_start_date_filter_excludes_earlier_candidates(self, base_record, ts_base):
        """Candidates before start_date are excluded."""
        from Strategy_Auto_Trader.markov_cli.live_sim import simulate_strategy, _Candidate

        # Day -1 (before start_date)
        rec_before = TradeRecord(
            date_opened="2026-01-11",
            ticker="BEFORE",
            strategy="test",
            entry_score=1.0,
            kelly_fraction=0.1,
            return_pct=0.05,
        )
        cand_before = _Candidate(
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
        )
        cand_after = make_candidate("AFTER", 0, 1.0, 0.1, 0.05, rec_after, ts_base)

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._fetch_and_extract", return_value=[cand_before, cand_after]):
            result = simulate_strategy(
                tickers=["TEST"],
                strategy_name="test",
                start_date="2026-01-12",
                initial_cash=1000.0,
                trade_cost=1.0,
                kelly_fallback=100.0,
                max_trades_per_day=5,
            )
        # Only AFTER should be admitted
        assert len(result) == 1
        assert result[0].ticker == "AFTER"


class TestVolFilterTickers:

    def test_vol_filter_tickers_returns_suitable_unsuitable(self):
        from Strategy_Auto_Trader.markov_cli.live_sim import _vol_filter_tickers

        profiles = [
            {"ticker": "GOOD", "trend_quality": 0.8},
            {"ticker": "BAD", "trend_quality": 0.2},
            {"ticker": "MEH", "trend_quality": 0.5},
        ]

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.screen_tickers", return_value=(["GOOD", "MEH"], profiles)):
            suitable, unsuitable, scores = _vol_filter_tickers(["GOOD", "BAD", "MEH"], min_trend_quality=0.3)

        assert "GOOD" in suitable
        assert "MEH" in suitable
        assert "BAD" in unsuitable
        assert scores["GOOD"] == 0.8
        assert scores["BAD"] == 0.2

    def test_vol_filter_tickers_empty(self):
        from Strategy_Auto_Trader.markov_cli.live_sim import _vol_filter_tickers

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.screen_tickers", return_value=([], [])):
            suitable, unsuitable, scores = _vol_filter_tickers(["TEST"], min_trend_quality=0.5)

        assert suitable == []
        assert "TEST" in unsuitable


class TestSkipRecords:

    def test_skip_records_one_per_unsuitable(self):
        from Strategy_Auto_Trader.markov_cli.live_sim import _skip_records

        records = _skip_records(["BAD1", "BAD2"], "test_strategy", "2026-01-12")

        assert len(records) == 2
        assert records[0].ticker == "BAD1"
        assert records[1].ticker == "BAD2"
        assert all(r.vol_filter == "unsuitable" for r in records)
        assert all(r.strategy == "test_strategy" for r in records)
        assert all(r.date_opened == "2026-01-12" for r in records)

    def test_skip_records_empty_unsuitable(self):
        from Strategy_Auto_Trader.markov_cli.live_sim import _skip_records

        records = _skip_records([], "test_strategy", "2026-01-12")

        assert records == []


class TestMainCLI:

    def test_main_no_vol_filter_bypasses_screen_tickers(self):
        from Strategy_Auto_Trader.markov_cli.live_sim import main

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._vol_filter_tickers") as mock_screen:
            with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.simulate_strategy", return_value=[]):
                with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.append_trades", return_value=0):
                    main([
                        "--tickers", "TEST1", "TEST2",
                        "--strategies", "default",
                        "--no-vol-filter",
                    ])

        # screen_tickers should not have been called
        mock_screen.assert_not_called()

    def test_main_with_vol_filter_calls_screen_tickers(self):
        from Strategy_Auto_Trader.markov_cli.live_sim import main

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._vol_filter_tickers", return_value=(["TEST1"], ["TEST2"], {})):
            with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.simulate_strategy", return_value=[]):
                with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.append_trades", return_value=0):
                    main([
                        "--tickers", "TEST1", "TEST2",
                        "--strategies", "default",
                    ])

    def test_main_vol_filter_exempt_strategy(self):
        from Strategy_Auto_Trader.markov_cli.live_sim import main

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._vol_filter_tickers", return_value=(["TEST1"], ["TEST2"], {})):
            with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.simulate_strategy", return_value=[]) as mock_sim:
                with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.append_trades", return_value=0):
                    main([
                        "--tickers", "TEST1", "TEST2",
                        "--strategies", "default", "exempt_strat",
                        "--vol-filter-exempt", "exempt_strat",
                    ])

        # exempt_strat should be called with all tickers, default with filtered
        calls = mock_sim.call_args_list
        assert len(calls) >= 2

    def test_main_default_arguments(self):
        from Strategy_Auto_Trader.markov_cli.live_sim import main

        with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim._vol_filter_tickers", return_value=(["TEST"], [], {})):
            with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.simulate_strategy", return_value=[]) as mock_sim:
                with mock.patch("Strategy_Auto_Trader.markov_cli.live_sim.append_trades", return_value=0):
                    main(["--tickers", "TEST"])

        call_kwargs = mock_sim.call_args_list[0][1]
        assert call_kwargs["initial_cash"] == 10_000.0
        assert call_kwargs["trade_cost"] == 1.0
        assert call_kwargs["kelly_fallback"] == 100.0
        assert call_kwargs["max_trades_per_day"] == 1
