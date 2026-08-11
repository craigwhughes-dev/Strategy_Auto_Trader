"""Parity test: live_sim.py's arbitrate() and the live daemon's
execute_signals() must make the same admit/reject/quantity decisions when
offered the same same-day candidates competing for the same cash pot.

This guards the invariant documented in .claude/rules/cli.md — live_sim's
capital arbitration mirrors PortfolioManager.compute_quantity() exactly —
after the entry_score-vs-kelly_fraction sort-order divergence found and
fixed 2026-08-11 (execute.py used to sort by kelly_fraction descending;
arbitrate() has always sorted by entry_score descending, which is what the
optimised_new backtest comparison that decided the live switch was run
against).
"""

from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

from Strategy_Auto_Trader.output.journal import TradeRecord


def _signal(score, kelly, price=500.0):
    return {
        "flag": "BUY", "close": price, "kelly_fraction": kelly, "score": score,
        "stop_level": price * 0.9, "target_level": price * 1.1,
    }


def _candidate(ticker, day, entry_score, kelly_fraction, price=500.0):
    from Strategy_Auto_Trader.markov_cli.live_sim import Candidate
    rec = TradeRecord(
        date_opened=str(day.date()), ticker=ticker, strategy="test",
        entry_score=entry_score, kelly_fraction=kelly_fraction, entry_price=price,
    )
    return Candidate(
        ticker=ticker, date_opened=day, date_closed=day + pd.Timedelta(days=5),
        entry_score=entry_score, kelly_fraction=kelly_fraction, return_pct=0.0,
        record=rec,
    )


class TestDaemonLiveSimParity:
    """Same candidates, same cash pot, same day — both paths must agree on
    who gets in and who gets shut out by cash exhaustion or kelly<=0."""

    def test_same_day_admission_matches(self, tmp_path):
        from Strategy_Auto_Trader.markov_cli.execute import execute_signals
        from Strategy_Auto_Trader.markov_cli.live_sim import arbitrate
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker

        # D has the highest score but kelly<=0 — must be rejected outright,
        # not merely deprioritised, in both paths.
        # A is next by score and (at kelly=1.0) consumes the entire £1000
        # pot on the daemon's whole-share sizing, so B and C must both be
        # cash-rejected in both paths too.
        signals = {
            "D": _signal(score=10.0, kelly=0.0),
            "A": _signal(score=5.0, kelly=1.0),
            "B": _signal(score=3.0, kelly=1.0),
            "C": _signal(score=1.0, kelly=1.0),
        }

        # -- daemon path ------------------------------------------------
        portfolio = PortfolioManager(1000.0, tmp_path / "state.json")
        broker = NullBroker(prices={t: 500.0 for t in signals})
        with mock.patch(
            "Strategy_Auto_Trader.broker.signal_reader.read_latest_signal",
            side_effect=lambda ticker, _dir: signals[ticker],
        ):
            buys, sells, skipped = execute_signals(
                list(signals), tmp_path, portfolio, portfolio.get_limit_tracker(), broker,
            )

        assert set(portfolio.positions) == {"A"}
        assert any("D(qty=0)" in s for s in skipped)
        assert any("B(qty=0)" in s for s in skipped)
        assert any("C(qty=0)" in s for s in skipped)

        # -- live_sim path ------------------------------------------------
        day = pd.Timestamp("2026-01-12", tz="UTC")
        candidates = [
            _candidate(t, day, sig["score"], sig["kelly_fraction"])
            for t, sig in signals.items()
        ]
        result = arbitrate(candidates, initial_cash=1000.0, trade_cost=0.0)

        assert [rec.ticker for rec in result["executed"]] == ["A"]
        assert result["n_rejected_kelly"] == 1  # D
        assert result["n_rejected_cash"] == 2  # B, C

    def test_admission_order_follows_entry_score_not_kelly_fraction(self, tmp_path):
        """Regression test for the 2026-08-11 sort-order fix: a low-kelly,
        high-score candidate must be admitted before a high-kelly, low-score
        one in both paths, since only one fits in the cash pot."""
        from Strategy_Auto_Trader.markov_cli.execute import execute_signals
        from Strategy_Auto_Trader.markov_cli.live_sim import arbitrate
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker

        # HIGH_SCORE has the weaker kelly_fraction but the stronger score;
        # LOW_SCORE is the reverse. Sorting by kelly_fraction (the old daemon
        # behaviour) would admit LOW_SCORE first; sorting by entry_score (the
        # backtest-validated behaviour) admits HIGH_SCORE first.
        signals = {
            "HIGH_SCORE": _signal(score=9.0, kelly=1.0),
            "LOW_SCORE": _signal(score=1.0, kelly=1.0),
        }

        portfolio = PortfolioManager(1000.0, tmp_path / "state.json")
        broker = NullBroker(prices={t: 500.0 for t in signals})
        with mock.patch(
            "Strategy_Auto_Trader.broker.signal_reader.read_latest_signal",
            side_effect=lambda ticker, _dir: signals[ticker],
        ):
            execute_signals(
                list(signals), tmp_path, portfolio, portfolio.get_limit_tracker(), broker,
            )
        assert set(portfolio.positions) == {"HIGH_SCORE"}

        day = pd.Timestamp("2026-01-12", tz="UTC")
        candidates = [
            _candidate(t, day, sig["score"], sig["kelly_fraction"])
            for t, sig in signals.items()
        ]
        result = arbitrate(candidates, initial_cash=1000.0, trade_cost=0.0)
        assert [rec.ticker for rec in result["executed"]] == ["HIGH_SCORE"]

    def test_quantity_matches_kelly_alloc(self, tmp_path):
        """Sizing formulas must agree in £ terms: daemon's whole-share
        compute_quantity() (available_cash*kelly/price, floored) and
        live_sim's continuous alloc=kelly*cash must land on the same £ spend
        when the sizing divides evenly into whole shares (no rounding noise
        to obscure a real formula mismatch)."""
        from Strategy_Auto_Trader.markov_cli.execute import execute_signals
        from Strategy_Auto_Trader.markov_cli.live_sim import arbitrate
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker

        cash, kelly, price = 10_000.0, 0.25, 100.0  # 10000*0.25/100 = 25 shares exactly
        signals = {"SOLO": _signal(score=1.0, kelly=kelly, price=price)}

        portfolio = PortfolioManager(cash, tmp_path / "state.json")
        broker = NullBroker(prices={"SOLO": price})
        with mock.patch(
            "Strategy_Auto_Trader.broker.signal_reader.read_latest_signal",
            side_effect=lambda ticker, _dir: signals[ticker],
        ):
            execute_signals(
                list(signals), tmp_path, portfolio, portfolio.get_limit_tracker(), broker,
            )
        daemon_qty = portfolio.positions["SOLO"]["quantity"]
        daemon_spend = daemon_qty * price

        day = pd.Timestamp("2026-01-12", tz="UTC")
        candidates = [_candidate("SOLO", day, 1.0, kelly, price)]
        result = arbitrate(candidates, initial_cash=cash, trade_cost=0.0)
        livesim_alloc = result["executed"][0].position_size_gbp

        assert daemon_qty == 25
        assert daemon_spend == pytest.approx(livesim_alloc)
        assert livesim_alloc == pytest.approx(cash * kelly)

    def test_kelly_leq_zero_rejected_outright_both_paths(self, tmp_path):
        """kelly_fraction<=0 must produce qty=0 / rejection in both paths —
        never a flat-fallback size (removed 2026-08-11, see
        .claude/rules/cli.md)."""
        from Strategy_Auto_Trader.markov_cli.execute import execute_signals
        from Strategy_Auto_Trader.markov_cli.live_sim import arbitrate
        from Strategy_Auto_Trader.broker.portfolio import PortfolioManager
        from Strategy_Auto_Trader.broker.null_adapter import NullBroker

        signals = {"ZERO": _signal(score=5.0, kelly=0.0), "NEG": _signal(score=5.0, kelly=-0.1)}

        portfolio = PortfolioManager(10_000.0, tmp_path / "state.json")
        broker = NullBroker(prices={t: 100.0 for t in signals})
        with mock.patch(
            "Strategy_Auto_Trader.broker.signal_reader.read_latest_signal",
            side_effect=lambda ticker, _dir: signals[ticker],
        ):
            execute_signals(
                list(signals), tmp_path, portfolio, portfolio.get_limit_tracker(), broker,
            )
        assert portfolio.positions == {}

        day = pd.Timestamp("2026-01-12", tz="UTC")
        candidates = [
            _candidate(t, day, sig["score"], sig["kelly_fraction"])
            for t, sig in signals.items()
        ]
        result = arbitrate(candidates, initial_cash=10_000.0, trade_cost=0.0)
        assert result["executed"] == []
        assert result["n_rejected_kelly"] == 2
