from __future__ import annotations

import pytest


class TestOptimisedStrategy:

    # -- Protocol conformance ---

    def test_optimised_entry_satisfies_protocol(self):
        from Strategy_Auto_Trader.strategy.optimised import OptimisedEntry
        from Strategy_Auto_Trader.strategy.base import EntryStrategyProtocol
        assert isinstance(OptimisedEntry(), EntryStrategyProtocol)

    def test_optimised_exit_satisfies_protocol(self):
        from Strategy_Auto_Trader.strategy.optimised import OptimisedExit
        from Strategy_Auto_Trader.strategy.base import ExitStrategyProtocol
        assert isinstance(OptimisedExit(), ExitStrategyProtocol)

    # -- Registry resolution --

    def test_resolve_strategy_optimised(self):
        from Strategy_Auto_Trader.strategy.base.registry import resolve_strategy
        from Strategy_Auto_Trader.strategy.optimised import OptimisedEntry, OptimisedExit
        entry, exit_ = resolve_strategy("optimised")
        assert isinstance(entry, OptimisedEntry)
        assert isinstance(exit_, OptimisedExit)

    # -- OptimisedEntry behaviour

    def test_optimised_entry_evaluate_returns_entry_decision(self):
        from Strategy_Auto_Trader.strategy.optimised import OptimisedEntry
        from Strategy_Auto_Trader.plugins.types import RegimeState, EntryDecision
        entry = OptimisedEntry()
        regime = RegimeState(p_bull=0.8, p_bear=0.2, p_bull_smooth=0.8,
                             regime_signal=0.6, hmm_vote=2)
        mom = {"cur_rsi": 60.0, "recent_cross_above_50": True,
               "recent_cross_below_40": False, "above_sma20": True,
               "above_sma50": True, "above_sma200": True, "volume_ratio": 1.3}
        decision = entry.evaluate(regime, mom, 1.3, currently_in=False)
        assert isinstance(decision, EntryDecision)
        assert decision.flag in ("BUY", "HOLD", "SELL")
        assert decision.raw_flag in ("BUY", "HOLD", "SELL")

    def test_optimised_entry_strong_bull_gives_buy(self):
        from Strategy_Auto_Trader.strategy.optimised import OptimisedEntry
        from Strategy_Auto_Trader.plugins.types import RegimeState
        entry = OptimisedEntry()
        regime = RegimeState(p_bull=0.9, p_bear=0.1, p_bull_smooth=0.9,
                             regime_signal=0.8, hmm_vote=2)
        mom = {"cur_rsi": 65.0, "recent_cross_above_50": True,
               "recent_cross_below_40": False, "above_sma20": True,
               "above_sma50": True, "above_sma200": True, "volume_ratio": 1.5}
        decision = entry.evaluate(regime, mom, 1.5, currently_in=False)
        assert decision.raw_flag == "BUY"
        assert decision.flag == "BUY"

    def test_optimised_entry_vetoes_overbought_rsi(self):
        from Strategy_Auto_Trader.strategy.optimised import OptimisedEntry
        from Strategy_Auto_Trader.plugins.types import RegimeState
        entry = OptimisedEntry()
        regime = RegimeState(p_bull=0.9, p_bear=0.1, p_bull_smooth=0.9,
                             regime_signal=0.8, hmm_vote=2)
        mom = {"cur_rsi": 75.0, "recent_cross_above_50": True,
               "recent_cross_below_40": False, "above_sma20": True,
               "above_sma50": True, "above_sma200": True, "volume_ratio": 1.5}
        decision = entry.evaluate(regime, mom, 1.5, currently_in=False)
        assert decision.flag == "HOLD"
        assert "RSI" in decision.reason

    def test_optimised_entry_vetoes_negative_regime(self):
        from Strategy_Auto_Trader.strategy.optimised import OptimisedEntry
        from Strategy_Auto_Trader.plugins.types import RegimeState
        entry = OptimisedEntry()
        regime = RegimeState(p_bull=0.9, p_bear=0.1, p_bull_smooth=0.9,
                             regime_signal=-0.1, hmm_vote=2)
        mom = {"cur_rsi": 65.0, "recent_cross_above_50": True,
               "recent_cross_below_40": False, "above_sma20": True,
               "above_sma50": True, "above_sma200": True, "volume_ratio": 1.5}
        decision = entry.evaluate(regime, mom, 1.5, currently_in=False)
        assert decision.flag != "BUY"

    def test_optimised_veto_does_not_block_exits_while_in(self):
        from Strategy_Auto_Trader.strategy.optimised import OptimisedEntry
        from Strategy_Auto_Trader.plugins.types import RegimeState
        regime = RegimeState(p_bull=0.9, p_bear=0.1, p_bull_smooth=0.9,
                             regime_signal=0.8, hmm_vote=2)
        mom = {"cur_rsi": 75.0, "recent_cross_above_50": True,
               "recent_cross_below_40": False, "above_sma20": True,
               "above_sma50": True, "above_sma200": True, "volume_ratio": 1.5}
        decision = OptimisedEntry().evaluate(regime, mom, 1.5, currently_in=True)
        assert "optimised veto" not in decision.reason

    def test_optimised_has_highest_buy_threshold_fraction(self):
        from Strategy_Auto_Trader.strategy.optimised import OptimisedEntry
        from Strategy_Auto_Trader.strategy.trend_follow import TrendEntry
        opt = OptimisedEntry.buy_threshold / sum(OptimisedEntry.weights.values())
        trend = TrendEntry.buy_threshold / sum(TrendEntry.weights.values())
        assert opt > trend

    def test_optimised_exit_stop_and_target(self):
        from Strategy_Auto_Trader.strategy.optimised import OptimisedExit
        ex = OptimisedExit()
        assert ex.stop_loss_pct == pytest.approx(0.08)
        assert ex.take_profit_pct == pytest.approx(0.30)

    # -- run.py arg parser

    def test_arg_parser_strategy_optimised(self):
        from Strategy_Auto_Trader.markov_cli.run import _build_arg_parser
        args = _build_arg_parser().parse_args(["--ticker", "SPY", "--strategy", "optimised"])
        assert args.strategy == "optimised"
