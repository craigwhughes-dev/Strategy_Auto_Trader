from __future__ import annotations

import pytest


class TestOptimisedAggressiveStrategy:

    # -- Protocol conformance ---

    def test_entry_satisfies_protocol(self):
        from Strategy_Auto_Trader.strategy.optimised_aggressive import OptimisedAggressiveEntry
        from Strategy_Auto_Trader.strategy.base import EntryStrategyProtocol
        assert isinstance(OptimisedAggressiveEntry(), EntryStrategyProtocol)

    def test_exit_satisfies_protocol(self):
        from Strategy_Auto_Trader.strategy.optimised_aggressive import OptimisedAggressiveExit
        from Strategy_Auto_Trader.strategy.base import ExitStrategyProtocol
        assert isinstance(OptimisedAggressiveExit(), ExitStrategyProtocol)

    # -- Registry resolution --

    def test_resolve_strategy_optimised_aggressive(self):
        from Strategy_Auto_Trader.strategy.base.registry import resolve_strategy
        from Strategy_Auto_Trader.strategy.optimised_aggressive import (
            OptimisedAggressiveEntry, OptimisedAggressiveExit,
        )
        entry, exit_ = resolve_strategy("optimised_aggressive")
        assert isinstance(entry, OptimisedAggressiveEntry)
        assert isinstance(exit_, OptimisedAggressiveExit)

    # -- Flip-guard override --

    def test_require_flip_entry_is_false(self):
        from Strategy_Auto_Trader.strategy.optimised_aggressive import OptimisedAggressiveEntry
        assert OptimisedAggressiveEntry.require_flip_entry is False

    def test_optimised_still_requires_flip(self):
        # Sanity check: the base strategy is untouched by this subclass.
        from Strategy_Auto_Trader.strategy.optimised import OptimisedEntry
        assert getattr(OptimisedEntry, "require_flip_entry", True) is True

    # -- Same signal/veto/exit behaviour as OptimisedEntry (inherited) --

    def test_entry_strong_bull_gives_buy(self):
        from Strategy_Auto_Trader.strategy.optimised_aggressive import OptimisedAggressiveEntry
        from Strategy_Auto_Trader.plugins.types import RegimeState
        entry = OptimisedAggressiveEntry()
        regime = RegimeState(p_bull=0.9, p_bear=0.1, p_bull_smooth=0.9,
                             regime_signal=0.8, hmm_vote=2)
        mom = {"cur_rsi": 65.0, "recent_cross_above_50": True,
               "recent_cross_below_40": False, "above_sma20": True,
               "above_sma50": True, "above_sma200": True, "volume_ratio": 1.5}
        decision = entry.evaluate(regime, mom, 1.5, currently_in=False)
        assert decision.raw_flag == "BUY"
        assert decision.flag == "BUY"

    def test_entry_vetoes_overbought_rsi(self):
        from Strategy_Auto_Trader.strategy.optimised_aggressive import OptimisedAggressiveEntry
        from Strategy_Auto_Trader.plugins.types import RegimeState
        entry = OptimisedAggressiveEntry()
        regime = RegimeState(p_bull=0.9, p_bear=0.1, p_bull_smooth=0.9,
                             regime_signal=0.8, hmm_vote=2)
        mom = {"cur_rsi": 75.0, "recent_cross_above_50": True,
               "recent_cross_below_40": False, "above_sma20": True,
               "above_sma50": True, "above_sma200": True, "volume_ratio": 1.5}
        decision = entry.evaluate(regime, mom, 1.5, currently_in=False)
        assert decision.flag == "HOLD"
        assert "RSI" in decision.reason

    def test_exit_stop_and_target_match_optimised(self):
        from Strategy_Auto_Trader.strategy.optimised_aggressive import OptimisedAggressiveExit
        ex = OptimisedAggressiveExit()
        assert ex.stop_loss_pct == pytest.approx(0.08)
        assert ex.take_profit_pct == pytest.approx(0.30)

    # -- run.py arg parser

    def test_arg_parser_strategy_optimised_aggressive(self):
        from Strategy_Auto_Trader.markov_cli.run import _build_arg_parser
        args = _build_arg_parser().parse_args(
            ["--ticker", "SPY", "--strategy", "optimised_aggressive"]
        )
        assert args.strategy == "optimised_aggressive"
