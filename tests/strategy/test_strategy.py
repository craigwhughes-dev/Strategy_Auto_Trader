from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestStrategy:

    # -- Protocol conformance ---------------------------------------------------

    def test_default_entry_satisfies_protocol(self):
        from Strategy_Auto_Trader.strategy.default import DefaultEntry
        from Strategy_Auto_Trader.strategy.base import EntryStrategyProtocol
        assert isinstance(DefaultEntry(), EntryStrategyProtocol)

    def test_default_exit_satisfies_protocol(self):
        from Strategy_Auto_Trader.strategy.default import DefaultExit
        from Strategy_Auto_Trader.strategy.base import ExitStrategyProtocol
        assert isinstance(DefaultExit(), ExitStrategyProtocol)

    def test_conservative_entry_satisfies_protocol(self):
        from Strategy_Auto_Trader.strategy.conservative import ConservativeEntry
        from Strategy_Auto_Trader.strategy.base import EntryStrategyProtocol
        assert isinstance(ConservativeEntry(), EntryStrategyProtocol)

    def test_conservative_exit_satisfies_protocol(self):
        from Strategy_Auto_Trader.strategy.conservative import ConservativeExit
        from Strategy_Auto_Trader.strategy.base import ExitStrategyProtocol
        assert isinstance(ConservativeExit(), ExitStrategyProtocol)

    def test_trend_entry_satisfies_protocol(self):
        from Strategy_Auto_Trader.strategy.trend_follow import TrendEntry
        from Strategy_Auto_Trader.strategy.base import EntryStrategyProtocol
        assert isinstance(TrendEntry(), EntryStrategyProtocol)

    def test_trend_exit_satisfies_protocol(self):
        from Strategy_Auto_Trader.strategy.trend_follow import TrendExit
        from Strategy_Auto_Trader.strategy.base import ExitStrategyProtocol
        assert isinstance(TrendExit(), ExitStrategyProtocol)

    def test_optimised_entry_satisfies_protocol(self):
        from Strategy_Auto_Trader.strategy.optimised import OptimisedEntry
        from Strategy_Auto_Trader.strategy.base import EntryStrategyProtocol
        assert isinstance(OptimisedEntry(), EntryStrategyProtocol)

    def test_optimised_exit_satisfies_protocol(self):
        from Strategy_Auto_Trader.strategy.optimised import OptimisedExit
        from Strategy_Auto_Trader.strategy.base import ExitStrategyProtocol
        assert isinstance(OptimisedExit(), ExitStrategyProtocol)

    # -- Registry ---------------------------------------------------------------

    def test_resolve_strategy_returns_entry_exit_pair(self):
        from Strategy_Auto_Trader.strategy.base.registry import resolve_strategy
        from Strategy_Auto_Trader.strategy.base import EntryStrategyProtocol, ExitStrategyProtocol
        entry, exit_ = resolve_strategy("default")
        assert isinstance(entry, EntryStrategyProtocol)
        assert isinstance(exit_, ExitStrategyProtocol)

    def test_resolve_strategy_conservative(self):
        from Strategy_Auto_Trader.strategy.base.registry import resolve_strategy
        from Strategy_Auto_Trader.strategy.conservative import ConservativeEntry, ConservativeExit
        entry, exit_ = resolve_strategy("conservative")
        assert isinstance(entry, ConservativeEntry)
        assert isinstance(exit_, ConservativeExit)

    def test_resolve_strategy_trend(self):
        from Strategy_Auto_Trader.strategy.base.registry import resolve_strategy
        from Strategy_Auto_Trader.strategy.trend_follow import TrendEntry, TrendExit
        entry, exit_ = resolve_strategy("trend")
        assert isinstance(entry, TrendEntry)
        assert isinstance(exit_, TrendExit)

    def test_resolve_strategy_optimised(self):
        from Strategy_Auto_Trader.strategy.base.registry import resolve_strategy
        from Strategy_Auto_Trader.strategy.optimised import OptimisedEntry, OptimisedExit
        entry, exit_ = resolve_strategy("optimised")
        assert isinstance(entry, OptimisedEntry)
        assert isinstance(exit_, OptimisedExit)

    def test_resolve_strategy_unknown_raises(self):
        from Strategy_Auto_Trader.strategy.base.registry import resolve_strategy
        with pytest.raises(KeyError, match="Unknown strategy"):
            resolve_strategy("banana")

    # -- DefaultEntry behaviour -------------------------------------------------

    def test_default_entry_evaluate_returns_entry_decision(self):
        from Strategy_Auto_Trader.strategy.default import DefaultEntry
        from Strategy_Auto_Trader.plugins.types import RegimeState, EntryDecision
        entry = DefaultEntry()
        regime = RegimeState(p_bull=0.8, p_bear=0.2, p_bull_smooth=0.8,
                             regime_signal=0.6, hmm_vote=2)
        mom = {"cur_rsi": 60.0, "recent_cross_above_50": True,
               "recent_cross_below_40": False, "above_sma20": True,
               "above_sma50": True, "above_sma200": True, "volume_ratio": 1.3}
        decision = entry.evaluate(regime, mom, 1.3, currently_in=False)
        assert isinstance(decision, EntryDecision)
        assert decision.flag in ("BUY", "HOLD", "SELL")
        assert decision.raw_flag in ("BUY", "HOLD", "SELL")
        assert isinstance(decision.score, float)

    def test_default_entry_strong_bull_gives_buy(self):
        from Strategy_Auto_Trader.strategy.default import DefaultEntry
        from Strategy_Auto_Trader.plugins.types import RegimeState
        entry = DefaultEntry()
        regime = RegimeState(p_bull=0.9, p_bear=0.1, p_bull_smooth=0.9,
                             regime_signal=0.8, hmm_vote=2)
        mom = {"cur_rsi": 65.0, "recent_cross_above_50": True,
               "recent_cross_below_40": False, "above_sma20": True,
               "above_sma50": True, "above_sma200": True, "volume_ratio": 1.5}
        decision = entry.evaluate(regime, mom, 1.5, currently_in=False)
        assert decision.raw_flag == "BUY"

    def test_default_entry_strong_bear_gives_sell(self):
        from Strategy_Auto_Trader.strategy.default import DefaultEntry
        from Strategy_Auto_Trader.plugins.types import RegimeState
        entry = DefaultEntry()
        regime = RegimeState(p_bull=0.1, p_bear=0.9, p_bull_smooth=0.1,
                             regime_signal=-0.8, hmm_vote=0)
        mom = {"cur_rsi": 25.0, "recent_cross_above_50": False,
               "recent_cross_below_40": True, "above_sma20": False,
               "above_sma50": False, "above_sma200": False, "volume_ratio": 0.5}
        decision = entry.evaluate(regime, mom, 0.5, currently_in=False)
        assert decision.raw_flag == "SELL"

    # -- OptimisedEntry behaviour ----------------------------------------------

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
        # veto only fires if the gate let a BUY through; either way never BUY
        assert decision.flag != "BUY"

    def test_optimised_veto_does_not_block_exits_while_in(self):
        from Strategy_Auto_Trader.strategy.optimised import OptimisedEntry
        from Strategy_Auto_Trader.plugins.types import RegimeState
        # overbought RSI while already in a position: decision must match the
        # un-vetoed base behaviour (veto applies to new entries only)
        regime = RegimeState(p_bull=0.9, p_bear=0.1, p_bull_smooth=0.9,
                             regime_signal=0.8, hmm_vote=2)
        mom = {"cur_rsi": 75.0, "recent_cross_above_50": True,
               "recent_cross_below_40": False, "above_sma20": True,
               "above_sma50": True, "above_sma200": True, "volume_ratio": 1.5}
        decision = OptimisedEntry().evaluate(regime, mom, 1.5, currently_in=True)
        assert "optimised veto" not in decision.reason

    def test_optimised_has_highest_buy_threshold_fraction(self):
        # conviction bar (threshold as fraction of max score) is the strictest
        from Strategy_Auto_Trader.strategy.optimised import OptimisedEntry
        from Strategy_Auto_Trader.strategy.trend_follow import TrendEntry
        opt = OptimisedEntry.buy_threshold / sum(OptimisedEntry.weights.values())
        trend = TrendEntry.buy_threshold / sum(TrendEntry.weights.values())
        assert opt > trend

    # -- ExitStrategy stop/target properties -----------------------------------

    def test_default_exit_stop_and_target(self):
        from Strategy_Auto_Trader.strategy.default import DefaultExit
        ex = DefaultExit()
        assert ex.stop_loss_pct == pytest.approx(0.05)
        assert ex.take_profit_pct == pytest.approx(0.15)

    def test_conservative_exit_stop_and_target(self):
        from Strategy_Auto_Trader.strategy.conservative import ConservativeExit
        ex = ConservativeExit()
        assert ex.stop_loss_pct == pytest.approx(0.03)
        assert ex.take_profit_pct == pytest.approx(0.10)

    def test_trend_exit_stop_and_target(self):
        from Strategy_Auto_Trader.strategy.trend_follow import TrendExit
        ex = TrendExit()
        assert ex.stop_loss_pct == pytest.approx(0.08)
        assert ex.take_profit_pct == pytest.approx(0.30)

    def test_optimised_exit_stop_and_target(self):
        from Strategy_Auto_Trader.strategy.optimised import OptimisedExit
        ex = OptimisedExit()
        assert ex.stop_loss_pct == pytest.approx(0.08)
        assert ex.take_profit_pct == pytest.approx(0.30)

    # -- Conservative weights are stricter ------------------------------------

    def test_conservative_has_higher_buy_threshold(self):
        from Strategy_Auto_Trader.strategy.conservative import ConservativeEntry
        from Strategy_Auto_Trader.strategy.default import DefaultEntry
        assert ConservativeEntry.buy_threshold > DefaultEntry.buy_threshold

    def test_conservative_weights_sma200_heavier(self):
        from Strategy_Auto_Trader.strategy.conservative import ConservativeEntry
        from Strategy_Auto_Trader.strategy.default import DefaultEntry
        assert ConservativeEntry.weights["sma200"] > DefaultEntry.weights["sma200"]

    # -- run.py arg parser accepts --strategy ---------------------------------

    def test_arg_parser_strategy_default(self):
        from Strategy_Auto_Trader.markov_cli.run import _build_arg_parser
        args = _build_arg_parser().parse_args(["--ticker", "SPY"])
        assert args.strategy == "default"

    def test_arg_parser_strategy_conservative(self):
        from Strategy_Auto_Trader.markov_cli.run import _build_arg_parser
        args = _build_arg_parser().parse_args(["--ticker", "SPY", "--strategy", "conservative"])
        assert args.strategy == "conservative"

    def test_arg_parser_strategy_trend(self):
        from Strategy_Auto_Trader.markov_cli.run import _build_arg_parser
        args = _build_arg_parser().parse_args(["--ticker", "SPY", "--strategy", "trend"])
        assert args.strategy == "trend"

    def test_arg_parser_strategy_optimised(self):
        from Strategy_Auto_Trader.markov_cli.run import _build_arg_parser
        args = _build_arg_parser().parse_args(["--ticker", "SPY", "--strategy", "optimised"])
        assert args.strategy == "optimised"

    def test_arg_parser_unknown_strategy_rejected(self):
        from Strategy_Auto_Trader.markov_cli.run import _build_arg_parser
        with pytest.raises(SystemExit):
            _build_arg_parser().parse_args(["--ticker", "SPY", "--strategy", "neural_net"])

    # -- batch._build_argv passes --strategy -----------------------------------

    def test_build_argv_strategy_conservative(self):
        from Strategy_Auto_Trader.markov_cli.batch import _build_argv
        cfg = {"ticker": "AAPL", "strategy": "conservative"}
        argv = _build_argv(cfg, {})
        assert "--strategy" in argv
        assert argv[argv.index("--strategy") + 1] == "conservative"

    def test_build_argv_no_strategy_key(self):
        from Strategy_Auto_Trader.markov_cli.batch import _build_argv
        cfg = {"ticker": "AAPL"}
        argv = _build_argv(cfg, {})
        assert "--strategy" not in argv

    # -- Engine integration: entry_strategy kwarg wires through ---------------

    def test_engine_accepts_strategy_kwargs(self):
        from Strategy_Auto_Trader.quant_hmm.consolidated_engine import consolidated_backtest
        from Strategy_Auto_Trader.strategy.default import DefaultEntry, DefaultExit
        close = np.concatenate([
            np.linspace(100, 130, 400),
            np.linspace(130, 120, 200),
            np.linspace(120, 140, 200),
        ])
        dates = pd.date_range("2022-01-03", periods=800, freq="h")
        df = pd.DataFrame({"Close": close, "Volume": np.full(800, 1e6)}, index=dates)
        bt = consolidated_backtest(df, entry_strategy=DefaultEntry(), exit_strategy=DefaultExit())
        assert "n_bars" in bt
        assert bt["n_bars"] >= 0
