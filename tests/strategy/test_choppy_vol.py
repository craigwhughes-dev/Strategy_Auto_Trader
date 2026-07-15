from __future__ import annotations

import pytest


class TestChoppyVolStrategy:

    # -- Protocol conformance ---

    def test_choppy_vol_entry_satisfies_protocol(self):
        from Strategy_Auto_Trader.strategy.choppy_vol import ChoppyVolEntry
        from Strategy_Auto_Trader.strategy.base import EntryStrategyProtocol
        assert isinstance(ChoppyVolEntry(), EntryStrategyProtocol)

    def test_choppy_vol_exit_satisfies_protocol(self):
        from Strategy_Auto_Trader.strategy.choppy_vol import ChoppyVolExit
        from Strategy_Auto_Trader.strategy.base import ExitStrategyProtocol
        assert isinstance(ChoppyVolExit(), ExitStrategyProtocol)

    # -- Registry resolution --

    def test_resolve_strategy_choppy_vol(self):
        from Strategy_Auto_Trader.strategy.base.registry import resolve_strategy
        from Strategy_Auto_Trader.strategy.choppy_vol import ChoppyVolEntry, ChoppyVolExit
        entry, exit_ = resolve_strategy("choppy_vol")
        assert isinstance(entry, ChoppyVolEntry)
        assert isinstance(exit_, ChoppyVolExit)

    # -- ChoppyVolEntry: entry side --

    def _regime(self):
        from Strategy_Auto_Trader.plugins.types import RegimeState
        return RegimeState(p_bull=0.5, p_bear=0.5, p_bull_smooth=0.5,
                           regime_signal=0.0, hmm_vote=1)

    def test_entry_evaluate_returns_entry_decision(self):
        from Strategy_Auto_Trader.strategy.choppy_vol import ChoppyVolEntry
        from Strategy_Auto_Trader.plugins.types import EntryDecision
        entry = ChoppyVolEntry()
        mom = {"cur_rsi": 25.0, "consolidation": True}
        decision = entry.evaluate(self._regime(), mom, 1.0, currently_in=False)
        assert isinstance(decision, EntryDecision)
        assert decision.flag in ("BUY", "HOLD", "SELL")

    def test_entry_buys_on_oversold_and_consolidation(self):
        from Strategy_Auto_Trader.strategy.choppy_vol import ChoppyVolEntry
        entry = ChoppyVolEntry()
        mom = {"cur_rsi": 30.0, "consolidation": True}
        decision = entry.evaluate(self._regime(), mom, 1.0, currently_in=False)
        assert decision.flag == "BUY"
        assert decision.raw_flag == "BUY"

    def test_entry_holds_when_rsi_not_oversold(self):
        from Strategy_Auto_Trader.strategy.choppy_vol import ChoppyVolEntry
        entry = ChoppyVolEntry()
        mom = {"cur_rsi": 45.0, "consolidation": True}
        decision = entry.evaluate(self._regime(), mom, 1.0, currently_in=False)
        assert decision.flag == "HOLD"

    def test_entry_holds_when_not_consolidating(self):
        from Strategy_Auto_Trader.strategy.choppy_vol import ChoppyVolEntry
        entry = ChoppyVolEntry()
        mom = {"cur_rsi": 25.0, "consolidation": False}
        decision = entry.evaluate(self._regime(), mom, 1.0, currently_in=False)
        assert decision.flag == "HOLD"

    def test_entry_buys_regardless_of_bb_pctb(self):
        """bb_pctb was tested as a separate candidate, never combined with
        rsi<35 AND consolidation — the winning rule doesn't gate on it."""
        from Strategy_Auto_Trader.strategy.choppy_vol import ChoppyVolEntry
        entry = ChoppyVolEntry()
        mom = {"cur_rsi": 25.0, "bb_pctb": 0.9, "consolidation": True}
        decision = entry.evaluate(self._regime(), mom, 1.0, currently_in=False)
        assert decision.flag == "BUY"

    def test_entry_holds_on_missing_indicators(self):
        from Strategy_Auto_Trader.strategy.choppy_vol import ChoppyVolEntry
        entry = ChoppyVolEntry()
        decision = entry.evaluate(self._regime(), {}, 1.0, currently_in=False)
        assert decision.flag == "HOLD"

    def test_entry_ignores_vol_filter_ok_false(self):
        """This strategy IS the alternative to the veto — it must still buy
        even when vol_filter_ok=False, unlike DefaultEntry/OptimisedEntry."""
        from Strategy_Auto_Trader.strategy.choppy_vol import ChoppyVolEntry
        entry = ChoppyVolEntry(vol_filter_ok=False)
        mom = {"cur_rsi": 30.0, "bb_pctb": 0.03, "consolidation": True}
        decision = entry.evaluate(self._regime(), mom, 1.0, currently_in=False)
        assert decision.flag == "BUY"

    def test_entry_while_in_position_exits_on_rsi_reversion(self):
        """RSI-reversion exit now fires via the entry path when currently_in=True
        and cur_rsi >= 50."""
        from Strategy_Auto_Trader.strategy.choppy_vol import ChoppyVolEntry
        entry = ChoppyVolEntry()
        mom = {"cur_rsi": 55.0, "bb_pctb": 0.9, "consolidation": False}
        decision = entry.evaluate(self._regime(), mom, 1.0, currently_in=True)
        assert decision.flag == "SELL"
        assert decision.raw_flag == "SELL"
        assert "RSI" in decision.reason and "reversion" in decision.reason

    def test_entry_while_in_position_holds_when_rsi_not_reverted(self):
        """When in position but RSI hasn't reverted yet, hold."""
        from Strategy_Auto_Trader.strategy.choppy_vol import ChoppyVolEntry
        entry = ChoppyVolEntry()
        mom = {"cur_rsi": 45.0, "consolidation": False}
        decision = entry.evaluate(self._regime(), mom, 1.0, currently_in=True)
        assert decision.flag == "HOLD"

    # -- ChoppyVolExit --

    def _trade(self, days_in_trade=1, entry_price=100.0):
        from Strategy_Auto_Trader.plugins.types import TradeState
        return TradeState(
            position=1.0, entry_price=entry_price, stop_level=entry_price * 0.96,
            target_level=entry_price * 1.06, peak_price_since_entry=entry_price,
            days_in_trade=days_in_trade, entry_bar=0,
        )

    def _bar(self, cur_close=101.0, cur_rsi=None):
        from Strategy_Auto_Trader.plugins.types import BarData
        return BarData(
            t=10, cur_close=cur_close, daily_vol_t=0.01, use_sar_stop=False,
            sar_val=None, need_exit=False, macd_bc=False, rsi_ob=False,
            rsi_ml=False, consol=False, cur_rsi=cur_rsi,
        )

    def test_exit_stop_and_target_values(self):
        from Strategy_Auto_Trader.strategy.choppy_vol import ChoppyVolExit
        ex = ChoppyVolExit()
        assert ex.stop_loss_pct == pytest.approx(0.04)
        assert ex.take_profit_pct == pytest.approx(0.06)

    def test_exit_does_not_fire_on_rsi_reversion(self):
        """RSI reversion exit now lives in ChoppyVolEntry.evaluate()'s
        currently_in=True path, not in ChoppyVolExit.check(). ChoppyVolExit
        should not fire on high RSI."""
        from Strategy_Auto_Trader.strategy.choppy_vol import ChoppyVolExit
        ex = ChoppyVolExit()
        result = ex.check(self._trade(days_in_trade=2), self._bar(cur_close=102.0, cur_rsi=55.0))
        assert result.exit_hit is False

    def test_exit_holds_when_rsi_not_reverted(self):
        from Strategy_Auto_Trader.strategy.choppy_vol import ChoppyVolExit
        ex = ChoppyVolExit()
        result = ex.check(self._trade(days_in_trade=2), self._bar(cur_close=99.5, cur_rsi=40.0))
        assert result.exit_hit is False

    def test_exit_hard_stop_still_fires_below_reversion(self):
        from Strategy_Auto_Trader.strategy.choppy_vol import ChoppyVolExit
        ex = ChoppyVolExit()
        # entry 100, stop level 96 -> close of 95 should trip the hard stop
        trade = self._trade(days_in_trade=3, entry_price=100.0)
        result = ex.check(trade, self._bar(cur_close=95.0, cur_rsi=30.0))
        assert result.exit_hit is True
        assert "rr_stop_loss" in result.sell_reason

    def test_exit_max_hold_bars_backstop(self):
        from Strategy_Auto_Trader.strategy.choppy_vol import ChoppyVolExit
        ex = ChoppyVolExit()
        trade = self._trade(days_in_trade=39, entry_price=100.0)
        result = ex.check(trade, self._bar(cur_close=100.5, cur_rsi=42.0))
        assert result.exit_hit is True
        assert "max_hold" in result.sell_reason

    # -- run.py arg parser

    def test_arg_parser_strategy_choppy_vol(self):
        from Strategy_Auto_Trader.markov_cli.run import _build_arg_parser
        args = _build_arg_parser().parse_args(["--ticker", "SPY", "--strategy", "choppy_vol"])
        assert args.strategy == "choppy_vol"
