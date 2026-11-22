from __future__ import annotations

from Strategy_Auto_Trader.plugins.types import BarData, RegimeState, TradeState
from Strategy_Auto_Trader.strategy.base import EntryStrategyProtocol, ExitStrategyProtocol
from Strategy_Auto_Trader.strategy.turtle_breakout import TurtleEntry, TurtleExit

_REGIME = RegimeState(p_bull=0.0, p_bear=0.0, p_bull_smooth=0.0, regime_signal=None, hmm_vote=None)


class TestTurtleEntry:

    def test_satisfies_protocol(self):
        assert isinstance(TurtleEntry(), EntryStrategyProtocol)

    def test_buy_when_close_above_breakout_level(self):
        entry = TurtleEntry()
        mom = {"cur_close": 105.0, "breakout_entry_level": 100.0}
        decision = entry.evaluate(_REGIME, mom, 1.0, currently_in=False)
        assert decision.flag == "BUY"
        assert decision.raw_flag == "BUY"

    def test_hold_when_close_at_or_below_breakout_level(self):
        entry = TurtleEntry()
        mom = {"cur_close": 100.0, "breakout_entry_level": 100.0}
        decision = entry.evaluate(_REGIME, mom, 1.0, currently_in=False)
        assert decision.flag == "HOLD"

    def test_hold_when_level_not_yet_warmed_up(self):
        entry = TurtleEntry()
        mom = {"cur_close": 105.0}
        decision = entry.evaluate(_REGIME, mom, 1.0, currently_in=False)
        assert decision.flag == "HOLD"
        assert "insufficient history" in decision.reason

    def test_vol_filter_vetoes_to_hold(self):
        entry = TurtleEntry(vol_filter_ok=False)
        mom = {"cur_close": 105.0, "breakout_entry_level": 100.0}
        decision = entry.evaluate(_REGIME, mom, 1.0, currently_in=False)
        assert decision.flag == "HOLD"
        assert "vol_filter" in decision.reason

    def test_weights_all_zero_so_hmm_and_rsi_are_skipped(self):
        entry = TurtleEntry()
        assert all(w == 0.0 for w in entry._weights.values())

    def test_volume_min_ratio_is_zero(self):
        assert TurtleEntry.volume_min_ratio == 0.0

    def test_entry_lookback_days_override_flows_through(self):
        entry = TurtleEntry(entry_lookback_days=30)
        assert entry.entry_lookback_days == 30


class TestTurtleExit:

    def test_satisfies_protocol(self):
        assert isinstance(TurtleExit(), ExitStrategyProtocol)

    def test_take_profit_is_infinite(self):
        assert TurtleExit().take_profit_pct == float("inf")

    def test_entry_levels_sets_atr_stop_below_entry(self):
        exit_ = TurtleExit(atr_mult=2.0)
        stop, target = exit_.entry_levels(entry_price=100.0, atr_at_entry=5.0)
        assert stop == 100.0 - 2.0 * 5.0
        assert target == float("inf")

    def test_entry_levels_falls_back_to_flat_stop_without_atr(self):
        exit_ = TurtleExit()
        stop, target = exit_.entry_levels(entry_price=100.0, atr_at_entry=None)
        assert stop == 95.0
        assert target == float("inf")

    def test_check_exits_on_fixed_stop_when_no_trailing_level(self):
        exit_ = TurtleExit()
        trade = TradeState(position=1.0, entry_price=100.0, stop_level=90.0, target_level=float("inf"),
                            peak_price_since_entry=100.0, days_in_trade=1, entry_bar=0)
        bar = BarData(t=1, cur_close=89.0, daily_vol_t=0.0, use_sar_stop=False, sar_val=None,
                      need_exit=False, macd_bc=False, rsi_ob=False, rsi_ml=False, consol=False,
                      breakout_exit_level=None)
        result = exit_.check(trade, bar)
        assert result.exit_hit is True
        assert result.sell_reason == "turtle_fixed_stop"

    def test_check_uses_trailing_level_when_it_is_the_nearer_stop(self):
        exit_ = TurtleExit()
        # Fixed stop at 90, trailing stop (rolling low) risen to 95 — trailing
        # is nearer (higher) so it should govern the exit.
        trade = TradeState(position=1.0, entry_price=100.0, stop_level=90.0, target_level=float("inf"),
                            peak_price_since_entry=110.0, days_in_trade=5, entry_bar=0)
        bar = BarData(t=5, cur_close=94.0, daily_vol_t=0.0, use_sar_stop=False, sar_val=None,
                      need_exit=False, macd_bc=False, rsi_ob=False, rsi_ml=False, consol=False,
                      breakout_exit_level=95.0)
        result = exit_.check(trade, bar)
        assert result.exit_hit is True
        assert result.sell_reason == "turtle_trailing_stop"

    def test_check_no_exit_when_close_above_both_stops(self):
        exit_ = TurtleExit()
        trade = TradeState(position=1.0, entry_price=100.0, stop_level=90.0, target_level=float("inf"),
                            peak_price_since_entry=110.0, days_in_trade=5, entry_bar=0)
        bar = BarData(t=5, cur_close=96.0, daily_vol_t=0.0, use_sar_stop=False, sar_val=None,
                      need_exit=False, macd_bc=False, rsi_ob=False, rsi_ml=False, consol=False,
                      breakout_exit_level=95.0)
        result = exit_.check(trade, bar)
        assert result.exit_hit is False

    def test_min_hold_bars_is_zero(self):
        assert TurtleExit.min_hold_bars == 0


class TestTurtleRegistry:

    def test_resolve_strategy_turtle_breakout(self):
        from Strategy_Auto_Trader.strategy.base.registry import resolve_strategy
        entry, exit_ = resolve_strategy("turtle_breakout")
        assert isinstance(entry, TurtleEntry)
        assert isinstance(exit_, TurtleExit)
