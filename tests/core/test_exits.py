from __future__ import annotations

import numpy as np
import pytest


class TestExits:

    # --- _effective_stop_for_bar -------------------------------------------

    def test_effective_stop_for_bar_vol_scaled(self):
        from Strategy_Auto_Trader.core.exits import _effective_stop_for_bar
        stop = _effective_stop_for_bar(
            vol_stop_mult=2.0, daily_vol_t=0.02, vol_stop_window=20, trailing_stop=0.0,
            current_position=0.0, profit_stop_scale=0.0, entry_price=0.0, cur_close=100.0,
            min_stop_pct=0.05,
        )
        assert abs(stop - (2.0 * 0.02 * np.sqrt(20))) < 1e-9

    def test_effective_stop_for_bar_vol_unavailable_falls_back_to_fixed(self):
        from Strategy_Auto_Trader.core.exits import _effective_stop_for_bar
        stop = _effective_stop_for_bar(
            vol_stop_mult=2.0, daily_vol_t=float("nan"), vol_stop_window=20, trailing_stop=0.30,
            current_position=0.0, profit_stop_scale=0.0, entry_price=0.0, cur_close=100.0,
            min_stop_pct=0.05,
        )
        assert stop == 0.30

    def test_effective_stop_for_bar_fixed_only(self):
        from Strategy_Auto_Trader.core.exits import _effective_stop_for_bar
        stop = _effective_stop_for_bar(
            vol_stop_mult=0.0, daily_vol_t=0.02, vol_stop_window=20, trailing_stop=0.30,
            current_position=0.0, profit_stop_scale=0.0, entry_price=0.0, cur_close=100.0,
            min_stop_pct=0.05,
        )
        assert stop == 0.30

    def test_effective_stop_for_bar_profit_scale_narrows_stop(self):
        from Strategy_Auto_Trader.core.exits import _effective_stop_for_bar
        # entry 100, now 140 -> 40% unrealised gain; base stop 0.30, scale=0.5
        # -> 0.30 - 0.40*0.5 = 0.10
        stop = _effective_stop_for_bar(
            vol_stop_mult=0.0, daily_vol_t=0.0, vol_stop_window=20, trailing_stop=0.30,
            current_position=1.0, profit_stop_scale=0.5, entry_price=100.0, cur_close=140.0,
            min_stop_pct=0.05,
        )
        assert abs(stop - 0.10) < 1e-9

    def test_effective_stop_for_bar_profit_scale_floored_at_min(self):
        from Strategy_Auto_Trader.core.exits import _effective_stop_for_bar
        # Huge unrealised gain would drive stop negative -> floored at min_stop_pct
        stop = _effective_stop_for_bar(
            vol_stop_mult=0.0, daily_vol_t=0.0, vol_stop_window=20, trailing_stop=0.30,
            current_position=1.0, profit_stop_scale=0.5, entry_price=100.0, cur_close=1000.0,
            min_stop_pct=0.05,
        )
        assert stop == 0.05

    # --- _check_exit_conditions ---------------------------------------------

    def test_check_exit_conditions_flat_returns_no_exit(self):
        from Strategy_Auto_Trader.core.exits import _check_exit_conditions
        hit, reason, peak, days = _check_exit_conditions(
            current_position=0.0, cur_close=100.0, entry_price=0.0,
            rr_ratio=0.0, rr_risk=0.05, rr_stop_level=0.0, rr_target_level=0.0,
            effective_stop=0.0, peak_price_since_entry=0.0,
            max_hold_days=0, days_in_trade=0,
            use_sar_stop=False, sar_val=None,
            need_exit=False, macd_bearish_cross=False,
            exit_on_macd_cross=False, exit_on_rsi_reversal=False,
            rsi_overbought_exit=False, rsi_momentum_loss=False,
            exit_on_consolidation=False, consolidating=False,
        )
        assert hit is False and reason == "" and peak == 0.0 and days == 0

    def test_check_exit_conditions_rr_stop_loss_fires(self):
        from Strategy_Auto_Trader.core.exits import _check_exit_conditions
        hit, reason, peak, days = _check_exit_conditions(
            current_position=1.0, cur_close=94.0, entry_price=100.0,
            rr_ratio=3.0, rr_risk=0.05, rr_stop_level=95.0, rr_target_level=115.0,
            effective_stop=0.0, peak_price_since_entry=100.0,
            max_hold_days=0, days_in_trade=0,
            use_sar_stop=False, sar_val=None,
            need_exit=False, macd_bearish_cross=False,
            exit_on_macd_cross=False, exit_on_rsi_reversal=False,
            rsi_overbought_exit=False, rsi_momentum_loss=False,
            exit_on_consolidation=False, consolidating=False,
        )
        assert hit is True
        assert "rr_stop_loss" in reason

    def test_check_exit_conditions_rr_take_profit_fires(self):
        from Strategy_Auto_Trader.core.exits import _check_exit_conditions
        hit, reason, peak, days = _check_exit_conditions(
            current_position=1.0, cur_close=120.0, entry_price=100.0,
            rr_ratio=3.0, rr_risk=0.05, rr_stop_level=95.0, rr_target_level=115.0,
            effective_stop=0.0, peak_price_since_entry=120.0,
            max_hold_days=0, days_in_trade=0,
            use_sar_stop=False, sar_val=None,
            need_exit=False, macd_bearish_cross=False,
            exit_on_macd_cross=False, exit_on_rsi_reversal=False,
            rsi_overbought_exit=False, rsi_momentum_loss=False,
            exit_on_consolidation=False, consolidating=False,
        )
        assert hit is True
        assert "rr_take_profit" in reason

    def test_check_exit_conditions_rr_takes_priority_over_trailing_stop(self):
        from Strategy_Auto_Trader.core.exits import _check_exit_conditions
        # Both R:R stop AND trailing stop would fire this bar -> R:R should win
        hit, reason, peak, days = _check_exit_conditions(
            current_position=1.0, cur_close=94.0, entry_price=100.0,
            rr_ratio=3.0, rr_risk=0.05, rr_stop_level=95.0, rr_target_level=115.0,
            effective_stop=0.05, peak_price_since_entry=100.0,
            max_hold_days=0, days_in_trade=0,
            use_sar_stop=False, sar_val=None,
            need_exit=False, macd_bearish_cross=False,
            exit_on_macd_cross=False, exit_on_rsi_reversal=False,
            rsi_overbought_exit=False, rsi_momentum_loss=False,
            exit_on_consolidation=False, consolidating=False,
        )
        assert hit is True
        assert "rr_stop_loss" in reason

    def test_check_exit_conditions_trailing_stop_fires(self):
        from Strategy_Auto_Trader.core.exits import _check_exit_conditions
        # peak 120, now 110 -> 8.3% drop from peak, still +10% from entry of 100
        hit, reason, peak, days = _check_exit_conditions(
            current_position=1.0, cur_close=110.0, entry_price=100.0,
            rr_ratio=0.0, rr_risk=0.05, rr_stop_level=0.0, rr_target_level=0.0,
            effective_stop=0.05, peak_price_since_entry=120.0,
            max_hold_days=0, days_in_trade=0,
            use_sar_stop=False, sar_val=None,
            need_exit=False, macd_bearish_cross=False,
            exit_on_macd_cross=False, exit_on_rsi_reversal=False,
            rsi_overbought_exit=False, rsi_momentum_loss=False,
            exit_on_consolidation=False, consolidating=False,
        )
        assert hit is True
        assert "trailing_stop" in reason
        assert peak == 120.0

    def test_check_exit_conditions_trailing_stop_updates_peak(self):
        from Strategy_Auto_Trader.core.exits import _check_exit_conditions
        # New high at 130 -> peak updates even though stop doesn't fire
        hit, reason, peak, days = _check_exit_conditions(
            current_position=1.0, cur_close=130.0, entry_price=100.0,
            rr_ratio=0.0, rr_risk=0.05, rr_stop_level=0.0, rr_target_level=0.0,
            effective_stop=0.30, peak_price_since_entry=120.0,
            max_hold_days=0, days_in_trade=0,
            use_sar_stop=False, sar_val=None,
            need_exit=False, macd_bearish_cross=False,
            exit_on_macd_cross=False, exit_on_rsi_reversal=False,
            rsi_overbought_exit=False, rsi_momentum_loss=False,
            exit_on_consolidation=False, consolidating=False,
        )
        assert hit is False
        assert peak == 130.0

    def test_check_exit_conditions_max_hold_days_fires(self):
        from Strategy_Auto_Trader.core.exits import _check_exit_conditions
        hit, reason, peak, days = _check_exit_conditions(
            current_position=1.0, cur_close=105.0, entry_price=100.0,
            rr_ratio=0.0, rr_risk=0.05, rr_stop_level=0.0, rr_target_level=0.0,
            effective_stop=0.0, peak_price_since_entry=105.0,
            max_hold_days=5, days_in_trade=4,
            use_sar_stop=False, sar_val=None,
            need_exit=False, macd_bearish_cross=False,
            exit_on_macd_cross=False, exit_on_rsi_reversal=False,
            rsi_overbought_exit=False, rsi_momentum_loss=False,
            exit_on_consolidation=False, consolidating=False,
        )
        assert hit is True
        assert reason == "max_hold(5d)"
        assert days == 5

    def test_check_exit_conditions_sar_stop_fires(self):
        from Strategy_Auto_Trader.core.exits import _check_exit_conditions
        hit, reason, peak, days = _check_exit_conditions(
            current_position=1.0, cur_close=95.0, entry_price=100.0,
            rr_ratio=0.0, rr_risk=0.05, rr_stop_level=0.0, rr_target_level=0.0,
            effective_stop=0.0, peak_price_since_entry=110.0,
            max_hold_days=0, days_in_trade=0,
            use_sar_stop=True, sar_val=96.0,
            need_exit=False, macd_bearish_cross=False,
            exit_on_macd_cross=False, exit_on_rsi_reversal=False,
            rsi_overbought_exit=False, rsi_momentum_loss=False,
            exit_on_consolidation=False, consolidating=False,
        )
        assert hit is True
        assert "sar_stop" in reason

    def test_check_exit_conditions_macd_cross_exit_fires(self):
        from Strategy_Auto_Trader.core.exits import _check_exit_conditions
        hit, reason, peak, days = _check_exit_conditions(
            current_position=1.0, cur_close=100.0, entry_price=100.0,
            rr_ratio=0.0, rr_risk=0.05, rr_stop_level=0.0, rr_target_level=0.0,
            effective_stop=0.0, peak_price_since_entry=100.0,
            max_hold_days=0, days_in_trade=0,
            use_sar_stop=False, sar_val=None,
            need_exit=True, macd_bearish_cross=True,
            exit_on_macd_cross=True, exit_on_rsi_reversal=False,
            rsi_overbought_exit=False, rsi_momentum_loss=False,
            exit_on_consolidation=False, consolidating=False,
        )
        assert hit is True
        assert reason == "exit_macd_cross"

    def test_check_exit_conditions_rsi_overbought_exit_fires(self):
        from Strategy_Auto_Trader.core.exits import _check_exit_conditions
        hit, reason, peak, days = _check_exit_conditions(
            current_position=1.0, cur_close=100.0, entry_price=100.0,
            rr_ratio=0.0, rr_risk=0.05, rr_stop_level=0.0, rr_target_level=0.0,
            effective_stop=0.0, peak_price_since_entry=100.0,
            max_hold_days=0, days_in_trade=0,
            use_sar_stop=False, sar_val=None,
            need_exit=True, macd_bearish_cross=False,
            exit_on_macd_cross=False, exit_on_rsi_reversal=True,
            rsi_overbought_exit=True, rsi_momentum_loss=False,
            exit_on_consolidation=False, consolidating=False,
        )
        assert hit is True
        assert reason == "exit_rsi_overbought_exit"

    def test_check_exit_conditions_consolidation_exit_fires(self):
        from Strategy_Auto_Trader.core.exits import _check_exit_conditions
        hit, reason, peak, days = _check_exit_conditions(
            current_position=1.0, cur_close=100.0, entry_price=100.0,
            rr_ratio=0.0, rr_risk=0.05, rr_stop_level=0.0, rr_target_level=0.0,
            effective_stop=0.0, peak_price_since_entry=100.0,
            max_hold_days=0, days_in_trade=0,
            use_sar_stop=False, sar_val=None,
            need_exit=True, macd_bearish_cross=False,
            exit_on_macd_cross=False, exit_on_rsi_reversal=False,
            rsi_overbought_exit=False, rsi_momentum_loss=False,
            exit_on_consolidation=True, consolidating=True,
        )
        assert hit is True
        assert reason == "exit_consolidation"
