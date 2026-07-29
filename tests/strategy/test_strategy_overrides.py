from __future__ import annotations

import pytest

from Strategy_Auto_Trader.plugins.types import RegimeState


class TestEntryOverrides:
    """Every Entry class except ChoppyVolEntry accepts buy_threshold/
    sell_threshold/quality_gate_enabled overrides, None-sentinel, falling
    back to its own class-level default when omitted."""

    @pytest.mark.parametrize("module,cls_name,default_buy,default_sell,default_gate", [
        ("Strategy_Auto_Trader.strategy.default", "DefaultEntry", 3.0, -3.0, True),
        ("Strategy_Auto_Trader.strategy.conservative", "ConservativeEntry", 4.5, -4.5, True),
        ("Strategy_Auto_Trader.strategy.optimised", "OptimisedEntry", 6.0, -4.5, True),
        ("Strategy_Auto_Trader.strategy.trend_follow", "TrendEntry", 4.5, -4.0, True),
        ("Strategy_Auto_Trader.strategy.breakout_momentum", "BreakoutMomentumEntry", 4.0, -3.0, True),
    ])
    def test_no_override_preserves_class_default(self, module, cls_name, default_buy, default_sell, default_gate):
        import importlib
        cls = getattr(importlib.import_module(module), cls_name)
        entry = cls()
        assert entry._buy_t == default_buy
        assert entry._sell_t == default_sell
        assert entry._quality_gate_enabled is default_gate

    @pytest.mark.parametrize("module,cls_name", [
        ("Strategy_Auto_Trader.strategy.default", "DefaultEntry"),
        ("Strategy_Auto_Trader.strategy.conservative", "ConservativeEntry"),
        ("Strategy_Auto_Trader.strategy.optimised", "OptimisedEntry"),
        ("Strategy_Auto_Trader.strategy.trend_follow", "TrendEntry"),
        ("Strategy_Auto_Trader.strategy.breakout_momentum", "BreakoutMomentumEntry"),
    ])
    def test_override_takes_effect(self, module, cls_name):
        import importlib
        cls = getattr(importlib.import_module(module), cls_name)
        entry = cls(buy_threshold=99.0, sell_threshold=-99.0, quality_gate_enabled=False)
        assert entry._buy_t == 99.0
        assert entry._sell_t == -99.0
        assert entry._quality_gate_enabled is False

    def test_mean_reversion_no_override_preserves_default(self):
        from Strategy_Auto_Trader.strategy.mean_reversion import MeanReversionEntry
        entry = MeanReversionEntry()
        assert entry._buy_t == 2.5
        assert entry._sell_t == -2.5
        assert entry._quality_gate_enabled is False

    def test_mean_reversion_override_takes_effect(self):
        from Strategy_Auto_Trader.strategy.mean_reversion import MeanReversionEntry
        entry = MeanReversionEntry(buy_threshold=99.0, quality_gate_enabled=True)
        assert entry._buy_t == 99.0
        assert entry._quality_gate_enabled is True

    def test_mean_reversion_gate_enabled_vetoes_its_own_buy_setup(self):
        """The whole point of this strategy is to buy exactly the context
        the shared weak-buy veto flags as weak — confirms quality_gate_enabled
        is genuinely wired (not decorative) and documents the near-always-veto
        effect the module docstring warns about."""
        from Strategy_Auto_Trader.strategy.mean_reversion import MeanReversionEntry
        mom = {"cur_rsi": 30.0, "pct_from_sma20": -0.04, "volume_ratio": 1.5,
               "above_sma20": False, "above_sma50": False, "above_sma200": False,
               "recent_cross_above_50": False}
        regime = RegimeState(p_bull=0.3, p_bear=0.3, p_bull_smooth=0.3,
                             regime_signal=0.1, hmm_vote=0)

        default_entry = MeanReversionEntry(vol_filter_ok=False)
        default_decision = default_entry.evaluate(regime, mom, 1.5, currently_in=False)
        assert default_decision.flag == "BUY"

        gated_entry = MeanReversionEntry(vol_filter_ok=False, quality_gate_enabled=True)
        gated_decision = gated_entry.evaluate(regime, mom, 1.5, currently_in=False)
        assert gated_decision.flag == "HOLD"
        assert "weak buy context" in gated_decision.reason

    def test_choppy_vol_entry_rejects_composite_overrides(self):
        """Pins the deliberate exclusion: no composite score exists here to
        threshold or gate against, unlike every other strategy."""
        from Strategy_Auto_Trader.strategy.choppy_vol import ChoppyVolEntry
        with pytest.raises(TypeError):
            ChoppyVolEntry(buy_threshold=1.0)
        with pytest.raises(TypeError):
            ChoppyVolEntry(quality_gate_enabled=True)
        with pytest.raises(TypeError):
            ChoppyVolEntry(gate_sensitivity=1)
        # vol_filter_ok is the only accepted param, and is a no-op by design.
        ChoppyVolEntry(vol_filter_ok=True)

    @pytest.mark.parametrize("module,cls_name", [
        ("Strategy_Auto_Trader.strategy.default", "DefaultEntry"),
        ("Strategy_Auto_Trader.strategy.conservative", "ConservativeEntry"),
        ("Strategy_Auto_Trader.strategy.optimised", "OptimisedEntry"),
        ("Strategy_Auto_Trader.strategy.trend_follow", "TrendEntry"),
        ("Strategy_Auto_Trader.strategy.breakout_momentum", "BreakoutMomentumEntry"),
        ("Strategy_Auto_Trader.strategy.mean_reversion", "MeanReversionEntry"),
    ])
    def test_gate_sensitivity_no_override_preserves_default(self, module, cls_name):
        import importlib
        cls = getattr(importlib.import_module(module), cls_name)
        entry = cls()
        assert entry._gate_sensitivity == 2

    @pytest.mark.parametrize("module,cls_name", [
        ("Strategy_Auto_Trader.strategy.default", "DefaultEntry"),
        ("Strategy_Auto_Trader.strategy.conservative", "ConservativeEntry"),
        ("Strategy_Auto_Trader.strategy.optimised", "OptimisedEntry"),
        ("Strategy_Auto_Trader.strategy.trend_follow", "TrendEntry"),
        ("Strategy_Auto_Trader.strategy.breakout_momentum", "BreakoutMomentumEntry"),
        ("Strategy_Auto_Trader.strategy.mean_reversion", "MeanReversionEntry"),
    ])
    def test_gate_sensitivity_override_takes_effect(self, module, cls_name):
        import importlib
        cls = getattr(importlib.import_module(module), cls_name)
        entry = cls(gate_sensitivity=1)
        assert entry._gate_sensitivity == 1

    def test_gate_sensitivity_changes_default_entry_gate_decision(self):
        """Confirms gate_sensitivity is genuinely wired into the veto path,
        not just stored — a single weak condition vetoes at sensitivity=1
        but not at the default sensitivity=2."""
        from Strategy_Auto_Trader.strategy.default import DefaultEntry
        mom = {"cur_rsi": 60.0, "recent_cross_above_50": False,
               "recent_cross_below_40": False, "above_sma20": True,
               "above_sma50": False, "above_sma200": True, "volume_ratio": 1.5}
        regime = RegimeState(p_bull=0.6, p_bear=0.1, p_bull_smooth=0.6,
                             regime_signal=0.5, hmm_vote=1)

        default_entry = DefaultEntry(buy_threshold=0.5)
        decision = default_entry.evaluate(regime, mom, 1.5, currently_in=False)
        assert decision.flag == "BUY"

        strict_entry = DefaultEntry(buy_threshold=0.5, gate_sensitivity=1)
        decision = strict_entry.evaluate(regime, mom, 1.5, currently_in=False)
        assert decision.flag == "HOLD"

    def test_mean_reversion_gate_sensitivity_reaches_inline_adverse_exit_check(self):
        """mean_reversion's default (quality_gate_enabled=False) path still
        calls _is_adverse_exit_context inline — gate_sensitivity must reach
        that call too, not just the opt-in _apply_quality_gate path."""
        from Strategy_Auto_Trader.strategy.mean_reversion import MeanReversionEntry
        mom = {"cur_rsi": 35.0, "recent_cross_above_50": False,
               "recent_cross_below_40": False, "above_sma20": True,
               "above_sma50": True, "above_sma200": True, "volume_ratio": 1.5}
        regime = RegimeState(p_bull=0.4, p_bear=0.3, p_bull_smooth=0.4,
                             regime_signal=0.1, hmm_vote=0)

        default_entry = MeanReversionEntry(vol_filter_ok=False)
        decision = default_entry.evaluate(regime, mom, 1.5, currently_in=True)
        assert decision.flag != "SELL"  # 1 adverse condition, not enough at sensitivity=2

        strict_entry = MeanReversionEntry(vol_filter_ok=False, gate_sensitivity=1)
        decision = strict_entry.evaluate(regime, mom, 1.5, currently_in=True)
        assert decision.flag == "SELL"


class TestExitOverrides:
    """Every Exit class accepts every StandardExitRules-backed override,
    None-sentinel, falling back to its own class-level default when omitted."""

    @pytest.mark.parametrize("module,cls_name,default_stop,default_target", [
        ("Strategy_Auto_Trader.strategy.default", "DefaultExit", 0.05, 0.15),
        ("Strategy_Auto_Trader.strategy.conservative", "ConservativeExit", 0.03, 0.10),
        ("Strategy_Auto_Trader.strategy.mean_reversion", "MeanReversionExit", 0.03, 0.08),
        ("Strategy_Auto_Trader.strategy.optimised", "OptimisedExit", 0.08, 0.30),
        ("Strategy_Auto_Trader.strategy.trend_follow", "TrendExit", 0.08, 0.30),
        ("Strategy_Auto_Trader.strategy.breakout_momentum", "BreakoutMomentumExit", 0.06, 0.25),
        ("Strategy_Auto_Trader.strategy.choppy_vol", "ChoppyVolExit", 0.04, 0.06),
    ])
    def test_no_override_preserves_class_default(self, module, cls_name, default_stop, default_target):
        import importlib
        cls = getattr(importlib.import_module(module), cls_name)
        exit_ = cls()
        assert exit_.stop_loss_pct == default_stop
        assert exit_.take_profit_pct == default_target

    @pytest.mark.parametrize("module,cls_name", [
        ("Strategy_Auto_Trader.strategy.default", "DefaultExit"),
        ("Strategy_Auto_Trader.strategy.conservative", "ConservativeExit"),
        ("Strategy_Auto_Trader.strategy.mean_reversion", "MeanReversionExit"),
        ("Strategy_Auto_Trader.strategy.optimised", "OptimisedExit"),
        ("Strategy_Auto_Trader.strategy.trend_follow", "TrendExit"),
        ("Strategy_Auto_Trader.strategy.breakout_momentum", "BreakoutMomentumExit"),
        ("Strategy_Auto_Trader.strategy.choppy_vol", "ChoppyVolExit"),
    ])
    def test_override_takes_effect(self, module, cls_name):
        import importlib
        cls = getattr(importlib.import_module(module), cls_name)
        exit_ = cls(stop_loss_pct=0.5, take_profit_pct=1.5, trailing_stop=0.2,
                    profit_stop_scale=0.4, min_stop_pct=0.01, max_hold_days=7)
        assert exit_.stop_loss_pct == 0.5
        assert exit_.take_profit_pct == 1.5
        assert exit_._impl._trailing_stop == 0.2
        assert exit_._impl._profit_stop_scale == 0.4
        assert exit_._impl._min_stop_pct == 0.01
        assert exit_._impl._max_hold_days == 7

    def test_unsupported_exit_override_raises(self):
        from Strategy_Auto_Trader.strategy.default import DefaultExit
        with pytest.raises(TypeError):
            DefaultExit(not_a_real_param=1)

    @pytest.mark.parametrize("module,cls_name,default_macd,default_rsi", [
        ("Strategy_Auto_Trader.strategy.default", "DefaultExit", False, False),
        ("Strategy_Auto_Trader.strategy.conservative", "ConservativeExit", False, False),
        ("Strategy_Auto_Trader.strategy.mean_reversion", "MeanReversionExit", False, True),
        ("Strategy_Auto_Trader.strategy.optimised", "OptimisedExit", False, False),
        ("Strategy_Auto_Trader.strategy.trend_follow", "TrendExit", False, False),
        ("Strategy_Auto_Trader.strategy.breakout_momentum", "BreakoutMomentumExit", True, True),
        ("Strategy_Auto_Trader.strategy.choppy_vol", "ChoppyVolExit", False, False),
    ])
    def test_exit_on_macd_rsi_properties_expose_own_defaults(self, module, cls_name, default_macd, default_rsi):
        """Item 2 fix: consolidated_engine.py reads these public properties via
        getattr(exit_strategy, ...) to OR against the CLI flag — pins each
        strategy's own hardcoded default is actually reachable, not private
        to StandardExitRules."""
        import importlib
        cls = getattr(importlib.import_module(module), cls_name)
        exit_ = cls()
        assert exit_.exit_on_macd_cross is default_macd
        assert exit_.exit_on_rsi_reversal is default_rsi

    def test_exit_on_macd_rsi_override_takes_effect(self):
        from Strategy_Auto_Trader.strategy.default import DefaultExit
        exit_ = DefaultExit(exit_on_macd_cross=True, exit_on_rsi_reversal=True)
        assert exit_.exit_on_macd_cross is True
        assert exit_.exit_on_rsi_reversal is True
