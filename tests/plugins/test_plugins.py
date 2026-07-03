from __future__ import annotations

import numpy as np
import pytest


class TestPlugins:

    # -- types ----------------------------------------------------------------

    def test_regime_state_dataclass(self):
        from Strategy_Auto_Trader.plugins.types import RegimeState
        rs = RegimeState(p_bull=0.7, p_bear=0.1, p_bull_smooth=0.65,
                         regime_signal=0.55, hmm_vote=2)
        assert rs.p_bull == 0.7
        assert rs.hmm_vote == 2

    def test_trade_state_dataclass(self):
        from Strategy_Auto_Trader.plugins.types import TradeState
        ts = TradeState(position=0.1, entry_price=100.0, stop_level=95.0,
                        target_level=115.0, peak_price_since_entry=102.0,
                        days_in_trade=3, entry_bar=10)
        assert ts.stop_level == 95.0

    def test_bar_data_dataclass(self):
        from Strategy_Auto_Trader.plugins.types import BarData
        bd = BarData(t=50, cur_close=105.0, daily_vol_t=0.01, use_sar_stop=False,
                     sar_val=None, need_exit=False, macd_bc=False,
                     rsi_ob=False, rsi_ml=False, consol=False)
        assert bd.cur_close == 105.0

    def test_exit_result_dataclass(self):
        from Strategy_Auto_Trader.plugins.types import ExitResult
        er = ExitResult(exit_hit=True, sell_reason="stop_loss",
                        peak_price_since_entry=110.0, days_in_trade=5)
        assert er.exit_hit is True

    # -- Protocol conformance -------------------------------------------------

    def test_hmm_regime_satisfies_protocol(self):
        from Strategy_Auto_Trader.plugins.hmm_regime import HMMRegimeModel
        from Strategy_Auto_Trader.plugins.protocols import RegimeModelProtocol
        m = HMMRegimeModel()
        assert isinstance(m, RegimeModelProtocol)

    def test_composite_signal_generator_satisfies_protocol(self):
        from Strategy_Auto_Trader.plugins.vote_signal import CompositeSignalGenerator
        from Strategy_Auto_Trader.plugins.protocols import SignalGeneratorProtocol
        sg = CompositeSignalGenerator(weights={}, buy_threshold=3.0, sell_threshold=-3.0)
        assert isinstance(sg, SignalGeneratorProtocol)

    def test_quality_gate_plugin_satisfies_protocol(self):
        from Strategy_Auto_Trader.plugins.quality_gate import QualityGatePlugin, NullQualityGate
        from Strategy_Auto_Trader.plugins.protocols import QualityGateProtocol
        assert isinstance(QualityGatePlugin(), QualityGateProtocol)
        assert isinstance(NullQualityGate(), QualityGateProtocol)

    def test_standard_exit_rules_satisfies_protocol(self):
        from Strategy_Auto_Trader.plugins.exit_rules import StandardExitRules
        from Strategy_Auto_Trader.plugins.protocols import ExitRulesProtocol
        er = StandardExitRules(stop_loss_pct=0.05)
        assert isinstance(er, ExitRulesProtocol)

    def test_kelly_sizer_satisfies_protocol(self):
        from Strategy_Auto_Trader.plugins.kelly_sizer import KellySizer
        from Strategy_Auto_Trader.plugins.protocols import PositionSizerProtocol
        assert isinstance(KellySizer(), PositionSizerProtocol)

    def test_sentiment_adjuster_satisfies_protocol(self):
        from Strategy_Auto_Trader.plugins.context_adjuster import SentimentAdjuster, NullAdjuster
        from Strategy_Auto_Trader.plugins.protocols import ContextAdjusterProtocol
        assert isinstance(SentimentAdjuster(), ContextAdjusterProtocol)
        assert isinstance(NullAdjuster(), ContextAdjusterProtocol)

    def test_vol_prescreen_satisfies_protocol(self):
        from Strategy_Auto_Trader.plugins.prescreen import VolatilityPrescreen, NullPrescreen
        from Strategy_Auto_Trader.plugins.protocols import PrescreenProtocol
        assert isinstance(VolatilityPrescreen(), PrescreenProtocol)
        assert isinstance(NullPrescreen(), PrescreenProtocol)

    # -- KellySizer -----------------------------------------------------------

    def test_kelly_sizer_size_pure_wins(self):
        from Strategy_Auto_Trader.plugins.kelly_sizer import KellySizer
        sizer = KellySizer()
        k = sizer.size([0.10, 0.08, 0.12, 0.05, 0.09])
        assert 0 <= k <= 0.25

    def test_kelly_sizer_size_empty_returns_zero(self):
        from Strategy_Auto_Trader.plugins.kelly_sizer import KellySizer
        assert KellySizer().size([]) == 0.0

    def test_kelly_sizer_record_updates_position(self):
        from Strategy_Auto_Trader.plugins.kelly_sizer import KellySizer
        sizer = KellySizer(lookback=5)
        initial = sizer.position
        for pl in [0.05, 0.04, 0.06, -0.02, 0.03]:
            sizer.record(pl)
        assert sizer.current_kelly >= 0.0
        assert sizer.position >= 0.02

    def test_kelly_sizer_position_floored_at_min(self):
        from Strategy_Auto_Trader.plugins.kelly_sizer import KellySizer
        sizer = KellySizer(default=0.01, min_position=0.02)
        assert sizer.position == 0.02

    def test_kelly_sizer_use_kelly_false_stays_at_default(self):
        from Strategy_Auto_Trader.plugins.kelly_sizer import KellySizer
        sizer = KellySizer(use_kelly=False, default=0.10)
        for pl in [0.1] * 30:
            sizer.record(pl)
        assert sizer.current_kelly == 0.10  # never updated

    def test_kelly_sizer_trade_results_accumulated(self):
        from Strategy_Auto_Trader.plugins.kelly_sizer import KellySizer
        sizer = KellySizer()
        sizer.record(0.05)
        sizer.record(-0.03)
        assert len(sizer.trade_results) == 2
        assert sizer.trade_results[0] == pytest.approx(0.05)

    # -- QualityGatePlugin ----------------------------------------------------

    def test_quality_gate_plugin_passes_strong_buy(self):
        from Strategy_Auto_Trader.plugins.quality_gate import QualityGatePlugin
        gate = QualityGatePlugin()
        sig = {"flag": "BUY", "score": 5.0}
        mom = {
            "cur_rsi": 60, "recent_cross_above_50": True, "recent_cross_below_40": False,
            "above_sma20": True, "above_sma50": True, "above_sma200": True,
            "volume_ratio": 1.5,
        }
        result = gate.apply(sig, mom, regime_signal=0.5, currently_in=False)
        assert result["flag"] == "BUY"

    def test_null_gate_passes_through_unchanged(self):
        from Strategy_Auto_Trader.plugins.quality_gate import NullQualityGate
        gate = NullQualityGate()
        sig = {"flag": "BUY", "score": 1.0}
        mom: dict = {}
        result = gate.apply(sig, mom, regime_signal=-0.9, currently_in=False)
        assert result["flag"] == "BUY"

    # -- StandardExitRules ----------------------------------------------------

    def test_standard_exit_rules_fires_stop_loss(self):
        from Strategy_Auto_Trader.plugins.exit_rules import StandardExitRules
        from Strategy_Auto_Trader.plugins.types import BarData, TradeState
        rules = StandardExitRules(stop_loss_pct=0.05)
        trade = TradeState(
            position=0.1, entry_price=100.0, stop_level=95.0,
            target_level=115.0, peak_price_since_entry=101.0,
            days_in_trade=2, entry_bar=10,
        )
        bar = BarData(
            t=15, cur_close=94.0, daily_vol_t=0.01, use_sar_stop=False,
            sar_val=None, need_exit=False, macd_bc=False,
            rsi_ob=False, rsi_ml=False, consol=False,
        )
        result = rules.check(trade, bar)
        assert result.exit_hit is True
        assert "rr_stop_loss" in result.sell_reason

    def test_standard_exit_rules_fires_take_profit(self):
        from Strategy_Auto_Trader.plugins.exit_rules import StandardExitRules
        from Strategy_Auto_Trader.plugins.types import BarData, TradeState
        rules = StandardExitRules(stop_loss_pct=0.05)
        trade = TradeState(
            position=0.1, entry_price=100.0, stop_level=95.0,
            target_level=115.0, peak_price_since_entry=110.0,
            days_in_trade=5, entry_bar=5,
        )
        bar = BarData(
            t=20, cur_close=116.0, daily_vol_t=0.01, use_sar_stop=False,
            sar_val=None, need_exit=False, macd_bc=False,
            rsi_ob=False, rsi_ml=False, consol=False,
        )
        result = rules.check(trade, bar)
        assert result.exit_hit is True
        assert "rr_take_profit" in result.sell_reason

    def test_standard_exit_rules_no_exit_in_quiet_bar(self):
        from Strategy_Auto_Trader.plugins.exit_rules import StandardExitRules
        from Strategy_Auto_Trader.plugins.types import BarData, TradeState
        rules = StandardExitRules(stop_loss_pct=0.05)
        trade = TradeState(
            position=0.1, entry_price=100.0, stop_level=95.0,
            target_level=115.0, peak_price_since_entry=102.0,
            days_in_trade=1, entry_bar=5,
        )
        bar = BarData(
            t=10, cur_close=103.0, daily_vol_t=0.01, use_sar_stop=False,
            sar_val=None, need_exit=False, macd_bc=False,
            rsi_ob=False, rsi_ml=False, consol=False,
        )
        result = rules.check(trade, bar)
        assert result.exit_hit is False

    # -- CompositeSignalGenerator ---------------------------------------------

    def test_vote_signal_generate_returns_flag(self):
        from Strategy_Auto_Trader.plugins.vote_signal import CompositeSignalGenerator
        from Strategy_Auto_Trader.plugins.types import RegimeState
        from Strategy_Auto_Trader.quant_hmm.consolidated_engine import _CONSOLIDATED_WEIGHTS
        gen = CompositeSignalGenerator(
            weights=_CONSOLIDATED_WEIGHTS, buy_threshold=3.0, sell_threshold=-3.0,
        )
        regime = RegimeState(p_bull=0.7, p_bear=0.1, p_bull_smooth=0.7,
                             regime_signal=0.6, hmm_vote=2)
        mom = {
            "cur_rsi": 60, "recent_cross_above_50": True, "recent_cross_below_40": False,
            "above_sma20": True, "above_sma50": True, "above_sma200": True,
            "volume_ratio": 1.3,
        }
        sig = gen.generate(regime, mom)
        assert sig["flag"] in ("BUY", "HOLD", "SELL")
        assert "votes" in sig

    def test_vote_signal_matches_direct_composite_signal(self):
        from Strategy_Auto_Trader.plugins.vote_signal import CompositeSignalGenerator
        from Strategy_Auto_Trader.plugins.types import RegimeState
        from Strategy_Auto_Trader.core.momentum import composite_signal
        from Strategy_Auto_Trader.quant_hmm.consolidated_engine import _CONSOLIDATED_WEIGHTS
        gen = CompositeSignalGenerator(weights=_CONSOLIDATED_WEIGHTS)
        regime = RegimeState(p_bull=0.7, p_bear=0.1, p_bull_smooth=0.7,
                             regime_signal=0.6, hmm_vote=2)
        mom = {
            "cur_rsi": 55, "recent_cross_above_50": True, "recent_cross_below_40": False,
            "above_sma20": True, "above_sma50": True, "volume_ratio": 1.1,
        }
        plugin_sig = gen.generate(regime, mom)
        direct_sig = composite_signal(0.0, mom, hmm_state=2, weights=_CONSOLIDATED_WEIGHTS)
        assert plugin_sig["flag"] == direct_sig["flag"]
        assert plugin_sig["score"] == direct_sig["score"]

    # -- ContextAdjuster ------------------------------------------------------

    def test_sentiment_adjuster_matches_compute_effective_thresholds(self):
        from Strategy_Auto_Trader.plugins.context_adjuster import SentimentAdjuster
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _compute_effective_thresholds
        adj = SentimentAdjuster()
        result = adj.adjust(0.65, 0.40, 0.05, sentiment_score=1.0, vix_signal=-1)
        expected = _compute_effective_thresholds(0.65, 0.40, 0.05, 1.0, -1)
        assert result == pytest.approx(expected)

    def test_null_adjuster_is_identity(self):
        from Strategy_Auto_Trader.plugins.context_adjuster import NullAdjuster
        adj = NullAdjuster()
        ep, xp, sp = adj.adjust(0.70, 0.35, 0.06)
        assert ep == pytest.approx(0.70)
        assert xp == pytest.approx(0.35)
        assert sp == pytest.approx(0.06)

    # -- NullPrescreen --------------------------------------------------------

    def test_null_prescreen_always_returns_dict(self):
        from Strategy_Auto_Trader.plugins.prescreen import NullPrescreen
        ps = NullPrescreen()
        assert ps("AAPL") == {}
        assert ps("FAKE_TICKER_XYZ") == {}
