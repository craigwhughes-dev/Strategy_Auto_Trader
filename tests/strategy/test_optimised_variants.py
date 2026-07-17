"""Tests for the whole standalone optimised family.

Base pair (journal-derived weights, RSI>70 + regime_signal<=0 vetoes):
  optimised                        — flip-guard on (engine default)
  optimised_aggressive             — require_flip_entry = False

Conviction-gate pair (score >= 6.0 gate over the same logic):
  optimised_optimised              — flip-guard on
  optimised_aggressive_optimised   — require_flip_entry = False

Journal-derived variants, each ONE stricter gate on top of optimised:
  optimised_volume  — volume_ratio >= 1.5 hard veto
  optimised_regime  — regime_signal > 0.75 required
  optimised_rsi     — RSI must sit in [60, 70]
  optimised_score7  — buy_threshold raised to 7.0
  optimised_pbull   — P(Bull) > 0.6 required
"""

from __future__ import annotations

import pytest

from Strategy_Auto_Trader.plugins.types import EntryDecision, RegimeState
from Strategy_Auto_Trader.strategy.optimised import OptimisedEntry, OptimisedExit
from Strategy_Auto_Trader.strategy.optimised_aggressive import (
    OptimisedAggressiveEntry,
    OptimisedAggressiveExit,
)
from Strategy_Auto_Trader.strategy.optimised_aggressive_optimised import (
    OptimisedAggressiveOptimisedEntry,
    OptimisedAggressiveOptimisedExit,
)
from Strategy_Auto_Trader.strategy.optimised_optimised import (
    OptimisedOptimisedEntry,
    OptimisedOptimisedExit,
)
from Strategy_Auto_Trader.strategy.optimised_pbull import OptimisedPbullEntry, OptimisedPbullExit
from Strategy_Auto_Trader.strategy.optimised_regime import OptimisedRegimeEntry, OptimisedRegimeExit
from Strategy_Auto_Trader.strategy.optimised_rsi import OptimisedRsiEntry, OptimisedRsiExit
from Strategy_Auto_Trader.strategy.optimised_score7 import OptimisedScore7Entry, OptimisedScore7Exit
from Strategy_Auto_Trader.strategy.optimised_volume import OptimisedVolumeEntry, OptimisedVolumeExit

VARIANTS = {
    "optimised": (OptimisedEntry, OptimisedExit),
    "optimised_aggressive": (OptimisedAggressiveEntry, OptimisedAggressiveExit),
    "optimised_aggressive_optimised": (
        OptimisedAggressiveOptimisedEntry, OptimisedAggressiveOptimisedExit),
    "optimised_optimised": (OptimisedOptimisedEntry, OptimisedOptimisedExit),
    "optimised_pbull": (OptimisedPbullEntry, OptimisedPbullExit),
    "optimised_regime": (OptimisedRegimeEntry, OptimisedRegimeExit),
    "optimised_rsi": (OptimisedRsiEntry, OptimisedRsiExit),
    "optimised_score7": (OptimisedScore7Entry, OptimisedScore7Exit),
    "optimised_volume": (OptimisedVolumeEntry, OptimisedVolumeExit),
}

#: Regime + momentum that produce a clean BUY under base optimised
#: (score 9.0: all components vote) and pass every variant gate.
STRONG_REGIME = RegimeState(p_bull=0.9, p_bear=0.05, p_bull_smooth=0.9,
                            regime_signal=0.85, hmm_vote=2)
STRONG_MOM = {
    "cur_rsi": 65.0, "recent_cross_above_50": True,
    "recent_cross_below_40": False, "above_sma20": True,
    "above_sma50": True, "above_sma200": True, "volume_ratio": 1.8,
}


class TestVariantConformance:

    @pytest.mark.parametrize("name", sorted(VARIANTS))
    def test_entry_satisfies_protocol(self, name):
        from Strategy_Auto_Trader.strategy.base import EntryStrategyProtocol
        entry_cls, _ = VARIANTS[name]
        assert isinstance(entry_cls(), EntryStrategyProtocol)

    @pytest.mark.parametrize("name", sorted(VARIANTS))
    def test_exit_satisfies_protocol(self, name):
        from Strategy_Auto_Trader.strategy.base import ExitStrategyProtocol
        _, exit_cls = VARIANTS[name]
        assert isinstance(exit_cls(), ExitStrategyProtocol)

    @pytest.mark.parametrize("name", sorted(VARIANTS))
    def test_registry_resolution(self, name):
        from Strategy_Auto_Trader.strategy.base.registry import resolve_strategy
        entry_cls, exit_cls = VARIANTS[name]
        entry, exit_ = resolve_strategy(name)
        assert isinstance(entry, entry_cls)
        assert isinstance(exit_, exit_cls)

    @pytest.mark.parametrize("name", sorted(VARIANTS))
    def test_arg_parser_accepts_variant(self, name):
        from Strategy_Auto_Trader.markov_cli.run import _build_arg_parser
        args = _build_arg_parser().parse_args(["--ticker", "SPY", "--strategy", name])
        assert args.strategy == name

    @pytest.mark.parametrize("name", sorted(VARIANTS))
    def test_exit_stop_and_target_match_optimised(self, name):
        _, exit_cls = VARIANTS[name]
        ex = exit_cls()
        assert ex.stop_loss_pct == pytest.approx(0.08)
        assert ex.take_profit_pct == pytest.approx(0.30)

    @pytest.mark.parametrize("name", sorted(VARIANTS))
    def test_vol_filter_veto(self, name):
        entry_cls, _ = VARIANTS[name]
        decision = entry_cls(vol_filter_ok=False).evaluate(
            STRONG_REGIME, dict(STRONG_MOM), 1.8, currently_in=False)
        assert decision.flag == "HOLD"
        assert "vol_filter" in decision.reason


class TestVariantGates:
    """Each gate fires under its trigger condition and passes under STRONG."""

    @pytest.mark.parametrize("name", sorted(VARIANTS))
    def test_strong_conditions_give_buy(self, name):
        entry_cls, _ = VARIANTS[name]
        decision = entry_cls().evaluate(STRONG_REGIME, dict(STRONG_MOM), 1.8,
                                        currently_in=False)
        assert decision.flag == "BUY", decision.reason

    def test_volume_gate_fires_below_1_5(self):
        mom = dict(STRONG_MOM, volume_ratio=1.2)
        decision = OptimisedVolumeEntry().evaluate(STRONG_REGIME, mom, 1.2,
                                                   currently_in=False)
        assert decision.flag == "HOLD"
        assert "volume veto" in decision.reason

    def test_volume_gate_missing_data_fails_safe(self):
        mom = {k: v for k, v in STRONG_MOM.items() if k != "volume_ratio"}
        decision = OptimisedVolumeEntry().evaluate(STRONG_REGIME, mom, 0.0,
                                                   currently_in=False)
        assert decision.flag == "HOLD"

    def test_regime_gate_fires_at_0_5(self):
        regime = RegimeState(p_bull=0.9, p_bear=0.05, p_bull_smooth=0.9,
                             regime_signal=0.5, hmm_vote=2)
        decision = OptimisedRegimeEntry().evaluate(regime, dict(STRONG_MOM), 1.8,
                                                   currently_in=False)
        assert decision.flag == "HOLD"
        assert "regime veto" in decision.reason

    def test_regime_gate_boundary_0_75_vetoed(self):
        regime = RegimeState(p_bull=0.9, p_bear=0.05, p_bull_smooth=0.9,
                             regime_signal=0.75, hmm_vote=2)
        decision = OptimisedRegimeEntry().evaluate(regime, dict(STRONG_MOM), 1.8,
                                                   currently_in=False)
        assert decision.flag == "HOLD"

    def test_rsi_gate_fires_below_band(self):
        mom = dict(STRONG_MOM, cur_rsi=55.0)
        decision = OptimisedRsiEntry().evaluate(STRONG_REGIME, mom, 1.8,
                                                currently_in=False)
        assert decision.flag == "HOLD"
        assert "rsi band veto" in decision.reason

    def test_rsi_gate_band_edges_admit(self):
        for rsi in (60.0, 70.0):
            mom = dict(STRONG_MOM, cur_rsi=rsi)
            decision = OptimisedRsiEntry().evaluate(STRONG_REGIME, mom, 1.8,
                                                    currently_in=False)
            assert decision.flag == "BUY", f"RSI {rsi} should be in band"

    def test_rsi_gate_fires_above_band(self):
        mom = dict(STRONG_MOM, cur_rsi=72.0)
        decision = OptimisedRsiEntry().evaluate(STRONG_REGIME, mom, 1.8,
                                                currently_in=False)
        assert decision.flag == "HOLD"

    def test_score7_needs_full_confluence(self):
        # Drop volume + RSI votes: score falls to 7.0 max minus both -> < 7.0.
        # RSI at 45 (no RSI vote) and thin volume (no volume vote): score 7.0
        # requires all but one 1.0-weight component, so this must HOLD.
        mom = dict(STRONG_MOM, cur_rsi=45.0, recent_cross_above_50=False,
                   volume_ratio=0.8)
        decision = OptimisedScore7Entry().evaluate(STRONG_REGIME, mom, 0.8,
                                                   currently_in=False)
        assert decision.flag != "BUY"

    def test_score7_threshold_is_7(self):
        assert OptimisedScore7Entry.buy_threshold == pytest.approx(7.0)

    def test_pbull_gate_fires_at_0_5(self):
        regime = RegimeState(p_bull=0.5, p_bear=0.2, p_bull_smooth=0.7,
                             regime_signal=0.5, hmm_vote=2)
        decision = OptimisedPbullEntry().evaluate(regime, dict(STRONG_MOM), 1.8,
                                                  currently_in=False)
        assert decision.flag == "HOLD"
        assert "regime_gate" in decision.reason

    def test_pbull_gate_boundary_0_6_vetoed(self):
        regime = RegimeState(p_bull=0.6, p_bear=0.2, p_bull_smooth=0.7,
                             regime_signal=0.5, hmm_vote=2)
        decision = OptimisedPbullEntry().evaluate(regime, dict(STRONG_MOM), 1.8,
                                                  currently_in=False)
        assert decision.flag == "HOLD"


class TestVariantVetoesOnlyBlockNewEntries:
    """No variant gate may suppress SELL/exit signalling while in a position."""

    @pytest.mark.parametrize("name", sorted(VARIANTS))
    def test_gate_not_applied_while_in_position(self, name):
        entry_cls, _ = VARIANTS[name]
        # Conditions that trip every variant gate at once, while in a trade.
        regime = RegimeState(p_bull=0.3, p_bear=0.4, p_bull_smooth=0.4,
                             regime_signal=0.1, hmm_vote=1)
        mom = dict(STRONG_MOM, cur_rsi=45.0, volume_ratio=0.9)
        decision = entry_cls().evaluate(regime, mom, 0.9, currently_in=True)
        assert isinstance(decision, EntryDecision)
        for marker in ("veto", "regime_gate", "signal_gate", "conviction gate"):
            assert marker not in decision.reason, (
                f"{name} applied an entry gate while in a position: {decision.reason}")

    @pytest.mark.parametrize("name", sorted(VARIANTS))
    def test_flip_guard_setting(self, name):
        entry_cls, _ = VARIANTS[name]
        expected = "aggressive" not in name
        assert getattr(entry_cls, "require_flip_entry", True) is expected

    @pytest.mark.parametrize("name", sorted(VARIANTS))
    def test_base_rsi_overbought_veto_retained(self, name):
        if name == "optimised_rsi":
            pytest.skip("band veto replaces the plain >70 veto (same effect)")
        entry_cls, _ = VARIANTS[name]
        mom = dict(STRONG_MOM, cur_rsi=75.0)
        decision = entry_cls().evaluate(STRONG_REGIME, mom, 1.8, currently_in=False)
        assert decision.flag == "HOLD"
        assert "RSI" in decision.reason
