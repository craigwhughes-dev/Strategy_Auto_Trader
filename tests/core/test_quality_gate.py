from __future__ import annotations

import pytest


class TestQualityGate:

    def _mom(self, **overrides):
        base = {
            "cur_rsi": 60.0,
            "recent_cross_above_50": False,
            "recent_cross_below_40": False,
            "above_sma20": True,
            "above_sma50": True,
            "above_sma200": True,
            "volume_ratio": 1.5,
        }
        base.update(overrides)
        return base

    def test_is_weak_buy_context_strong_context_false(self):
        from Strategy_Auto_Trader.core.quality_gate import _is_weak_buy_context
        assert _is_weak_buy_context(0.5, self._mom()) is False

    def test_is_weak_buy_context_two_weak_conditions_true(self):
        from Strategy_Auto_Trader.core.quality_gate import _is_weak_buy_context
        mom = self._mom(above_sma50=False, volume_ratio=0.8)
        assert _is_weak_buy_context(0.25, mom) is True

    def test_is_adverse_exit_context_calm_context_false(self):
        from Strategy_Auto_Trader.core.quality_gate import _is_adverse_exit_context
        assert _is_adverse_exit_context(0.5, self._mom()) is False

    def test_is_adverse_exit_context_two_adverse_conditions_true(self):
        from Strategy_Auto_Trader.core.quality_gate import _is_adverse_exit_context
        mom = self._mom(cur_rsi=35.0, above_sma20=False, above_sma50=False)
        assert _is_adverse_exit_context(-0.25, mom) is True

    def test_apply_quality_gate_vetoes_weak_buy_candidates(self):
        from Strategy_Auto_Trader.core.quality_gate import _apply_quality_gate
        sig = {"flag": "BUY", "score": 3.0, "max_score": 4.0}
        mom = self._mom(above_sma50=False, above_sma200=False, volume_ratio=0.8)
        gate = _apply_quality_gate(sig, mom, 0.25, False)
        assert gate["flag"] == "HOLD"
        assert "quality_gate" in gate["reason"]

    def test_apply_quality_gate_early_exit_on_adverse_context(self):
        from Strategy_Auto_Trader.core.quality_gate import _apply_quality_gate
        sig = {"flag": "HOLD", "score": 0.0, "max_score": 4.0}
        mom = self._mom(cur_rsi=35.0, recent_cross_below_40=True,
                         above_sma20=False, above_sma50=False,
                         above_sma200=False, volume_ratio=0.6)
        gate = _apply_quality_gate(sig, mom, -0.25, True)
        assert gate["flag"] == "SELL"
        assert "quality_gate" in gate["reason"]

    def test_apply_quality_gate_passes_through_strong_buy(self):
        from Strategy_Auto_Trader.core.quality_gate import _apply_quality_gate
        sig = {"flag": "BUY", "score": 4.0, "max_score": 4.0}
        gate = _apply_quality_gate(sig, self._mom(), 0.5, False)
        assert gate["flag"] == "BUY"

    def test_apply_quality_gate_no_veto_when_not_in_position_and_not_buy(self):
        from Strategy_Auto_Trader.core.quality_gate import _apply_quality_gate
        sig = {"flag": "HOLD", "score": 0.0, "max_score": 4.0}
        mom = self._mom(cur_rsi=35.0, above_sma20=False, above_sma50=False)
        gate = _apply_quality_gate(sig, mom, -0.25, False)
        assert gate["flag"] == "HOLD"

    # -- markov_sig=None (HMM indicator disabled) --------------------------

    def test_is_weak_buy_context_none_markov_does_not_count(self):
        from Strategy_Auto_Trader.core.quality_gate import _is_weak_buy_context
        # Exactly one other weak condition (SMA50) — with markov_sig < 0.25 this
        # would reach 2 and veto; with None it must stay at 1 and pass.
        mom = self._mom(above_sma50=False)
        assert _is_weak_buy_context(0.10, mom) is True
        assert _is_weak_buy_context(None, mom) is False

    def test_is_adverse_exit_context_none_markov_does_not_count(self):
        from Strategy_Auto_Trader.core.quality_gate import _is_adverse_exit_context
        mom = self._mom(cur_rsi=35.0)
        assert _is_adverse_exit_context(-0.25, mom) is True
        assert _is_adverse_exit_context(None, mom) is False

    def test_apply_quality_gate_none_markov_uses_remaining_conditions(self):
        from Strategy_Auto_Trader.core.quality_gate import _apply_quality_gate
        # Two non-markov weak conditions still veto with markov_sig=None.
        sig = {"flag": "BUY", "score": 3.0, "max_score": 4.0}
        mom = self._mom(above_sma50=False, volume_ratio=0.8)
        gate = _apply_quality_gate(sig, mom, None, False)
        assert gate["flag"] == "HOLD"
        assert "quality_gate" in gate["reason"]

    def test_apply_quality_gate_none_markov_passes_strong_context(self):
        from Strategy_Auto_Trader.core.quality_gate import _apply_quality_gate
        sig = {"flag": "BUY", "score": 4.0, "max_score": 4.0}
        gate = _apply_quality_gate(sig, self._mom(), None, False)
        assert gate["flag"] == "BUY"
