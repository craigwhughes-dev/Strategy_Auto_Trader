"""Parameter-sweep test variants of optimised_new for 2026-09-13 full live_sim sweep.

Six groups, each isolating ONE change vs. current optimised_new defaults:
  Group 1 — buy_threshold / sell_threshold pairs (symmetric):
    on_bt50 (5.0/-5.0), on_bt55 (5.5/-5.5), on_bt65 (6.5/-6.5), on_bt70 (7.0/-7.0)
    Note: min_entry_score=7.0 is the binding entry gate; these vary the SELL threshold.
  Group 2 — SMA200 weight (baseline 3.0):
    on_sma2 (2.0), on_sma25 (2.5)  [on_sma4=4.0 already tested, lost]
  Group 3 — RSI weight (baseline 1.0):
    on_rsi05 (0.5), on_rsi15 (1.5)  [on_rsi2=2.0 already tested, lost]
  Group 4 — HMM weight (baseline 2.0):
    on_hmm1 (1.0), on_hmm15 (1.5), on_hmm25 (2.5), on_hmm3 (3.0)
  Group 5 — Regime-signal veto:
    on_no_regime_veto (removed)
  Group 6 — RSI overbought veto threshold (baseline 70):
    on_rsi_veto_off, on_rsi_veto60, on_rsi_veto65, on_rsi_veto75, on_rsi_veto80

All variants use OptimisedNewExit unchanged.
These are test variants, not live strategies. Do not use as --strategy on the daemon.
"""

from __future__ import annotations

from ..plugins.types import EntryDecision, RegimeState
from .optimised_new import OptimisedNewEntry, OptimisedNewExit

__all__ = [
    "OptimisedNewExit",
    # Group 1: threshold pairs
    "OnBt50Entry", "OnBt55Entry", "OnBt65Entry", "OnBt70Entry",
    # Group 2: SMA200 weight
    "OnSma2Entry", "OnSma25Entry",
    # Group 3: RSI weight
    "OnRsi05Entry", "OnRsi15Entry",
    # Group 4: HMM weight
    "OnHmm1Entry", "OnHmm15Entry", "OnHmm25Entry", "OnHmm3Entry",
    # Group 5: regime veto
    "OnNoRegimeVetoEntry",
    # Group 6: RSI veto threshold
    "OnRsiVetoOffEntry", "OnRsiVeto60Entry", "OnRsiVeto65Entry",
    "OnRsiVeto75Entry", "OnRsiVeto80Entry",
]


# ── Group 1: buy_threshold / sell_threshold pairs ──────────────────────────────

class OnBt50Entry(OptimisedNewEntry):
    """buy_threshold=5.0, sell_threshold=-5.0 (looser sell exit vs baseline -6.0)."""
    buy_threshold: float = 5.0
    sell_threshold: float = -5.0


class OnBt55Entry(OptimisedNewEntry):
    """buy_threshold=5.5, sell_threshold=-5.5."""
    buy_threshold: float = 5.5
    sell_threshold: float = -5.5


class OnBt65Entry(OptimisedNewEntry):
    """buy_threshold=6.5, sell_threshold=-6.5 (stricter sell exit vs baseline -6.0)."""
    buy_threshold: float = 6.5
    sell_threshold: float = -6.5


class OnBt70Entry(OptimisedNewEntry):
    """buy_threshold=7.0, sell_threshold=-7.0 (strictest sell exit)."""
    buy_threshold: float = 7.0
    sell_threshold: float = -7.0


# ── Group 2: SMA200 weight ─────────────────────────────────────────────────────

class OnSma2Entry(OptimisedNewEntry):
    """sma200 weight 3.0 -> 2.0."""
    weights: dict[str, float] = {**OptimisedNewEntry.weights, "sma200": 2.0}


class OnSma25Entry(OptimisedNewEntry):
    """sma200 weight 3.0 -> 2.5."""
    weights: dict[str, float] = {**OptimisedNewEntry.weights, "sma200": 2.5}


# ── Group 3: RSI weight ────────────────────────────────────────────────────────

class OnRsi05Entry(OptimisedNewEntry):
    """rsi weight 1.0 -> 0.5."""
    weights: dict[str, float] = {**OptimisedNewEntry.weights, "rsi": 0.5}


class OnRsi15Entry(OptimisedNewEntry):
    """rsi weight 1.0 -> 1.5."""
    weights: dict[str, float] = {**OptimisedNewEntry.weights, "rsi": 1.5}


# ── Group 4: HMM weight ────────────────────────────────────────────────────────

class OnHmm1Entry(OptimisedNewEntry):
    """hmm weight 2.0 -> 1.0."""
    weights: dict[str, float] = {**OptimisedNewEntry.weights, "hmm": 1.0}


class OnHmm15Entry(OptimisedNewEntry):
    """hmm weight 2.0 -> 1.5."""
    weights: dict[str, float] = {**OptimisedNewEntry.weights, "hmm": 1.5}


class OnHmm25Entry(OptimisedNewEntry):
    """hmm weight 2.0 -> 2.5."""
    weights: dict[str, float] = {**OptimisedNewEntry.weights, "hmm": 2.5}


class OnHmm3Entry(OptimisedNewEntry):
    """hmm weight 2.0 -> 3.0."""
    weights: dict[str, float] = {**OptimisedNewEntry.weights, "hmm": 3.0}


# ── Groups 5 & 6: veto variants ────────────────────────────────────────────────

class _VetoConfigEntry(OptimisedNewEntry):
    """Internal base for veto-threshold variants; not registered directly.

    Subclasses override _rsi_veto (None = disabled) and _regime_veto_enabled
    (False = skip regime_signal <= 0 check). The evaluate() body is identical
    to OptimisedNewEntry except it reads these class attributes instead of the
    module-level constants.
    """
    _rsi_veto: float | None = 70.0
    _regime_veto_enabled: bool = True

    def evaluate(
        self,
        regime: RegimeState,
        mom: dict,
        _volume_ratio: float,
        currently_in: bool = False,
    ) -> EntryDecision:
        from ..core.momentum import composite_signal
        from ..core.quality_gate import _apply_quality_gate

        if not self._vol_filter_ok:
            return EntryDecision(
                flag="HOLD", raw_flag="HOLD", score=0.0,
                reason="vol_filter: unsuitable (choppy/mean-reverting)",
            )
        raw = composite_signal(
            markov_signal=0.0, mom=mom, hmm_state=regime.hmm_vote,
            buy_threshold=self._buy_t, sell_threshold=self._sell_t,
            weights=self._weights,
        )
        if self._quality_gate_enabled:
            gated = _apply_quality_gate(
                raw, mom, regime.regime_signal,
                currently_in=currently_in, gate_sensitivity=self._gate_sensitivity,
            )
        else:
            gated = dict(raw, reason="", gate_fired=False)
        decision = EntryDecision(
            flag=gated["flag"], raw_flag=raw["flag"],
            score=float(raw.get("score", 0.0)),
            reason=gated.get("reason", ""),
            gate_fired=gated.get("gate_fired", False),
        )
        if currently_in or decision.flag != "BUY":
            return decision
        if self._rsi_veto is not None and float(mom.get("cur_rsi", 50.0)) > self._rsi_veto:
            return EntryDecision(
                flag="HOLD", raw_flag=decision.raw_flag, score=decision.score,
                reason=f"optimised_new veto: RSI > {self._rsi_veto:.0f} (overbought entries lose)",
            )
        if self._regime_veto_enabled and regime.regime_signal is not None and regime.regime_signal <= 0.0:
            return EntryDecision(
                flag="HOLD", raw_flag=decision.raw_flag, score=decision.score,
                reason="optimised_new veto: regime_signal <= 0 (no bull-regime confirmation)",
            )
        if self._min_entry_score is not None and decision.score < self._min_entry_score:
            return EntryDecision(
                flag="HOLD", raw_flag=decision.raw_flag, score=decision.score,
                reason=f"optimised_new veto: score {decision.score:.1f} < min_entry_score {self._min_entry_score:.1f}",
            )
        return decision


class OnNoRegimeVetoEntry(_VetoConfigEntry):
    """regime_signal <= 0 veto removed; all else identical to optimised_new."""
    _regime_veto_enabled: bool = False


class OnRsiVetoOffEntry(_VetoConfigEntry):
    """RSI overbought veto disabled; all else identical to optimised_new."""
    _rsi_veto: float | None = None


class OnRsiVeto60Entry(_VetoConfigEntry):
    """RSI overbought veto threshold tightened 70 -> 60."""
    _rsi_veto: float | None = 60.0


class OnRsiVeto65Entry(_VetoConfigEntry):
    """RSI overbought veto threshold tightened 70 -> 65."""
    _rsi_veto: float | None = 65.0


class OnRsiVeto75Entry(_VetoConfigEntry):
    """RSI overbought veto threshold relaxed 70 -> 75."""
    _rsi_veto: float | None = 75.0


class OnRsiVeto80Entry(_VetoConfigEntry):
    """RSI overbought veto threshold relaxed 70 -> 80."""
    _rsi_veto: float | None = 80.0
