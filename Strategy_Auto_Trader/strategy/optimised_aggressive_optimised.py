"""Optimised-Aggressive + high-conviction entry gate.

Differentiator: Winners return 0.044%, losers 0.004% (r=0.889).
Gate: Only enter on strong composite signal score (> 6.0).
Flip-guard: disabled — re-enters on every qualifying bar.
"""

from __future__ import annotations

from ..core.momentum import composite_signal
from ..core.quality_gate import _apply_quality_gate
from ..plugins.exit_rules import StandardExitRules
from ..plugins.types import BarData, EntryDecision, ExitResult, RegimeState, TradeState

#: Entry vetoes derived from the journal analysis (see module docstring).
_RSI_OVERBOUGHT = 70.0
_MIN_REGIME_SIGNAL = 0.0
_MIN_SIGNAL_SCORE = 6.0


class OptimisedAggressiveOptimisedEntry:
    """Optimised entry with high-conviction gate + aggressive re-entry.

    Satisfies EntryStrategyProtocol.
    """

    weights: dict[str, float] = {
        "markov": 0.0,
        "rsi":    1.0,
        "trend":  2.0,
        "sma200": 3.0,
        "volume": 1.0,
        "hmm":    2.0,
    }
    buy_threshold: float = 6.0
    sell_threshold: float = -4.5
    #: Whether core/quality_gate._apply_quality_gate runs on top of the vote.
    quality_gate_enabled: bool = True
    #: Aggressive: no flip-guard — re-enter on every BUY bar.
    require_flip_entry: bool = False

    def __init__(self, vol_filter_ok: bool = True) -> None:
        self._weights = self.weights
        self._buy_t = self.buy_threshold
        self._sell_t = self.sell_threshold
        self._vol_filter_ok = vol_filter_ok

    def evaluate(
        self,
        regime: RegimeState,
        mom: dict,
        _volume_ratio: float,
        currently_in: bool = False,
    ) -> EntryDecision:
        """Score a bar, veto overbought/bear-regime entries, then apply conviction gate."""
        if not self._vol_filter_ok:
            return EntryDecision(
                flag="HOLD", raw_flag="HOLD", score=0.0,
                reason="vol_filter: unsuitable (choppy/mean-reverting)",
            )
        raw = composite_signal(
            markov_signal=0.0,
            mom=mom,
            hmm_state=regime.hmm_vote,
            buy_threshold=self._buy_t,
            sell_threshold=self._sell_t,
            weights=self._weights,
        )
        if self.quality_gate_enabled:
            gated = _apply_quality_gate(raw, mom, regime.regime_signal, currently_in=currently_in)
        else:
            gated = dict(raw, reason="", gate_fired=False)
        decision = EntryDecision(
            flag=gated["flag"],
            raw_flag=raw["flag"],
            score=float(raw.get("score", 0.0)),
            reason=gated.get("reason", ""),
            gate_fired=gated.get("gate_fired", False),
        )
        if currently_in or decision.flag != "BUY":
            return decision
        if float(mom.get("cur_rsi", 50.0)) > _RSI_OVERBOUGHT:
            return EntryDecision(
                flag="HOLD", raw_flag=decision.raw_flag, score=decision.score,
                reason=f"optimised veto: RSI > {_RSI_OVERBOUGHT:.0f} (overbought entries lose)",
            )
        if regime.regime_signal is not None and regime.regime_signal <= _MIN_REGIME_SIGNAL:
            return EntryDecision(
                flag="HOLD", raw_flag=decision.raw_flag, score=decision.score,
                reason="optimised veto: regime_signal <= 0 (no bull-regime confirmation)",
            )
        # High-conviction gate: only enter on strong signal score
        if decision.score < _MIN_SIGNAL_SCORE:
            return EntryDecision(
                flag="HOLD", raw_flag=decision.raw_flag, score=decision.score,
                reason=f"conviction gate: weak signal (score={decision.score:.2f}, need >={_MIN_SIGNAL_SCORE})",
            )
        return decision


class OptimisedAggressiveOptimisedExit:
    """Wide stop (8%) and target (30%) with vol-scaled trailing stop.

    Identical shape to OptimisedExit.

    Satisfies ExitStrategyProtocol.
    """

    _stop: float = 0.08
    _target: float = 0.30
    use_kelly: bool = True
    kelly_lookback: int = 20

    def __init__(self) -> None:
        self._impl = StandardExitRules(
            stop_loss_pct=self._stop,
            trailing_stop=0.0,
            vol_stop_mult=2.0,
            vol_stop_window=20,
            profit_stop_scale=0.5,
            min_stop_pct=0.04,
            max_hold_days=0,
            exit_on_macd_cross=False,
            exit_on_rsi_reversal=False,
            exit_on_consolidation=False,
            use_sar_stop=False,
        )

    @property
    def stop_loss_pct(self) -> float:
        """Hard stop-loss fraction (8%)."""
        return self._stop

    @property
    def take_profit_pct(self) -> float:
        """Hard take-profit fraction (30%)."""
        return self._target

    def check(self, trade: TradeState, bar_data: BarData) -> ExitResult:
        """Evaluate exit conditions for the current car."""
        return self._impl.check(trade, bar_data)
