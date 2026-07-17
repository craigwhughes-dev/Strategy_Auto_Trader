"""Optimised-aggressive — same signal/exit logic as `optimised`, no flip-guard.

What this strategy is trying to do
------------------------------------
Identical entry scoring, thresholds, vetoes and exit rules to
[[optimised]] (see that module's docstring for the journal analysis behind
the weights). The only difference: `require_flip_entry = False`, so a BUY
signal is acted on even when the previous bar was already BUY. `optimised`
only enters on a non-BUY -> BUY transition (see
quant_hmm/consolidated_engine.py's transition guard), which means it can
sit flat for long stretches while the score stays above threshold. This
variant re-enters on every qualifying bar instead, trading the same edge
with more market exposure.

Entry
-----
Same as `optimised`: HMM (2.0) + RSI (1.0) + SMA200 (3.0) + trend (2.0) +
volume (1.0), buy threshold 6.0, sell -4.5, RSI>70 and regime_signal<=0
vetoes. Flip-guard disabled.

Exit
----
Identical to `optimised` (8% stop, 30% target, vol-scaled trailing stop).

Known weaknesses: more time in market means more exposure to whipsaws
between the buy threshold and the flip point; not validated separately
from `optimised` — same in-sample caveat applies.
"""

from __future__ import annotations

from ..core.momentum import composite_signal
from ..core.quality_gate import _apply_quality_gate
from ..plugins.exit_rules import StandardExitRules
from ..plugins.types import BarData, EntryDecision, ExitResult, RegimeState, TradeState

#: Entry vetoes derived from the journal analysis (see optimised's docstring).
_RSI_OVERBOUGHT = 70.0
_MIN_REGIME_SIGNAL = 0.0


class OptimisedAggressiveEntry:
    """Optimised entry logic with the flip-guard disabled (re-enters every BUY bar).

    Satisfies EntryStrategyProtocol.
    """

    weights: dict[str, float] = {
        "markov": 0.0,
        "rsi":    1.0,   # kept low — but see the RSI > 70 veto below
        "trend":  2.0,
        "sma200": 3.0,
        "volume": 1.0,   # raised from trend's 0.5 — high volume carried the P&L
        "hmm":    2.0,   # raised from 1.5 — strong regimes hit 60%
    }
    buy_threshold: float = 6.0
    sell_threshold: float = -4.5
    #: Whether core/quality_gate._apply_quality_gate runs on top of the vote.
    quality_gate_enabled: bool = True
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
        """Score a bar, then veto overbought / bear-regime NEW entries."""
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
        return decision


class OptimisedAggressiveExit:
    """Identical exit rules to optimised's exit.

    Wide stop (8%) and target (30%) with vol-scaled trailing stop.

    Satisfies ExitStrategyProtocol.
    """

    _stop: float = 0.08
    _target: float = 0.30
    use_kelly: bool = True
    kelly_lookback: int = 20

    def __init__(self) -> None:
        self._impl = StandardExitRules(
            stop_loss_pct=self._stop,
            trailing_stop=0.0,        # use vol-stop rather than fixed trail
            vol_stop_mult=2.0,        # 2 × realised-vol trailing stop
            vol_stop_window=20,
            profit_stop_scale=0.5,    # tighten trail as profit grows
            min_stop_pct=0.04,        # floor: never tighter than 4%
            max_hold_days=0,          # no forced exit — winners ran to 27 days
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
        """Evaluate exit conditions for the current bar."""
        return self._impl.check(trade, bar_data)
