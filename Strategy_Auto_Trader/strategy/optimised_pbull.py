"""Optimised + HMM bull-probability gate (P(Bull) > 0.6 at entry).

What this strategy is trying to do
------------------------------------
Standalone variant of `optimised` (full copy, no inheritance) that adds the
same P(Bull) regime-strength gate used by the other `*_optimised` variants
(conservative_optimised, trend_optimised): only enter when the HMM assigns
strong bull probability. Base `optimised` already vetoes regime_signal <= 0
(p_bull_smooth - p_bear); this variant additionally requires the raw
per-bar bull probability to exceed 0.6, filtering entries where the
smoothed signal is positive but the current bar's regime conviction is weak.

Entry
-----
Weighted vote: HMM (2.0) + RSI (1.0) + SMA200 (3.0) + trend SMA20/50 (2.0)
+ volume (1.0). Buy threshold 6.0 (out of max 9.0), sell -4.5.
Entry vetoes (BUY -> HOLD when flat):
  * RSI > 70 (overbought entries lose)
  * regime_signal <= 0 (no bull-regime confirmation)
  * P(Bull) <= 0.6 (THE VARIANT GATE — raw bull probability must be strong)
Vetoes only block NEW entries; unlike conservative/trend_optimised (which
gate before scoring), this gate runs after scoring and only demotes BUY,
so SELL/exit signalling while a position is open is never suppressed.
Standard quality gate applies on top.

Exit
----
Identical to optimised: wide hard stop 8%, take-profit 30%, vol-scaled
trailing stop (vol_stop_mult=2.0, window 20), profit-stop tightening
(scale 0.5), floor 4%, no max hold. Kelly position sizing on.

Best suited to: same trending names as optimised in decisively bull regimes.
Known weaknesses: UNTESTED as a variant — largely overlaps the base's
regime_signal veto plus HMM weight 2.0, so the marginal effect may be
small. Backtest before trusting.
"""

from __future__ import annotations

from ..core.momentum import composite_signal
from ..core.quality_gate import _apply_quality_gate
from ..plugins.exit_rules import StandardExitRules
from ..plugins.types import BarData, EntryDecision, ExitResult, RegimeState, TradeState

_RSI_OVERBOUGHT = 70.0
_MIN_REGIME_SIGNAL = 0.0
#: Raw HMM bull probability required at entry.
_MIN_P_BULL = 0.6


class OptimisedPbullEntry:
    """Optimised entry + P(Bull) > 0.6 veto on new entries.

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
    quality_gate_enabled: bool = True

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
        """Score a bar, then veto overbought / bear-regime / weak-P(Bull) NEW entries."""
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
        p_bull = regime.p_bull if regime.p_bull is not None else 0.0
        if p_bull <= _MIN_P_BULL:
            return EntryDecision(
                flag="HOLD", raw_flag=decision.raw_flag, score=decision.score,
                reason=f"regime_gate: weak bull (P_Bull={p_bull:.2f}, need >{_MIN_P_BULL})",
            )
        return decision


class OptimisedPbullExit:
    """Wide stop (8%) and target (30%) with vol-scaled trailing stop.

    Identical shape to optimised's exit — the variant only changes entry.

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
            max_hold_days=0,          # no forced exit
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
