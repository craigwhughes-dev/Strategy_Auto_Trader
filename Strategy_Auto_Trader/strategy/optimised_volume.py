"""Optimised + hard volume-conviction veto (volume_ratio >= 1.5).

What this strategy is trying to do
------------------------------------
Standalone variant of `optimised` (full copy, no inheritance) that promotes
the journal's strongest unconsumed finding into a hard entry veto: in the
2,122-trade live.csv journal analysis, entries with volume_ratio > 1.5
carried the ENTIRE net profit (+£2,293); everything below 1.5 was net
negative. The base `optimised` only raised the volume *weight* (0.5 -> 1.0);
this variant requires the volume conviction outright.

Entry
-----
Weighted vote: HMM (2.0) + RSI (1.0) + SMA200 (3.0) + trend SMA20/50 (2.0)
+ volume (1.0). Buy threshold 6.0 (out of max 9.0), sell -4.5.
Entry vetoes (BUY -> HOLD when flat):
  * RSI > 70 (overbought entries lose)
  * regime_signal <= 0 (no bull-regime confirmation)
  * volume_ratio < 1.5 (THE VARIANT GATE — low-volume entries were net
    negative in the journal; missing volume data also vetoes, fail-safe)
Vetoes only block NEW entries; they never suppress SELL/exit signalling
while a position is open. Standard quality gate applies on top.

Exit
----
Identical to optimised: wide hard stop 8%, take-profit 30%, vol-scaled
trailing stop (vol_stop_mult=2.0, window 20), profit-stop tightening
(scale 0.5), floor 4%, no max hold. Kelly position sizing on.

Best suited to: same trending names as optimised, but only on
volume-confirmed breakout bars. Expect fewer trades.
Known weaknesses: UNTESTED as a variant — the 1.5 cutoff is fitted
in-sample on the journal window; backtest before trusting.
"""

from __future__ import annotations

from ..core.momentum import composite_signal
from ..core.quality_gate import _apply_quality_gate
from ..plugins.exit_rules import StandardExitRules
from ..plugins.types import BarData, EntryDecision, ExitResult, RegimeState, TradeState

_RSI_OVERBOUGHT = 70.0
_MIN_REGIME_SIGNAL = 0.0
#: Journal: entries with volume_ratio > 1.5 carried the entire net profit.
_MIN_VOLUME_RATIO = 1.5


class OptimisedVolumeEntry:
    """Optimised entry + hard volume_ratio >= 1.5 veto on new entries.

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
        """Score a bar, then veto overbought / bear-regime / low-volume NEW entries."""
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
        vr = mom.get("volume_ratio")
        if vr is None:
            vr = _volume_ratio
        if float(vr or 0.0) < _MIN_VOLUME_RATIO:
            return EntryDecision(
                flag="HOLD", raw_flag=decision.raw_flag, score=decision.score,
                reason=(
                    f"volume veto: volume_ratio {float(vr or 0.0):.2f} < {_MIN_VOLUME_RATIO} "
                    "(low-volume entries were net negative)"
                ),
            )
        return decision


class OptimisedVolumeExit:
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
