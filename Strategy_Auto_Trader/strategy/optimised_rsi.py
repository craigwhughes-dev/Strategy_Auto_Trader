"""Optimised + RSI sweet-spot band entry (60 <= RSI <= 70).

What this strategy is trying to do
------------------------------------
Standalone variant of `optimised` (full copy, no inheritance) that restricts
entries to the RSI band the journal analysis identified as the sweet spot:
RSI 60-70 at entry had profit factor 1.71, while RSI > 70 lost £1,123 over
402 trades (PF 1.02). The base `optimised` only vetoes the overbought tail
(> 70); this variant also vetoes everything below 60, admitting only the
measured winning band.

Entry
-----
Weighted vote: HMM (2.0) + RSI (1.0) + SMA200 (3.0) + trend SMA20/50 (2.0)
+ volume (1.0). Buy threshold 6.0 (out of max 9.0), sell -4.5.
Entry vetoes (BUY -> HOLD when flat):
  * RSI outside [60, 70] (THE VARIANT GATE — journal sweet spot only;
    missing RSI data defaults to 50 and is vetoed, fail-safe)
  * regime_signal <= 0 (no bull-regime confirmation)
Vetoes only block NEW entries; they never suppress SELL/exit signalling
while a position is open. Standard quality gate applies on top.

Exit
----
Identical to optimised: wide hard stop 8%, take-profit 30%, vol-scaled
trailing stop (vol_stop_mult=2.0, window 20), profit-stop tightening
(scale 0.5), floor 4%, no max hold. Kelly position sizing on.

Best suited to: momentum entries already in motion (RSI 60-70) without
chasing overbought moves. Expect fewer trades than optimised.
Known weaknesses: UNTESTED as a variant — the 60-70 band is fitted
in-sample on the journal window; the narrow band may cut trade count
sharply. Backtest before trusting.
"""

from __future__ import annotations

from ..core.momentum import composite_signal
from ..core.quality_gate import _apply_quality_gate
from ..plugins.exit_rules import StandardExitRules
from ..plugins.types import BarData, EntryDecision, ExitResult, RegimeState, TradeState

#: Journal: RSI 60-70 at entry was the sweet spot (PF 1.71); >70 lost money.
_RSI_BAND_LOW = 60.0
_RSI_BAND_HIGH = 70.0
_MIN_REGIME_SIGNAL = 0.0


class OptimisedRsiEntry:
    """Optimised entry restricted to the RSI 60-70 sweet-spot band.

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
        """Score a bar, then veto out-of-band-RSI / bear-regime NEW entries."""
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
        cur_rsi = float(mom.get("cur_rsi", 50.0))
        if not (_RSI_BAND_LOW <= cur_rsi <= _RSI_BAND_HIGH):
            return EntryDecision(
                flag="HOLD", raw_flag=decision.raw_flag, score=decision.score,
                reason=(
                    f"rsi band veto: RSI {cur_rsi:.1f} outside "
                    f"[{_RSI_BAND_LOW:.0f}, {_RSI_BAND_HIGH:.0f}] sweet spot (journal PF 1.71)"
                ),
            )
        if regime.regime_signal is not None and regime.regime_signal <= _MIN_REGIME_SIGNAL:
            return EntryDecision(
                flag="HOLD", raw_flag=decision.raw_flag, score=decision.score,
                reason="optimised veto: regime_signal <= 0 (no bull-regime confirmation)",
            )
        return decision


class OptimisedRsiExit:
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
