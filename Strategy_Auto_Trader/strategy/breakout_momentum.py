"""Breakout momentum strategy — trade strong breakouts with vol confirmation.

Entry
-----
Uses a momentum-weighted composite signal biased towards breakouts: larger
weight for short-term trend and volume. Requires the short-term trend to be
aligned (above SMA20 & SMA50) and rewards volume spikes. Quality gate
(quality_gate_enabled=True) applies on top of the trend-adjusted score.

Exit
----
Wider stop and large take-profit to let strong breakouts run; uses a
vol-scaled trailing stop to protect gains.
Kelly position sizing on (use_kelly=True, kelly_lookback=20).

Best suited to: no live/backtest validation yet.
Known weaknesses: UNTESTED — added alongside ai_strategy and mean_reversion,
none of the three have a backtest run against them. Treat as experimental
until validated (see choppy_vol for what an untested strategy looks like
after backtesting turned it negative).
"""

from __future__ import annotations

from ..core.momentum import composite_signal
from ..core.quality_gate import _apply_quality_gate
from ..plugins.exit_rules import StandardExitRules
from ..plugins.types import BarData, EntryDecision, ExitResult, RegimeState, TradeState


class BreakoutMomentumEntry:
    """Momentum breakout entry using a tuned composite signal."""

    weights: dict[str, float] = {
        "markov": 0.0,
        "rsi": 2.0,
        "trend": 2.5,
        "sma200": 1.0,
        "volume": 1.5,
        "hmm": 1.5,
    }
    buy_threshold: float = 4.0
    sell_threshold: float = -3.0
    #: Whether core/quality_gate._apply_quality_gate runs on top of the score.
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
        volume_ratio: float,
        currently_in: bool = False,
    ) -> EntryDecision:
        if not self._vol_filter_ok:
            return EntryDecision(
                flag="HOLD",
                raw_flag="HOLD",
                score=0.0,
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

        # Reward volume spikes by nudging the score — applied before the
        # trend-alignment gate so the reported score stays consistent
        # across both the HOLD and admitted branches.
        vr = mom.get("volume_ratio") or volume_ratio or 1.0
        score = float(raw.get("score", 0.0)) + (0.8 if vr > 1.3 else 0.0)

        # Require short-term trend alignment for breakout
        if not (mom.get("above_sma20") and mom.get("above_sma50")):
            return EntryDecision(
                flag="HOLD", raw_flag=raw["flag"], score=round(score, 2),
                reason="needs short-term trend alignment",
            )

        final_flag = "BUY" if score >= self._buy_t else ("SELL" if score <= self._sell_t else "HOLD")

        if self.quality_gate_enabled:
            gated = _apply_quality_gate(
                {"flag": final_flag, "score": score},
                mom, regime.regime_signal, currently_in=currently_in,
            )
        else:
            gated = {"flag": final_flag, "reason": "", "gate_fired": False}
        return EntryDecision(
            flag=gated["flag"], raw_flag=raw["flag"], score=round(score, 2),
            reason=gated.get("reason", ""), gate_fired=gated.get("gate_fired", False),
        )


class BreakoutMomentumExit:
    _stop: float = 0.06
    _target: float = 0.25
    use_kelly: bool = True
    kelly_lookback: int = 20

    def __init__(self) -> None:
        self._impl = StandardExitRules(
            stop_loss_pct=self._stop,
            trailing_stop=0.0,
            vol_stop_mult=1.5,
            vol_stop_window=20,
            profit_stop_scale=0.5,
            min_stop_pct=0.04,
            max_hold_days=0,
            exit_on_macd_cross=True,
            exit_on_rsi_reversal=True,
            exit_on_consolidation=False,
            use_sar_stop=False,
        )

    @property
    def stop_loss_pct(self) -> float:
        return self._stop

    @property
    def take_profit_pct(self) -> float:
        return self._target

    def check(self, trade: TradeState, bar: BarData) -> ExitResult:
        return self._impl.check(trade, bar)
