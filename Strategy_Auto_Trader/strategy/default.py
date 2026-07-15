"""Default strategy — balanced, moderate entry, fixed stop/target.

What this strategy is trying to do
------------------------------------
This is the baseline strategy. It looks for stocks where several independent
signals all point in the same direction at the same time: the HMM regime model
believes we are in a bull market, RSI momentum is positive, the price is above
its long-term (200-bar) moving average, the short-term trend (SMA20 + SMA50) is
bullish, and volume is above average. No single signal is dominant — all five
contribute roughly equally, with the SMA200 carrying slightly more weight (2.0)
because long-term trend alignment is the most reliable filter against whipsaws.

A secondary quality gate acts as a veto: even if the vote score reaches the buy
threshold, the entry is blocked if the market context looks weak (e.g. RSI below
50, below the SMA200, and thin volume — any two of five weak conditions blocks).
While in a trade the same gate watches for adverse conditions and forces an early
exit if at least two deterioration signals fire together, waiting min_hold_bars
(~2 trading days) same as a plain composite-signal SELL.

Risk management is simple: take profit at +15% or cut the loss at -5%. There is
no trailing stop — the strategy relies on the signal gate to exit rather than on
price-action trailing.

Best suited to: trending markets with clear regime changes. Will struggle in
choppy, sideways markets where the HMM oscillates without a clear bull/bear
signal.

Entry
-----
Weighted vote: HMM (1.5) + RSI (1.5) + SMA200 (2.0) + trend SMA20/50 (1.0)
+ volume (1.0).  Markov slot zeroed (HMM carries that role).
Buy threshold 3.0 (out of max ~7.5), sell -3.0.
Quality gate vetoes BUY if >= 2 of 5 weak-context signals are true;
forces SELL if >= 2 of 5 adverse-exit signals are true while in a position
(after min_hold_bars — backtesting confirmed bypassing this hurts P&L).

Exit
----
Hard stop-loss 5%, hard take-profit 15%.
No trailing stop, no vol-stop, no indicator exits.
"""

from __future__ import annotations

from ..core.momentum import composite_signal
from ..core.quality_gate import _apply_quality_gate
from ..plugins.exit_rules import StandardExitRules
from ..plugins.types import BarData, EntryDecision, ExitResult, RegimeState, TradeState

_WEIGHTS: dict[str, float] = {
    "markov": 0.0,
    "rsi":    1.5,
    "trend":  1.0,
    "sma200": 2.0,
    "volume": 1.0,
    "hmm":    1.5,
}


class DefaultEntry:
    """Balanced HMM + RSI + SMA200 entry with the standard quality gate.

    Satisfies EntryStrategyProtocol.
    """

    #: Weight overrides — subclasses can override the class attribute.
    weights: dict[str, float] = _WEIGHTS
    buy_threshold: float = 3.0
    sell_threshold: float = -3.0

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        buy_threshold: float | None = None,
        sell_threshold: float | None = None,
        vol_filter_ok: bool = True,
    ) -> None:
        self._weights = {**self.weights, **(weights or {})}
        self._buy_t = buy_threshold if buy_threshold is not None else self.buy_threshold
        self._sell_t = sell_threshold if sell_threshold is not None else self.sell_threshold
        self._vol_filter_ok = vol_filter_ok

    def evaluate(
        self,
        regime: RegimeState,
        mom: dict,
        _volume_ratio: float,
        currently_in: bool = False,
    ) -> EntryDecision:
        """Score a bar and return an entry decision (pre- and post-gate flag)."""
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
        gated = _apply_quality_gate(raw, mom, regime.regime_signal, currently_in=currently_in)
        return EntryDecision(
            flag=gated["flag"],
            raw_flag=raw["flag"],
            score=float(raw.get("score", 0.0)),
            reason=gated.get("reason", ""),
            gate_fired=gated.get("gate_fired", False),
        )


class DefaultExit:
    """Hard 5% stop-loss, 15% take-profit.  No trailing or indicator exits.

    Satisfies ExitStrategyProtocol.
    """

    _stop: float = 0.05
    _target: float = 0.15

    def __init__(self) -> None:
        self._impl = StandardExitRules(
            stop_loss_pct=self._stop,
            trailing_stop=0.0,
            vol_stop_mult=0.0,
            vol_stop_window=20,
            profit_stop_scale=0.0,
            min_stop_pct=0.05,
            max_hold_days=0,
            exit_on_macd_cross=False,
            exit_on_rsi_reversal=False,
            exit_on_consolidation=False,
            use_sar_stop=False,
        )

    @property
    def stop_loss_pct(self) -> float:
        """Hard stop-loss fraction (5%)."""
        return self._stop

    @property
    def take_profit_pct(self) -> float:
        """Hard take-profit fraction (15%)."""
        return self._target

    def check(self, trade: TradeState, bar_data: BarData) -> ExitResult:
        """Evaluate exit conditions for the current bar."""
        return self._impl.check(trade, bar_data)
