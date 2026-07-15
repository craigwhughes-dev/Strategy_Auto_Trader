"""Conservative strategy — fewer, higher-conviction entries, tighter risk management.

What this strategy is trying to do
------------------------------------
This strategy tries to avoid entering trades unless a strong, multi-signal
consensus exists. It is designed for investors who would rather miss a move
than take a losing trade.

The key idea is that the two most reliable macro filters — the long-term trend
(SMA200, weight 3.0) and the HMM regime model (weight 2.0) — must both agree
strongly before entry is considered. Short-term trend and volume confirmation
are also weighted higher than the default (1.5 each), so all five signal groups
must lean bullish at the same time. The buy threshold is raised to 4.5 (vs.
default 3.0), which means roughly 60% of the maximum possible vote score is
required — in practice this eliminates most marginal signals.

Risk management is tighter than the default: stops are cut at 3% (not 5%),
and profits are banked at 10% (not 15%). The logic is that if the signal is
this strong, a 3% adverse move is more likely to mean the thesis is wrong than
just noise, so cutting earlier is rational. The tighter take-profit also means
each winning trade books a smaller but more certain gain, which works well when
the entry bar is high and losses per trade are infrequent.

Best suited to: blue-chip, lower-volatility stocks where a 3% stop is sensible
and where false signals are expensive (e.g. large-cap ETFs, defensive sectors).
Will under-perform in fast-moving growth stocks where 3% noise is normal and
15%+ moves are the reward.

Entry
-----
Weighted vote: HMM (2.0) + RSI (1.5) + SMA200 (3.0) + trend SMA20/50 (1.5)
+ volume (1.5).  Buy threshold 4.5 (out of max ~10.5).
Quality gate (quality_gate_enabled=True) still applies on top (same 2-of-5
veto threshold as default).

Exit
----
Tighter stop-loss 3%, take-profit 10%.  No trailing stop.
Kelly position sizing on (use_kelly=True, kelly_lookback=20).
"""

from __future__ import annotations

from ..core.momentum import composite_signal
from ..core.quality_gate import _apply_quality_gate
from ..plugins.exit_rules import StandardExitRules
from ..plugins.types import BarData, EntryDecision, ExitResult, RegimeState, TradeState


class ConservativeEntry:
    """High-conviction entry: SMA200 and HMM must strongly agree.

    Satisfies EntryStrategyProtocol.
    """

    weights: dict[str, float] = {
        "markov": 0.0,
        "rsi":    1.5,
        "trend":  1.5,
        "sma200": 3.0,   # dominant long-term trend gate
        "volume": 1.5,
        "hmm":    2.0,   # regime must be clearly bullish
    }
    buy_threshold: float = 4.5
    sell_threshold: float = -4.5
    #: Whether core/quality_gate._apply_quality_gate runs on top of the vote.
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
        if self.quality_gate_enabled:
            gated = _apply_quality_gate(raw, mom, regime.regime_signal, currently_in=currently_in)
        else:
            gated = dict(raw, reason="", gate_fired=False)
        return EntryDecision(
            flag=gated["flag"],
            raw_flag=raw["flag"],
            score=float(raw.get("score", 0.0)),
            reason=gated.get("reason", ""),
            gate_fired=gated.get("gate_fired", False),
        )


class ConservativeExit:
    """Tighter 3% stop-loss and 10% take-profit.  No trailing or indicator exits.

    Satisfies ExitStrategyProtocol.
    """

    _stop: float = 0.03
    _target: float = 0.10
    use_kelly: bool = True
    kelly_lookback: int = 20

    def __init__(self) -> None:
        self._impl = StandardExitRules(
            stop_loss_pct=self._stop,
            trailing_stop=0.0,
            vol_stop_mult=0.0,
            vol_stop_window=20,
            profit_stop_scale=0.0,
            min_stop_pct=0.03,
            max_hold_days=0,
            exit_on_macd_cross=False,
            exit_on_rsi_reversal=False,
            exit_on_consolidation=False,
            use_sar_stop=False,
        )

    @property
    def stop_loss_pct(self) -> float:
        """Hard stop-loss fraction (3%)."""
        return self._stop

    @property
    def take_profit_pct(self) -> float:
        """Hard take-profit fraction (10%)."""
        return self._target

    def check(self, trade: TradeState, bar_data: BarData) -> ExitResult:
        """Evaluate exit conditions for the current bar."""
        return self._impl.check(trade, bar_data)
