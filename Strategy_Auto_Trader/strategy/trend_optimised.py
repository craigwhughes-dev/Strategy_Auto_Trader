"""Trend + HMM regime strength filter.

What this strategy is trying to do
------------------------------------
Trend-following strategy wins on stocks in strong bull regimes but loses money
on weak/choppy stocks. Analysis shows Trend already filters entries heavily
through SMA200 (weight 3.0) + trend SMAs (weight 2.0), but the real differentiator
between winners and losers is **underlying stock momentum/bull regime strength**
(r=+0.956 for winners vs -0.069 for losers).

This variant adds a regime-strength gate: only enter if the HMM model detects
a **strong bull probability** (P_Bull > 0.6). This eliminates entries in
weak/choppy markets where the price is above moving averages but the regime
lacks conviction.

Best suited to: same as Trend but avoiding weak regimes. Fewer trades but
higher win rate on the entries that pass.

Entry
-----
Same as Trend (SMA200 3.0 + trend SMA20/50 2.0 + HMM 1.5 + RSI 1.0 + volume 0.5,
threshold 4.5) PLUS additional regime gate: **P(Bull) > 0.6** at entry.

Exit
----
Identical to Trend: 8% stop-loss, 30% take-profit. Vol-scaled trailing stop
(vol_stop_mult=2.0). Kelly position sizing.
"""

from __future__ import annotations

from ..core.momentum import composite_signal
from ..core.quality_gate import _apply_quality_gate
from ..plugins.exit_rules import StandardExitRules
from ..plugins.types import BarData, EntryDecision, ExitResult, RegimeState, TradeState


class TrendOptimisedEntry:
    """SMA-trend-led entry with strong bull regime gate.

    Satisfies EntryStrategyProtocol.
    """

    weights: dict[str, float] = {
        "markov": 0.0,
        "rsi":    1.0,   # reduced — overbought RSI is normal in trends
        "trend":  2.0,   # price above SMA20 AND SMA50 strongly weighted
        "sma200": 3.0,   # must be in long-term uptrend
        "volume": 0.5,   # reduced — volume less decisive for trend entries
        "hmm":    1.5,
    }
    buy_threshold: float = 4.5
    sell_threshold: float = -4.0
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
        """Score a bar and return an entry decision (pre- and post-gate flag).

        Additional health filter: only BUY if HMM regime is strongly bullish (P_Bull > 0.6).
        Filters out entries in weak/choppy markets despite above-MA price.
        """
        if not self._vol_filter_ok:
            return EntryDecision(
                flag="HOLD", raw_flag="HOLD", score=0.0,
                reason="vol_filter: unsuitable (choppy/mean-reverting)",
            )

        # Regime strength gate: require strong bull regime (P_Bull > 0.6)
        p_bull = regime.p_bull if regime.p_bull is not None else 0.0
        if p_bull <= 0.6:
            return EntryDecision(
                flag="HOLD", raw_flag="HOLD", score=0.0,
                reason=f"regime_gate: weak bull signal (P_Bull={p_bull:.2f}, need >0.6)",
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


class TrendOptimisedExit:
    """Identical to Trend: 8% stop-loss, 30% take-profit with vol-scaled trailing stop.

    The vol-scaled trailing stop (vol_stop_mult=2.0) tracks realised volatility
    and narrows as the trade matures (profit_stop_scale=0.5), keeping the
    strategy in the trend while protecting most of the accumulated gain.

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
            max_hold_days=0,          # no forced exit — let the trend run
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
