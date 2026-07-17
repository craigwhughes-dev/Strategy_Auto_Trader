"""Conservative_Optimised — Conservative + HMM regime strength filter.

What this strategy is trying to do
------------------------------------
Conservative strategy wins on stocks in strong bull regimes but loses money on
weak/choppy stocks. Analysis shows Conservative already filters 100% of trades
to above-SMA200 (SMA200 weight 3.0), but the real differentiator between winners
and losers is **underlying stock momentum/bull regime strength** (r=+0.95 with
B&H return).

This variant adds a regime-strength gate: only enter if the HMM model detects
a **strong bull probability** (P_Bull > 0.6, vs default 0.0 threshold). This
eliminates entries in weak/choppy markets where the stock is sideways despite
above-SMA200 price action.

Analysis shows:
- Conservative losers: weak B&H return (+0.27%), choppy (trend_quality -3.03)
- Conservative winners: strong B&H return (+1.71%), less choppy (trend_quality -2.58)
- Adding HMM regime strength gate filters out weak-momentum stocks while keeping strong ones

Best suited to: same as Conservative but avoiding chop. Fewer trades but higher
win rate on the entries that pass (Kelly fraction increases on winners).

Entry
-----
Same as Conservative (HMM 2.0 + RSI 1.5 + SMA200 3.0 + trend 1.5 + volume 1.5,
threshold 4.5) PLUS additional regime gate: **P(Bull) > 0.6** at entry.

Exit
----
Identical to Conservative: 3% stop-loss, 10% take-profit. Kelly position sizing.
"""

from __future__ import annotations

from ..core.momentum import composite_signal
from ..core.quality_gate import _apply_quality_gate
from ..plugins.exit_rules import StandardExitRules
from ..plugins.types import BarData, EntryDecision, ExitResult, RegimeState, TradeState


class ConservativeOptimisedEntry:
    """Conservative entry + SMA200 health gate for uptrend confirmation.

    Satisfies EntryStrategyProtocol.
    """

    weights: dict[str, float] = {
        "markov": 0.0,
        "rsi":    1.5,
        "trend":  1.5,
        "sma200": 3.0,
        "volume": 1.5,
        "hmm":    2.0,
    }
    buy_threshold: float = 4.5
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
        """Score a bar and return an entry decision (pre- and post-gate flag).

        Additional health filter: only BUY if HMM regime is strongly bullish (P_Bull > 0.6).
        Filters out entries in weak/choppy markets despite above-SMA200 price.
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


class ConservativeOptimisedExit:
    """Identical to Conservative: 3% stop-loss, 10% take-profit."""

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
