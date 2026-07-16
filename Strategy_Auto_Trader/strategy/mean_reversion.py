"""Mean-reversion strategy — trade pullbacks and RSI overshoots.

This strategy is designed to trade mean-reverting / choppy tickers where
short-term overshoots tend to revert back to the short-term mean (SMA20).
It prefers a choppy environment (vol_filter OK == False) — if the global
volatility/regime filter marks a ticker as trending, the entry will veto.

Entry
-----
Look for RSI below ~35 together with price a few percent below SMA20. A
moderate volume boost increases confidence. The raw numeric score is a
small heuristic rather than a learned model.

Exit
----
Small stop (3%) and modest target (8%) — keep trades tight and frequent.
Kelly position sizing on (use_kelly=True, kelly_lookback=20).

Score is a custom heuristic (RSI undershoot + SMA20 distance + volume
boost), not the shared composite_signal/weights-dict path other strategies
use — deliberately, since the weights dict's "trend"/"sma200" keys assume
a trend-following signal, which is the opposite of what this strategy
wants.

Quality gate: quality_gate_enabled=False (documentation-only — the shared
_apply_quality_gate is never called). Only the adverse-exit escalation half
is applied inline (see _is_adverse_exit_context in core/quality_gate.py).
The weak-buy-context veto half is skipped — it counts "price below
SMA20/50" and "RSI<50 without a cross above 50" as weak-context signals,
which is exactly the setup this strategy buys, so applying it would veto
nearly every entry.
"""

from __future__ import annotations

from ..core.quality_gate import _is_adverse_exit_context
from ..plugins.exit_rules import StandardExitRules
from ..plugins.types import BarData, EntryDecision, ExitResult, RegimeState, TradeState


class MeanReversionEntry:
    """Entry that buys RSI overshoots against the short-term mean.

    Satisfies EntryStrategyProtocol.
    """

    buy_threshold: float = 2.5
    sell_threshold: float = -2.5
    #: Documentation-only — this strategy never calls _apply_quality_gate;
    #: only its adverse-exit half is applied inline in evaluate() (see module
    #: docstring for why the weak-buy veto half is skipped).
    quality_gate_enabled: bool = False
    #: Trades the low-trend-quality names the default vol_screen vetoes —
    #: overnight_scope.py's stage-1 screen inverts to keep those tickers
    #: instead of excluding them when the market's strategy sets this.
    wants_low_trend_quality: bool = True

    def __init__(self, vol_filter_ok: bool = True) -> None:
        # mean-reversion prefers choppy tickers; if vol_filter_ok is True
        # it means the environment looks trendy — veto entries in that case.
        self._vol_filter_ok = vol_filter_ok

    def evaluate(
        self,
        regime: RegimeState,
        mom: dict,
        _volume_ratio: float,
        currently_in: bool = False,
    ) -> EntryDecision:
        if self._vol_filter_ok:
            return EntryDecision(
                flag="HOLD",
                raw_flag="HOLD",
                score=0.0,
                reason="mean_reversion: unsuitable in trending environment",
            )

        rsi = float(mom.get("cur_rsi", 50.0))
        pct_sma20 = float(mom.get("pct_from_sma20", 0.0) or 0.0)
        vol_ratio = float(mom.get("volume_ratio", 1.0) or 1.0)

        # Heuristic score: undershoot depth + RSI undershoot + volume boost
        score = 0.0
        # RSI contribution: below 50 is bullish for reversion; stronger below 40
        score += max(0.0, (50.0 - rsi) / 10.0)
        # Price distance from SMA20: deeper below -> larger score. Multiplier
        # is 30 (not 10) so a realistic 3-5% dip contributes comparably to
        # the RSI term instead of being dwarfed by it — at 10x, a 10% dip
        # only scored 1.0 against RSI's max ~5.0, making this term decorative.
        score += max(0.0, -pct_sma20 * 30.0)
        # Volume boost
        if vol_ratio > 1.2:
            score += 0.8

        # Map to BUY/HOLD/SELL similar to composite_signal thresholds
        if score >= self.buy_threshold:
            raw_flag = "BUY"
        elif score <= self.sell_threshold:
            raw_flag = "SELL"
        else:
            raw_flag = "HOLD"

        # Adverse-exit escalation only (see module docstring for why the
        # weak-buy veto half of the shared gate is skipped for this strategy).
        if currently_in and _is_adverse_exit_context(regime.regime_signal, mom):
            return EntryDecision(
                flag="SELL",
                raw_flag=raw_flag,
                score=float(round(score, 2)),
                reason="quality_gate: adverse exit context",
                gate_fired=True,
            )

        return EntryDecision(
            flag=raw_flag,
            raw_flag=raw_flag,
            score=float(round(score, 2)),
            reason="",
        )


class MeanReversionExit:
    """Tight stop and modest take-profit for mean reversion trades.

    Satisfies ExitStrategyProtocol.
    """

    _stop: float = 0.03
    _target: float = 0.08
    use_kelly: bool = True
    kelly_lookback: int = 20

    def __init__(self) -> None:
        self._impl = StandardExitRules(
            stop_loss_pct=self._stop,
            trailing_stop=0.0,
            vol_stop_mult=0.0,
            vol_stop_window=10,
            profit_stop_scale=0.0,
            min_stop_pct=0.02,
            max_hold_days=5,
            exit_on_macd_cross=False,
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
