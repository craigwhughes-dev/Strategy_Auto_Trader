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

Quality gate: quality_gate_enabled=False by default — only the adverse-exit
escalation half runs (see _is_adverse_exit_context in core/quality_gate.py),
unconditionally, same as always. Passing quality_gate_enabled=True is
accepted but NOT RECOMMENDED for this strategy: the full gate's weak-buy
veto counts "price below SMA20/50" and "RSI<50 without a cross above 50" as
weak-context signals — exactly the setup this strategy buys — so enabling
it functions correctly but vetoes nearly every entry, effectively disabling
the strategy rather than tuning it.

buy_threshold/sell_threshold and every exit value above (stop_loss_pct/
take_profit_pct/vol_stop_window/exit_on_rsi_reversal/etc.) are this
strategy's own defaults, not fixed — overridable via the matching CLI flag
regardless of which --strategy is selected; an omitted flag leaves this
strategy's default untouched. No `weights` override exists here (see above:
this strategy doesn't use the shared weights-dict path).
"""

from __future__ import annotations

from ..core.quality_gate import _apply_quality_gate, _is_adverse_exit_context
from ..plugins.types import BarData, EntryDecision, ExitResult, RegimeState, TradeState
from .base.exit_overrides import build_standard_exit_rules


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
    #: Number of weak-context/adverse-exit conditions (of 5) needed to fire
    #: the gate (applies to the always-on inline adverse-exit check below,
    #: as well as the opt-in full gate). Overridable via CLI --gate-sensitivity.
    gate_sensitivity: int = 2
    #: Trades the low-trend-quality names the default vol_screen vetoes —
    #: overnight_scope.py's stage-1 screen inverts to keep those tickers
    #: instead of excluding them when the market's strategy sets this.
    wants_low_trend_quality: bool = True

    def __init__(
        self,
        buy_threshold: float | None = None,
        sell_threshold: float | None = None,
        vol_filter_ok: bool = True,
        quality_gate_enabled: bool | None = None,
        gate_sensitivity: int | None = None,
    ) -> None:
        # mean-reversion prefers choppy tickers; if vol_filter_ok is True
        # it means the environment looks trendy — veto entries in that case.
        self._buy_t = buy_threshold if buy_threshold is not None else self.buy_threshold
        self._sell_t = sell_threshold if sell_threshold is not None else self.sell_threshold
        self._vol_filter_ok = vol_filter_ok
        self._quality_gate_enabled = (
            quality_gate_enabled if quality_gate_enabled is not None else self.quality_gate_enabled
        )
        self._gate_sensitivity = (
            gate_sensitivity if gate_sensitivity is not None else self.gate_sensitivity
        )

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
        if score >= self._buy_t:
            raw_flag = "BUY"
        elif score <= self._sell_t:
            raw_flag = "SELL"
        else:
            raw_flag = "HOLD"

        if self._quality_gate_enabled:
            # See module docstring: this vetoes nearly every entry for this
            # strategy specifically. Accepted for uniformity, not recommended.
            gated = _apply_quality_gate(
                {"flag": raw_flag}, mom, regime.regime_signal, currently_in=currently_in,
                gate_sensitivity=self._gate_sensitivity,
            )
            return EntryDecision(
                flag=gated["flag"],
                raw_flag=raw_flag,
                score=float(round(score, 2)),
                reason=gated.get("reason", ""),
                gate_fired=gated.get("gate_fired", False),
            )

        # Adverse-exit escalation only (see module docstring for why the
        # weak-buy veto half of the shared gate is skipped for this strategy).
        if currently_in and _is_adverse_exit_context(regime.regime_signal, mom, self._gate_sensitivity):
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

    def __init__(
        self,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        trailing_stop: float | None = None,
        vol_stop_mult: float | None = None,
        vol_stop_window: int | None = None,
        profit_stop_scale: float | None = None,
        min_stop_pct: float | None = None,
        max_hold_days: int | None = None,
        exit_on_macd_cross: bool | None = None,
        exit_on_rsi_reversal: bool | None = None,
        exit_on_consolidation: bool | None = None,
        use_sar_stop: bool | None = None,
    ) -> None:
        self._stop = stop_loss_pct if stop_loss_pct is not None else self._stop
        self._target = take_profit_pct if take_profit_pct is not None else self._target
        self._impl = build_standard_exit_rules(
            defaults={
                "stop_loss_pct": self._stop,
                "trailing_stop": 0.0,
                "vol_stop_mult": 0.0,
                "vol_stop_window": 10,
                "profit_stop_scale": 0.0,
                "min_stop_pct": 0.02,
                "max_hold_days": 5,
                "exit_on_macd_cross": False,
                "exit_on_rsi_reversal": True,
                "exit_on_consolidation": False,
                "use_sar_stop": False,
            },
            trailing_stop=trailing_stop,
            vol_stop_mult=vol_stop_mult,
            vol_stop_window=vol_stop_window,
            profit_stop_scale=profit_stop_scale,
            min_stop_pct=min_stop_pct,
            max_hold_days=max_hold_days,
            exit_on_macd_cross=exit_on_macd_cross,
            exit_on_rsi_reversal=exit_on_rsi_reversal,
            exit_on_consolidation=exit_on_consolidation,
            use_sar_stop=use_sar_stop,
        )

    @property
    def stop_loss_pct(self) -> float:
        return self._stop

    @property
    def take_profit_pct(self) -> float:
        return self._target

    @property
    def exit_on_macd_cross(self) -> bool:
        return self._impl.exit_on_macd_cross

    @property
    def exit_on_rsi_reversal(self) -> bool:
        return self._impl.exit_on_rsi_reversal

    def check(self, trade: TradeState, bar: BarData) -> ExitResult:
        return self._impl.check(trade, bar)
