"""Active strategy — higher-frequency variant of optimised_new targeting
~5-10 closed trades/week across a top-70 portfolio.

Problem solved
--------------
optimised_new makes almost zero trades in some months because gates stack:

  1. vix_entry_gate_threshold=20.0 — blocks all entries when VIX >= 20.
     The 2022 bear market had VIX 20-40 nearly the full year. Every tariff
     shock, carry-unwind, or macro panic triggers it. Validated -0.15 Sharpe
     cost in normal conditions; eliminates whole calendar months in crises.
  2. min_entry_score=7.0 with max=8.0 — 87.5% of max required. Rare outside
     strong bull legs where all five signals align simultaneously.
  3. regime_signal <= 0 veto — blocks entries in neutral HMM regimes, even
     when composite signal is bullish.
  4. max_hold_days=0 (no limit) — positions that drift sideways tie up capital
     indefinitely, reducing how many new entries can be admitted.

Previous redesign attempt (2026-09-13): buy_threshold=4.0 + require_flip_entry=
False produced 4,850 trades and -59% P&L. Root cause: at threshold=4.0, signals
fire on nearly every bar; without the flip guard, the strategy re-enters
immediately after every close and 27% of candidates had kelly<=0 (negative
historical edge). Lesson: require_flip_entry=True is critical at any threshold
below ~5.5; the quality improvement gates need to be preserved.

This version targets 2-4x more trades vs optimised_new by removing the four
blocking gates above while keeping the proven signal chain intact.

What this strategy is trying to do
------------------------------------
Same entry logic as optimised_new (HMM + RSI + trend + SMA200 + volume) but
with the over-restrictive portfolio-level blocks removed. Entries fire on
strong-enough composite signals (threshold=5.5, slightly lower than 6.0) in any
VIX environment. Positions are forced to close within 14 calendar days rather
than drifting indefinitely, which recycles capital and allows more entries.

Entry
-----
Weights identical to optimised_new:
  HMM (2.0) + RSI (1.0) + trend SMA20/50 (1.0) + SMA200 (3.0) + volume (1.0).
Max score = 8.0.  Buy threshold 5.5 (vs optimised_new's 6.0).

Gates removed vs optimised_new:
  - vix_entry_gate_threshold: None (no VIX block)
  - min_entry_score: None (no extra score gate above buy_threshold)
  - regime_signal <= 0 veto: removed

Gates preserved:
  - require_flip_entry=True (critical: prevents re-entry floods at low thresholds)
  - RSI > 70 overbought veto (avoids buying tops)
  - require_vol_filter_ok=True (trend-quality tickers only)
  - quality_gate_enabled=False (same as optimised_new — gate dominates exits)

Exit
----
Hard stop-loss 8% (slightly tighter than optimised_new's 10%).
Hard take-profit 20% (forces closure; no trailing stop for clean R:R).
max_hold_days=14 — recycles capital after ~10 trading days.
R:R = 2.5:1. Break-even above 29% win rate.
Kelly position sizing on (use_kelly=True, kelly_lookback=20).

Best suited to: same tickers as optimised_new (top-70, trend-quality universe)
but in any VIX environment. Accepts more entries at the cost of slightly looser
signal quality; the flip guard and overbought veto prevent entry floods.

Known weaknesses: no VIX gate means entries in 2022-style sustained bear
markets. Hard take-profit at 20% caps upside on strong winners (optimised_new
would have let those run via ratchet stop). Do not combine in the same pot as
optimised_new — the strategies compete for the same signals.

All values are this strategy's own defaults — overridable via CLI flags.
"""

from __future__ import annotations

from ..core.momentum import composite_signal
from ..core.quality_gate import _apply_quality_gate
from ..plugins.types import BarData, EntryDecision, ExitResult, RegimeState, TradeState
from .base.exit_overrides import build_standard_exit_rules

_RSI_OVERBOUGHT = 70.0
_MIN_REGIME_SIGNAL = 0.0


class ActiveEntry:
    """optimised_new signal chain with VIX/score/regime-signal gates removed.

    Satisfies EntryStrategyProtocol.
    """

    weights: dict[str, float] = {
        "markov": 0.0,
        "rsi":    1.0,
        "trend":  1.0,
        "sma200": 3.0,
        "volume": 1.0,
        "hmm":    2.0,
    }
    buy_threshold: float = 5.5
    sell_threshold: float = -5.5
    quality_gate_enabled: bool = False
    gate_sensitivity: int = 2
    require_flip_entry: bool = True     # preserved — critical at threshold < 6.0
    require_vol_filter_ok: bool = True
    volume_min_ratio: float = 1.0
    same_day_deployment_cap_pct: float | None = None
    vix_entry_gate_threshold: float | None = None   # removed
    skip_overnight_vol_screen: bool = True
    min_entry_score: float | None = None            # removed

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        buy_threshold: float | None = None,
        sell_threshold: float | None = None,
        vol_filter_ok: bool = True,
        quality_gate_enabled: bool | None = None,
        gate_sensitivity: int | None = None,
        min_entry_score: float | None = None,
    ) -> None:
        self._weights = {**self.weights, **(weights or {})}
        self._buy_t = buy_threshold if buy_threshold is not None else self.buy_threshold
        self._sell_t = sell_threshold if sell_threshold is not None else self.sell_threshold
        self._vol_filter_ok = vol_filter_ok if self.require_vol_filter_ok else True
        self._quality_gate_enabled = (
            quality_gate_enabled if quality_gate_enabled is not None else self.quality_gate_enabled
        )
        self._gate_sensitivity = (
            gate_sensitivity if gate_sensitivity is not None else self.gate_sensitivity
        )
        self._min_entry_score: float | None = (
            min_entry_score if min_entry_score is not None else self.min_entry_score
        )

    def evaluate(
        self,
        regime: RegimeState,
        mom: dict,
        _volume_ratio: float,
        currently_in: bool = False,
    ) -> EntryDecision:
        """Score bar; veto overbought only — no VIX gate, no regime_signal veto."""
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
        if self._quality_gate_enabled:
            gated = _apply_quality_gate(
                raw, mom, regime.regime_signal, currently_in=currently_in,
                gate_sensitivity=self._gate_sensitivity,
            )
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
                reason=f"active veto: RSI > {_RSI_OVERBOUGHT:.0f} (overbought entries lose)",
            )
        if self._min_entry_score is not None and decision.score < self._min_entry_score:
            return EntryDecision(
                flag="HOLD", raw_flag=decision.raw_flag, score=decision.score,
                reason=f"active veto: score {decision.score:.1f} < min_entry_score {self._min_entry_score:.1f}",
            )
        return decision


class ActiveExit:
    """Hard 8% stop-loss, 20% take-profit, 14-day maximum hold.

    Hard TP replaces optimised_new's ratchet trailing stop — cleaner R:R
    (2.5:1) and forces turnover. max_hold_days=14 recycles capital from
    drifting positions.

    Satisfies ExitStrategyProtocol.
    """

    _stop: float = 0.08
    _target: float = 0.20
    use_kelly: bool = True
    kelly_lookback: int = 20
    min_hold_bars: int = 0
    min_hold_bars_regime_exit: int | None = 6

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
        breakeven_trailing: bool | None = None,
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
                "vol_stop_window": 20,
                "profit_stop_scale": 0.0,
                "min_stop_pct": 0.08,
                "max_hold_days": 14,
                "breakeven_trailing": False,
                "exit_on_macd_cross": False,
                "exit_on_rsi_reversal": False,
                "exit_on_consolidation": False,
                "use_sar_stop": False,
            },
            trailing_stop=trailing_stop,
            vol_stop_mult=vol_stop_mult,
            vol_stop_window=vol_stop_window,
            profit_stop_scale=profit_stop_scale,
            min_stop_pct=min_stop_pct,
            max_hold_days=max_hold_days,
            breakeven_trailing=breakeven_trailing,
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

    def check(self, trade: TradeState, bar_data: BarData) -> ExitResult:
        return self._impl.check(trade, bar_data)
