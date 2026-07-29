"""Optimised-new strategy — ratchet-only exit variant of "optimised", for comparison.

>>> UNVALIDATED — single-ticker sanity test only, not a live-capital candidate <<<
Same entry logic as `optimised` (identical weights/thresholds/vetoes — see
strategy/optimised.py), copied standalone rather than shared so each can
evolve independently. The only difference is the exit shape, testing a
YouTube-inspired idea: instead of a hard take-profit ceiling, let a
profit-ratcheting trailing stop be the sole determinant of when a winning
trade closes (the hard stop-loss remains as the initial risk floor).

What changed vs. `optimised`, and why
--------------------------------------
* `take_profit_pct` effectively disabled (999 — no realistic trade reaches
  it) instead of the hard 30% ceiling. Winners are no longer capped; the
  trailing stop alone decides when to lock in gains.
* `quality_gate_enabled` defaults to False instead of True. Investigation
  this session found the quality gate's adverse-exit escalation was 100% of
  `optimised`'s exits on the tickers tested — with the gate on, the ratchet
  stop never got a chance to bind before the gate fired first. Turning the
  gate off is required for the ratchet mechanism to actually do anything.
* `profit_stop_scale` raised 0.5 -> ... no, kept 0.5's *shape* but the base
  mechanism (vol_stop_mult=2.0, unchanged from `optimised`) is what actually
  determines the trailing distance — `profit_stop_scale=0.30` narrows it as
  the trade becomes profitable. (A `trailing_stop=0.15` fixed-distance
  override was tried in the same test session but is a no-op whenever
  `vol_stop_mult>0`, per `core/exits.py::_effective_stop_for_bar` — the
  vol-scaled branch always wins when active. Not carried into this file to
  avoid a dead constant.)
* `min_stop_pct` tightened 0.04 -> 0.03 (the ratchet floor — never allow the
  trailing distance to narrow past this even at very high profit).
* `stop_loss_pct` (0.08), `vol_stop_mult` (2.0), `vol_stop_window` (20),
  `max_hold_days` (0) all unchanged from `optimised` — only the take-profit
  ceiling, gate, profit-stop-scale and min-stop floor differ.

Single-ticker result so far: AAPL, 2023-08 to 2026-07, hourly bars —
Sharpe 1.96 vs `optimised`'s 1.21 baseline on the same window, but only 28
trades vs 50 (fewer, larger-ratchet-managed trades). Not tested across a
universe, not tested for the tighter gate-driven whipsaw protection that
`optimised` relies on being permanently off — treat as an open comparison,
not a validated improvement (see choppy_vol.py for what a decisively-tested
regression looks like; this one just hasn't been tested broadly yet either
way).

Entry
-----
Identical to `optimised`: HMM (2.0) + RSI (1.0) + SMA200 (3.0) + trend
SMA20/50 (2.0) + volume (1.0). Buy threshold 6.0, sell -4.5. RSI>70 and
regime_signal<=0 entry vetoes. Quality gate defaults OFF (see above) but is
still overridable back on via --plugin-gate quality for further comparison.

Exit
----
Hard stop-loss 8% (unchanged floor). No hard take-profit (999, effectively
off). Vol-scaled trailing stop (vol_stop_mult=2.0, vol_stop_window=20),
profit-stop tightening (profit_stop_scale=0.30, floor 3%). No max hold.

All values above are this strategy's own defaults, not fixed — every one is
overridable via the matching CLI flag regardless of --strategy selected.
"""

from __future__ import annotations

from ..core.momentum import composite_signal
from ..core.quality_gate import _apply_quality_gate
from ..plugins.types import BarData, EntryDecision, ExitResult, RegimeState, TradeState
from .base.exit_overrides import build_standard_exit_rules

#: Entry vetoes, identical to optimised.py.
_RSI_OVERBOUGHT = 70.0
_MIN_REGIME_SIGNAL = 0.0


class OptimisedNewEntry:
    """Trend-style entry, identical to OptimisedEntry except the quality
    gate defaults off (see module docstring for why).

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
    #: Defaults OFF here (unlike optimised.py) — the gate's adverse-exit
    #: escalation dominated every exit in testing, leaving the ratchet stop
    #: below no chance to ever bind.
    quality_gate_enabled: bool = False
    gate_sensitivity: int = 2

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        buy_threshold: float | None = None,
        sell_threshold: float | None = None,
        vol_filter_ok: bool = True,
        quality_gate_enabled: bool | None = None,
        gate_sensitivity: int | None = None,
    ) -> None:
        self._weights = {**self.weights, **(weights or {})}
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
        """Score a bar, then veto overbought / bear-regime NEW entries."""
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
                reason=f"optimised_new veto: RSI > {_RSI_OVERBOUGHT:.0f} (overbought entries lose)",
            )
        if regime.regime_signal is not None and regime.regime_signal <= _MIN_REGIME_SIGNAL:
            return EntryDecision(
                flag="HOLD", raw_flag=decision.raw_flag, score=decision.score,
                reason="optimised_new veto: regime_signal <= 0 (no bull-regime confirmation)",
            )
        return decision


class OptimisedNewExit:
    """Hard stop-loss floor (8%) only — no hard take-profit. Winners are
    closed exclusively by the vol-scaled, profit-narrowing trailing stop.

    Satisfies ExitStrategyProtocol.
    """

    _stop: float = 0.08
    _target: float = 999.0  # effectively disabled — see module docstring
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
                "trailing_stop": 0.0,        # inert while vol_stop_mult>0 — see docstring
                "vol_stop_mult": 2.0,        # 2 × realised-vol trailing stop (unchanged)
                "vol_stop_window": 20,
                "profit_stop_scale": 0.30,   # tighten trail as profit grows (was 0.5)
                "min_stop_pct": 0.03,        # floor: never tighter than 3% (was 0.04)
                "max_hold_days": 0,
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
            exit_on_macd_cross=exit_on_macd_cross,
            exit_on_rsi_reversal=exit_on_rsi_reversal,
            exit_on_consolidation=exit_on_consolidation,
            use_sar_stop=use_sar_stop,
        )

    @property
    def stop_loss_pct(self) -> float:
        """Hard stop-loss fraction (8%)."""
        return self._stop

    @property
    def take_profit_pct(self) -> float:
        """Effectively-disabled hard take-profit (999)."""
        return self._target

    @property
    def exit_on_macd_cross(self) -> bool:
        return self._impl.exit_on_macd_cross

    @property
    def exit_on_rsi_reversal(self) -> bool:
        return self._impl.exit_on_rsi_reversal

    def check(self, trade: TradeState, bar_data: BarData) -> ExitResult:
        """Evaluate exit conditions for the current bar."""
        return self._impl.check(trade, bar_data)
