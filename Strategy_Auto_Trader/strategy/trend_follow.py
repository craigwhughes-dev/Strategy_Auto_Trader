"""Trend-following strategy — ride confirmed uptrends with wide stops.

What this strategy is trying to do
------------------------------------
This strategy is built on a single core belief: when a stock is in a confirmed
uptrend (above SMA20, SMA50, and SMA200 simultaneously), the best approach is
to get in and stay in, tolerating normal pullbacks rather than being stopped out
by noise. It is designed to capture the bulk of large trend moves, not to time
perfect entries and exits.

The entry signal is dominated by structural trend alignment: SMA200 (weight 3.0)
and the SMA20+SMA50 trend pair (weight 2.0) together account for 62% of the
maximum possible vote. This means the strategy simply will not enter unless the
price is clearly above all major moving averages. The HMM regime adds confirmation
(1.5) but RSI is de-weighted (1.0) because overbought RSI is normal in a strong
trend — you do not want to be excluded from the best-performing part of a move
because RSI hit 75. Volume is also de-weighted (0.5) for the same reason: trending
stocks often have irregular volume patterns.

Exit is deliberately wide: the hard stop is 8% (to survive normal trend
pullbacks) and the take-profit is 30% (to let winners run). A vol-scaled trailing
stop (2 × realised volatility × sqrt of window) activates from bar 1 and narrows
as unrealised profit grows (profit_stop_scale = 0.5), so the trail tightens
progressively as the position matures — locking in gains without exiting early.
The minimum trail floor is 4% to prevent over-tightening in low-vol conditions.

Best suited to: growth stocks and ETFs in strong secular uptrends (e.g. NVDA,
QQQ). Will take larger losses on individual failing trades than the default or
conservative strategies, but the expectation is that these are offset by fewer
exits from winning trades.

Entry
-----
Weighted vote: HMM (1.5) + RSI (1.0) + SMA200 (3.0) + trend SMA20/50 (2.0)
+ volume (0.5).  Buy threshold 4.5, sell threshold -4.0.
Quality gate (quality_gate_enabled=True) applies but trend-heavy weights
mean SMA200/trend dominate.

Exit
----
Wide hard stop 8%, wide take-profit 30%.
Vol-scaled trailing stop (vol_stop_mult=2.0, vol_stop_window=20).
Profit-stop tightening (profit_stop_scale=0.5), floor 4%.
No max hold limit — let the trend determine exit timing.
Kelly position sizing on (use_kelly=True, kelly_lookback=20).

All entry and exit values above are this strategy's own defaults, not
fixed — every one is overridable via the matching CLI flag (e.g.
--trailing-stop, --profit-stop-scale, --plugin-gate) regardless of which
--strategy is selected; an omitted flag leaves this strategy's default
untouched.
"""

from __future__ import annotations

from ..core.momentum import composite_signal
from ..core.quality_gate import _apply_quality_gate
from ..plugins.types import BarData, EntryDecision, ExitResult, RegimeState, TradeState
from .base.exit_overrides import build_standard_exit_rules


class TrendEntry:
    """SMA-trend-led entry: above all major MAs with regime confirmation.

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
    #: Number of weak-context/adverse-exit conditions (of 5) needed to fire
    #: the gate. Overridable at construction (e.g. CLI --gate-sensitivity).
    gate_sensitivity: int = 2
    require_flip_entry: bool = True
    require_vol_filter_ok: bool = True
    volume_min_ratio: float = 1.0

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
        self._vol_filter_ok = vol_filter_ok if self.require_vol_filter_ok else True
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
        if self._quality_gate_enabled:
            gated = _apply_quality_gate(
                raw, mom, regime.regime_signal, currently_in=currently_in,
                gate_sensitivity=self._gate_sensitivity,
            )
        else:
            gated = dict(raw, reason="", gate_fired=False)
        return EntryDecision(
            flag=gated["flag"],
            raw_flag=raw["flag"],
            score=float(raw.get("score", 0.0)),
            reason=gated.get("reason", ""),
            gate_fired=gated.get("gate_fired", False),
        )


class TrendExit:
    """Wide stop (8%) and target (30%) with vol-scaled trailing stop.

    The vol-scaled trailing stop (vol_stop_mult=2.0) tracks realised volatility
    and narrows as the trade matures (profit_stop_scale=0.5), keeping the
    strategy in the trend while protecting most of the accumulated gain.

    Satisfies ExitStrategyProtocol.
    """

    _stop: float = 0.08
    _target: float = 0.30
    use_kelly: bool = True
    kelly_lookback: int = 20
    min_hold_bars: int = 48

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
                "trailing_stop": 0.0,        # use vol-stop rather than fixed trail
                "vol_stop_mult": 2.0,        # 2 × realised-vol trailing stop
                "vol_stop_window": 20,
                "profit_stop_scale": 0.5,    # tighten trail as profit grows
                "min_stop_pct": 0.04,        # floor: never tighter than 4%
                "max_hold_days": 0,          # no forced exit — let the trend run
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
        """Hard take-profit fraction (30%)."""
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
