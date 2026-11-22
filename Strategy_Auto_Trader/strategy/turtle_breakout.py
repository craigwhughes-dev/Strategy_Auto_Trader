"""Turtle breakout — long-only 20-day-high / 10-day-low price-action breakout.

What this strategy is trying to do
------------------------------------
Pure price-action trend-following, adapted from the classic 1980s Turtle
system (long-only, no pyramiding). No HMM, no RSI, no composite vote — one
rule: buy when price breaks above the highest high of the last N days; exit
at whichever of two stops is nearer: a fixed ATR stop set at entry, or a
trailing stop at the lowest low of the last M days.

Approximations versus the source (daily-bar) system
-----------------------------------------------------
This engine runs hourly bars only. entry_lookback_days / exit_lookback_days /
atr_period_days are declared here in *trading days* and converted to an
hourly-bar count by the engine (consolidated_engine.py's
_estimate_bars_per_day), using the bars-per-day implied by the ticker's own
data — so it adapts to whatever session length the exchange/data feed gives
it (e.g. LSE vs NYSE hours) rather than hardcoding one. This is an
approximation of a true daily-bar replay, not an exact match.

Entry/exit also compare against the bar's *close*, not its high/low — same
convention as every stop/target check elsewhere in this engine (close-based).
The video and its companion TradingView indicator compare against the bar's
high; an intrabar breakout that reverses before the close is therefore not
caught here.

Entry
-----
BUY when close crosses above the rolling highest high of the last
entry_lookback_days (20 trading days), excluding the current bar.
require_flip_entry (default True, inherited) means only the crossing bar
triggers, not every bar price happens to stay above the band. No volume
filter (volume_min_ratio=0.0) — the source strategy has none.
vol_filter_ok still vetoes to HOLD on choppy/mean-reverting tickers, the
same convention every other trend-following strategy in this registry uses.

Exit
----
No take-profit (there isn't one in the source rule). Two stops, exit at
whichever is nearer (the higher of the two, for a long):
1. Fixed: entry_price - atr_mult (2.0) * Wilder ATR(atr_period_days=20),
   set once at entry and never recalculated.
2. Trailing: rolling lowest low of the last exit_lookback_days (10 trading
   days), recalculated every bar.
Inherits the engine's Kelly position sizing (no override).

Best suited to: assets with real multi-week trend episodes — the strategy's
own thesis is that most breakouts fail small and a few runners pay for the
losses. Will whipsaw in sideways/choppy markets; vol_filter_ok is the main
defence against that, same as every other trend strategy here.

Known weaknesses: day-count approximation of a daily-bar system run on
hourly bars (see above) — results will not exactly match a true daily
backtest of the same rule.
"""

from __future__ import annotations

from ..plugins.types import BarData, EntryDecision, ExitResult, RegimeState, TradeState


class TurtleEntry:
    """Long-only N-day-high breakout entry. No HMM/RSI/volume vote.

    Satisfies EntryStrategyProtocol.
    """

    entry_lookback_days: int = 20
    volume_min_ratio: float = 0.0
    require_flip_entry: bool = True
    #: Tells consolidated_engine.py to precompute the rolling high/low/ATR
    #: bands (see _needs_breakout_levels in consolidated_backtest).
    needs_breakout_levels: bool = True
    #: All weights zero so skip_unused_indicators skips HMM refit + RSI
    #: compute for this strategy — it uses neither.
    _weights: dict[str, float] = {
        "markov": 0.0, "rsi": 0.0, "trend": 0.0,
        "sma200": 0.0, "volume": 0.0, "hmm": 0.0,
    }

    def __init__(self, vol_filter_ok: bool = True, entry_lookback_days: int | None = None) -> None:
        self._vol_filter_ok = vol_filter_ok
        self.entry_lookback_days = (
            entry_lookback_days if entry_lookback_days is not None else self.entry_lookback_days
        )

    def evaluate(
        self,
        regime: RegimeState,
        mom: dict,
        _volume_ratio: float,
        currently_in: bool = False,
    ) -> EntryDecision:
        if not self._vol_filter_ok:
            return EntryDecision(
                flag="HOLD", raw_flag="HOLD", score=0.0,
                reason="vol_filter: unsuitable (choppy/mean-reverting)",
            )
        level = mom.get("breakout_entry_level")
        cur_close = mom.get("cur_close")
        if level is None or cur_close is None:
            return EntryDecision(
                flag="HOLD", raw_flag="HOLD", score=0.0,
                reason="insufficient history for breakout band",
            )
        breakout = cur_close > level
        flag = "BUY" if breakout else "HOLD"
        return EntryDecision(
            flag=flag, raw_flag=flag, score=1.0 if breakout else 0.0,
            reason="breakout" if breakout else "",
        )


class TurtleExit:
    """Fixed ATR stop + trailing M-day-low stop. No take-profit, no signal exit.

    Satisfies ExitStrategyProtocol.
    """

    exit_lookback_days: int = 10
    atr_period_days: int = 20
    atr_mult: float = 2.0
    min_hold_bars: int = 0
    needs_breakout_levels: bool = True

    def __init__(
        self,
        exit_lookback_days: int | None = None,
        atr_period_days: int | None = None,
        atr_mult: float | None = None,
    ) -> None:
        self.exit_lookback_days = (
            exit_lookback_days if exit_lookback_days is not None else self.exit_lookback_days
        )
        self.atr_period_days = (
            atr_period_days if atr_period_days is not None else self.atr_period_days
        )
        self._atr_mult = atr_mult if atr_mult is not None else self.atr_mult

    @property
    def stop_loss_pct(self) -> float:
        # Placeholder only — entry_levels() below sets real price levels, this
        # property exists so code that reads it generically (e.g. the context
        # adjuster's base-stop input) doesn't crash.
        return 0.0

    @property
    def take_profit_pct(self) -> float:
        return float("inf")

    def entry_levels(self, entry_price: float, atr_at_entry: float | None) -> tuple[float, float]:
        """Called by the engine at BUY time instead of the pct-based formula.

        Returns (stop_level, target_level) in price terms. No take-profit,
        so target_level is +inf (never triggers). Falls back to a flat 5%
        stop if ATR isn't warmed up yet (shouldn't happen in practice since
        TurtleEntry's own breakout band needs a similar warmup window).
        """
        if atr_at_entry is None or atr_at_entry <= 0:
            return entry_price * (1 - 0.05), float("inf")
        return entry_price - self._atr_mult * atr_at_entry, float("inf")

    def check(self, trade: TradeState, bar: BarData) -> ExitResult:
        fixed_stop = trade.stop_level
        trailing_stop = bar.breakout_exit_level
        if trailing_stop is not None and trailing_stop >= fixed_stop:
            eff_stop = trailing_stop
            reason = "turtle_trailing_stop"
        else:
            eff_stop = fixed_stop
            reason = "turtle_fixed_stop"
        exit_hit = bar.cur_close <= eff_stop
        return ExitResult(
            exit_hit=exit_hit,
            sell_reason=reason if exit_hit else "",
            peak_price_since_entry=trade.peak_price_since_entry,
            days_in_trade=trade.days_in_trade + 1,
        )
