"""Choppy-vol strategy — mean reversion for tickers the trend strategies veto.

>>> NOT VALIDATED — DO NOT USE FOR LIVE CAPITAL <<<
A signal-level (vectorized) study of this exact entry/exit rule showed a
positive profit factor (see below), but running it through the real
event-driven consolidated_backtest engine across the full 204-ticker choppy
cohort (reports/full_scan/summary.csv, efficiency_ratio<0.05, ann_vol<0.35)
was decisively negative: 199/203 tickers lost money, -$38,327 aggregate P&L
across 1,049 trades ($10k-$20k accounts, mean -$189/ticker). The signal-level
study never modelled a stop-loss interrupting a trade before RSI could
revert — chop names getting MORE oversold before reverting ("catching a
falling knife") means the -4% stop fires on many trades the vectorized study
counted as eventual winners. Kept in the registry as a documented,
tested-but-losing starting point, not a recommendation. See HANDOFF.md /
project memory for the full validation trail before iterating further.

What this strategy is trying to do
------------------------------------
`resolve_strategy()` currently vetoes any ticker with low `trend_quality`
(choppy/mean-reverting character) to permanent HOLD for every trend-following
strategy (default/conservative/trend/optimised) — see
`strategy/base/registry.py`. That veto is correct for those strategies (the
HMM/trend vote whipsaws in chop), but it leaves choppy tickers with no
strategy at all. This one is meant to be used *instead of* the veto for those
names, not layered on top of it — it deliberately does not check
`trend_quality` or `vol_filter_ok` itself.

Derived from a vectorized study (not the event-driven backtest engine) of
`reports/full_scan/hourly/*.csv` for the 25 tickers with the worst
`sharpe_strategy` under the existing "default" strategy, restricted to
genuinely choppy names (`efficiency_ratio < 0.05`, `ann_vol < 0.35` — this
excludes hyper-trending high-vol growth stocks like SMCI/COIN that the
`trend_quality` formula mislabels "choppy" but which actually trended and
made money under the trend strategies).

Four entry candidates were tested (RSI oversold, Bollinger %b, both combined,
RSI+consolidation), each against fixed-horizon forward returns (5/10/20/40
bars) and an RSI-mean-reversion exit (hold until RSI recovers to >=50, capped
at 40 bars). Findings:
  * Fixed 5/10-bar horizons were noise for every candidate (hit rate ~50-52%,
    profit factor ~1.03-1.07) — not worth trading alone.
  * The RSI-reversion exit beat every fixed horizon on every candidate, and
    was also faster (16-21 bars average vs. a 40-bar wait).
  * `RSI < 35 AND consolidation` (Bollinger-squeeze + low-ATR) was the
    strongest candidate: 449 signal events (well diversified, no ticker over
    ~14% of events), profit factor 1.44 on the RSI-reversion exit, only 8.7%
    of trades never reverted within 40 bars.
  * `RSI < 30 AND bb_pctb < 0.05` (price at/below the lower Bollinger band)
    was the runner-up: more trade frequency (1,620 events) at a slightly
    lower profit factor (1.38), but was NOT tested combined with the winning
    candidate above — stacking untested conditions together would trade a
    stricter rule than anything actually measured, so this strategy uses the
    winning candidate exactly as tested rather than layering the two.

Entry
-----
BUY when (not already in a position): `cur_rsi < 35` AND `consolidation` is
True (Bollinger-width squeeze + ATR below its own 20-bar average) — the
exact winning candidate from the study, nothing added on top. No HMM/SMA200/
volume vote — those are trend signals and this is deliberately not a trend
strategy.
No entry vetoes; `vol_filter_ok` is accepted for constructor-signature
compatibility with the registry but ignored (see module docstring — this
strategy IS the alternative to the veto).
SELL (while in a position) when `cur_rsi >= 50` — the reversion this
strategy is trading for.

Exit
----
Primary exit is the RSI reversion itself: `cur_rsi >= 50`. This is now
checked via the entry strategy's currently_in=True SELL signal path (in
ChoppyVolEntry.evaluate()), which fires immediately since this strategy's
ChoppyVolExit.min_hold_bars=0 overrides the engine's global default (no
hold gate needed). Backstopped by a tight hard stop -4% (chop ranges are
narrow; don't wait around for a big loss), modest take-profit +6%, and a
40-bar max hold (matches the study's reversion-search cap) in case RSI
never recovers, enforced via ChoppyVolExit.check(). No trailing stop — the
point is a quick in/out, not letting a winner run.
"""

from __future__ import annotations

from ..plugins.exit_rules import StandardExitRules
from ..plugins.types import BarData, EntryDecision, ExitResult, RegimeState, TradeState

_RSI_OVERSOLD = 35.0
_RSI_REVERSION = 50.0


class ChoppyVolEntry:
    """Mean-reversion entry: RSI oversold + Bollinger squeeze + lower-band touch.

    Satisfies EntryStrategyProtocol. Ignores vol_filter_ok by design — this
    strategy exists specifically to trade tickers the trend strategies veto
    for being choppy, so re-applying that veto here would defeat the point.
    """

    #: Not used for scoring (this strategy bypasses composite_signal
    #: entirely) — declared only so consolidated_backtest's skip_unused_
    #: indicators optimisation knows the HMM regime model isn't needed here.
    _weights: dict[str, float] = {"hmm": 0.0, "rsi": 1.0}

    def __init__(self, vol_filter_ok: bool = True) -> None:
        """vol_filter_ok is accepted only for registry constructor-signature
        compatibility (see class docstring) and otherwise unused."""

    def evaluate(
        self,
        _regime: RegimeState,
        mom: dict,
        _volume_ratio: float,
        currently_in: bool = False,
    ) -> EntryDecision:
        """Score a bar. While in a position, check for RSI reversion exit
        (cur_rsi >= 50). If reversion conditions met, return SELL; otherwise
        hold. The RSI-reversion exit fires via this path (min_hold_bars=0
        allows it immediately). Stop-loss/take-profit/max-hold backstops
        are enforced in ChoppyVolExit.check()."""
        if currently_in:
            cur_rsi = mom.get("cur_rsi")
            if cur_rsi is not None and cur_rsi >= _RSI_REVERSION:
                return EntryDecision(
                    flag="SELL", raw_flag="SELL", score=1.0,
                    reason="choppy_vol: RSI reversion (RSI>=50)",
                )
            return EntryDecision(flag="HOLD", raw_flag="HOLD", score=0.0, reason="")

        cur_rsi = mom.get("cur_rsi")
        consolidating = mom.get("consolidation", False)

        if cur_rsi is None:
            return EntryDecision(flag="HOLD", raw_flag="HOLD", score=0.0,
                                 reason="choppy_vol: insufficient data")

        if cur_rsi < _RSI_OVERSOLD and consolidating:
            return EntryDecision(
                flag="BUY", raw_flag="BUY", score=1.0,
                reason="choppy_vol: oversold + consolidation",
            )
        return EntryDecision(flag="HOLD", raw_flag="HOLD", score=0.0, reason="")


class ChoppyVolExit:
    """Tight stop/target/max-hold backstop.

    RSI-reversion exit has been moved to ChoppyVolEntry.evaluate()'s
    currently_in=True path, which fires immediately (min_hold_bars=0).
    This class now handles only the hard stop (-4%), hard take-profit (+6%),
    and max-hold-bars (40) backstop — the safety bounds for the trade.

    Satisfies ExitStrategyProtocol.
    """

    _stop: float = 0.04
    _target: float = 0.06
    _max_hold_bars: int = 40
    min_hold_bars: int = 0  # Allow signal-based SELL immediately (no hold gate)

    def __init__(self) -> None:
        self._impl = StandardExitRules(
            stop_loss_pct=self._stop,
            trailing_stop=0.0,
            vol_stop_mult=0.0,
            vol_stop_window=20,
            profit_stop_scale=0.0,
            min_stop_pct=self._stop,
            max_hold_days=self._max_hold_bars,
            exit_on_macd_cross=False,
            exit_on_rsi_reversal=False,
            exit_on_consolidation=False,
            use_sar_stop=False,
        )

    @property
    def stop_loss_pct(self) -> float:
        """Hard stop-loss fraction (4%)."""
        return self._stop

    @property
    def take_profit_pct(self) -> float:
        """Hard take-profit fraction (6%)."""
        return self._target

    def check(self, trade: TradeState, bar_data: BarData) -> ExitResult:
        """Delegate to the stop/target/max-hold backstop. RSI reversion exit
        is now handled by ChoppyVolEntry.evaluate()'s currently_in=True path."""
        return self._impl.check(trade, bar_data)
