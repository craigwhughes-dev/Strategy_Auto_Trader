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
"""

from __future__ import annotations

from ..plugins.exit_rules import StandardExitRules
from ..plugins.types import BarData, EntryDecision, ExitResult, RegimeState, TradeState


class MeanReversionEntry:
    """Entry that buys RSI overshoots against the short-term mean.

    Satisfies EntryStrategyProtocol.
    """

    buy_threshold: float = 2.5
    sell_threshold: float = -2.5

    def __init__(self, vol_filter_ok: bool = True) -> None:
        # mean-reversion prefers choppy tickers; if vol_filter_ok is True
        # it means the environment looks trendy — veto entries in that case.
        self._vol_filter_ok = vol_filter_ok

    def evaluate(
        self,
        _regime: RegimeState,
        mom: dict,
        _volume_ratio: float,
        _currently_in: bool = False,
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
        # Price distance from SMA20: deeper below -> larger score
        score += max(0.0, -pct_sma20 * 10.0)
        # Volume boost
        if vol_ratio > 1.2:
            score += 0.8

        # Map to BUY/HOLD/SELL similar to composite_signal thresholds
        raw_flag = "BUY" if score >= self.buy_threshold else ("SELL" if score <= self.sell_threshold else "HOLD")

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
