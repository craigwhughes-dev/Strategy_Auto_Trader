"""Exit-rules plugin: bundles _effective_stop_for_bar + _check_exit_conditions.

StandardExitRules is the default implementation.  It is initialized with all
stop/exit parameters at construction time, leaving the per-bar check()
method signature minimal.
"""

from __future__ import annotations

from ..core.exits import _check_exit_conditions, _effective_stop_for_bar
from .types import BarData, ExitResult, TradeState


class StandardExitRules:
    """Default exit rules: wraps core/exits functions.

    Satisfies ExitRulesProtocol.
    """

    def __init__(
        self,
        *,
        stop_loss_pct: float,
        trailing_stop: float = 0.0,
        vol_stop_mult: float = 0.0,
        vol_stop_window: int = 20,
        profit_stop_scale: float = 0.0,
        min_stop_pct: float = 0.05,
        max_hold_days: int = 0,
        exit_on_macd_cross: bool = False,
        exit_on_rsi_reversal: bool = False,
        exit_on_consolidation: bool = False,
        use_sar_stop: bool = False,
    ) -> None:
        self._stop_loss_pct = stop_loss_pct
        self._trailing_stop = trailing_stop
        self._vol_stop_mult = vol_stop_mult
        self._vol_stop_window = vol_stop_window
        self._profit_stop_scale = profit_stop_scale
        self._min_stop_pct = min_stop_pct
        self._max_hold_days = max_hold_days
        self._exit_on_macd_cross = exit_on_macd_cross
        self._exit_on_rsi_reversal = exit_on_rsi_reversal
        self._exit_on_consolidation = exit_on_consolidation
        self._use_sar_stop = use_sar_stop

    def check(self, trade: TradeState, bar: BarData) -> ExitResult:
        """Evaluate all exit conditions for one bar in priority order."""
        eff_stop = _effective_stop_for_bar(
            self._vol_stop_mult,
            bar.daily_vol_t,
            self._vol_stop_window,
            self._trailing_stop,
            trade.position,
            self._profit_stop_scale,
            trade.entry_price,
            bar.cur_close,
            self._min_stop_pct,
        )
        exit_hit, sell_reason, peak, days = _check_exit_conditions(
            trade.position,
            bar.cur_close,
            trade.entry_price,
            rr_ratio=1.0,
            rr_risk=self._stop_loss_pct,
            rr_stop_level=trade.stop_level,
            rr_target_level=trade.target_level,
            effective_stop=eff_stop,
            peak_price_since_entry=trade.peak_price_since_entry,
            max_hold_days=self._max_hold_days,
            days_in_trade=trade.days_in_trade,
            use_sar_stop=bar.use_sar_stop,
            sar_val=bar.sar_val,
            need_exit=bar.need_exit,
            macd_bearish_cross=bar.macd_bc,
            exit_on_macd_cross=self._exit_on_macd_cross,
            exit_on_rsi_reversal=self._exit_on_rsi_reversal,
            rsi_overbought_exit=bar.rsi_ob,
            rsi_momentum_loss=bar.rsi_ml,
            exit_on_consolidation=self._exit_on_consolidation,
            consolidating=bar.consol,
        )
        return ExitResult(
            exit_hit=exit_hit,
            sell_reason=sell_reason,
            peak_price_since_entry=peak,
            days_in_trade=days,
        )
