"""Bar-frequency-agnostic exit logic shared by the daily and hourly engines.

Both functions operate purely on prices, a position flag, and precomputed
indicator booleans/floats for the current bar — they don't care whether a
bar is a day or an hour, so the same logic applies to both engines.
"""

from __future__ import annotations

import numpy as np


def _effective_stop_for_bar(
    vol_stop_mult: float,
    daily_vol_t: float,
    vol_stop_window: int,
    trailing_stop: float,
    current_position: float,
    profit_stop_scale: float,
    entry_price: float,
    cur_close: float,
    min_stop_pct: float,
) -> float:
    """Today's trailing-stop fraction: vol-scaled (falls back to fixed if vol is
    unavailable), then narrowed by unrealised profit if profit_stop_scale is set."""
    if vol_stop_mult > 0.0:
        if np.isfinite(daily_vol_t) and daily_vol_t > 0:
            effective_stop = vol_stop_mult * daily_vol_t * np.sqrt(vol_stop_window)
        else:
            effective_stop = trailing_stop   # fallback if vol not yet available
    else:
        effective_stop = trailing_stop

    if (current_position > 0 and profit_stop_scale > 0.0
            and entry_price > 0 and effective_stop > 0.0):
        unrealised_pct = (cur_close - entry_price) / entry_price
        if unrealised_pct > 0:
            effective_stop = max(min_stop_pct, effective_stop - unrealised_pct * profit_stop_scale)

    return effective_stop


def _check_exit_conditions(
    current_position: float,
    cur_close: float,
    entry_price: float,
    rr_ratio: float,
    rr_risk: float,
    rr_stop_level: float,
    rr_target_level: float,
    effective_stop: float,
    peak_price_since_entry: float,
    max_hold_days: int,
    days_in_trade: int,
    use_sar_stop: bool,
    sar_val: float | None,
    need_exit: bool,
    macd_bearish_cross: bool,
    exit_on_macd_cross: bool,
    exit_on_rsi_reversal: bool,
    rsi_overbought_exit: bool,
    rsi_momentum_loss: bool,
    exit_on_consolidation: bool,
    consolidating: bool,
) -> tuple[bool, str, float, int]:
    """Check every exit condition for one bar, in priority order:
    R:R stop/target > trailing stop > max-hold-days > SAR stop > exit indicators.

    Returns (trailing_stop_hit, sell_reason, peak_price_since_entry, days_in_trade)
    — the latter two updated for carry-forward into the next bar.
    """
    if current_position == 0:
        return False, "", 0.0, 0

    sell_reason = ""
    trailing_stop_hit = False

    # R:R stop-loss and take-profit (hard floor/ceiling from entry) — checked
    # FIRST, since the R:R stop-loss is the maximum acceptable loss.
    if rr_ratio > 0.0 and entry_price > 0:
        if cur_close <= rr_stop_level:
            trailing_stop_hit = True
            loss_pct = (entry_price - cur_close) / entry_price * 100
            sell_reason = f"rr_stop_loss({loss_pct:.1f}% loss, risk={rr_risk*100:.0f}%)"
        elif cur_close >= rr_target_level:
            trailing_stop_hit = True
            gain_pct = (cur_close - entry_price) / entry_price * 100
            sell_reason = f"rr_take_profit({gain_pct:.1f}% gain, target={rr_risk*rr_ratio*100:.0f}%)"

    # Trailing stop (profit protection only). Only fires when the trade has
    # been profitable AND current price is still above entry — if price has
    # fallen below entry, the R:R stop handles it.
    if effective_stop > 0.0 and not trailing_stop_hit:
        peak_price_since_entry = max(peak_price_since_entry, cur_close)
        if peak_price_since_entry > entry_price and cur_close >= entry_price:
            drop_from_peak = (peak_price_since_entry - cur_close) / peak_price_since_entry
            if drop_from_peak >= effective_stop:
                trailing_stop_hit = True
                gain_pct = (cur_close - entry_price) / entry_price * 100
                sell_reason = (f"trailing_stop({drop_from_peak*100:.1f}% from peak, "
                              f"still +{gain_pct:.1f}% from entry)")

    # Max hold days
    if max_hold_days > 0 and not trailing_stop_hit:
        days_in_trade += 1
        if days_in_trade >= max_hold_days:
            trailing_stop_hit = True
            sell_reason = f"max_hold({days_in_trade}d)"

    # Parabolic SAR stop (only while in position, trade must have been profitable)
    if (use_sar_stop and sar_val is not None and not trailing_stop_hit
            and peak_price_since_entry > entry_price):
        if cur_close < sar_val:
            trailing_stop_hit = True
            pct_below = (sar_val - cur_close) / sar_val * 100
            sell_reason = f"sar_stop(price {cur_close:.2f} < SAR {sar_val:.2f}, -{pct_below:.1f}%)"

    # Exit indicator checks (optional, only while in position)
    if need_exit and not trailing_stop_hit:
        if exit_on_macd_cross and macd_bearish_cross:
            trailing_stop_hit = True
            sell_reason = "exit_macd_cross"
        elif exit_on_rsi_reversal and (rsi_overbought_exit or rsi_momentum_loss):
            trailing_stop_hit = True
            reason = "rsi_overbought_exit" if rsi_overbought_exit else "rsi_momentum_loss"
            sell_reason = f"exit_{reason}"
        elif exit_on_consolidation and consolidating:
            trailing_stop_hit = True
            sell_reason = "exit_consolidation"

    return trailing_stop_hit, sell_reason, peak_price_since_entry, days_in_trade
