"""Conservative veto layer shared by the daily and hourly engines.

Hard-vetoes weak BUY candidates and can force an early SELL when the
price/momentum context turns clearly adverse while already in a trade.
The gate never generates a new signal on its own — it only vetoes.

Operates purely on a `mom` dict (cur_rsi, recent_cross_above_50,
recent_cross_below_40, above_sma20, above_sma50, above_sma200,
volume_ratio) and a markov/regime signal float, so it is independent of
bar frequency — the same logic applies whether `mom` was built from daily
or hourly bars.
"""

from __future__ import annotations


def _is_weak_buy_context(markov_sig: float | None, mom: dict) -> bool:
    conditions = 0
    if markov_sig is not None and markov_sig < 0.25:
        conditions += 1
    if not (mom.get("above_sma20", False) and mom.get("above_sma50", False)):
        conditions += 1
    volume_ratio = mom.get("volume_ratio")
    if volume_ratio is not None and volume_ratio < 1.0:
        conditions += 1
    if mom.get("above_sma200") is False:
        conditions += 1
    if mom.get("cur_rsi", 50.0) < 50 and not mom.get("recent_cross_above_50", False):
        conditions += 1
    return conditions >= 2


def _is_adverse_exit_context(markov_sig: float | None, mom: dict) -> bool:
    conditions = 0
    if markov_sig is not None and markov_sig < -0.20:
        conditions += 1
    if mom.get("cur_rsi", 50.0) < 40 or mom.get("recent_cross_below_40", False):
        conditions += 1
    if not (mom.get("above_sma20", True) and mom.get("above_sma50", True)):
        conditions += 1
    volume_ratio = mom.get("volume_ratio")
    if volume_ratio is not None and volume_ratio < 0.8:
        conditions += 1
    if mom.get("above_sma200") is False:
        conditions += 1
    return conditions >= 2


def _apply_quality_gate(sig: dict, mom: dict, markov_sig: float | None, currently_in: bool) -> dict:
    """Apply the veto layer.

    sig.get("flag") == "BUY" and not currently_in and weak context -> HOLD.
    currently_in and adverse context -> SELL.
    Otherwise sig is returned unchanged (with a "reason" key added).
    """
    gated = dict(sig)
    gated["reason"] = sig.get("reason", "")

    if sig.get("flag") == "BUY" and not currently_in and _is_weak_buy_context(markov_sig, mom):
        gated["flag"] = "HOLD"
        gated["reason"] = "quality_gate: weak buy context"
        return gated

    if currently_in and _is_adverse_exit_context(markov_sig, mom):
        gated["flag"] = "SELL"
        gated["reason"] = "quality_gate: adverse exit context"
        return gated

    return gated
