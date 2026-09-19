"""Whipsaw filters applied to a raw daily tier series (1=Nasdaq ... 4=CSH2.L; higher = more defensive).

Pure functions over the tier sequence, so any allocator's signal, lag and cost model can be reused
unchanged. Semantics differ from the older multi_tier_allocator_{hysteresis,cadence}.py: those count
days where desired != current even if the desired tier keeps changing; here a switch needs the SAME
target tier to persist.

All filters start holding the first raw tier and only ever return tiers the raw series produced.
"""

from __future__ import annotations

import pandas as pd


def apply_hysteresis(raw: pd.Series, days: int) -> pd.Series:
    """Switch only after the same different target has been requested `days` days in a row."""
    return _run(raw, confirm_defensive=days, confirm_offensive=days, min_days_between=0)


def apply_asymmetric_hysteresis(raw: pd.Series, offensive_days: int) -> pd.Series:
    """Move to a more defensive tier immediately; move to a riskier tier only after `offensive_days` in a row."""
    return _run(raw, confirm_defensive=1, confirm_offensive=offensive_days, min_days_between=0)


def apply_buy_cadence(raw: pd.Series, cadence_days: int) -> pd.Series:
    """Defensive moves immediate; riskier moves only if >= `cadence_days` since the last switch."""
    return _run(raw, confirm_defensive=1, confirm_offensive=1, min_days_between=cadence_days)


def _run(raw: pd.Series, confirm_defensive: int, confirm_offensive: int, min_days_between: int) -> pd.Series:
    values = raw.tolist()
    if not values:
        return raw.copy()
    current = values[0]
    since_switch = min_days_between  # first offensive move is not blocked by the cadence
    pending, streak = None, 0
    out = []
    for target in values:
        if target == current:
            pending, streak = None, 0
        else:
            streak = streak + 1 if target == pending else 1
            pending = target
            defensive = target > current
            need = confirm_defensive if defensive else confirm_offensive
            if streak >= need and (defensive or since_switch >= min_days_between):
                current, pending, streak, since_switch = target, None, 0, 0
        since_switch += 1
        out.append(current)
    return pd.Series(out, index=raw.index)
