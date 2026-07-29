"""Shared override-resolution helper for Exit classes.

A plain function, not a base class — strategies stay standalone (no
strategy-to-strategy inheritance); each Exit class still declares its own
defaults inline in its own file. This only eliminates the repeated
None-coalescing + StandardExitRules(**kwargs) boilerplate that would
otherwise be copied into every one of the 7 Exit classes.

Note: `stop_loss_pct` has two independent consumers — the Exit class's own
`self._stop`/`stop_loss_pct` property (which the engine reads directly to
set the R:R stop level) and StandardExitRules' `rr_risk` (used inside
`_check_exit_conditions`). Each Exit class's `__init__` should resolve the
override into `self._stop` first, then pass that resolved value as this
dict's `"stop_loss_pct"` default — not re-pass the raw constructor arg as a
second override to this helper, which would just be resolving it twice.
"""

from __future__ import annotations

from ...plugins.exit_rules import StandardExitRules

_STANDARD_EXIT_RULES_PARAMS = (
    "stop_loss_pct", "trailing_stop", "vol_stop_mult", "vol_stop_window",
    "profit_stop_scale", "min_stop_pct", "max_hold_days",
    "exit_on_macd_cross", "exit_on_rsi_reversal", "exit_on_consolidation",
    "use_sar_stop",
)


def build_standard_exit_rules(defaults: dict, **overrides) -> StandardExitRules:
    """Resolve `defaults` merged with any non-None `overrides` into a
    StandardExitRules instance. Unknown override keys raise TypeError (same
    as calling StandardExitRules(**kwargs) directly would)."""
    unknown = set(overrides) - set(_STANDARD_EXIT_RULES_PARAMS)
    if unknown:
        raise TypeError(f"build_standard_exit_rules: unknown override(s) {sorted(unknown)}")
    resolved = dict(defaults)
    for key, value in overrides.items():
        if value is not None:
            resolved[key] = value
    return StandardExitRules(**resolved)
