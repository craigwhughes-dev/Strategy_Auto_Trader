"""Strategy registry — maps name strings to Entry/Exit class pairs.

Usage
-----
    from Strategy_Auto_Trader.strategy.base.registry import resolve_strategy
    entry, exit_ = resolve_strategy("conservative")
    bt = consolidated_backtest(df, entry_strategy=entry, exit_strategy=exit_)

To register a new strategy:
    1. Create strategy/<name>.py with an Entry class and an Exit class.
    2. Import them here and add an entry to STRATEGY_REGISTRY.
"""

from __future__ import annotations

from ..conservative import ConservativeEntry, ConservativeExit
from ..default import DefaultEntry, DefaultExit
from ..optimised import OptimisedEntry, OptimisedExit
from ..trend_follow import TrendEntry, TrendExit

STRATEGY_REGISTRY: dict[str, dict[str, type]] = {
    "default": {
        "entry": DefaultEntry,
        "exit":  DefaultExit,
    },
    "conservative": {
        "entry": ConservativeEntry,
        "exit":  ConservativeExit,
    },
    "trend": {
        "entry": TrendEntry,
        "exit":  TrendExit,
    },
    "optimised": {
        "entry": OptimisedEntry,
        "exit":  OptimisedExit,
    },
}


def resolve_strategy(
    name: str,
    ticker: str | None = None,
    vol_filter_ok: bool | None = None,
    min_trend_quality: float = 0.0,
) -> tuple[object, object]:
    """Instantiate the entry and exit classes for a named strategy.

    The volatility/choppiness pre-screen (quant_hmm.vol_screen) is baked into
    every strategy's entry decision — a ticker classified as choppy/mean-
    reverting is vetoed to permanent HOLD, regardless of caller.

    vol_filter_ok, if given explicitly, overrides the computed check (True
    forces the filter off for this instance, e.g. a --vol-filter-exempt
    strategy; False forces a veto without a lookup). Otherwise, if `ticker`
    is given, trend_quality is computed and the filter applied automatically.
    With neither, the filter defaults to "on"/permissive (True) since there
    is no ticker to evaluate.

    Returns (entry_instance, exit_instance).
    Raises KeyError for unknown names.
    """
    if name not in STRATEGY_REGISTRY:
        raise KeyError(
            f"Unknown strategy '{name}'. Available: {sorted(STRATEGY_REGISTRY)}"
        )
    cls_map = STRATEGY_REGISTRY[name]

    if vol_filter_ok is None:
        vol_filter_ok = True
        if ticker is not None:
            from ...quant_hmm.vol_screen import volatility_profile
            prof = volatility_profile(ticker)
            if prof is not None:
                vol_filter_ok = prof["trend_quality"] >= min_trend_quality

    return cls_map["entry"](vol_filter_ok=vol_filter_ok), cls_map["exit"]()
