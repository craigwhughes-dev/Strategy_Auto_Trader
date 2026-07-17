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

from ..ai_optimised import AiOptimisedEntry, AiOptimisedExit
from ..ai_strategy import AiEntry, AiExit
from ..breakout_momentum import BreakoutMomentumEntry, BreakoutMomentumExit
from ..breakout_momentum_optimised import (
    BreakoutMomentumOptimisedEntry,
    BreakoutMomentumOptimisedExit,
)
from ..choppy_vol import ChoppyVolEntry, ChoppyVolExit
from ..choppy_vol_optimised import ChoppyVolOptimisedEntry, ChoppyVolOptimisedExit
from ..conservative import ConservativeEntry, ConservativeExit
from ..conservative_optimised import ConservativeOptimisedEntry, ConservativeOptimisedExit
from ..default import DefaultEntry, DefaultExit
from ..mean_reversion import MeanReversionEntry, MeanReversionExit
from ..optimised import OptimisedEntry, OptimisedExit
from ..optimised_aggressive import OptimisedAggressiveEntry, OptimisedAggressiveExit
from ..optimised_aggressive_optimised import (
    OptimisedAggressiveOptimisedEntry,
    OptimisedAggressiveOptimisedExit,
)
from ..optimised_optimised import OptimisedOptimisedEntry, OptimisedOptimisedExit
from ..optimised_pbull import OptimisedPbullEntry, OptimisedPbullExit
from ..optimised_regime import OptimisedRegimeEntry, OptimisedRegimeExit
from ..optimised_rsi import OptimisedRsiEntry, OptimisedRsiExit
from ..optimised_score7 import OptimisedScore7Entry, OptimisedScore7Exit
from ..optimised_volume import OptimisedVolumeEntry, OptimisedVolumeExit
from ..trend_follow import TrendEntry, TrendExit
from ..trend_optimised import TrendOptimisedEntry, TrendOptimisedExit

STRATEGY_REGISTRY: dict[str, dict[str, type]] = {
    "ai": {
        "entry": AiEntry,
        "exit":  AiExit,
    },
    "ai_optimised": {
        "entry": AiOptimisedEntry,
        "exit":  AiOptimisedExit,
    },
    "breakout_momentum": {
        "entry": BreakoutMomentumEntry,
        "exit":  BreakoutMomentumExit,
    },
    "breakout_momentum_optimised": {
        "entry": BreakoutMomentumOptimisedEntry,
        "exit":  BreakoutMomentumOptimisedExit,
    },
    "choppy_vol": {
        "entry": ChoppyVolEntry,
        "exit":  ChoppyVolExit,
    },
    "choppy_vol_optimised": {
        "entry": ChoppyVolOptimisedEntry,
        "exit":  ChoppyVolOptimisedExit,
    },
    "conservative": {
        "entry": ConservativeEntry,
        "exit":  ConservativeExit,
    },
    "conservative_optimised": {
        "entry": ConservativeOptimisedEntry,
        "exit":  ConservativeOptimisedExit,
    },
    "default": {
        "entry": DefaultEntry,
        "exit":  DefaultExit,
    },
    "mean_reversion": {
        "entry": MeanReversionEntry,
        "exit":  MeanReversionExit,
    },
    "optimised": {
        "entry": OptimisedEntry,
        "exit":  OptimisedExit,
    },
    "optimised_aggressive": {
        "entry": OptimisedAggressiveEntry,
        "exit":  OptimisedAggressiveExit,
    },
    "optimised_aggressive_optimised": {
        "entry": OptimisedAggressiveOptimisedEntry,
        "exit":  OptimisedAggressiveOptimisedExit,
    },
    "optimised_optimised": {
        "entry": OptimisedOptimisedEntry,
        "exit":  OptimisedOptimisedExit,
    },
    "optimised_pbull": {
        "entry": OptimisedPbullEntry,
        "exit":  OptimisedPbullExit,
    },
    "optimised_regime": {
        "entry": OptimisedRegimeEntry,
        "exit":  OptimisedRegimeExit,
    },
    "optimised_rsi": {
        "entry": OptimisedRsiEntry,
        "exit":  OptimisedRsiExit,
    },
    "optimised_score7": {
        "entry": OptimisedScore7Entry,
        "exit":  OptimisedScore7Exit,
    },
    "optimised_volume": {
        "entry": OptimisedVolumeEntry,
        "exit":  OptimisedVolumeExit,
    },
    "trend": {
        "entry": TrendEntry,
        "exit":  TrendExit,
    },
    "trend_optimised": {
        "entry": TrendOptimisedEntry,
        "exit":  TrendOptimisedExit,
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
    every trend-following strategy's entry decision — a ticker classified as
    choppy/mean-reverting is vetoed to permanent HOLD, regardless of caller.
    "choppy_vol" is the exception: it ignores vol_filter_ok entirely (see
    strategy/choppy_vol.py) since it's the strategy meant to trade those
    vetoed tickers instead of leaving them idle — resolve it explicitly for
    a ticker whose trend_quality is low rather than relying on this filter.

    vol_filter_ok, if given explicitly, overrides the computed check (True
    forces the filter off for this instance, e.g. choppy_vol or another
    vol-filter-exempt strategy; False forces a veto without a lookup).
    Otherwise, if `ticker` is given, trend_quality is computed and the
    filter applied automatically. With neither, the filter defaults to
    "on"/permissive (True) since there is no ticker to evaluate.

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


def wants_low_trend_quality(name: str) -> bool:
    """True if the named strategy is meant to trade the low-trend-quality
    (choppy) tickers the default vol_screen vetoes, rather than the
    high-trend-quality names it keeps.

    Reads the Entry class's `wants_low_trend_quality` attribute (default
    False) instead of a hardcoded strategy-name list — set the flag on the
    strategy's Entry class, not here.
    """
    if name not in STRATEGY_REGISTRY:
        raise KeyError(
            f"Unknown strategy '{name}'. Available: {sorted(STRATEGY_REGISTRY)}"
        )
    return getattr(STRATEGY_REGISTRY[name]["entry"], "wants_low_trend_quality", False)
