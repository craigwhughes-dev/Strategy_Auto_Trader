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

import logging

from ..breakout_momentum import BreakoutMomentumEntry, BreakoutMomentumExit
from ..choppy_vol import ChoppyVolEntry, ChoppyVolExit
from ..conservative import ConservativeEntry, ConservativeExit
from ..default import DefaultEntry, DefaultExit
from ..mean_reversion import MeanReversionEntry, MeanReversionExit
from ..optimised import OptimisedEntry, OptimisedExit
from ..optimised_new import OptimisedNewEntry, OptimisedNewExit
from ..trend_follow import TrendEntry, TrendExit

logger = logging.getLogger(__name__)

STRATEGY_REGISTRY: dict[str, dict[str, type]] = {
    "breakout_momentum": {
        "entry": BreakoutMomentumEntry,
        "exit":  BreakoutMomentumExit,
    },
    "choppy_vol": {
        "entry": ChoppyVolEntry,
        "exit":  ChoppyVolExit,
    },
    "conservative": {
        "entry": ConservativeEntry,
        "exit":  ConservativeExit,
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
    "optimised_new": {
        "entry": OptimisedNewEntry,
        "exit":  OptimisedNewExit,
    },
    "trend": {
        "entry": TrendEntry,
        "exit":  TrendExit,
    },
}


def resolve_strategy(
    name: str,
    ticker: str | None = None,
    vol_filter_ok: bool | None = None,
    min_trend_quality: float = 0.0,
    entry_overrides: dict | None = None,
    exit_overrides: dict | None = None,
    source: str = "ibkr",
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

    entry_overrides/exit_overrides are splatted directly into the strategy's
    Entry/Exit constructors on top of vol_filter_ok — every strategy owns its
    own defaults (see its module), these only override what's explicitly
    passed. An override a given strategy's constructor doesn't accept (e.g.
    buy_threshold on "choppy_vol") raises a normal TypeError — no silent
    degrade, since nothing in this codebase applies one override set across
    varying --strategy values today.

    source is forwarded to volatility_profile's own fetch when computing the
    vol-filter check — must match whatever source the caller's own hourly
    fetch used, or the filter would silently evaluate a different data source
    than the one the strategy is actually trading on.

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
            prof = volatility_profile(ticker, source=source)
            if prof is not None:
                vol_filter_ok = prof["trend_quality"] >= min_trend_quality
                logger.debug(
                    f"resolve_strategy: {ticker} trend_quality={prof['trend_quality']:.3f} "
                    f"min_trend_quality={min_trend_quality} source={source} "
                    f"-> vol_filter_ok={vol_filter_ok}"
                )

    entry = cls_map["entry"](vol_filter_ok=vol_filter_ok, **(entry_overrides or {}))
    exit_ = cls_map["exit"](**(exit_overrides or {}))
    return entry, exit_


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


def wants_vol_screen_disabled(name: str) -> bool:
    """True if the named strategy opts out of the per-market vol_screen
    stage-1 sweep in overnight_scope.py. Strategy opt-out always wins over
    the vol_screen.enabled config flag.

    Reads the Entry class's `skip_overnight_vol_screen` attribute (default
    False) instead of a hardcoded name list — set the flag on the strategy's
    Entry class, not here.

    Note: per-ticker watchlist strategy overrides are not handled here —
    same limitation as wants_low_trend_quality().
    """
    if name not in STRATEGY_REGISTRY:
        raise KeyError(
            f"Unknown strategy '{name}'. Available: {sorted(STRATEGY_REGISTRY)}"
        )
    return getattr(STRATEGY_REGISTRY[name]["entry"], "skip_overnight_vol_screen", False)
