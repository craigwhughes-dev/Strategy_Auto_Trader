"""Weight-sweep test variants of optimised_new for 2026-09-13 isolated sweep.

Three variants, each changing exactly one weight vs optimised_new's defaults:
  - on_rsi2:      rsi weight 1.0 -> 2.0
  - on_sma4:      sma200 weight 3.0 -> 4.0
  - on_rsi2_sma4: both changes together

Identified as worth testing from per-ticker ablation sweep (run_unswept_param_sweep.py):
  - rsi=2.0: +0.116 Sharpe, +9pp win rate on 11-ticker basket
  - sma200=4.0: +0.061 Sharpe, +9pp win rate, more trades

These are test variants, not live strategies. Do not use as --strategy on the daemon.
"""

from __future__ import annotations

from .optimised_new import OptimisedNewEntry, OptimisedNewExit


class OnRsi2Entry(OptimisedNewEntry):
    """optimised_new with rsi weight raised 1.0 -> 2.0, all else unchanged."""
    weights: dict[str, float] = {**OptimisedNewEntry.weights, "rsi": 2.0}


class OnSma4Entry(OptimisedNewEntry):
    """optimised_new with sma200 weight raised 3.0 -> 4.0, all else unchanged."""
    weights: dict[str, float] = {**OptimisedNewEntry.weights, "sma200": 4.0}


class OnRsi2Sma4Entry(OptimisedNewEntry):
    """optimised_new with rsi=2.0 and sma200=4.0, all else unchanged."""
    weights: dict[str, float] = {**OptimisedNewEntry.weights, "rsi": 2.0, "sma200": 4.0}
