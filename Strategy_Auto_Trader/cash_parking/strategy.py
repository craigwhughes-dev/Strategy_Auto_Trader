"""Cash parking tier classification.

4-tier model: cash (XSTR.L) → gilts (IGLS.L) → hy_bonds (ISXF.L) → equity (ISF.L).
ISXF.L = iShares $ HY Corp Bond UCITS ETF GBP Hedged — USD HY credit risk, GBP-hedged.

Tuned via backtest.py sweep on ISXF.L 2026-09-14 (16yr synthetic data, 9,216 combos).
Optimal: Sharpe 0.289, Sortino 0.316, +24.53% return, -11.70% max DD.

LIQUID_FLOOR_PCT = 0.0 (no cash buffer needed; parking doesn't block main entries).
"""

from __future__ import annotations

LIQUID_FLOOR_PCT: float = 0.0

# Minimum GBP worth parking; avoids tiny orders on near-empty cash.
MIN_PARKING_AMOUNT: float = 500.0

# Only switch tiers when the required allocation shift exceeds this fraction of
# the initial pot. Prevents thrash on days where VIX sits near a boundary.
REBALANCE_THRESHOLD_PCT: float = 0.05


def tier_for(pbull_smooth: float, vix: float) -> str:
    """Map (pbull_smooth, vix) to parking tier name.

    Returns one of: "equity", "hy_bonds", "gilts", "cash".
    Boundaries optimised via 16yr ISXF.L sweep (2026-09-14).
    """
    if pbull_smooth >= 0.65 and vix < 12.0:   # strong bull, very low vol → equities
        return "equity"
    if pbull_smooth >= 0.55 and vix < 18.0:   # moderate bull → HY bonds (GBP-hedged)
        return "hy_bonds"
    if pbull_smooth >= 0.45 and vix < 22.0:   # mild bull → gilts
        return "gilts"
    return "cash"                               # capital preservation
