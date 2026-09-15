"""Cash parking tier classification.

3-tier model: cash (XSTR.L) → gilts (IGLS.L) → hy_bonds (ISXF.L).
ISXF.L = iShares $ HY Corp Bond UCITS ETF GBP Hedged — USD HY credit risk, GBP-hedged.

Tier decisions driven by VIX levels only (market volatility regime).

LIQUID_FLOOR_PCT = 0.0 (no cash buffer needed; parking doesn't block main entries).
"""

from __future__ import annotations

LIQUID_FLOOR_PCT: float = 0.0

# Minimum GBP worth parking; avoids tiny orders on near-empty cash.
MIN_PARKING_AMOUNT: float = 500.0

# Only switch tiers when the required allocation shift exceeds this fraction of
# the initial pot. Prevents thrash on days where VIX sits near a boundary.
REBALANCE_THRESHOLD_PCT: float = 0.05


def tier_for(vix: float) -> str:
    """Map VIX level to parking tier name.

    Returns one of: "hy_bonds", "gilts", "cash".
    """
    if vix < 12.0:          # very low vol → HY bonds
        return "hy_bonds"
    if vix < 18.0:          # moderate vol → gilts
        return "gilts"
    return "cash"           # high vol → capital preservation
