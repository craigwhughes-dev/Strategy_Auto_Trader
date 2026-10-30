"""Shadow benchmark: what a fixed GBP amount would be worth had it stayed in EQGB.L
(tier 1, the allocator's actual non-cash holding — tiers 2/3 are config-disabled, see
`lower_tiers_enabled`) instead of following the tier allocator, since the tier last
changed (`MultiTierAllocationManager`'s `tier_active_since`). Informational only —
never feeds back into allocation decisions."""

from __future__ import annotations

from datetime import date, timedelta

import yfinance as yf

from ..core.net_retry import call_with_timeout_retry

SHADOW_TICKER = "EQGB.L"
SHADOW_AMOUNT_GBP = 1000.0


def compute_shadow_benchmark(
    since: date,
    as_of: date,
    ticker: str = SHADOW_TICKER,
    amount_gbp: float = SHADOW_AMOUNT_GBP,
) -> dict | None:
    """P&L a GBP position opened at `ticker`'s close on `since` would show by `as_of`.

    Returns None if a close price isn't available for either end (fetch failure, or
    `since`/`as_of` land on a day the ETF didn't trade) — the caller should leave any
    previously persisted result in place rather than overwrite it with a gap.
    """
    def _fetch():
        return yf.Ticker(ticker).history(
            start=since.isoformat(), end=(as_of + timedelta(days=1)).isoformat(), interval="1d"
        )

    history = call_with_timeout_retry(_fetch)
    if history is None or history.empty:
        return None
    closes = history["Close"]
    entry_price = float(closes.iloc[0])
    latest_price = float(closes.iloc[-1])
    value_gbp = amount_gbp * latest_price / entry_price
    return {
        "ticker": ticker,
        "since": since.isoformat(),
        "as_of": as_of.isoformat(),
        "amount_gbp": amount_gbp,
        "entry_price": entry_price,
        "latest_price": latest_price,
        "value_gbp": round(value_gbp, 2),
        "pnl_gbp": round(value_gbp - amount_gbp, 2),
        "pnl_pct": round((value_gbp - amount_gbp) / amount_gbp * 100, 2),
    }
