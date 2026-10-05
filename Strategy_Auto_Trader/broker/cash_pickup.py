"""Release broker cash to the tier allocator once per daemon start."""

from __future__ import annotations

import logging

# The broker reports cash in GBP only; a tier allocator in another currency cannot use it.
RELEASE_CURRENCY = "GBP"


def release_uninvested_cash(allocation_mgr, broker, logger: logging.Logger, currency: str) -> float:
    """Release the broker's free GBP cash to the allocator. Returns the amount released.

    Cash that lands after this (dividends, deposits) waits until the next start, so the
    allocator never spends it during the day and manual rebalancing is not raced.
    """
    if currency != RELEASE_CURRENCY:
        logger.warning(f"Uninvested cash release skipped: pot currency is {currency}, "
                       f"broker cash is {RELEASE_CURRENCY}")
        return 0.0

    broker_cash = round(broker.get_available_cash(), 2)
    allocation_mgr.release_cash(broker_cash)
    logger.warning(f"Uninvested cash released to tier allocator: {broker_cash:.2f} {RELEASE_CURRENCY} "
                   f"(held until next start: any cash landing after this)")
    return broker_cash
