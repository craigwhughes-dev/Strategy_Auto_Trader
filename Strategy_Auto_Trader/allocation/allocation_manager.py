"""Multi-tier allocation manager for live daemon.

Manages daily rebalancing between SPY/ISF.L/CSH2.L based on VIX tiers.

Tier 1 (VIX ≤ 15): SPY (growth)
Tier 2 (15 < VIX ≤ 17.5): ISF.L (balanced)
Tier 3 (VIX > 17.5): CSH2.L (defensive/money-market)

Daily rebalance: if target asset differs from current, generate sell (current) + buy (target) orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import logging

import pandas as pd

_log = logging.getLogger(__name__)

TIER_ASSETS = {1: "SPY", 2: "ISF.L", 3: "CSH2.L"}
ASSET_TIERS = {"SPY": 1, "ISF.L": 2, "CSH2.L": 3}


@dataclass
class AllocationOrder:
    """Single buy/sell order for rebalancing."""
    ticker: str
    action: str  # "BUY" | "SELL"
    quantity: int
    limit_price: float | None = None
    reason: str = ""


@dataclass
class AllocationSignal:
    """Daily allocation decision."""
    date: date
    vix: float | None
    tier: int
    target_asset: str
    current_asset: str
    needs_rebalance: bool
    reason: str


class MultiTierAllocationManager:
    """Manage daily multi-tier allocation rebalances.

    Tracks current holding and generates buy/sell orders when target tier changes.
    """

    def __init__(
        self,
        vix_tier1: float = 15.0,
        vix_tier2: float = 17.5,
        commission_pct: float = 0.1,
    ):
        """Initialize allocation manager.

        Args:
            vix_tier1: VIX threshold for tier 1→2 boundary (VIX ≤ this → SPY)
            vix_tier2: VIX threshold for tier 2→3 boundary (VIX ≤ this → ISF.L)
            commission_pct: Commission as % of order value (e.g., 0.1 for 0.1%)
        """
        self.vix_tier1 = vix_tier1
        self.vix_tier2 = vix_tier2
        self.commission_pct = commission_pct
        self.current_asset = "SPY"  # Default starting position
        self.current_price = None  # Snapshot of entry price for tracking
        self.last_rebalance_date = None

    def signal(self, today: date, vix: float | None) -> AllocationSignal:
        """Compute daily allocation decision.

        Args:
            today: Current date
            vix: Daily VIX close (None if unavailable)

        Returns:
            AllocationSignal with tier, target asset, and rebalance decision
        """
        if vix is None:
            tier = 1
            target_asset = "SPY"
            reason = "no VIX data, default to SPY"
        elif vix <= self.vix_tier1:
            tier = 1
            target_asset = "SPY"
            reason = f"VIX={vix:.1f} ≤ {self.vix_tier1} → Tier 1 (SPY)"
        elif vix <= self.vix_tier2:
            tier = 2
            target_asset = "ISF.L"
            reason = f"VIX={vix:.1f} ∈ ({self.vix_tier1}, {self.vix_tier2}] → Tier 2 (ISF.L)"
        else:
            tier = 3
            target_asset = "SHV"
            reason = f"VIX={vix:.1f} > {self.vix_tier2} → Tier 3 (SHV)"

        needs_rebalance = target_asset != self.current_asset
        rebalance_reason = reason + (" (rebalance needed)" if needs_rebalance else " (holding)")

        return AllocationSignal(
            date=today,
            vix=vix,
            tier=tier,
            target_asset=target_asset,
            current_asset=self.current_asset,
            needs_rebalance=needs_rebalance,
            reason=rebalance_reason,
        )

    def rebalance(
        self,
        today: date,
        vix: float | None,
        current_price: dict[str, float],
        available_cash: float,
        positions: dict[str, int],
    ) -> list[AllocationOrder]:
        """Generate rebalance orders if target tier differs from current.

        Args:
            today: Current date
            vix: Daily VIX close
            current_price: Dict {ticker: price} for all three assets
            available_cash: Available cash to buy
            positions: Current positions {ticker: quantity}

        Returns:
            List of AllocationOrder (sell current + buy target, or empty if no rebalance needed)
        """
        signal = self.signal(today, vix)

        if not signal.needs_rebalance:
            _log.debug(f"[{today}] Allocation: {signal.reason} (no order)")
            return []

        orders = []

        # Sell current asset if we hold any
        current_qty = positions.get(signal.current_asset, 0)
        if current_qty > 0:
            sell_price = current_price.get(signal.current_asset, 0)
            if sell_price > 0:
                orders.append(
                    AllocationOrder(
                        ticker=signal.current_asset,
                        action="SELL",
                        quantity=current_qty,
                        limit_price=None,
                        reason=f"Rebalance: exit {signal.current_asset} (was tier {ASSET_TIERS.get(signal.current_asset, 0)})",
                    )
                )
                proceeds = current_qty * sell_price * (1 - self.commission_pct / 100)
                available_cash += proceeds
                _log.info(
                    f"[{today}] Allocation SELL {current_qty} {signal.current_asset} @ {sell_price:.2f} "
                    f"(proceeds ~{proceeds:.2f}, comm {self.commission_pct}%)"
                )

        # Buy target asset with available cash
        target_price = current_price.get(signal.target_asset, 0)
        if target_price > 0:
            # Buy as many whole shares as we can afford (after commission)
            cost_per_share = target_price * (1 + self.commission_pct / 100)
            buy_qty = int(available_cash / cost_per_share)

            if buy_qty > 0:
                orders.append(
                    AllocationOrder(
                        ticker=signal.target_asset,
                        action="BUY",
                        quantity=buy_qty,
                        limit_price=None,
                        reason=f"Rebalance: enter {signal.target_asset} (tier {signal.tier}), VIX={vix:.1f}",
                    )
                )
                cost = buy_qty * target_price * (1 + self.commission_pct / 100)
                _log.info(
                    f"[{today}] Allocation BUY {buy_qty} {signal.target_asset} @ {target_price:.2f} "
                    f"(cost ~{cost:.2f}, comm {self.commission_pct}%)"
                )
            else:
                _log.warning(
                    f"[{today}] Allocation: insufficient cash to buy {signal.target_asset} "
                    f"(have {available_cash:.2f}, need ≥{cost_per_share:.2f}/share)"
                )

        # Update internal state
        if orders:
            self.current_asset = signal.target_asset
            self.current_price = target_price
            self.last_rebalance_date = today
            _log.info(f"[{today}] Allocation rebalance: {signal.reason}")

        return orders

    def app_status_dict(self) -> dict:
        """Return current state for app_status.json."""
        return {
            "current_asset": self.current_asset,
            "current_price": self.current_price,
            "last_rebalance_date": self.last_rebalance_date.isoformat() if self.last_rebalance_date else None,
        }
