"""Multi-tier allocation manager for live daemon.

Manages daily rebalancing between Nasdaq/SPY/ISF.L/CSH2.L based on VXN + VIX tiers.

Tier 1 (VXN ≤ 18): Nasdaq/EQGB.L (growth)
Tier 2 (VIX ≤ 15): SPY (balanced)
Tier 3 (15 < VIX ≤ 17.5): ISF.L (defensive)
Tier 4 (VIX > 17.5): CSH2.L (money-market)

Daily rebalance: if target asset differs from current, generate sell (current) + buy (target) orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import logging
import time

import pandas as pd

_log = logging.getLogger(__name__)

TIER_ASSETS = {1: "EQGB.L", 2: "SPY", 3: "ISF.L", 4: "CSH2.L"}
ASSET_TIERS = {"EQGB.L": 1, "SPY": 2, "ISF.L": 3, "CSH2.L": 4}


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
    vxn: float | None
    tier: int
    target_asset: str
    current_asset: str
    needs_rebalance: bool
    reason: str


class MultiTierAllocationManager:
    """Manage daily 4-tier allocation rebalances with VXN + VIX gating.

    Tier 1: Nasdaq/EQGB.L (VXN ≤ vxn_threshold)
    Tier 2: SPY (VIX ≤ vix_tier1)
    Tier 3: ISF.L (vix_tier1 < VIX ≤ vix_tier2)
    Tier 4: CSH2.L (defensive, always available)
    """

    def __init__(
        self,
        vxn_threshold: float = 18.0,
        vix_tier1: float = 15.0,
        vix_tier2: float = 17.5,
        commission_pct: float = 0.1,
    ):
        """Initialize allocation manager.

        Args:
            vxn_threshold: VXN threshold for tier 1 (Nasdaq). VXN ≤ this → Nasdaq
            vix_tier1: VIX threshold for tier 2 (SPY). VIX ≤ this → SPY
            vix_tier2: VIX threshold for tier 3 (ISF.L). VIX ≤ this → ISF.L
            commission_pct: Commission as % of order value (e.g., 0.1 for 0.1%)
        """
        self.vxn_threshold = vxn_threshold
        self.vix_tier1 = vix_tier1
        self.vix_tier2 = vix_tier2
        self.commission_pct = commission_pct
        self.current_asset = "SPY"  # Default starting position
        self.current_price = None  # Snapshot of entry price for tracking
        self.last_rebalance_date = None
        self._last_known_cash = None  # Track cash for detecting user deposits
        self._last_cash_check_time = None  # Timestamp of last cash baseline

    def signal(self, today: date, vxn: float | None, vix: float | None) -> AllocationSignal:
        """Compute daily allocation decision.

        Args:
            today: Current date
            vxn: Daily VXN close (None if unavailable)
            vix: Daily VIX close (None if unavailable)

        Returns:
            AllocationSignal with tier, target asset, and rebalance decision
        """
        # Tier 1: Check VXN (Nasdaq)
        if vxn is not None and vxn <= self.vxn_threshold:
            tier = 1
            target_asset = "EQGB.L"
            reason = f"VXN={vxn:.1f} ≤ {self.vxn_threshold} → Tier 1 (Nasdaq)"
        # Tier 2-4: VIX-based fallback
        elif vix is None:
            tier = 4
            target_asset = "CSH2.L"
            reason = "no VIX data, fallback to CSH2.L"
        elif vix <= self.vix_tier1:
            tier = 2
            target_asset = "SPY"
            reason = f"VIX={vix:.1f} ≤ {self.vix_tier1} → Tier 2 (SPY)"
        elif vix <= self.vix_tier2:
            tier = 3
            target_asset = "ISF.L"
            reason = f"VIX={vix:.1f} ∈ ({self.vix_tier1}, {self.vix_tier2}] → Tier 3 (ISF.L)"
        else:
            tier = 4
            target_asset = "CSH2.L"
            reason = f"VIX={vix:.1f} > {self.vix_tier2} → Tier 4 (CSH2.L)"

        needs_rebalance = target_asset != self.current_asset
        rebalance_reason = reason + (" (rebalance needed)" if needs_rebalance else " (holding)")

        return AllocationSignal(
            date=today,
            vix=vix,
            vxn=vxn,
            tier=tier,
            target_asset=target_asset,
            current_asset=self.current_asset,
            needs_rebalance=needs_rebalance,
            reason=rebalance_reason,
        )

    def rebalance(
        self,
        today: date,
        vxn: float | None,
        vix: float | None,
        current_price: dict[str, float],
        available_cash: float,
        positions: dict[str, int],
    ) -> list[AllocationOrder]:
        """Generate rebalance orders if target tier differs from current.

        Args:
            today: Current date
            vxn: Daily VXN close
            vix: Daily VIX close
            current_price: Dict {ticker: price} for all four assets
            available_cash: Available cash to buy
            positions: Current positions {ticker: quantity}

        Returns:
            List of AllocationOrder (sell current + buy target, or empty if no rebalance needed)
        """
        # Exclude recent cash increases (deposits or manual sales) for 1 hour
        # to allow user to place manual trades before daemon allocates the cash
        allocatable_cash = self._filter_cash_for_allocation(available_cash)

        signal = self.signal(today, vxn, vix)

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
                allocatable_cash += proceeds
                _log.info(
                    f"[{today}] Allocation SELL {current_qty} {signal.current_asset} @ {sell_price:.2f} "
                    f"(proceeds ~{proceeds:.2f}, comm {self.commission_pct}%)"
                )

        # Buy target asset with allocatable cash
        target_price = current_price.get(signal.target_asset, 0)
        if target_price > 0:
            # Buy as many whole shares as we can afford (after commission)
            cost_per_share = target_price * (1 + self.commission_pct / 100)
            buy_qty = int(allocatable_cash / cost_per_share)

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
                    f"(have {allocatable_cash:.2f}, need ≥{cost_per_share:.2f}/share)"
                )

        # Update internal state
        if orders:
            self.current_asset = signal.target_asset
            self.current_price = target_price
            self.last_rebalance_date = today
            _log.info(f"[{today}] Allocation rebalance: {signal.reason}")

        return orders

    def _filter_cash_for_allocation(self, current_cash: float) -> float:
        """Filter cash to exclude recent deposits/increases.

        If cash increased within the last hour (user added funds or made manual trades),
        allocate only the baseline amount to give user time to deploy their own trades.
        After 1 hour, treat the increase as "user had time to deploy it" and allocate all.

        Returns: allocatable_cash (may be less than current_cash if increase is recent)
        """
        now = time.time()
        grace_period_seconds = 3600  # 1 hour

        if self._last_known_cash is None:
            # First call: initialize baseline
            self._last_known_cash = current_cash
            self._last_cash_check_time = now
            return current_cash

        increase = current_cash - self._last_known_cash
        time_since_baseline = now - self._last_cash_check_time

        if increase > 0 and time_since_baseline < grace_period_seconds:
            # Recent increase detected: allocate only the baseline
            _log.info(
                f"Cash increase detected (+{increase:.2f}); "
                f"grace period {grace_period_seconds - time_since_baseline:.0f}s remaining. "
                f"Allocating baseline {self._last_known_cash:.2f} only."
            )
            return self._last_known_cash
        elif increase > 0:
            # Grace period expired: update baseline and allocate all
            _log.info(f"Cash grace period expired; updating baseline to {current_cash:.2f}")
            self._last_known_cash = current_cash
            self._last_cash_check_time = now
            return current_cash
        else:
            # No increase (decrease or flat): update baseline if time has passed
            if time_since_baseline >= grace_period_seconds:
                self._last_known_cash = current_cash
                self._last_cash_check_time = now
            return current_cash

    def app_status_dict(self) -> dict:
        """Return current state for app_status.json."""
        return {
            "current_asset": self.current_asset,
            "current_price": self.current_price,
            "last_rebalance_date": self.last_rebalance_date.isoformat() if self.last_rebalance_date else None,
        }
