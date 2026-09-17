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

        # Daily cache for VIX/VXN to avoid reloading 80k rows every cycle
        self._vix_cache = None
        self._vix_cache_date = None
        self._vxn_cache = None
        self._vxn_cache_date = None

    def signal(self, today: date, vxn: float | None, vix: float | None, logger=None) -> AllocationSignal:
        """Compute daily allocation decision.

        Args:
            today: Current date
            vxn: Daily VXN close (None if unavailable)
            vix: Daily VIX close (None if unavailable)
            logger: Logger to use (if None, uses module logger)

        Returns:
            AllocationSignal with tier, target asset, and rebalance decision
        """
        log = logger or _log
        log.info(f"[{today}] Signal evaluation: VXN={vxn}, VIX={vix}, current_asset={self.current_asset}")

        # Evaluate all tiers and log each
        tier = None
        target_asset = None
        reason = None

        # Tier 1: Check VXN (Nasdaq)
        tier1_pass = vxn is not None and vxn <= self.vxn_threshold
        if vxn is not None:
            log.info(f"[{today}]   Tier 1 (EQGB.L/Nasdaq): VXN={vxn:.2f} vs gate={self.vxn_threshold} [{'PASS' if tier1_pass else 'FAIL'}]")
        else:
            log.info(f"[{today}]   Tier 1 (EQGB.L/Nasdaq): VXN unavailable [FAIL]")

        if tier1_pass:
            tier = 1
            target_asset = "EQGB.L"
            reason = f"Tier 1 (EQGB.L/Nasdaq): VXN={vxn:.2f} ≤ {self.vxn_threshold}"

        # Tier 2: VIX ≤ tier1 threshold (SPY)
        if tier is None:
            if vix is not None:
                tier2_pass = vix <= self.vix_tier1
                log.info(f"[{today}]   Tier 2 (SPY/Balanced): VIX={vix:.2f} vs gate={self.vix_tier1} [{'PASS' if tier2_pass else 'FAIL'}]")
                if tier2_pass:
                    tier = 2
                    target_asset = "SPY"
                    reason = f"Tier 2 (SPY/Balanced): VIX={vix:.2f} ≤ {self.vix_tier1}"
            else:
                log.info(f"[{today}]   Tier 2 (SPY/Balanced): VIX unavailable [FAIL]")

        # Tier 3: VIX in (tier1, tier2] (ISF.L)
        if tier is None:
            if vix is not None:
                tier3_pass = vix <= self.vix_tier2
                log.info(f"[{today}]   Tier 3 (ISF.L/Defensive): VIX={vix:.2f} in ({self.vix_tier1}, {self.vix_tier2}] [{'PASS' if tier3_pass else 'FAIL'}]")
                if tier3_pass:
                    tier = 3
                    target_asset = "ISF.L"
                    reason = f"Tier 3 (ISF.L/Defensive): {self.vix_tier1} < VIX={vix:.2f} ≤ {self.vix_tier2}"
            else:
                log.info(f"[{today}]   Tier 3 (ISF.L/Defensive): VIX unavailable [FAIL]")

        # Tier 4: VIX > tier2 or no VIX (CSH2.L) — always fallback
        if tier is None:
            if vix is not None:
                log.info(f"[{today}]   Tier 4 (CSH2.L/Money-Market): VIX={vix:.2f} > {self.vix_tier2} [PASS] ← SELECTED")
            else:
                log.info(f"[{today}]   Tier 4 (CSH2.L/Money-Market): no VIX data [PASS] ← SELECTED")
            tier = 4
            target_asset = "CSH2.L"
            reason = f"Tier 4 (CSH2.L/Money-Market): fallback"
        else:
            log.info(f"[{today}]   Tier 4 (CSH2.L/Money-Market): [SKIPPED, tier {tier} already selected]")

        needs_rebalance = target_asset != self.current_asset
        if needs_rebalance:
            log.info(f"[{today}]   ACTION: {self.current_asset} → {target_asset} (REBALANCE)")
        else:
            log.debug(f"[{today}]   ACTION: HOLD {target_asset} (no change)")

        return AllocationSignal(
            date=today,
            vix=vix,
            vxn=vxn,
            tier=tier,
            target_asset=target_asset,
            current_asset=self.current_asset,
            needs_rebalance=needs_rebalance,
            reason=reason,
        )

    def rebalance(
        self,
        today: date,
        vxn: float | None,
        vix: float | None,
        current_price: dict[str, float],
        available_cash: float,
        positions: dict[str, int],
        logger=None,
    ) -> list[AllocationOrder]:
        """Generate rebalance orders if target tier differs from current.

        Args:
            today: Current date
            vxn: Daily VXN close
            vix: Daily VIX close
            current_price: Dict {ticker: price} for all four assets
            available_cash: Available cash to buy
            positions: Current positions {ticker: quantity}
            logger: Logger to use (if None, uses module logger)

        Returns:
            List of AllocationOrder (sell current + buy target, or empty if no rebalance needed)
        """
        log = logger or _log
        log.info(f"[{today}] Rebalance: available_cash={available_cash:.2f}, positions={positions}")

        # Exclude recent cash increases (deposits or manual sales) for 1 hour
        # to allow user to place manual trades before daemon allocates the cash
        allocatable_cash = self._filter_cash_for_allocation(available_cash)
        log.info(f"[{today}]   allocatable_cash={allocatable_cash:.2f} (after grace period filter)")

        signal = self.signal(today, vxn, vix, logger=log)

        # Generate orders if: rebalancing tiers OR bootstrapping (no positions + cash available)
        has_any_position = any(positions.values())
        needs_action = signal.needs_rebalance or (not has_any_position and allocatable_cash > 0)

        if not needs_action:
            log.debug(f"[{today}] Allocation: {signal.reason} (no action)")
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
                log.info(
                    f"[{today}] Allocation SELL {current_qty} {signal.current_asset} @ {sell_price:.2f} "
                    f"(proceeds ~{proceeds:.2f}, comm {self.commission_pct}%)"
                )

        # Buy target asset with allocatable cash
        target_price = current_price.get(signal.target_asset, 0)
        _log.info(f"[{today}]   target={signal.target_asset}, price={target_price:.2f}")

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
                log.info(
                    f"[{today}]   BUY {buy_qty} shares @ {target_price:.2f} "
                    f"(total cost ~{cost:.2f}, incl {self.commission_pct}% commission)"
                )
            else:
                log.warning(
                    f"[{today}]   Insufficient cash: have {allocatable_cash:.2f}, "
                    f"need ≥{cost_per_share:.2f}/share to buy even 1"
                )
        else:
            log.error(f"[{today}]   Target price invalid for {signal.target_asset}: {target_price}")

        # Update internal state
        if orders:
            self.current_asset = signal.target_asset
            self.current_price = target_price
            self.last_rebalance_date = today
            action_type = "bootstrap" if not has_any_position else "rebalance"
            log.info(f"[{today}] Allocation {action_type}: {signal.reason}")

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

    def _get_vix_current(self, fetcher) -> float | None:
        """Get current VIX value with daily cache (fetches incremental on stale).

        Args:
            fetcher: callable that fetches VIX DataFrame from IBKR

        Returns:
            Current VIX close value, or None if unavailable
        """
        from datetime import date as _date
        today = _date.today()

        # If cache exists and is from today, use it
        if self._vix_cache is not None and self._vix_cache_date == today:
            try:
                return float(self._vix_cache["Close"].iloc[-1])
            except Exception as e:
                _log.warning(f"Failed to extract VIX from cache: {e}")
                return None

        # Cache is stale or missing; fetch fresh (incremental append to disk)
        try:
            df = fetcher()
            if df is not None and not df.empty:
                self._vix_cache = df
                self._vix_cache_date = today
                _log.debug(f"VIX cache refreshed: {len(df)} rows, date={today}")
                return float(df["Close"].iloc[-1])
        except Exception as e:
            _log.error(f"Failed to fetch VIX: {e}")
        return None

    def _get_vxn_current(self, fetcher) -> float | None:
        """Get current VXN value with daily cache (fetches incremental on stale).

        Args:
            fetcher: callable that fetches VXN DataFrame from IBKR

        Returns:
            Current VXN close value, or None if unavailable
        """
        from datetime import date as _date
        today = _date.today()

        # If cache exists and is from today, use it
        if self._vxn_cache is not None and self._vxn_cache_date == today:
            try:
                return float(self._vxn_cache["Close"].iloc[-1])
            except Exception as e:
                _log.warning(f"Failed to extract VXN from cache: {e}")
                return None

        # Cache is stale or missing; fetch fresh (incremental append to disk)
        try:
            df = fetcher()
            if df is not None and not df.empty:
                self._vxn_cache = df
                self._vxn_cache_date = today
                _log.debug(f"VXN cache refreshed: {len(df)} rows, date={today}")
                return float(df["Close"].iloc[-1])
        except Exception as e:
            _log.error(f"Failed to fetch VXN: {e}")
        return None

    def app_status_dict(self) -> dict:
        """Return current state for app_status.json."""
        return {
            "current_asset": self.current_asset,
            "current_price": self.current_price,
            "last_rebalance_date": self.last_rebalance_date.isoformat() if self.last_rebalance_date else None,
        }
