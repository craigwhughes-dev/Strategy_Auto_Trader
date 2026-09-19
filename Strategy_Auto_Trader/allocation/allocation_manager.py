"""Multi-tier allocation manager for live daemon.

Manages rebalancing between Nasdaq/SPY/ISF.L/CSH2.L based on VXN + VIX tiers. The signal is
re-read every cycle from the latest COMPLETED hourly VIX/VXN bar (see index_feed.py), so a tier
change can happen at any hourly boundary while the LSE is open.

Tier 1 (VXN ≤ vxn_threshold to enter, held until VXN > vxn_exit_threshold): Nasdaq/EQGB.L (growth)
Tier 2 (VIX ≤ vix_tier1): SPY (balanced)
Tier 3 (vix_tier1 < VIX ≤ vix_tier2): ISF.L (defensive)
Tier 4 (otherwise): CSH2.L (money-market)

All thresholds and the VIX/VXN refresh timing come from the `tier_allocation` section of
config/overnight_strategy.json (see `from_config`); nothing is hard-coded in the daemon.

Rebalance: if target asset differs from current, generate sell (current) + buy (target) orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import logging
import time

import pandas as pd

from .index_feed import MAX_OUTAGE_SECONDS, REFRESH_SECONDS, IndexFeed

_log = logging.getLogger(__name__)

# Keys accepted in the `tier_allocation` section of config/overnight_strategy.json. Anything else
# is rejected so a typo cannot silently leave a threshold at its built-in default.
_NUMBER_KEYS = frozenset({
    "vxn_threshold", "vxn_exit_threshold", "vix_tier1", "vix_tier2",
    "index_refresh_seconds", "index_max_outage_seconds", "commission_pct",
})
_BOOL_KEYS = frozenset({"lower_tiers_enabled"})
_CONFIG_KEYS = _NUMBER_KEYS | _BOOL_KEYS

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
        commission_pct: float = 0.05,
        vxn_exit_threshold: float | None = None,
        index_refresh_seconds: float = REFRESH_SECONDS,
        index_max_outage_seconds: float = MAX_OUTAGE_SECONDS,
        lower_tiers_enabled: bool = True,
    ):
        """Initialize allocation manager.

        Args:
            vxn_threshold: VXN level to ENTER tier 1 (Nasdaq). VXN ≤ this → Nasdaq
            vix_tier1: VIX threshold for tier 2 (SPY). VIX ≤ this → SPY
            vix_tier2: VIX threshold for tier 3 (ISF.L). VIX ≤ this → ISF.L
            commission_pct: Commission per side as % of order value. IBKR UK Tiered is 0.05 (0.05%),
                min GBP 1; UK ETFs pay no stamp duty or PTM levy, so this is the whole cost
            vxn_exit_threshold: deadband upper edge. While Nasdaq is HELD it stays tier 1 until
                VXN > this. None (default) means no deadband: exit at vxn_threshold.
            index_refresh_seconds: how often VIX/VXN hourly data is re-fetched
            index_max_outage_seconds: how long a failed fetch keeps the last good reading
            lower_tiers_enabled: when False the S&P (tier 2) and FTSE (tier 3) tiers never pass, so the
                allocation is Nasdaq or cash only. Their VIX cuts stay configured for when it is switched on.
        """
        if vxn_exit_threshold is None:
            vxn_exit_threshold = vxn_threshold
        if vxn_exit_threshold < vxn_threshold:
            raise ValueError(f"vxn_exit_threshold ({vxn_exit_threshold}) must be >= vxn_threshold ({vxn_threshold})")
        if vix_tier1 >= vix_tier2:
            raise ValueError(f"vix_tier1 ({vix_tier1}) must be < vix_tier2 ({vix_tier2})")
        if commission_pct < 0:
            raise ValueError(f"commission_pct ({commission_pct}) must be >= 0")
        self.vxn_threshold = vxn_threshold
        self.vxn_exit_threshold = vxn_exit_threshold
        self.vix_tier1 = vix_tier1
        self.vix_tier2 = vix_tier2
        self.lower_tiers_enabled = lower_tiers_enabled
        self.commission_pct = commission_pct
        self.current_asset = "SPY"  # Default starting position
        self.current_price = None  # Snapshot of entry price for tracking
        self.last_rebalance_date = None
        self._last_known_cash = None  # Track cash for detecting user deposits
        self._last_cash_check_time = None  # Timestamp of last cash baseline

        # Timer-refreshed readers (not a once-a-day cache): latest completed hourly bar
        self._vix_feed = IndexFeed("VIX", index_refresh_seconds, index_max_outage_seconds)
        self._vxn_feed = IndexFeed("VXN", index_refresh_seconds, index_max_outage_seconds)

        # Last tier allocation info for app_status.json
        self._last_vix = None
        self._last_vxn = None
        self._last_tier_num = None
        self._last_target_asset = None  # Target asset selected by last signal
        self._last_action = None  # "HOLD" | "BUY" | "SELL"

    @classmethod
    def from_config(cls, section: dict | None) -> "MultiTierAllocationManager":
        """Build from the `tier_allocation` section of overnight_strategy.json.

        A missing section falls back to the built-in defaults (with a warning) so an older config
        still starts; an unknown key or non-numeric value raises, since a silently ignored typo
        would trade on the wrong threshold.
        """
        if not section:
            _log.warning("config has no `tier_allocation` section — using built-in thresholds")
            return cls()
        unknown = set(section) - _CONFIG_KEYS
        if unknown:
            raise ValueError(f"unknown tier_allocation keys {sorted(unknown)}; allowed: {sorted(_CONFIG_KEYS)}")
        for key, value in section.items():
            if key in _BOOL_KEYS:
                if not isinstance(value, bool):
                    raise ValueError(f"tier_allocation.{key} must be true or false, got {value!r}")
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"tier_allocation.{key} must be a number, got {value!r}")
        return cls(**{key: (value if key in _BOOL_KEYS else float(value)) for key, value in section.items()})

    def thresholds_summary(self) -> str:
        lower = (f"S&P VIX<={self.vix_tier1:g}; FTSE VIX<={self.vix_tier2:g}" if self.lower_tiers_enabled
                 else "S&P/FTSE tiers OFF")
        return (f"Nasdaq enter VXN<={self.vxn_threshold:g}, hold until VXN>{self.vxn_exit_threshold:g}; "
                f"{lower}; else cash")

    def _lower_tier_detail(self, vix: float | None, gate: float) -> str:
        if not self.lower_tiers_enabled:
            return "disabled by config (lower_tiers_enabled=false)"
        return f"VIX={vix:.2f} vs gate={gate}" if vix is not None else "VIX unavailable"

    def _nasdaq_gate(self) -> float:
        """VXN level tier 1 must satisfy right now: the wider exit edge while Nasdaq is already held
        (deadband, so small wobbles around the entry level do not trade), else the entry level."""
        return self.vxn_exit_threshold if self.current_asset == "EQGB.L" else self.vxn_threshold

    def signal(
        self, today: date, vxn: float | None, vix: float | None, logger=None, verbose: bool = True
    ) -> AllocationSignal:
        """Compute daily allocation decision.

        Args:
            today: Current date
            vxn: Daily VXN close (None if unavailable)
            vix: Daily VIX close (None if unavailable)
            logger: Logger to use (if None, uses module logger)
            verbose: Log the full tier-by-tier breakdown at INFO level. Set False for
                a quiet lookup (e.g. speculative pre-checks) so the breakdown appears
                only once, at the call site that actually acts on it.

        Returns:
            AllocationSignal with tier, target asset, and rebalance decision
        """
        log = logger or _log
        log_level = log.info if verbose else log.debug
        log_level(f"[{today}] Signal evaluation: VXN={vxn}, VIX={vix}, current_asset={self.current_asset}")

        # Evaluate every tier independently (each computes its own pass/fail + reason),
        # log the result, then pick the best (lowest-numbered) tier that passed.
        nasdaq_gate = self._nasdaq_gate()
        tiers = [
            (1, "EQGB.L", "Nasdaq", "VXN",
             vxn, nasdaq_gate,
             vxn is not None and vxn <= nasdaq_gate,
             f"VXN={vxn:.2f} vs gate={nasdaq_gate:g} ({'hold' if self.current_asset == 'EQGB.L' else 'enter'})"
             if vxn is not None else "VXN unavailable"),
            (2, "SPY", "Balanced", "VIX",
             vix, self.vix_tier1,
             self.lower_tiers_enabled and vix is not None and vix <= self.vix_tier1,
             self._lower_tier_detail(vix, self.vix_tier1)),
            (3, "ISF.L", "Defensive", "VIX",
             vix, self.vix_tier2,
             self.lower_tiers_enabled and vix is not None and self.vix_tier1 < vix <= self.vix_tier2,
             self._lower_tier_detail(vix, self.vix_tier2)),
            (4, "CSH2.L", "Money-Market", "none", None, None, True, "always available fallback"),
        ]

        for tier_num, asset, label, index, curr_val, gate_val, passed, detail in tiers:
            log_level(f"[{today}]   Tier {tier_num} ({asset}/{label}): {detail} [{'PASS' if passed else 'FAIL'}]")

        tier, target_asset, reason = next(
            (t, asset, f"Tier {t} ({asset}/{label}): {detail}")
            for t, asset, label, index, curr_val, gate_val, passed, detail in tiers
            if passed
        )

        log_level(f"[{today}]   → SELECTED: Tier {tier} ({target_asset})")

        needs_rebalance = target_asset != self.current_asset
        if needs_rebalance:
            log_level(f"[{today}]   ACTION: {self.current_asset} → {target_asset} (REBALANCE)")
            action = "BUY"
        else:
            log.debug(f"[{today}]   ACTION: HOLD {target_asset} (no change)")
            action = "HOLD"

        # Store tier info for app_status_dict
        self._last_vix = vix
        self._last_vxn = vxn
        self._last_tier_num = tier
        self._last_target_asset = target_asset
        self._last_action = action

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

        # Verbose tier breakdown logged here (not at any earlier speculative signal()
        # call) so it lands right next to the resulting order, not 10-20s earlier
        # across an unrelated price-fetch gap.
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
            # Update action for app_status_dict
            if len(orders) > 1:  # SELL + BUY
                self._last_action = "REBALANCE"
            elif orders[0].action == "SELL":
                self._last_action = "SELL"
            elif orders[0].action == "BUY":
                self._last_action = "BUY"

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
        """Close of the latest completed hourly VIX bar (re-fetched at most every few minutes).

        Args:
            fetcher: callable that fetches the VIX hourly DataFrame from IBKR

        Returns:
            VIX close, or None if unavailable
        """
        return self._vix_feed.current(fetcher)

    def _get_vxn_current(self, fetcher) -> float | None:
        """Close of the latest completed hourly VXN bar. VXN has no London-morning print, so before
        US open this is the prior US session's last bar."""
        return self._vxn_feed.current(fetcher)

    def app_status_dict(self) -> dict:
        """Return current state for app_status.json."""
        tier_allocation = None
        if self._last_tier_num is not None:
            # Build tier breakdown for display
            tiers = [
                {
                    "tier_num": 1,
                    "asset": "EQGB.L",
                    "label": "Nasdaq",
                    "index": "VXN",
                    "gate_value": self._nasdaq_gate(),
                    "enter_gate_value": self.vxn_threshold,
                    "exit_gate_value": self.vxn_exit_threshold,
                    "current_value": self._last_vxn,
                    "passes": self._last_vxn is not None and self._last_vxn <= self._nasdaq_gate(),
                },
                {
                    "tier_num": 2,
                    "asset": "SPY",
                    "label": "Balanced" + ("" if self.lower_tiers_enabled else " (disabled)"),
                    "index": "VIX",
                    "gate_value": self.vix_tier1,
                    "enabled": self.lower_tiers_enabled,
                    "current_value": self._last_vix,
                    "passes": (self.lower_tiers_enabled and self._last_vix is not None
                               and self._last_vix <= self.vix_tier1),
                },
                {
                    "tier_num": 3,
                    "asset": "ISF.L",
                    "label": "Defensive" + ("" if self.lower_tiers_enabled else " (disabled)"),
                    "index": "VIX",
                    "gate_value": self.vix_tier2,
                    "enabled": self.lower_tiers_enabled,
                    "current_value": self._last_vix,
                    "passes": (self.lower_tiers_enabled and self._last_vix is not None
                               and self.vix_tier1 < self._last_vix <= self.vix_tier2),
                },
                {
                    "tier_num": 4,
                    "asset": "CSH2.L",
                    "label": "Money-Market",
                    "index": "none",
                    "gate_value": None,
                    "current_value": None,
                    "passes": True,
                },
            ]

            tier_allocation = {
                "vix_current": self._last_vix,
                "vxn_current": self._last_vxn,
                "tiers": tiers,
                "selected_tier_num": self._last_tier_num,
                "selected_asset": self._last_target_asset,
                "action": self._last_action,
            }

        return {
            "current_asset": self.current_asset,
            "current_price": self.current_price,
            "last_rebalance_date": self.last_rebalance_date.isoformat() if self.last_rebalance_date else None,
            "tier_allocation": tier_allocation,
        }
