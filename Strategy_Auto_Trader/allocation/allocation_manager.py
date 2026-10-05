"""Multi-tier allocation manager for live daemon.

Manages rebalancing between Nasdaq/VUSA/ISF.L/CSH2.L based on VXN + VIX tiers. The signal is
re-read every cycle from the latest COMPLETED hourly VIX/VXN bar (see index_feed.py), so a tier
change can happen at any hourly boundary while the LSE is open.

Tier 1 (VXN ≤ gate to enter, held until VXN > exit gate): Nasdaq/EQGB.L (growth). The (enter, exit)
    VXN pair is picked by the day's confirmed VVIX band (see live_daemon.py's `update_vvix_band`,
    which owns the N-consecutive-day confirmation and is out of this class's scope — this class
    only picks the pair once told which band applies): 'calm' -> (vxn_calm_enter, vxn_calm_exit)
    (wider, more permissive), 'base' -> (vxn_threshold, vxn_exit_threshold) (the original rule,
    unaffected by VVIX), 'stressed' -> (vxn_stressed_enter, vxn_stressed_exit) (narrower). A
    daemon never passing `vvix_band` (or always passing "base") reproduces the pre-VVIX behavior
    exactly — this is a strict superset, not a replacement.
Tier 2 (VIX ≤ vix_tier1): VUSA (balanced)
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
import math
import time

import pandas as pd

from .index_feed import MAX_OUTAGE_SECONDS, REFRESH_SECONDS, IndexFeed

_log = logging.getLogger(__name__)

# Keys accepted in the `tier_allocation` section of config/overnight_strategy.json. Anything else
# is rejected so a typo cannot silently leave a threshold at its built-in default.
_NUMBER_KEYS = frozenset({
    "vxn_threshold", "vxn_exit_threshold", "vix_tier1", "vix_tier2",
    "index_refresh_seconds", "index_max_outage_seconds", "commission_pct",
    "vvix_edge_low", "vvix_edge_high", "vvix_confirm_days",
    "vxn_calm_enter", "vxn_calm_exit", "vxn_stressed_enter", "vxn_stressed_exit",
    "min_hold_gbp",
})
_BOOL_KEYS = frozenset({"lower_tiers_enabled"})
_CONFIG_KEYS = _NUMBER_KEYS | _BOOL_KEYS

TIER_ASSETS = {1: "EQGB.L", 2: "VUSA", 3: "ISF.L", 4: "CSH2.L"}
ASSET_TIERS = {"EQGB.L": 1, "VUSA": 2, "ISF.L": 3, "CSH2.L": 4}
# Smallest fractional trade the allocator places (2 dp, accepted by T212 for every mapped tier asset)
MIN_TRADE_UNITS = 0.01
# Share of allocatable cash the allocator leaves unspent, for ask-over-quote fills on market buys
MIN_HOLD_CASH_HEADROOM = 0.01


@dataclass
class AllocationOrder:
    """Single buy/sell order for rebalancing. Fractional units (T212)."""
    ticker: str
    action: str  # "BUY" | "SELL"
    quantity: float
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
    vvix_band: str = "base"  # "calm" | "base" | "stressed" — which VXN pair tier 1 used


class MultiTierAllocationManager:
    """Manage daily 4-tier allocation rebalances with VXN + VIX gating.

    Tier 1: Nasdaq/EQGB.L (VXN ≤ vxn_threshold)
    Tier 2: VUSA (VIX ≤ vix_tier1)
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
        vvix_edge_low: float = 76.0,
        vvix_edge_high: float = 122.0,
        vvix_confirm_days: float = 3.0,
        vxn_calm_enter: float | None = None,
        vxn_calm_exit: float | None = None,
        vxn_stressed_enter: float | None = None,
        vxn_stressed_exit: float | None = None,
        min_hold_gbp: float = 10.0,
    ):
        """Initialize allocation manager.

        Args:
            vxn_threshold: VXN level to ENTER tier 1 (Nasdaq). VXN ≤ this → Nasdaq. This is also
                the 'base' band's enter level (see vvix_* args below) — unaffected by VVIX.
            vix_tier1: VIX threshold for tier 2 (VUSA). VIX ≤ this → VUSA
            vix_tier2: VIX threshold for tier 3 (ISF.L). VIX ≤ this → ISF.L
            commission_pct: Commission per side as % of order value. IBKR UK Tiered is 0.05 (0.05%),
                min GBP 1; UK ETFs pay no stamp duty or PTM levy, so this is the whole cost
            vxn_exit_threshold: deadband upper edge. While Nasdaq is HELD it stays tier 1 until
                VXN > this. None (default) means no deadband: exit at vxn_threshold. Also the
                'base' band's exit level.
            index_refresh_seconds: how often VIX/VXN hourly data is re-fetched
            index_max_outage_seconds: how long a failed fetch keeps the last good reading
            lower_tiers_enabled: when False the S&P (tier 2) and FTSE (tier 3) tiers never pass, so the
                allocation is Nasdaq or cash only. Their VIX cuts stay configured for when it is switched on.
            vvix_edge_low: VVIX level below which (once confirmed, see live_daemon.py) the 'calm'
                band applies to tier 1's VXN pair. Only has an effect when `signal()`/`rebalance()`
                are called with vvix_band != "base" — otherwise unused.
            vvix_edge_high: VVIX level at/above which (once confirmed) the 'stressed' band applies.
            vvix_confirm_days: consecutive daily VVIX readings past an edge required before the
                calm/stressed band is considered confirmed (live_daemon.py owns the actual
                counting; this class just stores the threshold for it to read).
            vxn_calm_enter / vxn_calm_exit: tier 1's (enter, exit) VXN pair while the 'calm' VVIX
                band is confirmed — wider than vxn_threshold/vxn_exit_threshold (more permissive).
                None (default) falls back to vxn_threshold/vxn_exit_threshold unchanged, i.e. VVIX
                has no effect unless these are explicitly configured.
            vxn_stressed_enter / vxn_stressed_exit: tier 1's pair while 'stressed' is confirmed —
                narrower than the base pair. Same None-means-no-effect default as the calm pair.
            min_hold_gbp: every tier asset is held at this GBP value and the target takes the rest
                (fractional units, see `_rebalance_tiers`). Must be > 0.
        """
        if min_hold_gbp <= 0:
            raise ValueError(f"min_hold_gbp ({min_hold_gbp}) must be > 0")
        if vxn_exit_threshold is None:
            vxn_exit_threshold = vxn_threshold
        if vxn_exit_threshold < vxn_threshold:
            raise ValueError(f"vxn_exit_threshold ({vxn_exit_threshold}) must be >= vxn_threshold ({vxn_threshold})")
        if vix_tier1 >= vix_tier2:
            raise ValueError(f"vix_tier1 ({vix_tier1}) must be < vix_tier2 ({vix_tier2})")
        if commission_pct < 0:
            raise ValueError(f"commission_pct ({commission_pct}) must be >= 0")
        if vvix_edge_high <= vvix_edge_low:
            raise ValueError(f"vvix_edge_high ({vvix_edge_high}) must be > vvix_edge_low ({vvix_edge_low})")
        if vvix_confirm_days < 1:
            raise ValueError(f"vvix_confirm_days ({vvix_confirm_days}) must be >= 1")
        if vxn_calm_enter is None:
            vxn_calm_enter = vxn_threshold
        if vxn_calm_exit is None:
            vxn_calm_exit = vxn_exit_threshold
        if vxn_calm_exit < vxn_calm_enter:
            raise ValueError(f"vxn_calm_exit ({vxn_calm_exit}) must be >= vxn_calm_enter ({vxn_calm_enter})")
        if vxn_stressed_enter is None:
            vxn_stressed_enter = vxn_threshold
        if vxn_stressed_exit is None:
            vxn_stressed_exit = vxn_exit_threshold
        if vxn_stressed_exit < vxn_stressed_enter:
            raise ValueError(f"vxn_stressed_exit ({vxn_stressed_exit}) must be >= vxn_stressed_enter ({vxn_stressed_enter})")
        self.vxn_threshold = vxn_threshold
        self.vxn_exit_threshold = vxn_exit_threshold
        self.vix_tier1 = vix_tier1
        self.vix_tier2 = vix_tier2
        self.lower_tiers_enabled = lower_tiers_enabled
        self.commission_pct = commission_pct
        self.vvix_edge_low = vvix_edge_low
        self.vvix_edge_high = vvix_edge_high
        self.vvix_confirm_days = vvix_confirm_days
        self.vxn_calm_enter = vxn_calm_enter
        self.vxn_calm_exit = vxn_calm_exit
        self.vxn_stressed_enter = vxn_stressed_enter
        self.vxn_stressed_exit = vxn_stressed_exit
        self.min_hold_gbp = min_hold_gbp
        self.current_asset = "VUSA"  # Default starting position
        self.current_price = None  # Snapshot of entry price for tracking
        self.last_rebalance_date = None
        self._last_known_cash = None  # Track cash for detecting user deposits
        self._last_cash_check_time = None  # Timestamp of last cash baseline
        # Min-hold only: broker cash the allocator may spend. None = all broker cash.
        self.cash_budget: float | None = None

        # Timer-refreshed readers (not a once-a-day cache): latest completed hourly bar
        self._vix_feed = IndexFeed("VIX", index_refresh_seconds, index_max_outage_seconds)
        self._vxn_feed = IndexFeed("VXN", index_refresh_seconds, index_max_outage_seconds)

        # Last tier allocation info for app_status.json
        self._last_vix = None
        self._last_vxn = None
        self._last_vvix_band = "base"
        self._last_tier_num = None
        self._last_target_asset = None  # Target asset selected by last signal
        self._last_action = None  # "HOLD" | "BUY" | "SELL"
        self._data_outage = False  # both indices lost for longer than the outage limit
        self.outage_limit_hours = index_max_outage_seconds / 3600

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
        vvix_note = (
            f"; VVIX band: calm {self.vxn_calm_enter:g}/{self.vxn_calm_exit:g}, "
            f"base {self.vxn_threshold:g}/{self.vxn_exit_threshold:g}, "
            f"stressed {self.vxn_stressed_enter:g}/{self.vxn_stressed_exit:g} "
            f"(edges {self.vvix_edge_low:g}/{self.vvix_edge_high:g})"
            if (self.vxn_calm_enter, self.vxn_calm_exit, self.vxn_stressed_enter, self.vxn_stressed_exit)
            != (self.vxn_threshold, self.vxn_exit_threshold, self.vxn_threshold, self.vxn_exit_threshold)
            else ""
        )
        return (f"Nasdaq enter VXN<={self.vxn_threshold:g}, hold until VXN>{self.vxn_exit_threshold:g}; "
                f"{lower}; else cash{vvix_note}")

    def _lower_tier_detail(self, vix: float | None, gate: float) -> str:
        if not self.lower_tiers_enabled:
            return "disabled by config (lower_tiers_enabled=false)"
        return f"VIX={vix:.2f} vs gate={gate}" if vix is not None else "VIX unavailable"

    def _nasdaq_pair(self, vvix_band: str) -> tuple[float, float]:
        """(enter, exit) VXN pair for the given confirmed VVIX band. 'base' always reproduces the
        plain vxn_threshold/vxn_exit_threshold rule, regardless of any vvix_* config — a daemon
        that never passes a non-"base" band trades identically to before this feature existed."""
        if vvix_band == "calm":
            return self.vxn_calm_enter, self.vxn_calm_exit
        if vvix_band == "stressed":
            return self.vxn_stressed_enter, self.vxn_stressed_exit
        if vvix_band != "base":
            raise ValueError(f"unknown vvix_band {vvix_band!r}; expected 'calm', 'base', or 'stressed'")
        return self.vxn_threshold, self.vxn_exit_threshold

    def _nasdaq_gate(self, vvix_band: str = "base") -> float:
        """VXN level tier 1 must satisfy right now, for the given confirmed VVIX band: the wider
        exit edge while Nasdaq is already held (deadband, so small wobbles around the entry level
        do not trade), else the entry level."""
        enter_at, exit_above = self._nasdaq_pair(vvix_band)
        return exit_above if self.current_asset == "EQGB.L" else enter_at

    def signal(
        self,
        today: date,
        vxn: float | None,
        vix: float | None,
        vvix_band: str = "base",
        logger=None,
        verbose: bool = True,
    ) -> AllocationSignal:
        """Compute daily allocation decision.

        Args:
            today: Current date
            vxn: Daily VXN close (None if unavailable)
            vix: Daily VIX close (None if unavailable)
            vvix_band: "calm" | "base" | "stressed" — today's confirmed VVIX regime, owned and
                computed by the caller (live_daemon.py's `update_vvix_band`, which persists the
                N-consecutive-day confirmation across restarts). This class does not fetch VVIX or
                track confirmation state itself — it only picks the VXN pair for the band it's
                told. Default "base" reproduces pre-VVIX behavior exactly.
            logger: Logger to use (if None, uses module logger)
            verbose: Log the full tier-by-tier breakdown at INFO level. Set False for
                a quiet lookup (e.g. speculative pre-checks) so the breakdown appears
                only once, at the call site that actually acts on it.

        Returns:
            AllocationSignal with tier, target asset, and rebalance decision
        """
        log = logger or _log
        log_level = log.info if verbose else log.debug
        log_level(
            f"[{today}] Signal evaluation: VXN={vxn}, VIX={vix}, vvix_band={vvix_band}, "
            f"current_asset={self.current_asset}"
        )

        # Evaluate every tier independently (each computes its own pass/fail + reason),
        # log the result, then pick the best (lowest-numbered) tier that passed.
        nasdaq_gate = self._nasdaq_gate(vvix_band)
        tiers = [
            (1, "EQGB.L", "Nasdaq", "VXN",
             vxn, nasdaq_gate,
             vxn is not None and vxn <= nasdaq_gate,
             f"VXN={vxn:.2f} vs gate={nasdaq_gate:g} [{vvix_band}] "
             f"({'hold' if self.current_asset == 'EQGB.L' else 'enter'})"
             if vxn is not None else "VXN unavailable"),
            (2, "VUSA", "Balanced", "VIX",
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
        self._last_vvix_band = vvix_band
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
            vvix_band=vvix_band,
        )

    def rebalance(
        self,
        today: date,
        vxn: float | None,
        vix: float | None,
        current_price: dict[str, float],
        available_cash: float,
        positions: dict[str, int],
        vvix_band: str = "base",
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
            vvix_band: "calm" | "base" | "stressed" — see `signal()`. Default "base" is a no-op.
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
        signal = self.signal(today, vxn, vix, vvix_band=vvix_band, logger=log)

        return self._rebalance_tiers(today, signal, current_price, allocatable_cash, positions, log)

    def enabled_assets(self) -> list[str]:
        """Tier assets the strategy trades. The S&P and FTSE tiers are off unless lower_tiers_enabled."""
        return [a for a, tier in ASSET_TIERS.items() if self.lower_tiers_enabled or tier in (1, 4)]

    def adopt_held_tier(self, positions: dict[str, float], prices: dict[str, float | None], log=None) -> None:
        """Set current_asset to the enabled tier holding the most value, as the broker reports it.

        Min-hold keeps every enabled tier at the minimum, so several tiers are held at once. The
        target holds the rest, so it is the largest. Disabled tiers are ignored entirely. If an
        enabled held tier has no price, the value cannot be compared, so current_asset is left unchanged.
        """
        log = log or _log
        enabled = self.enabled_assets()
        held = {a: float(positions.get(a) or 0) for a in enabled if float(positions.get(a) or 0) > 0}
        if not held:
            return
        unpriced = [a for a in held if not prices.get(a) or prices[a] <= 0]
        if unpriced:
            log.warning(f"Allocation: held tier(s) {unpriced} unpriced at startup — current_asset left at {self.current_asset}")
            return
        values = {a: units * prices[a] for a, units in held.items()}
        self.current_asset = max(values, key=values.get)
        log.info(f"Allocation: adopted held tier {self.current_asset} (value £{values[self.current_asset]:.2f}) from broker")

    def release_cash(self, amount: float) -> None:
        """Set the cash the min-hold allocator may spend, until the next release (daily start)."""
        self.cash_budget = max(0.0, amount)

    def spendable_cash(self, broker_cash: float) -> float:
        """Broker cash capped at the released budget. Cash that landed after the release waits."""
        if self.cash_budget is None:
            return broker_cash
        return min(broker_cash, self.cash_budget)

    def note_fill(self, action: str, value: float) -> None:
        """Track a min-hold fill against the budget: buys spend it, sells replenish it."""
        if self.cash_budget is None:
            return
        net = value * (1 - self.commission_pct / 100)
        self.cash_budget = max(0.0, self.cash_budget + (net if action == "SELL" else -value * (1 + self.commission_pct / 100)))

    def _rebalance_tiers(
        self,
        today: date,
        signal: AllocationSignal,
        current_price: dict[str, float],
        allocatable_cash: float,
        positions: dict[str, float],
        log,
    ) -> list[AllocationOrder]:
        """Rebalance tier assets: every tier asset is held at `min_hold_gbp`, the target takes the rest.

        On a tier change the other assets are trimmed or topped to the minimum. Otherwise they keep any
        value above it, and only the shortfall below it is bought. Either way the target takes what the
        others leave, so spare cash is invested and daily price drift does not trade.
        Quantities are rounded to 2 dp, which T212 accepts for every mapped tier asset. Sells come
        before buys so the cash they free funds the buys.
        """
        enabled = self.enabled_assets()
        missing = [a for a in enabled if not current_price.get(a) or current_price[a] <= 0]
        if missing:
            log.warning(f"[{today}] min-hold rebalance: no price for enabled {missing} — skipping")
            return []

        # A disabled tier with no price is left out entirely: it can neither block nor be traded
        priced = [a for a in ASSET_TIERS if current_price.get(a) and current_price[a] > 0]
        held_units = {a: float(positions.get(a, 0) or 0) for a in priced}
        held_value = {a: held_units[a] * current_price[a] for a in priced}
        total_value = allocatable_cash + sum(held_value.values())
        target = signal.target_asset
        others = [a for a in priced if a != target]

        # Enabled others sit at the minimum on a tier change. Disabled tiers are only ever topped up to the
        # minimum, never trimmed. Otherwise others keep any value above it. The target takes what the others
        # leave, so spare cash is invested without trading on price drift.
        desired = {}
        for a in others:
            if a in enabled and signal.needs_rebalance:
                desired[a] = self.min_hold_gbp
            else:
                desired[a] = max(held_value[a], self.min_hold_gbp)
        desired[target] = max(total_value - sum(desired.values()), 0.0)

        trades = {}
        for asset in priced:
            units = round((desired[asset] - held_value[asset]) / current_price[asset], 2)
            if abs(units) >= MIN_TRADE_UNITS:
                trades[asset] = units

        orders: list[AllocationOrder] = []
        # Market buys fill at the ask, above the quoted price, and T212 rejects a buy that exceeds free cash
        budget = allocatable_cash * (1 - MIN_HOLD_CASH_HEADROOM)
        for asset, units in trades.items():
            if units < 0 and asset in enabled:  # disabled tiers are never trimmed
                orders.append(AllocationOrder(
                    ticker=asset, action="SELL", quantity=-units, limit_price=None,
                    reason=f"Min-hold: trim {asset} to £{self.min_hold_gbp:g}",
                ))
                budget += -units * current_price[asset] * (1 - self.commission_pct / 100)

        # Minimum top-ups first, target last, so the target cannot spend the cash the minimums need
        buy_order = sorted(trades, key=lambda a: a == target)
        for asset in buy_order:
            units = trades[asset]
            if units <= 0:
                continue
            unit_cost = current_price[asset] * (1 + self.commission_pct / 100)
            affordable = math.floor(min(units, budget / unit_cost) * 100) / 100
            if affordable < MIN_TRADE_UNITS:
                log.warning(f"[{today}]   Insufficient cash for {asset}: budget {budget:.2f}, need {units * unit_cost:.2f}")
                continue
            orders.append(AllocationOrder(
                ticker=asset, action="BUY", quantity=affordable, limit_price=None,
                reason=f"Min-hold: {'enter' if asset == target else 'top up'} {asset} (tier {ASSET_TIERS[asset]})",
            ))
            budget -= affordable * unit_cost

        if orders:
            self.current_asset = target
            self.current_price = current_price[target]
            self.last_rebalance_date = today
            self._last_action = "REBALANCE" if len(orders) > 1 else orders[0].action
            log.info(f"[{today}] Min-hold rebalance: {signal.reason}; orders={[(o.action, o.ticker, o.quantity) for o in orders]}")
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

    def data_outage(self) -> tuple[bool, bool]:
        """(outage active, outage newly started this call).

        An outage is both VIX and VXN having had no successful fetch for longer than the feeds' outage limit.
        While active, the daemon must still call rebalance() with both readings None so the signal falls to
        cash; when either feed recovers the flag clears and the normal rule resumes from cash.
        """
        lost = self._vix_feed.is_lost() and self._vxn_feed.is_lost()
        newly = lost and not self._data_outage
        self._data_outage = lost
        return lost, newly

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

    def app_status_dict(self, daemon_state: dict | None = None) -> dict:
        """Return current state for app_status.json.

        Args:
            daemon_state: live_daemon's persisted state dict, read-only here, for the raw VVIX
                reading and streak counters `update_vvix_band` writes there (this class only
                tracks the confirmed band it was told, not VVIX itself). None (e.g. in tests that
                construct this class standalone) omits the `vvix` block's daemon_state-sourced
                fields rather than raising.
        """
        tier_allocation = None
        if self._last_tier_num is not None:
            # Build tier breakdown for display
            enter_at, exit_above = self._nasdaq_pair(self._last_vvix_band)
            tiers = [
                {
                    "tier_num": 1,
                    "asset": "EQGB.L",
                    "label": "Nasdaq",
                    "index": "VXN",
                    "gate_value": self._nasdaq_gate(self._last_vvix_band),
                    "enter_gate_value": enter_at,
                    "exit_gate_value": exit_above,
                    "current_value": self._last_vxn,
                    "vvix_band": self._last_vvix_band,
                    "passes": self._last_vxn is not None and self._last_vxn <= self._nasdaq_gate(self._last_vvix_band),
                },
                {
                    "tier_num": 2,
                    "asset": "VUSA",
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
                "vvix_band": self._last_vvix_band,
                "tiers": tiers,
                "selected_tier_num": self._last_tier_num,
                "selected_asset": self._last_target_asset,
                "action": self._last_action,
                "vvix": {
                    "current": (daemon_state or {}).get("vvix_last_value"),
                    "band": self._last_vvix_band,
                    "calm_streak_days": (daemon_state or {}).get("vvix_calm_streak_days", 0),
                    "stressed_streak_days": (daemon_state or {}).get("vvix_stressed_streak_days", 0),
                    "confirm_days": self.vvix_confirm_days,
                    "edge_low": self.vvix_edge_low,
                    "edge_high": self.vvix_edge_high,
                    "last_date": (daemon_state or {}).get("vvix_last_date"),
                    "vxn_thresholds": {
                        "calm": {"enter": self.vxn_calm_enter, "exit": self.vxn_calm_exit},
                        "base": {"enter": self.vxn_threshold, "exit": self.vxn_exit_threshold},
                        "stressed": {"enter": self.vxn_stressed_enter, "exit": self.vxn_stressed_exit},
                    },
                },
            }

        return {
            "current_asset": self.current_asset,
            "current_price": self.current_price,
            "last_rebalance_date": self.last_rebalance_date.isoformat() if self.last_rebalance_date else None,
            "data_outage": self._data_outage,
            "tier_allocation": tier_allocation,
        }
