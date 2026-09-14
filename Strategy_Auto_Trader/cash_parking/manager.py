"""CashParkingManager — idle-cash tier rebalancing for the live daemon.

Called once per cycle after execute_signals(), when both pbull_smooth and VIX
are available. Returns 0-2 OrderRequests; the caller places them.

State is persisted to state/cash_parking_state.json so T+2 settlement blocks
and current holding survive daemon restarts.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

from ..broker.types import OrderRequest
from .instruments import PARKING_TICKERS
from .strategy import (
    LIQUID_FLOOR_PCT,
    MIN_PARKING_AMOUNT,
    REBALANCE_THRESHOLD_PCT,
    tier_for,
)

logger = logging.getLogger(__name__)

_BROKER_SYMBOLS_PER_POUND = 100.0  # LSE prices quoted in pence; pot is in £

_STATE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "state" / "cash_parking_state.json"
)


@dataclass
class ParkingState:
    tier: str = "cash"
    ticker: str | None = None     # yfinance (.L) ticker currently held
    quantity: int = 0
    entry_price: float = 0.0      # GBP per share at time of buy
    settling_until: str | None = None  # ISO date; buy blocked until this date passes


class CashParkingManager:
    """Manages idle-cash allocation across XSTR/IGLS/ISXF/ISF based on regime signals.

    One instance per daemon run. Persists state to disk so T+2 settlement gates
    survive restarts.
    """

    def __init__(
        self,
        initial_cash: float,
        liquid_floor_pct: float = LIQUID_FLOOR_PCT,
        state_path: Path | None = None,
    ) -> None:
        self._initial_cash = initial_cash
        self._liquid_floor_pct = liquid_floor_pct
        self._state_path = state_path or _STATE_PATH
        self._state = self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rebalance(
        self,
        broker: object,
        available_cash: float,
        pbull_smooth: float,
        vix_level: float,
        current_positions: dict,
        today: date,
    ) -> list[OrderRequest]:
        """Compute required tier change and return orders (0-2).

        Gates (in order):
        1. Liquid floor — only the investable fraction can be parked.
        2. Min amount — skip if investable cash < MIN_PARKING_AMOUNT.
        3. Dedup — if equity tier and ISF.L already in main positions, degrade to gilts.
        4. Same ticker — no change if target equals current holding.
        5. Rebalance threshold — only switch if allocation delta > threshold * pot.
        6. T+2 block — can sell an existing position, but new buy deferred until settled.
        """
        desired_tier = tier_for(pbull_smooth, vix_level)

        # Dedup: ISF.L held by main strategy → downgrade equity to hy_bonds instead
        if desired_tier == "equity" and "ISF.L" in current_positions:
            desired_tier = "hy_bonds"

        desired_ticker = PARKING_TICKERS[desired_tier]

        parkable = available_cash * (1 - self._liquid_floor_pct)

        if parkable < MIN_PARKING_AMOUNT:
            logger.debug(
                f"CashParking: parkable £{parkable:.0f} < MIN_PARKING_AMOUNT — skipping"
            )
            return []

        current_ticker = self._state.ticker
        current_qty = self._state.quantity

        # Same holding — check if a material buy-more opportunity exists
        if current_ticker == desired_ticker:
            if current_qty > 0:
                logger.debug(
                    f"CashParking: already in {current_ticker} (tier={desired_tier}) — no change"
                )
                return []
            # Not yet holding anything in this tier — fall through to buy

        # Is switching worth the friction?
        if not self._exceeds_threshold(parkable, current_ticker, current_qty, broker):
            logger.debug(
                f"CashParking: delta below threshold — staying in "
                f"{current_ticker or 'nothing'}"
            )
            return []

        orders: list[OrderRequest] = []

        # Sell existing parking position (if any)
        if current_ticker and current_qty > 0:
            orders.append(OrderRequest(current_ticker, "SELL", current_qty))
            logger.info(
                f"CashParking: SELL {current_qty}x {current_ticker} "
                f"(switching {self._state.tier} → {desired_tier})"
            )

        # Block new buy if T+2 settlement still in progress
        if self._state.settling_until is not None:
            settling_date = date.fromisoformat(self._state.settling_until)
            if today <= settling_date:
                logger.info(
                    f"CashParking: T+2 gate — sell settling until {self._state.settling_until}, "
                    f"buy deferred"
                )
                return orders

        # Size the buy
        price_gbp = self._get_price_gbp(broker, desired_ticker)
        if price_gbp <= 0:
            logger.warning(
                f"CashParking: cannot get price for {desired_ticker} — buy skipped"
            )
            return orders

        qty = max(0, math.floor(parkable / price_gbp))
        if qty < 1:
            logger.debug(
                f"CashParking: parkable £{parkable:.0f} / price £{price_gbp:.2f} = {qty} — too few shares"
            )
            return orders

        orders.append(OrderRequest(desired_ticker, "BUY", qty))
        logger.info(
            f"CashParking: BUY {qty}x {desired_ticker} @ ~£{price_gbp:.2f} "
            f"(tier={desired_tier}, parkable=£{parkable:.0f})"
        )

        # Optimistically update state (actual fill confirmed by record_* methods
        # if the caller wires those up, but we need tier/ticker for next cycle).
        self._state.tier = desired_tier
        self._state.ticker = desired_ticker
        self._state.quantity = qty
        self._state.entry_price = price_gbp
        self._save_state()

        return orders

    def record_sell_settled(self, ticker: str, sell_date: date) -> None:
        """Call after a parking SELL order is placed.

        Sets settling_until to sell_date + 2 business days, blocking the
        subsequent BUY until settlement clears.
        """
        settling = _add_business_days(sell_date, 2)
        self._state.settling_until = settling.isoformat()
        self._state.ticker = None
        self._state.quantity = 0
        self._state.entry_price = 0.0
        self._save_state()
        logger.info(
            f"CashParking: sold {ticker}, settling {settling.isoformat()} — buy blocked until then"
        )

    def record_buy_filled(self, ticker: str, qty: int, fill_price_gbp: float) -> None:
        """Update state after a BUY fill is confirmed."""
        self._state.ticker = ticker
        self._state.quantity = qty
        self._state.entry_price = fill_price_gbp
        self._state.settling_until = None
        self._save_state()

    def app_status_dict(self) -> dict:
        """Returns dict for inclusion in app_status.json under 'cash_parking'."""
        return {
            "tier": self._state.tier,
            "ticker": self._state.ticker,
            "qty": self._state.quantity,
            "entry_price_gbp": self._state.entry_price,
            "settling_until": self._state.settling_until,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _exceeds_threshold(
        self,
        parkable: float,
        current_ticker: str | None,
        current_qty: int,
        broker: object,
    ) -> bool:
        """True if switching allocation would shift more than REBALANCE_THRESHOLD_PCT of pot."""
        if current_ticker is None or current_qty == 0:
            return True  # no current position, any buy is a material change

        current_value = self._get_price_gbp(broker, current_ticker) * current_qty
        delta = abs(parkable - current_value)
        threshold = REBALANCE_THRESHOLD_PCT * self._initial_cash
        return delta > threshold

    @staticmethod
    def _get_price_gbp(broker: object, ticker: str) -> float:
        """Fetch current price for a parking ticker, converted to GBP."""
        try:
            price_raw = broker.get_last_price(ticker)
            if price_raw and price_raw > 0:
                # LSE prices from IBKR are in pence; convert to pounds
                if ticker.upper().endswith(".L"):
                    return price_raw / _BROKER_SYMBOLS_PER_POUND
                return float(price_raw)
        except Exception as exc:
            logger.warning(f"CashParking: price fetch failed for {ticker}: {exc}")
        return 0.0

    def _load_state(self) -> ParkingState:
        try:
            if self._state_path.exists():
                data = json.loads(self._state_path.read_text(encoding="utf-8"))
                return ParkingState(**data)
        except Exception as exc:
            logger.warning(f"CashParking: could not load state ({exc}), starting fresh")
        return ParkingState()

    def _save_state(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps(asdict(self._state), indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning(f"CashParking: could not save state: {exc}")


def _add_business_days(d: date, n: int) -> date:
    """Add n business days (Mon-Fri) to date d."""
    result = d
    added = 0
    while added < n:
        result += timedelta(days=1)
        if result.weekday() < 5:  # Mon=0..Fri=4
            added += 1
    return result
