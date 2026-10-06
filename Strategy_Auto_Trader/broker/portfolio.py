"""PortfolioManager — execution_state.json I/O, sizing, and capacity checks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from math import floor
from pathlib import Path

from .types import FillResult, round_quantity
from ..core.atomic_io import atomic_write_json
from ..plugins.costs import IbkrTieredCost


def slippage_bps(signal_price: float, fill_price: float, action: str) -> float | None:
    """Execution slippage in basis points, positive = worse than signal price.

    BUY: paying above the signal close costs; SELL: filling below it costs.
    Returns None when either price is missing (dry-run brokers may fill at 0).
    """
    if signal_price <= 0 or fill_price <= 0:
        return None
    raw = (fill_price - signal_price) / signal_price * 10_000
    return round(raw if action == "BUY" else -raw, 1)


class PortfolioManager:
    """Manages a fixed capital pot across a capped number of concurrent positions.

    Reads and writes state/execution_state.json.  Separate from trade_state.json
    (which belongs to the email alert system and is not touched here).
    """

    def __init__(
        self,
        capital_pot: float,
        state_path: Path,
        currency: str = "GBP",
    ) -> None:
        self._capital_pot = capital_pot
        self._path = state_path
        self._currency = currency
        self._state: dict = self._load()

    # -- Persistence --------------------------------------------------------

    def _load(self) -> dict:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data.setdefault("positions", {})
                    data.setdefault("trade_log", [])
                    data.setdefault("interest_accrued", 0.0)
                    data.setdefault("trades_today", {
                        "date": datetime.now(timezone.utc).date().isoformat(),
                        "buys": 0,
                        "sells": 0,
                    })
                    for pos in data.get("positions", {}).values():
                        pos.setdefault("stop_perm_id", None)
                        pos.setdefault("stop_price", None)
                        pos.setdefault("stop_managed", True)
                    return data
            except Exception:
                pass
        return {
            "positions": {},
            "trade_log": [],
            "interest_accrued": 0.0,
            "trades_today": {
                "date": datetime.now(timezone.utc).date().isoformat(),
                "buys": 0,
                "sells": 0,
            },
        }

    def save(self) -> None:
        """Write current state to execution_state.json (atomically)."""
        atomic_write_json(self._path, self._state)

    # -- Read-only accessors ------------------------------------------------

    @property
    def positions(self) -> dict[str, dict]:
        return self._state["positions"]

    @property
    def trade_log(self) -> list[dict]:
        return self._state["trade_log"]

    @property
    def interest_accrued(self) -> float:
        """Total interest earned to date."""
        return self._state.get("interest_accrued", 0.0)

    @property
    def available_cash(self) -> float:
        """Cash not deployed in open positions."""
        deployed = sum(pos.get("cost_value", 0.0) for pos in self.positions.values())
        return self._capital_pot - deployed

    def get_limit_tracker(self) -> DailyLimitTracker:
        """Return a DailyLimitTracker for this portfolio's state."""
        from .daily_limits import DailyLimitTracker
        return DailyLimitTracker(self._state)

    # -- Capacity and sizing ------------------------------------------------

    def can_open(self, ticker: str) -> bool:
        """True if ticker has no open position (capacity is cash-gated only,
        via compute_quantity returning 0 when unaffordable)."""
        return ticker not in self.positions

    def compute_quantity(self, kelly_fraction: float, price: float) -> int:
        """Shares to buy: available_cash × kelly / price, floored to whole
        shares. Returns 0 if even 1 share isn't affordable — sizing must
        never round up to a purchase the pot can't cover."""
        if price <= 0 or kelly_fraction <= 0:
            return 0
        cash = self.available_cash
        if cash < price:
            return 0
        return max(1, int(floor(cash * kelly_fraction / price)))

    # -- State mutations ----------------------------------------------------

    def record_entry(
        self,
        ticker: str,
        fill: FillResult,
        kelly_fraction: float,
        stop_level: float,
        target_level: float,
        signal_price: float = 0.0,
        market: str = "",
        currency: str = "",
        stop_managed: bool = True,
    ) -> None:
        """Record a new open position after a BUY fill.

        stop_managed: False for positions that don't use a per-trade protective
        stop (e.g. tier-allocation holdings, which exit on regime change, not
        a stop-loss) — check_protective_stops() skips these entirely.
        """
        today = datetime.now(timezone.utc).date().isoformat()
        quantity = round_quantity(fill.quantity)
        cost_value = fill.fill_price * quantity
        entry_cost = IbkrTieredCost(ticker).cost(cost_value, is_buy=True)
        self._state["positions"][ticker] = {
            "entry_date": today,
            "fill_price": fill.fill_price,
            "quantity": quantity,
            "cost_value": cost_value,
            "entry_cost": entry_cost,
            "market": market,
            "currency": currency,
            "kelly_fraction": kelly_fraction,
            "stop_level": stop_level,
            "target_level": target_level,
            "stop_managed": stop_managed,
            "stop_perm_id": None,
            "stop_price": None,
        }
        self._state["trade_log"].append({
            "ticker": ticker,
            "action": "BUY",
            "date": today,
            "fill_price": fill.fill_price,
            "quantity": fill.quantity,
            "signal_price": signal_price,
            "slippage_bps": slippage_bps(signal_price, fill.fill_price, "BUY"),
            "cost": round(entry_cost, 2),
        })

    def record_exit(self, ticker: str, fill: FillResult, signal_price: float = 0.0,
                    exit_type: str = "strategy_exit") -> None:
        """Remove position and log realised P&L after a SELL fill.

        exit_type: 'strategy_exit' | 'stop_loss' | 'reconciled_stop_loss'
        """
        pos = self._state["positions"].pop(ticker, None)
        entry_price = pos["fill_price"] if pos else 0.0
        entry_cost = pos.get("entry_cost", 0.0) if pos else 0.0
        exit_value = fill.fill_price * fill.quantity
        exit_cost = IbkrTieredCost(ticker).cost(exit_value, is_buy=False)
        gross_pl = (fill.fill_price - entry_price) * fill.quantity
        pl = round(gross_pl - entry_cost - exit_cost, 2)
        log_entry = {
            "ticker": ticker,
            "action": "SELL",
            "date": datetime.now(timezone.utc).date().isoformat(),
            "fill_price": fill.fill_price,
            "quantity": fill.quantity,
            "pl": pl,
            "gross_pl": round(gross_pl, 2),
            "cost": round(entry_cost + exit_cost, 2),
            "signal_price": signal_price,
            "slippage_bps": slippage_bps(signal_price, fill.fill_price, "SELL"),
            "exit_type": exit_type,
        }
        if exit_type != "strategy_exit" and pos:
            log_entry["stop_price"] = pos.get("stop_price")
        self._state["trade_log"].append(log_entry)

    def record_tier_fill(self, ticker: str, fill: FillResult, action: str, market: str, currency: str) -> None:
        """Apply a confirmed min-hold fill to the tier's ledger position.

        A BUY tops up the average-cost position (or opens it); a SELL reduces it and closes it at zero.
        Tier holdings have no protective stop, so the position stays stop_managed=False.
        """
        pos = self._state["positions"].get(ticker)
        if action == "BUY":
            if pos is None:
                self.record_entry(ticker, fill, kelly_fraction=1.0, stop_level=0.0, target_level=0.0,
                                  market=market, currency=currency, stop_managed=False)
                return
            quantity = round_quantity(fill.quantity)
            cost_value = fill.fill_price * quantity
            entry_cost = IbkrTieredCost(ticker).cost(cost_value, is_buy=True)
            pos["cost_value"] += cost_value
            pos["entry_cost"] += entry_cost
            pos["quantity"] = round_quantity(pos["quantity"] + quantity)
            pos["fill_price"] = pos["cost_value"] / pos["quantity"]
            self._log_tier_trade(ticker, "BUY", fill, quantity, entry_cost, pl=None)
            return

        if pos is None:
            return  # caller logs this: the broker sold a tier position the ledger never held
        quantity = round_quantity(min(fill.quantity, pos["quantity"]))
        share_of_entry_cost = pos["entry_cost"] * quantity / pos["quantity"]
        exit_cost = IbkrTieredCost(ticker).cost(fill.fill_price * quantity, is_buy=False)
        gross_pl = (fill.fill_price - pos["fill_price"]) * quantity
        self._log_tier_trade(ticker, "SELL", fill, quantity,
                             share_of_entry_cost + exit_cost, pl=round(gross_pl - share_of_entry_cost - exit_cost, 2))
        remaining = round_quantity(pos["quantity"] - quantity)
        if remaining <= 0:
            del self._state["positions"][ticker]
            return
        pos["quantity"] = remaining
        pos["cost_value"] = pos["fill_price"] * remaining
        pos["entry_cost"] -= share_of_entry_cost

    def sync_tier_positions(
        self,
        broker_units: dict[str, float],
        prices: dict[str, float | None],
        tier_assets: list[str],
        market: str,
        currency: str,
    ) -> list[str]:
        """Make the ledger's tier quantities equal the broker's; the broker is the source of truth.

        Corrects drift from fills the daemon could not confirm (an order that filled after a 429 is one),
        so reconciliation does not halt entries over it. A new position is opened at the current price,
        because the cost basis of a holding the ledger never recorded is unknown. Returns one line per change.
        """
        notes: list[str] = []
        today = datetime.now(timezone.utc).date().isoformat()
        for ticker in tier_assets:
            held = round_quantity(broker_units.get(ticker) or 0.0)
            pos = self._state["positions"].get(ticker)
            ledger = pos["quantity"] if pos else 0.0
            if round_quantity(held - ledger) == 0:
                continue
            if held <= 0:
                del self._state["positions"][ticker]
                notes.append(f"{ticker}: ledger {ledger} → broker 0, position removed")
            elif pos is not None:
                pos["quantity"] = held
                pos["cost_value"] = pos["fill_price"] * held
                notes.append(f"{ticker}: ledger {ledger} → broker {held}")
            else:
                price = prices.get(ticker)
                if not price or price <= 0:
                    notes.append(f"{ticker}: broker holds {held} but no price — not adopted")
                    continue
                self._state["positions"][ticker] = {
                    "entry_date": today,
                    "fill_price": price,
                    "quantity": held,
                    "cost_value": price * held,
                    "entry_cost": 0.0,
                    "market": market,
                    "currency": currency,
                    "kelly_fraction": 1.0,
                    "stop_level": 0.0,
                    "target_level": 0.0,
                    "stop_managed": False,
                    "stop_perm_id": None,
                    "stop_price": None,
                }
                notes.append(f"{ticker}: adopted {held} at {price} (broker holds, ledger empty)")
        return notes

    def _log_tier_trade(self, ticker: str, action: str, fill: FillResult, quantity: float,
                        cost: float, pl: float | None) -> None:
        entry = {
            "ticker": ticker,
            "action": action,
            "date": datetime.now(timezone.utc).date().isoformat(),
            "fill_price": fill.fill_price,
            "quantity": quantity,
            "cost": round(cost, 2),
        }
        if pl is not None:
            entry["pl"] = pl
        self._state["trade_log"].append(entry)

    def set_stop_order(self, ticker: str, perm_id: int, stop_price: float) -> None:
        """Record a placed protective stop order for an open position."""
        if ticker in self._state["positions"]:
            self._state["positions"][ticker]["stop_perm_id"] = perm_id
            self._state["positions"][ticker]["stop_price"] = stop_price

    def clear_stop_order(self, ticker: str) -> None:
        """Clear stop order tracking for a position (when it's cancelled or executed)."""
        if ticker in self._state["positions"]:
            self._state["positions"][ticker]["stop_perm_id"] = None
            self._state["positions"][ticker]["stop_price"] = None

