"""PortfolioManager — execution_state.json I/O, sizing, and capacity checks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from math import floor
from pathlib import Path

from .types import FillResult


class PortfolioManager:
    """Manages a fixed capital pot across a capped number of concurrent positions.

    Reads and writes state/execution_state.json.  Separate from trade_state.json
    (which belongs to the email alert system and is not touched here).
    """

    def __init__(
        self,
        capital_pot: float,
        max_positions: int,
        state_path: Path,
    ) -> None:
        self._capital_pot = capital_pot
        self._max_positions = max_positions
        self._path = state_path
        self._state: dict = self._load()

    # -- Persistence --------------------------------------------------------

    def _load(self) -> dict:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data.setdefault("positions", {})
                    data.setdefault("trade_log", [])
                    data.setdefault("trades_today", {
                        "date": datetime.now(timezone.utc).date().isoformat(),
                        "buys": 0,
                        "sells": 0,
                    })
                    return data
            except Exception:
                pass
        return {
            "positions": {},
            "trade_log": [],
            "trades_today": {
                "date": datetime.now(timezone.utc).date().isoformat(),
                "buys": 0,
                "sells": 0,
            },
        }

    def save(self) -> None:
        """Write current state to execution_state.json."""
        self._path.write_text(
            json.dumps(self._state, indent=2), encoding="utf-8"
        )

    # -- Read-only accessors ------------------------------------------------

    @property
    def positions(self) -> dict[str, dict]:
        return self._state["positions"]

    @property
    def trade_log(self) -> list[dict]:
        return self._state["trade_log"]

    def get_limit_tracker(self) -> DailyLimitTracker:
        """Return a DailyLimitTracker for this portfolio's state."""
        from .daily_limits import DailyLimitTracker
        return DailyLimitTracker(self._state)

    # -- Capacity and sizing ------------------------------------------------

    def can_open(self, ticker: str) -> bool:
        """True if ticker has no open position and the portfolio has capacity."""
        return (
            ticker not in self.positions
            and len(self.positions) < self._max_positions
        )

    def compute_quantity(self, kelly_fraction: float, price: float) -> int:
        """Shares to buy: slot_value × kelly / price, floored to whole shares."""
        if price <= 0 or kelly_fraction <= 0:
            return 0
        slot_value = self._capital_pot / self._max_positions
        return max(1, int(floor(slot_value * kelly_fraction / price)))

    # -- State mutations ----------------------------------------------------

    def record_entry(
        self,
        ticker: str,
        fill: FillResult,
        kelly_fraction: float,
        stop_level: float,
        target_level: float,
    ) -> None:
        """Record a new open position after a BUY fill."""
        today = datetime.now(timezone.utc).date().isoformat()
        self._state["positions"][ticker] = {
            "entry_date": today,
            "fill_price": fill.fill_price,
            "quantity": fill.quantity,
            "kelly_fraction": kelly_fraction,
            "stop_level": stop_level,
            "target_level": target_level,
        }
        self._state["trade_log"].append({
            "ticker": ticker,
            "action": "BUY",
            "date": today,
            "fill_price": fill.fill_price,
            "quantity": fill.quantity,
        })

    def record_exit(self, ticker: str, fill: FillResult) -> None:
        """Remove position and log realised P&L after a SELL fill."""
        pos = self._state["positions"].pop(ticker, None)
        entry_price = pos["fill_price"] if pos else 0.0
        pl = round((fill.fill_price - entry_price) * fill.quantity, 2)
        self._state["trade_log"].append({
            "ticker": ticker,
            "action": "SELL",
            "date": datetime.now(timezone.utc).date().isoformat(),
            "fill_price": fill.fill_price,
            "quantity": fill.quantity,
            "pl": pl,
        })
