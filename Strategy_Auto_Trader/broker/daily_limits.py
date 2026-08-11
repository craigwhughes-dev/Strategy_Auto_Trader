"""Daily trade count tracking — records BUY/SELL execution counts per day.

Counting only, not enforcement — nothing reads these counts to cap trading.
The daily_buy_limit/daily_sell_limit config keys some watchlists carried were
dead configuration and have been removed; trades_today still feeds the
app_status.json monitoring snapshot (live_daemon.py's write_app_status_snapshot)."""

from __future__ import annotations

from datetime import datetime, timezone


class DailyLimitTracker:
    """Tracks daily BUY/SELL execution counts, resets at UTC midnight.

    Integrates with execution_state.json via the state dict:
    {
        "trades_today": {
            "date": "2026-07-02",
            "buys": 1,
            "sells": 0
        }
    }
    """

    def __init__(self, state: dict) -> None:
        self._state = state
        self._ensure_today()

    def _ensure_today(self) -> None:
        """Initialize or reset trades_today if date has changed."""
        today = datetime.now(timezone.utc).date().isoformat()
        if "trades_today" not in self._state:
            self._state["trades_today"] = {"date": today, "buys": 0, "sells": 0}
        elif self._state["trades_today"].get("date") != today:
            self._state["trades_today"] = {"date": today, "buys": 0, "sells": 0}

    def record_buy(self) -> None:
        """Increment today's BUY count."""
        self._ensure_today()
        self._state["trades_today"]["buys"] += 1

    def record_sell(self) -> None:
        """Increment today's SELL count."""
        self._ensure_today()
        self._state["trades_today"]["sells"] += 1
