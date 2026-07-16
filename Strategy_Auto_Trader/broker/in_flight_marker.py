"""In-flight order marker for safe order tracking across process restarts.

When an order call into the broker is in progress, a marker file is written
to disk atomically. If the process dies mid-call, the marker persists on disk
as evidence that the order's outcome is ambiguous (may have been filled
server-side). On restart, reconciliation detects the marker and escalates
immediately rather than waiting until evening.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..core.atomic_io import atomic_write_json


def write_marker(path: Path, ticker: str, action: str, quantity: int) -> None:
    """Write an in-flight order marker atomically.

    Called immediately before broker.place_order() to mark that an order call
    is in progress. Marker content includes timestamp so reconciliation can
    distinguish old (stale) from new (recent) markers.
    """
    marker = {
        "ticker": ticker,
        "action": action,
        "quantity": quantity,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(path, marker)


def read_marker(path: Path) -> dict | None:
    """Read the in-flight order marker, if present.

    Returns marker dict on success, or None if file missing or corrupt.
    """
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def clear_marker(path: Path) -> None:
    """Remove the in-flight order marker.

    Called immediately after broker.place_order() returns (successfully filled
    or not-filled both count as resolution). unlink(missing_ok=True) so this
    is safe even if the marker never existed (race condition in tests, etc.).
    """
    path.unlink(missing_ok=True)
