"""Read the latest signal for a ticker from the run output files.

Mirrors the pattern in markov_cli/batch._collect_results() but returns
only the fields needed by the execution engine, with a staleness guard.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

_STALE_HOURS = 24


def read_latest_signal(ticker: str, data_dir: Path) -> dict | None:
    """Return signal dict for the most recent run of *ticker*, or None.

    Returns None when:
    - no run directory exists for the ticker
    - the latest run is older than _STALE_HOURS
    - the CSV or gate file is missing / unreadable

    Returned dict keys:
        flag           — "BUY" | "SELL" | "HOLD" (from qualityGate.json)
        close          — last bar close price
        kelly_fraction — last bar Kelly fraction (defaults to 0.10)
        stop_level     — last bar hard stop price
        target_level   — last bar take-profit price
    """
    ticker_safe = ticker.replace("/", "-").replace("\\", "-")
    dirs = sorted(data_dir.glob(f"{ticker_safe}_*"), key=lambda p: p.name)
    if not dirs:
        return None
    latest = dirs[-1]

    mtime = datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc)
    if datetime.now(timezone.utc) - mtime > timedelta(hours=_STALE_HOURS):
        return None

    csv_path = latest / "compositeBacktest.csv"
    if not csv_path.exists():
        return None

    try:
        detail = pd.read_csv(csv_path, index_col=0)
        if detail.empty:
            return None
        last = detail.iloc[-1]

        flag = "HOLD"
        gate_path = latest / "qualityGate.json"
        if gate_path.exists():
            with open(gate_path, encoding="utf-8") as fh:
                gate = json.load(fh)
            flag = gate.get("flag", "HOLD")

        return {
            "flag": flag,
            "close": float(last.get("close", 0.0)),
            "kelly_fraction": float(last.get("kelly_fraction", 0.10)),
            "stop_level": float(last.get("stop_level", 0.0)),
            "target_level": float(last.get("target_level", 0.0)),
        }
    except Exception:
        return None
