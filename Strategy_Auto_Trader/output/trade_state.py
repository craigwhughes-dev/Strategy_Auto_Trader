"""Persistent trade state: tracks which tickers have had BUY alerts sent.

A SELL alert is only sent if a BUY was previously recorded for that ticker
since the reference date. After a SELL alert, the ticker is removed and the
round trip is appended to the live trading journal (output/journal.py).
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from pathlib import Path

from .journal import LIVE_JOURNAL, TradeRecord, append_trades

STATE_FILE = Path(__file__).resolve().parent.parent.parent / "state" / "trade_state.json"


def _load() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"buys": {}}


def _save(state: dict) -> None:
    # Atomic write: write to temp file then rename (prevents corruption on concurrent access)
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=STATE_FILE.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, STATE_FILE)
    except Exception:
        os.unlink(tmp)
        raise


def record_buy(ticker: str, context: dict | None = None) -> None:
    """Record that a BUY alert was sent for this ticker.

    context: optional entry details (signal, score, gate_flag, price, regime,
    rsi, volume_ratio, kelly_fraction) — stored so record_sell() can complete
    a full journal entry when the position closes.
    """
    state = _load()
    entry = {"date": str(date.today())}
    entry.update(context or {})
    state["buys"][ticker] = entry
    _save(state)


def has_open_buy(ticker: str) -> bool:
    """Check if a BUY was previously recorded (and not yet closed by SELL)."""
    state = _load()
    return ticker in state.get("buys", {})


def record_sell(ticker: str, context: dict | None = None) -> None:
    """Record that a SELL alert was sent — removes the BUY entry and logs the trade.

    context: optional exit details (price, reason, regime/score at exit are
    not tracked — only entry-side context is journaled for the live book).
    """
    state = _load()
    entry = state["buys"].pop(ticker, None)
    _save(state)

    if entry is None:
        return

    context = context or {}
    entry_price = float(entry.get("price", 0) or 0)
    exit_price = float(context.get("price", 0) or 0)
    return_pct = (exit_price - entry_price) / entry_price if entry_price else 0.0

    # Market's move over the hold: bh_return is cumulative, so compound the
    # entry-time and exit-time values (0.0 when entry predates the field)
    bh_at_entry = float(entry.get("bh_return", 0) or 0)
    bh_at_exit = float(context.get("bh_return", 0) or 0)
    market_ret = ((1 + bh_at_exit) / (1 + bh_at_entry) - 1.0) if bh_at_entry > -1 else 0.0
    date_opened = entry.get("date", "")
    date_closed = str(date.today())
    try:
        from pandas import Timestamp
        days_held = (Timestamp(date_closed) - Timestamp(date_opened)).days
    except Exception:
        days_held = 0

    record = TradeRecord(
        date_opened=date_opened,
        ticker=ticker,
        strategy=str(entry.get("strategy", "")),
        vol_filter=str(entry.get("vol_filter", "")),
        entry_signal=str(entry.get("signal", "")),
        entry_score=float(entry.get("score", 0) or 0),
        entry_gate_flag=str(entry.get("gate_flag", "")),
        entry_price=entry_price,
        regime_at_entry=float(entry.get("regime", 0) or 0),
        rsi_at_entry=float(entry.get("rsi", 0) or 0),
        volume_ratio=float(entry.get("volume_ratio", 0) or 0),
        kelly_fraction=float(entry.get("kelly_fraction", 0) or 0),
        date_closed=date_closed,
        exit_reason=str(context.get("reason", "")),
        exit_price=exit_price,
        stop_level=float(entry.get("stop_level", 0) or 0),
        target_level=float(entry.get("target_level", 0) or 0),
        pnl_usd=return_pct * float(entry.get("kelly_fraction", 0) or 0) * float(entry.get("portfolio_value", 0) or 0),
        return_pct=return_pct,
        days_held=days_held,
        strategy_return=float(context.get("strategy_return", 0) or 0),
        bh_return=float(context.get("bh_return", 0) or 0),
        market_ret_during_hold=market_ret,
    )
    append_trades(LIVE_JOURNAL, [record])


def get_open_positions() -> dict[str, str]:
    """Return all tickers with open BUY alerts and their entry dates."""
    state = _load()
    return {ticker: v.get("date", "") if isinstance(v, dict) else v
            for ticker, v in state.get("buys", {}).items()}
