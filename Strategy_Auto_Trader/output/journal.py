"""Lightweight trading journal: CSV logs of closed trades.

Two journals are maintained independently:
  data/journals/backtest.csv — trades extracted from compositeBacktest.csv after each batch run
  data/journals/live.csv     — trades recorded via trade_state.py's record_buy/record_sell

Both share the same column schema so they can be analysed the same way.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd

JOURNAL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "journals"

BACKTEST_JOURNAL = JOURNAL_DIR / "backtest.csv"
LIVE_JOURNAL = JOURNAL_DIR / "live.csv"


@dataclass
class TradeRecord:
    """One closed (or still-open) trade. Field order defines the CSV column order."""

    date_opened: str
    ticker: str
    strategy: str = ""
    vol_filter: str = ""
    entry_signal: str = ""
    entry_score: float = 0.0
    entry_gate_flag: str = ""
    entry_price: float = 0.0
    regime_at_entry: float = 0.0
    rsi_at_entry: float = 0.0
    volume_ratio: float = 0.0
    kelly_fraction: float = 0.0
    date_closed: str = ""
    exit_reason: str = ""
    exit_price: float = 0.0
    stop_level: float = 0.0
    target_level: float = 0.0
    pnl_usd: float = 0.0
    return_pct: float = 0.0
    days_held: int = 0
    peak_gain: float = 0.0
    peak_loss: float = 0.0
    strategy_return: float = 0.0
    bh_return: float = 0.0
    market_ret_during_hold: float = 0.0
    notes: str = ""


JOURNAL_FIELDNAMES = [f.name for f in fields(TradeRecord)]


def _trade_key(row: dict) -> tuple:
    """Identity used to dedupe rows already present in a journal."""
    return (row.get("ticker", ""), row.get("strategy", ""), row.get("date_opened", ""), row.get("date_closed", ""))


def _existing_keys(journal_path: Path) -> set[tuple]:
    if not journal_path.exists():
        return set()
    with open(journal_path, newline="", encoding="utf-8") as f:
        return {_trade_key(row) for row in csv.DictReader(f)}


def _migrate_journal(journal_path: Path) -> None:
    """Rewrite a journal whose header predates the current schema.

    New columns are appended over time (e.g. market_ret_during_hold); old rows
    get empty values for them so DictWriter stays aligned with the header.
    """
    with open(journal_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or list(reader.fieldnames) == JOURNAL_FIELDNAMES:
            return
        rows = list(reader)

    with open(journal_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in JOURNAL_FIELDNAMES})


def append_trades(journal_path: Path, trades: list[TradeRecord]) -> int:
    """Append trades to a journal CSV, skipping any already recorded. Returns count appended."""
    if not trades:
        return 0

    journal_path.parent.mkdir(parents=True, exist_ok=True)
    if journal_path.exists():
        _migrate_journal(journal_path)
    existing = _existing_keys(journal_path)
    write_header = not journal_path.exists()

    new_rows = [t for t in trades if _trade_key(t.__dict__) not in existing]
    if not new_rows:
        return 0

    with open(journal_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for t in new_rows:
            writer.writerow(t.__dict__)

    return len(new_rows)


def extract_trades_from_csv(
    ticker: str, csv_path: Path, strategy: str = "", vol_filter: str = ""
) -> list[TradeRecord]:
    """Read a compositeBacktest.csv and extract round-trip trades. See extract_trades_from_detail."""
    if not csv_path.exists():
        return []
    detail = pd.read_csv(csv_path)
    return extract_trades_from_detail(ticker, detail, strategy=strategy, vol_filter=vol_filter)


def extract_trades_from_detail(
    ticker: str, detail: pd.DataFrame, strategy: str = "", vol_filter: str = ""
) -> list[TradeRecord]:
    """Walk a compositeBacktest detail DataFrame and build one TradeRecord per BUY-to-SELL round trip.

    A trade opens on the bar where trade_event == 'BUY' and closes on the next
    bar where trade_event == 'SELL'. Trades still open at the end of the file
    are skipped (date_closed would be empty and P&L undefined).
    """
    if detail.empty or "trade_event" not in detail.columns:
        return []

    trades: list[TradeRecord] = []
    open_trade: dict | None = None
    trough_price: float = float("inf")
    peak_price: float = float("-inf")

    for _, row in detail.iterrows():
        event = row.get("trade_event", "")

        if open_trade is not None:
            close = float(row.get("close", 0) or 0)
            trough_price = min(trough_price, close)
            peak_price = max(peak_price, close)

        if event == "BUY" and open_trade is None:
            open_trade = {
                "date_opened": str(row.get("date", "")),
                "ticker": ticker,
                "entry_signal": str(row.get("signal_flag", "")),
                "entry_score": float(row.get("signal_score", 0) or 0),
                "entry_gate_flag": str(row.get("gate_flag", "")),
                "entry_price": float(row.get("entry_price", 0) or row.get("close", 0) or 0),
                "regime_at_entry": float(row.get("regime_signal", 0) or 0),
                "rsi_at_entry": float(row.get("rsi", 0) or 0),
                "volume_ratio": float(row.get("volume_ratio", 0) or 0),
                "kelly_fraction": float(row.get("kelly_fraction", 0) or 0),
                "stop_level": float(row.get("stop_level", 0) or 0),
                "target_level": float(row.get("target_level", 0) or 0),
                "portfolio_value": float(row.get("portfolio_value", 0) or 0),
                "bh_equity": float(row.get("bh_equity", 0) or 0),
            }
            trough_price = open_trade["entry_price"]
            peak_price = open_trade["entry_price"]

        elif event == "SELL" and open_trade is not None:
            entry_price = open_trade["entry_price"]
            exit_price = float(row.get("close", 0) or 0)
            return_pct = (exit_price - entry_price) / entry_price if entry_price else 0.0

            date_opened = pd.Timestamp(open_trade["date_opened"])
            date_closed = pd.Timestamp(row.get("date", ""))
            days_held = (date_closed - date_opened).days

            # Market's own move over the hold window (for market-adjusted P&L)
            bh_at_entry = open_trade["bh_equity"]
            bh_at_exit = float(row.get("bh_equity", 0) or 0)
            market_ret = (bh_at_exit / bh_at_entry - 1.0) if bh_at_entry > 0 else 0.0

            trades.append(TradeRecord(
                date_opened=open_trade["date_opened"],
                ticker=ticker,
                strategy=strategy,
                vol_filter=vol_filter,
                entry_signal=open_trade["entry_signal"],
                entry_score=open_trade["entry_score"],
                entry_gate_flag=open_trade["entry_gate_flag"],
                entry_price=entry_price,
                regime_at_entry=open_trade["regime_at_entry"],
                rsi_at_entry=open_trade["rsi_at_entry"],
                volume_ratio=open_trade["volume_ratio"],
                kelly_fraction=open_trade["kelly_fraction"],
                date_closed=str(row.get("date", "")),
                exit_reason=str(row.get("sell_reason", "") or ""),
                exit_price=exit_price,
                stop_level=open_trade["stop_level"],
                target_level=open_trade["target_level"],
                pnl_usd=return_pct * open_trade["kelly_fraction"] * open_trade["portfolio_value"],
                return_pct=return_pct,
                days_held=days_held,
                peak_gain=(peak_price - entry_price) / entry_price if entry_price else 0.0,
                peak_loss=(trough_price - entry_price) / entry_price if entry_price else 0.0,
                strategy_return=float(row.get("strategy_equity", 1.0) or 1.0) - 1.0,
                bh_return=float(row.get("bh_equity", 1.0) or 1.0) - 1.0,
                market_ret_during_hold=market_ret,
            ))
            open_trade = None
            trough_price = float("inf")
            peak_price = float("-inf")

    return trades
