"""Execution engine — reads latest signals and submits orders to IBKR.

Run independently of batch.py (separate process, separate tests).

Usage:
    # Dry run — NullBroker, no state written, safe to run any time
    uv run python -m Strategy_Auto_Trader.markov_cli.execute --dry-run

    # Paper account — TWS must be running on localhost:7497
    uv run python -m Strategy_Auto_Trader.markov_cli.execute
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
STATE_DIR = ROOT / "state"
CONFIG_DIR = ROOT / "config"

logger = logging.getLogger("live_daemon.execute")


class ExecutionInterrupted(Exception):
    """Raised when execute_signals() fails partway through a batch.

    Carries whatever buys/sells/skipped were already recorded before the
    failure, plus the tickers whose outcome is unknown, so the caller can
    tell "nothing happened yet, safe to retry" from "something may have
    already reached the broker, do not blindly resubmit."
    """

    def __init__(
        self,
        original: Exception,
        buys: list[str],
        sells: list[str],
        skipped: list[str],
        unresolved: list[str],
    ) -> None:
        super().__init__(str(original))
        self.original = original
        self.buys = buys
        self.sells = sells
        self.skipped = skipped
        self.unresolved = unresolved


def _load_watchlist(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="IBKR execution engine — reads signals and places orders."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Use NullBroker (no real orders, no state changes).",
    )
    p.add_argument(
        "--watchlist",
        default=str(CONFIG_DIR / "watchlist.json"),
        help="Path to watchlist JSON (default: config/watchlist.json).",
    )
    p.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Directory containing per-ticker run subdirectories.",
    )
    p.add_argument(
        "--state-dir",
        default=str(STATE_DIR),
        help="Directory for execution_state.json (default: state/).",
    )
    return p


def execute_signals(
    tickers: list[str],
    data_dir: Path,
    portfolio: object,
    limit_tracker: object,
    broker: object,
    daily_buy_limit: int | None = 2,
    daily_sell_limit: int | None = None,
    market_name: str = "",
    market_currency: str = "",
    marker_path: Path | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Execute BUY/SELL signals for the given tickers.

    Returns (buys, sells, skipped) lists of strings for logging/display.
    Modifies portfolio and limit_tracker state in place.
    """
    from ..broker.in_flight_marker import write_marker, clear_marker
    from ..broker.signal_reader import read_latest_signal
    from ..broker.symbols import sizing_price
    from ..broker.types import FillResult, OrderRequest

    if marker_path is None:
        marker_path = STATE_DIR / "order_in_flight.json"

    buys: list[str] = []
    sells: list[str] = []
    skipped: list[str] = []
    resolved: set[str] = set()

    buy_signals: list[tuple[str, dict]] = []
    sell_signals: list[tuple[str, dict]] = []

    for ticker in tickers:
        signal = read_latest_signal(ticker, data_dir)
        if signal is None or signal["flag"] == "HOLD":
            skipped.append(ticker)
            resolved.add(ticker)
            continue
        if signal["flag"] == "BUY":
            buy_signals.append((ticker, signal))
        elif signal["flag"] == "SELL":
            sell_signals.append((ticker, signal))

    buy_signals.sort(key=lambda x: x[1]["kelly_fraction"], reverse=True)

    for ticker, signal in buy_signals:
        try:
            if not limit_tracker.can_buy(daily_buy_limit):
                skipped.append(f"{ticker}(daily limit reached)")
                resolved.add(ticker)
                continue
            if not portfolio.can_open(ticker):
                skipped.append(f"{ticker}(at capacity)")
                resolved.add(ticker)
                continue
            qty = portfolio.compute_quantity(
                signal["kelly_fraction"], sizing_price(ticker, signal["close"])
            )
            if qty < 1:
                skipped.append(f"{ticker}(qty=0)")
                resolved.add(ticker)
                continue
            logger.info(f"About to place order: BUY {qty}x {ticker} — marker written")
            write_marker(marker_path, ticker, "BUY", qty)
            try:
                fill = broker.place_order(OrderRequest(ticker, "BUY", qty))
            except Exception as e:
                logger.warning(f"Order call for {ticker} raised before returning — in-flight marker left in place: {e}")
                raise
            try:
                clear_marker(marker_path)
            except Exception as e:
                logger.warning(f"Failed to clear in-flight marker for {ticker} after a successful order: {e}")

            if fill is not None:
                logger.info(f"Order placed: BUY {qty}x {ticker} @ {fill.fill_price} — marker cleared")
            else:
                logger.info(f"Order not filled: BUY {qty}x {ticker} (status not Filled) — marker cleared")

            # Order not filled (cancelled, partially filled, etc.)
            if fill is None:
                skipped.append(f"{ticker}(order not filled)")
                resolved.add(ticker)
                continue

            # Recompute stop/target based on fill price, not signal close.
            # Derive the percentage distances used by the signal:
            signal_close = signal["close"]
            if signal_close > 0:
                stop_pct = (signal["stop_level"] - signal_close) / signal_close
                target_pct = (signal["target_level"] - signal_close) / signal_close
            else:
                stop_pct, target_pct = -0.05, 0.15  # fallback to defaults

            adjusted_stop = fill.fill_price * (1 + stop_pct)
            adjusted_target = fill.fill_price * (1 + target_pct)

            # Check for severe slippage: if fill price breaches the recomputed stop,
            # treat as same-bar stop-out instead of recording a broken stop level.
            portfolio.record_entry(
                ticker, fill,
                signal["kelly_fraction"],
                adjusted_stop,
                adjusted_target,
                signal_price=signal["close"],
                market=market_name,
                currency=market_currency,
            )

            limit_tracker.record_buy()

            # Immediate stop-out: compare the fill against the ORIGINAL signal-time
            # stop level, not the recomputed adjusted_stop (which is always below
            # the fill price by construction and can never itself be breached).
            if fill.fill_price <= signal["stop_level"]:
                loss_pct = (signal_close - fill.fill_price) / signal_close * 100 if signal_close > 0 else 0.0
                exit_fill = FillResult(
                    ticker=ticker, action="SELL", fill_price=adjusted_stop,
                    quantity=fill.quantity, timestamp=fill.timestamp,
                )
                portfolio.record_exit(
                    ticker, exit_fill, signal_price=signal["close"]
                )
                # Log as entry but flag the severe slippage condition
                slippage_note = (f" (SEVERE SLIPPAGE: stopped out on entry, "
                               f"-{loss_pct:.1f}% from signal price)")
                buys.append(f"{ticker} x{qty} @ {fill.fill_price:.2f}{slippage_note}")
            else:
                buys.append(f"{ticker} x{qty} @ {fill.fill_price:.2f}"
                            f"{_slippage_tag(signal['close'], fill.fill_price, 'BUY')}")
            resolved.add(ticker)
        except Exception as e:
            raise ExecutionInterrupted(
                e, buys, sells, skipped,
                [t for t in tickers if t not in resolved]
            )

    for ticker, signal in sell_signals:
        try:
            if not limit_tracker.can_sell(daily_sell_limit):
                skipped.append(f"{ticker}(daily sell limit reached)")
                resolved.add(ticker)
                continue
            if ticker not in portfolio.positions:
                skipped.append(f"{ticker}(no position)")
                resolved.add(ticker)
                continue
            qty = portfolio.positions[ticker]["quantity"]
            logger.info(f"About to place order: SELL {qty}x {ticker} — marker written")
            write_marker(marker_path, ticker, "SELL", qty)
            try:
                fill = broker.place_order(OrderRequest(ticker, "SELL", qty))
            except Exception as e:
                logger.warning(f"Order call for {ticker} raised before returning — in-flight marker left in place: {e}")
                raise
            try:
                clear_marker(marker_path)
            except Exception as e:
                logger.warning(f"Failed to clear in-flight marker for {ticker} after a successful order: {e}")

            if fill is not None:
                logger.info(f"Order placed: SELL {qty}x {ticker} @ {fill.fill_price} — marker cleared")
            else:
                logger.info(f"Order not filled: SELL {qty}x {ticker} (status not Filled) — marker cleared")

            # Order not filled (cancelled, partially filled, etc.)
            if fill is None:
                skipped.append(f"{ticker}(order not filled)")
                resolved.add(ticker)
                continue

            portfolio.record_exit(ticker, fill, signal_price=signal["close"])
            limit_tracker.record_sell()
            sells.append(f"{ticker} x{qty} @ {fill.fill_price:.2f}"
                         f"{_slippage_tag(signal['close'], fill.fill_price, 'SELL')}")
            resolved.add(ticker)
        except Exception as e:
            raise ExecutionInterrupted(
                e, buys, sells, skipped,
                [t for t in tickers if t not in resolved]
            )

    return buys, sells, skipped


def _slippage_tag(signal_price: float, fill_price: float, action: str) -> str:
    """Suffix like ' (slippage +3.2bps)' for order log lines, '' when unknown."""
    from ..broker.portfolio import slippage_bps
    bps = slippage_bps(signal_price, fill_price, action)
    return f" (slippage {bps:+.1f}bps)" if bps is not None else ""


def main(argv: list[str] | None = None) -> int:
    from ..broker.portfolio import PortfolioManager
    from ..broker.signal_reader import read_latest_signal

    args = _build_arg_parser().parse_args(argv)

    if not args.dry_run:
        # Execution reads precomputed signals (no HMM here), but a real
        # order run must verify the broker library before touching state.
        from ..core.self_check import SelfCheckError, run_startup_checks
        try:
            run_startup_checks(require_hmm=False, require_broker=True)
        except SelfCheckError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

    data_dir = Path(args.data_dir)
    watchlist = _load_watchlist(Path(args.watchlist))
    defaults = watchlist.get("defaults", {})

    capital_pot = float(defaults.get("capital_pot", 20_000))
    max_positions = int(defaults.get("max_positions", 5))
    daily_buy_limit = defaults.get("daily_buy_limit", 2)
    if daily_buy_limit is not None:
        daily_buy_limit = int(daily_buy_limit)
    daily_sell_limit = defaults.get("daily_sell_limit", None)
    if daily_sell_limit is not None:
        daily_sell_limit = int(daily_sell_limit)
    tickers = [t["ticker"] for t in watchlist.get("tickers", [])]

    state_path = Path(args.state_dir) / "execution_state.json"
    portfolio = PortfolioManager(capital_pot, max_positions, state_path)
    limit_tracker = portfolio.get_limit_tracker()

    if args.dry_run:
        from ..broker.null_adapter import NullBroker
        prices = {}
        for ticker in tickers:
            sig = read_latest_signal(ticker, data_dir)
            if sig:
                prices[ticker] = sig["close"]
        broker = NullBroker(prices=prices)
    else:
        from ..broker.ibkr_adapter import IBKRAdapter
        broker = IBKRAdapter()

    broker.connect()
    try:
        buys, sells, skipped = execute_signals(
            tickers, data_dir, portfolio, limit_tracker, broker,
            daily_buy_limit, daily_sell_limit
        )
    finally:
        broker.disconnect()

    if not args.dry_run:
        portfolio.save()

    _print_summary(buys, sells, skipped, portfolio, dry_run=args.dry_run)
    return 0


def _print_summary(
    buys: list[str],
    sells: list[str],
    skipped: list[str],
    portfolio: object,
    *,
    dry_run: bool,
) -> None:
    tag = "[DRY RUN] " if dry_run else ""
    print(f"\n{tag}Execution summary")
    print(f"  BUY  orders : {len(buys)}")
    for b in buys:
        print(f"    {b}")
    print(f"  SELL orders : {len(sells)}")
    for s in sells:
        print(f"    {s}")
    print(f"  Skipped     : {len(skipped)}")
    open_pos = portfolio.positions  # type: ignore[attr-defined]
    print(f"  Open positions ({len(open_pos)}):")
    for ticker, pos in open_pos.items():
        print(
            f"    {ticker}: {pos['quantity']} shares @ "
            f"{pos['fill_price']:.2f} (entered {pos['entry_date']})"
        )


if __name__ == "__main__":
    sys.exit(main())
