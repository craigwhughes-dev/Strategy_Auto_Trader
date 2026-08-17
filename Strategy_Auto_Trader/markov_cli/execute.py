"""Execution engine — reads latest signals and submits orders to IBKR.

Run independently of batch.py (separate process, separate tests).

Usage:
    # Dry run — NullBroker, no state written, safe to run any time
    uv run python -m Strategy_Auto_Trader.markov_cli.execute --dry-run

    # Paper account — IB Gateway/TWS must be running (see config/overnight_strategy.json "broker")
    uv run python -m Strategy_Auto_Trader.markov_cli.execute
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from ..core.cli_logging import setup_cli_logger

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


def _place_order_with_retry(
    broker: object,
    req: object,
    ticker: str,
    max_retries: int = 5,
    retry_delay: float = 30.0,
):
    """Place an order, retrying on a dropped socket only (not on order rejects).

    A ConnectionError here means the TCP session to TWS/Gateway died before
    the broker call could confirm anything reached IBKR — the caller's
    in-flight marker is still set, so it's safe to reconnect and resubmit.
    Any other exception (bad contract, order reject, etc.) is not a
    connectivity issue and is raised immediately without retrying.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return broker.place_order(req)
        except ConnectionError as e:
            if attempt >= max_retries:
                logger.warning(
                    f"Order call for {ticker} raised before returning — "
                    f"in-flight marker left in place after {attempt} attempt(s): {e}"
                )
                raise
            logger.warning(
                f"Order call for {ticker} raised before returning "
                f"(attempt {attempt}/{max_retries}) — retrying in {retry_delay:.0f}s: {e}"
            )
            time.sleep(retry_delay)
            if not broker.is_connected():
                try:
                    broker.connect()
                except Exception as connect_err:
                    logger.warning(f"Reconnect attempt for {ticker} failed: {connect_err}")


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
    allow_new_entries: bool = True,
    market_name: str = "",
    market_currency: str = "",
    marker_path: Path | None = None,
    protective_stops: bool = False,
    stop_buffer_pct: float = 1.5,
) -> tuple[list[str], list[str], list[str]]:
    """Execute BUY/SELL signals for the given tickers.

    Returns (buys, sells, skipped) lists of strings for logging/display.
    Modifies portfolio and limit_tracker state in place.
    allow_new_entries=False blocks all buys (reconciliation halt or user pause).
    """
    from ..broker.in_flight_marker import write_marker, clear_marker
    from ..broker.signal_reader import read_latest_signal
    from ..broker.symbols import normalize_fill_price, sizing_price
    from ..broker.types import FillResult, OrderRequest, StopOrderRequest

    if marker_path is None:
        marker_path = STATE_DIR / "order_in_flight.json"

    buys: list[str] = []
    sells: list[str] = []
    skipped: list[str] = []
    resolved: set[str] = set()

    buy_signals: list[tuple[str, dict]] = []
    sell_signals: list[tuple[str, dict]] = []
    hold_details: dict[str, list[str]] = {}
    rejected_details: dict[str, list[str]] = {}

    for ticker in tickers:
        signal = read_latest_signal(ticker, data_dir)
        if signal is None or signal["flag"] == "HOLD":
            if signal is None:
                bucket = "no_data"
                detail = ticker
            else:
                raw = signal.get("signal_flag", "HOLD")
                reason = signal.get("reason", "")
                score = signal.get("score", 0.0)
                detail = f"{ticker}={score:.1f}"
                if raw == "SELL":
                    bucket = "bearish"
                elif raw == "BUY":
                    bucket = "entry_blocked"
                elif reason:
                    bucket = reason
                elif score > 0:
                    bucket = "below_threshold"
                else:
                    bucket = "no_signal"
            hold_details.setdefault(bucket, []).append(detail)
            skipped.append(ticker)
            resolved.add(ticker)
            continue
        # Normalize signal prices to pot currency (pounds for .L) once, here.
        # quote_close keeps the original exchange-units close as the reference
        # for disambiguating IBKR's inconsistent LSE fill-price units.
        signal["quote_close"] = signal["close"]
        for key in ("close", "stop_level", "target_level"):
            signal[key] = sizing_price(ticker, signal.get(key, 0.0) or 0.0)
        if signal["flag"] == "BUY":
            buy_signals.append((ticker, signal))
        elif signal["flag"] == "SELL":
            sell_signals.append((ticker, signal))

    if hold_details:
        parts = []
        for bucket, details in sorted(hold_details.items(), key=lambda x: -len(x[1])):
            parts.append(f"{bucket}({', '.join(details)})={len(details)}")
        logger.info(f"  HOLD breakdown: {', '.join(parts)}")

    # Sort order must match live_sim.py's arbitrate() (entry_score descending) —
    # that's the backtest-validated ordering the optimised_new switch was decided on.
    buy_signals.sort(key=lambda x: x[1]["score"], reverse=True)

    for ticker, signal in buy_signals:
        try:
            if not allow_new_entries:
                skipped.append(f"{ticker}(new entries blocked)")
                rejected_details.setdefault("new entries blocked", []).append(ticker)
                resolved.add(ticker)
                continue
            if not portfolio.can_open(ticker):
                skipped.append(f"{ticker}(at capacity)")
                rejected_details.setdefault("at capacity", []).append(ticker)
                resolved.add(ticker)
                continue
            qty = portfolio.compute_quantity(
                signal["kelly_fraction"], signal["close"]
            )
            if qty < 1:
                skipped.append(f"{ticker}(qty=0)")
                rejected_details.setdefault("qty=0", []).append(ticker)
                resolved.add(ticker)
                continue
            logger.info(f"About to place order: BUY {qty}x {ticker} — marker written")
            write_marker(marker_path, ticker, "BUY", qty)
            fill = _place_order_with_retry(broker, OrderRequest(ticker, "BUY", qty), ticker)
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
                rejected_details.setdefault("order not filled", []).append(ticker)
                resolved.add(ticker)
                continue

            fill = FillResult(
                ticker=fill.ticker, action=fill.action,
                fill_price=normalize_fill_price(ticker, fill.fill_price, signal["quote_close"]),
                quantity=fill.quantity, timestamp=fill.timestamp,
            )

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
                    ticker, exit_fill, signal_price=signal["close"],
                    exit_type="strategy_exit"
                )
                # Log as entry but flag the severe slippage condition
                slippage_note = (f" (SEVERE SLIPPAGE: stopped out on entry, "
                               f"-{loss_pct:.1f}% from signal price)")
                buys.append(f"{ticker} x{qty} @ {fill.fill_price:.2f}{slippage_note}")
            else:
                buys.append(f"{ticker} x{qty} @ {fill.fill_price:.2f}"
                            f"{_slippage_tag(signal['close'], fill.fill_price, 'BUY')}")
                if protective_stops:
                    resting_stop = adjusted_stop * (1 - stop_buffer_pct / 100)
                    req = StopOrderRequest(ticker, qty, resting_stop)
                    try:
                        result = broker.place_stop_order(req)
                        if result is not None:
                            portfolio.set_stop_order(ticker, result.perm_id, result.stop_price)
                            logger.info(f"{ticker}: protective stop placed @ {result.stop_price}")
                        else:
                            logger.warning(f"{ticker}: resting stop rejected — will retry next poll")
                    except Exception as e:
                        logger.warning(f"{ticker}: stop placement error: {e}")
            resolved.add(ticker)
        except Exception as e:
            raise ExecutionInterrupted(
                e, buys, sells, skipped,
                [t for t in tickers if t not in resolved]
            )

    for ticker, signal in sell_signals:
        try:
            if ticker not in portfolio.positions:
                skipped.append(f"{ticker}(no position)")
                rejected_details.setdefault("no position", []).append(ticker)
                resolved.add(ticker)
                continue
            qty = portfolio.positions[ticker]["quantity"]

            if protective_stops:
                perm_id = portfolio.positions[ticker].get("stop_perm_id")
                if perm_id:
                    outcome = broker.cancel_stop_order(perm_id)
                    if outcome == "Filled":
                        fill = broker.get_stop_fill(perm_id)
                        if fill is not None:
                            fill = FillResult(
                                ticker=fill.ticker, action=fill.action,
                                fill_price=normalize_fill_price(
                                    ticker, fill.fill_price, signal["quote_close"]),
                                quantity=fill.quantity, timestamp=fill.timestamp,
                            )
                        if fill is None:
                            # Broker returned "Filled" but fill lookup failed; synthesize
                            # an estimated FillResult from the position's recorded stop price.
                            stop_price = portfolio.positions[ticker].get("stop_price")
                            if not stop_price or stop_price <= 0:
                                # Fall back to signal close as last resort
                                stop_price = signal["close"]
                            fill = FillResult(
                                ticker=ticker, action="SELL", fill_price=stop_price,
                                quantity=qty, timestamp=""
                            )
                            exit_type = "reconciled_stop_loss"
                        else:
                            exit_type = "stop_loss"
                        portfolio.record_exit(ticker, fill, signal_price=signal["close"],
                                            exit_type=exit_type)
                        portfolio.clear_stop_order(ticker)
                        sells.append(f"{ticker} x{qty} @ {fill.fill_price:.2f} (stop_loss)")
                        resolved.add(ticker)
                        continue
                    elif outcome == "Error":
                        # Cancel failed; do not attempt market sell
                        skipped.append(f"{ticker}(stop cancel error)")
                        rejected_details.setdefault("stop cancel error", []).append(ticker)
                        resolved.add(ticker)
                        continue
                    portfolio.clear_stop_order(ticker)

            logger.info(f"About to place order: SELL {qty}x {ticker} — marker written")
            write_marker(marker_path, ticker, "SELL", qty)
            fill = _place_order_with_retry(broker, OrderRequest(ticker, "SELL", qty), ticker)
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
                rejected_details.setdefault("order not filled", []).append(ticker)
                resolved.add(ticker)
                continue

            fill = FillResult(
                ticker=fill.ticker, action=fill.action,
                fill_price=normalize_fill_price(ticker, fill.fill_price, signal["quote_close"]),
                quantity=fill.quantity, timestamp=fill.timestamp,
            )
            portfolio.record_exit(ticker, fill, signal_price=signal["close"],
                                exit_type="strategy_exit")
            limit_tracker.record_sell()
            sells.append(f"{ticker} x{qty} @ {fill.fill_price:.2f}"
                         f"{_slippage_tag(signal['close'], fill.fill_price, 'SELL')}")
            resolved.add(ticker)
        except Exception as e:
            raise ExecutionInterrupted(
                e, buys, sells, skipped,
                [t for t in tickers if t not in resolved]
            )

    if rejected_details:
        parts = []
        for reason, details in sorted(rejected_details.items(), key=lambda x: -len(x[1])):
            parts.append(f"{reason}({', '.join(details)})={len(details)}")
        logger.info(f"  REJECTED breakdown: {', '.join(parts)}")

    return buys, sells, skipped


def _slippage_tag(signal_price: float, fill_price: float, action: str) -> str:
    """Suffix like ' (slippage +3.2bps)' for order log lines, '' when unknown."""
    from ..broker.portfolio import slippage_bps
    bps = slippage_bps(signal_price, fill_price, action)
    return f" (slippage {bps:+.1f}bps)" if bps is not None else ""


def main(argv: list[str] | None = None) -> int:
    setup_cli_logger("execute")

    from ..broker.portfolio import PortfolioManager
    from ..broker.signal_reader import read_latest_signal

    args = _build_arg_parser().parse_args(argv)

    broker_cfg = {}
    config_path = CONFIG_DIR / "overnight_strategy.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            broker_cfg = json.load(f).get("broker", {})
    broker_host = broker_cfg.get("host", "127.0.0.1")
    broker_port = broker_cfg.get("port", 7497)
    broker_client_id = broker_cfg.get("client_id", 1)

    if not args.dry_run:
        # Execution reads precomputed signals (no HMM here), but a real
        # order run must verify the broker library before touching state.
        from ..core.self_check import SelfCheckError, run_startup_checks
        try:
            run_startup_checks(
                require_hmm=False, require_broker=True,
                broker_host=broker_host, broker_port=broker_port,
            )
        except SelfCheckError as e:
            logger.error(f"ERROR: {e}")
            return 1

    data_dir = Path(args.data_dir)
    watchlist = _load_watchlist(Path(args.watchlist))
    defaults = watchlist.get("defaults", {})

    capital_pot = float(defaults.get("capital_pot", 20_000))
    tickers = [t["ticker"] for t in watchlist.get("tickers", [])]

    state_path = Path(args.state_dir) / "execution_state.json"
    # Infer currency from tickers (all FTSE if .L suffix, else USD)
    currency = "GBP" if any(t.endswith(".L") for t in tickers) else "USD"
    portfolio = PortfolioManager(capital_pot, state_path, currency=currency)
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
        broker = IBKRAdapter(host=broker_host, port=broker_port, client_id=broker_client_id)

    broker.connect()
    try:
        buys, sells, skipped = execute_signals(
            tickers, data_dir, portfolio, limit_tracker, broker,
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
    logger.info(f"\n{tag}Execution summary")
    logger.info(f"  BUY  orders : {len(buys)}")
    for b in buys:
        logger.info(f"    {b}")
    logger.info(f"  SELL orders : {len(sells)}")
    for s in sells:
        logger.info(f"    {s}")
    logger.info(f"  Skipped     : {len(skipped)}")
    open_pos = portfolio.positions  # type: ignore[attr-defined]
    logger.info(f"  Open positions ({len(open_pos)}):")
    for ticker, pos in open_pos.items():
        logger.info(
            f"    {ticker}: {pos['quantity']} shares @ "
            f"{pos['fill_price']:.2f} (entered {pos['entry_date']})"
        )


if __name__ == "__main__":
    sys.exit(main())
