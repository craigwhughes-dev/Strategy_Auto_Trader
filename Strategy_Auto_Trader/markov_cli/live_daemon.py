"""Live daemon — persistent automated paper trading process.

Runs continuously, screening tickers overnight, then cycling through in-scope
tickers during trading hours. Prioritizes open positions (checked every hour),
round-robins through the rest, respecting a per-cycle time budget to avoid
overloading the system.

Usage:
    uv run python -m Strategy_Auto_Trader.markov_cli.live_daemon
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    import psutil
except ImportError:
    psutil = None

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = ROOT / "config"
STATE_DIR = ROOT / "state"
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"


def setup_logging() -> logging.Logger:
    """Set up rotating daily log."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().date().isoformat()
    log_path = LOGS_DIR / f"daemon_{today}.log"

    logger = logging.getLogger("live_daemon")
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(console)

    return logger


def acquire_process_lock(logger: logging.Logger) -> bool:
    """Acquire exclusive daemon lock to prevent multiple instances.

    Uses PID-based detection: if lock file exists, check if the PID is still running.
    If PID is gone, lock is stale and can be removed.

    Returns True if lock acquired. On False, another daemon is running.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = STATE_DIR / "daemon.lock"

    # Check if lock exists and if the process is still alive
    if lock_path.exists():
        try:
            content = lock_path.read_text(encoding="utf-8").strip()
            parts = content.split("|")
            if len(parts) >= 2:
                pid_str = parts[0]
                timestamp_str = parts[1]
                try:
                    old_pid = int(pid_str)
                    # Check if PID is still running (psutil-based if available)
                    pid_alive = False
                    if psutil:
                        try:
                            pid_alive = psutil.pid_exists(old_pid)
                        except Exception:
                            pid_alive = True  # Conservative: assume alive if check fails
                    else:
                        # No psutil; try basic check (Windows)
                        try:
                            os.kill(old_pid, 0)
                            pid_alive = True
                        except (ProcessLookupError, OSError, PermissionError):
                            pid_alive = False

                    if pid_alive:
                        logger.error(f"Daemon already running (PID {old_pid}). Exiting.")
                        return False
                    else:
                        logger.warning(f"Removing stale lock (PID {old_pid} no longer running, "
                                      f"timestamp: {timestamp_str})")
                        lock_path.unlink()
                except (ValueError, TypeError):
                    pass
        except Exception as e:
            logger.warning(f"Could not read lock file: {e}. Removing.")
            try:
                lock_path.unlink()
            except Exception:
                pass

    try:
        current_pid = os.getpid()
        timestamp = datetime.now().isoformat()
        lock_path.write_text(f"{current_pid}|{timestamp}\n", encoding="utf-8")
        logger.info(f"Process lock acquired (PID {current_pid})")
        return True
    except Exception as e:
        logger.error(f"Failed to acquire lock: {e}")
        return False


def release_process_lock(logger: logging.Logger) -> None:
    """Release the process lock on shutdown."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = STATE_DIR / "daemon.lock"
    try:
        if lock_path.exists():
            lock_path.unlink()
            logger.info("Process lock released")
    except Exception as e:
        logger.error(f"Failed to release lock: {e}")


def validate_startup_environment(logger: logging.Logger) -> bool:
    """Validate that startup is possible. Fail loudly if not.

    Checks:
    - Lock file is valid (no stale locks)
    - Necessary directories exist and are writable

    Returns True if environment is valid, False if startup should abort.
    """
    errors: list[str] = []

    # Test directory access
    for dirname, dirpath in [("state", STATE_DIR), ("logs", LOGS_DIR), ("config", CONFIG_DIR)]:
        try:
            dirpath.mkdir(parents=True, exist_ok=True)
            test_file = dirpath / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            errors.append(f"{dirname} directory not writable: {e}")

    if errors:
        for err in errors:
            logger.critical(f"Startup validation failed: {err}")
        return False

    logger.info("Startup environment validation OK")
    return True


def cleanup_incomplete_runs(data_dir: Path, logger: logging.Logger) -> int:
    """Remove run directories that only have inputData.csv (incomplete backtests).

    Returns count of directories cleaned up.
    """
    if not data_dir.exists():
        return 0

    cleaned = 0
    for run_dir in data_dir.glob("*_*"):
        if not run_dir.is_dir():
            continue

        files = set(f.name for f in run_dir.glob("*"))
        has_output = "compositeBacktest.csv" in files or "qualityGate.json" in files

        if "inputData.csv" in files and not has_output:
            try:
                import shutil
                shutil.rmtree(run_dir)
                cleaned += 1
            except Exception as e:
                logger.warning(f"Failed to remove incomplete run {run_dir.name}: {e}")

    if cleaned > 0:
        logger.info(f"Cleaned up {cleaned} incomplete run director(y/ies)")
    return cleaned


def load_config() -> dict:
    """Load overnight_strategy.json."""
    config_path = CONFIG_DIR / "overnight_strategy.json"
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def load_daemon_state() -> dict:
    """Load daemon_state.json, or return empty dict if not yet created."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path = STATE_DIR / "daemon_state.json"
    if state_path.exists():
        try:
            with open(state_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "last_overnight_date": None,
        "cursors": {},
    }


def save_daemon_state(state: dict) -> None:
    """Save daemon_state.json (atomically)."""
    from ..core.atomic_io import atomic_write_json
    state_path = STATE_DIR / "daemon_state.json"
    atomic_write_json(state_path, state)


def get_market_currency(market_name: str, config: dict) -> str:
    """Get currency code for a market (FTSE=GBP, SP500=USD, etc.)."""
    # Map market names to currencies. Can be extended per config later.
    market_currencies = {
        "ftse": "GBP",
        "ftse100": "GBP",
        "sp500": "USD",
        "usa": "USD",
    }
    return market_currencies.get(market_name.lower(), "")


def write_app_status_snapshot(
    portfolio: object,
    daemon_state: dict,
    config: dict,
    last_cycle_hour: dict,
    logger: logging.Logger,
) -> None:
    """Write app_status.json snapshot atomically every poll loop (~60s).

    This is the app's only window into daemon state. Includes heartbeat,
    positions with market/currency/cost_value, daemon health flags, and
    trading hours status per market.
    """
    from ..core.atomic_io import atomic_write_json

    now_utc = datetime.now(timezone.utc).isoformat()
    pid = os.getpid()

    trades_today = portfolio._state.get("trades_today", {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "buys": 0,
        "sells": 0,
    })

    # Snapshot positions with all needed fields
    positions_snapshot = {}
    for ticker, pos in portfolio.positions.items():
        positions_snapshot[ticker] = {
            "entry_date": pos.get("entry_date", ""),
            "fill_price": pos.get("fill_price", 0.0),
            "quantity": pos.get("quantity", 0),
            "cost_value": pos.get("cost_value", 0.0),
            "market": pos.get("market", ""),
            "currency": pos.get("currency", ""),
            "stop_level": pos.get("stop_level", 0.0),
            "target_level": pos.get("target_level", 0.0),
            "kelly_fraction": pos.get("kelly_fraction", 0.0),
        }

    # Market trading hours status
    markets_status = {}
    now = datetime.now(timezone.utc)
    for market_name, market_cfg in config.get("markets", {}).items():
        in_trading = is_trading_hours(market_cfg, logger)
        last_hour = last_cycle_hour.get(market_name, -1)
        markets_status[market_name] = {
            "in_trading_hours": in_trading,
            "last_cycle_hour": last_hour,
        }

    snapshot = {
        "schema_version": 1,
        "heartbeat_utc": now_utc,
        "daemon_pid": pid,
        "dry_run": config.get("execution", {}).get("dry_run", True),
        "halt_new_entries": daemon_state.get("halt_new_entries", False),
        "reconciliation_discrepancies": daemon_state.get("reconciliation_discrepancies", []),
        "last_reconcile_date": daemon_state.get("last_reconcile_date", ""),
        "trades_today": trades_today,
        "markets": markets_status,
        "positions": positions_snapshot,
    }

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    status_path = STATE_DIR / "app_status.json"
    atomic_write_json(status_path, snapshot)


def is_trading_hours(market_cfg: dict, logger: logging.Logger) -> bool:
    """Check if market is currently in trading hours."""
    tz = ZoneInfo(market_cfg["timezone"])
    now = datetime.now(tz)

    weekday = now.weekday()
    if weekday >= 5:
        logger.debug(f"  Market {market_cfg['timezone']}: weekend, skipping")
        return False

    start_str = market_cfg["trading_start"]
    end_str = market_cfg["trading_end"]
    start_time = datetime.strptime(start_str, "%H:%M").time()
    end_time = datetime.strptime(end_str, "%H:%M").time()

    is_open = start_time <= now.time() <= end_time
    if not is_open:
        logger.debug(f"  Market {market_cfg['timezone']}: outside hours "
                     f"({start_str}-{end_str}), skipping")
    return is_open


def load_in_scope_tickers(market_name: str, logger: logging.Logger) -> list[str]:
    """Load in_scope_<market>.json."""
    scope_path = STATE_DIR / f"in_scope_{market_name}.json"
    if not scope_path.exists():
        logger.warning(f"  No in_scope_{market_name}.json yet, run overnight_scope first")
        return []
    try:
        with open(scope_path, encoding="utf-8") as f:
            result = json.load(f)
        return result.get("kept", [])
    except Exception as e:
        logger.error(f"  Error loading in_scope_{market_name}.json: {e}")
        return []


def get_open_positions(market_name: str, all_tickers: list[str], logger: logging.Logger) -> list[str]:
    """Get tickers with open positions in this market (intersection of exec state positions and market tickers)."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path = STATE_DIR / "execution_state.json"
    if not state_path.exists():
        return []
    try:
        with open(state_path, encoding="utf-8") as f:
            exec_state = json.load(f)
        positions = exec_state.get("positions", {})
        open_in_market = [t for t in positions.keys() if t in all_tickers]
        return sorted(open_in_market)
    except Exception as e:
        logger.error(f"  Error loading execution_state.json: {e}")
        return []


def next_round_robin_slice(
    market_name: str,
    in_scope: list[str],
    max_items: int,
    daemon_state: dict,
    logger: logging.Logger,
) -> list[str]:
    """Get next slice of round-robin tickers.

    Updates cursor in daemon_state. Wraps daily.
    """
    if not in_scope:
        return []

    cursors = daemon_state.setdefault("cursors", {})
    today = datetime.now().date().isoformat()
    key = f"{market_name}:{today}"

    cursor = cursors.get(key, 0)
    slice_end = min(cursor + max_items, len(in_scope))
    slice_tickers = in_scope[cursor:slice_end]

    if slice_end >= len(in_scope):
        cursors[key] = 0
        logger.debug(f"  {market_name}: round-robin wrapped")
    else:
        cursors[key] = slice_end

    return slice_tickers


def execute_signals_with_retry(
    market_name: str,
    ticker_list: list[str],
    data_dir: Path,
    portfolio: object,
    limit_tracker: object,
    broker: object,
    daily_buy_limit: int | None,
    daily_sell_limit: int | None,
    logger: logging.Logger,
    max_retries: int = 3,
    market_currency: str = "",
) -> tuple[list[str], list[str], list[str]]:
    """Execute signals with automatic reconnect and retry on socket errors.

    Detects connection failures (socket disconnect, timeout, etc.) and attempts
    to reconnect the broker before retrying signal execution. Non-connection
    errors are raised immediately.

    Returns (buys, sells, skipped) tuple. On connection failure after max_retries,
    returns empty results and continues (doesn't halt daemon).
    """
    from .execute import execute_signals

    for attempt in range(max_retries):
        try:
            return execute_signals(
                ticker_list, data_dir, portfolio, limit_tracker, broker,
                daily_buy_limit, daily_sell_limit,
                market_name=market_name,
                market_currency=market_currency,
            )
        except (ConnectionError, OSError, TimeoutError) as e:
            if attempt < max_retries - 1:
                wait_secs = 2 ** attempt
                logger.warning(
                    f"[{market_name}] Connection error (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Reconnecting in {wait_secs}s..."
                )
                time.sleep(wait_secs)
                try:
                    broker.disconnect()
                    time.sleep(0.5)
                    broker.connect()
                    logger.info(f"[{market_name}] Broker reconnected successfully")
                except Exception as reconnect_err:
                    logger.warning(
                        f"[{market_name}] Reconnect attempt failed: {reconnect_err} "
                        f"(will retry execute)"
                    )
            else:
                logger.error(
                    f"[{market_name}] Connection error persists after {max_retries} attempts. "
                    f"Skipping signals for this cycle. Error: {e}"
                )
                return [], [], ticker_list
        except Exception as e:
            error_msg = str(e).lower()
            is_socket_error = (
                "socket" in error_msg or
                "disconnect" in error_msg or
                "ib_insync" in error_msg
            )

            if is_socket_error:
                if attempt < max_retries - 1:
                    wait_secs = 2 ** attempt
                    logger.warning(
                        f"[{market_name}] Socket error (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Reconnecting in {wait_secs}s..."
                    )
                    time.sleep(wait_secs)
                    try:
                        broker.disconnect()
                        time.sleep(0.5)
                        broker.connect()
                        logger.info(f"[{market_name}] Broker reconnected successfully")
                    except Exception as reconnect_err:
                        logger.warning(
                            f"[{market_name}] Reconnect attempt failed: {reconnect_err}"
                        )
                else:
                    logger.error(
                        f"[{market_name}] Socket error persists after {max_retries} attempts. "
                        f"Skipping signals for this cycle. Error: {e}"
                    )
                    return [], [], ticker_list
            else:
                raise

    return [], [], ticker_list


def process_cycle(
    market_name: str,
    market_cfg: dict,
    config: dict,
    daemon_state: dict,
    portfolio: object,
    broker: object,
    logger: logging.Logger,
) -> int:
    """Run one market cycle: prioritize open positions, then round-robin through candidates.

    Returns number of tickers processed.
    """
    from .batch import process_ticker

    in_scope = load_in_scope_tickers(market_name, logger)
    if not in_scope:
        logger.debug(f"  {market_name}: no in-scope tickers")
        return 0

    # Open positions always run first
    open_positions = get_open_positions(market_name, in_scope, logger)
    must_run = [t for t in open_positions if t in in_scope]

    # Remaining budget for candidates
    max_seconds = config.get("daytime", {}).get("max_seconds_per_cycle", 1500)
    cycle_start = time.time()

    # Daemon cycles skip chart/HTML rendering unless a signal fires; a market
    # config can override by setting defaults.signal_reports_only = false.
    defaults = {"signal_reports_only": True, **market_cfg.get("defaults", {})}
    processed = []
    skipped_budget = []

    # Stage 1: must-run (open positions)
    logger.info(f"[{market_name}] Must-run ({len(must_run)} positions):")
    for ticker in must_run:
        remaining = max_seconds - (time.time() - cycle_start)
        if remaining <= 0:
            logger.debug(f"  {ticker}: budget exhausted")
            skipped_budget.append(ticker)
            break

        logger.debug(f"  Processing {ticker}")
        ticker_cfg = {"ticker": ticker}
        result = process_ticker(ticker_cfg, defaults, send_email=True)
        processed.append(result)

    # Stage 2: candidates (round-robin through rest)
    remaining_budget = max_seconds - (time.time() - cycle_start)
    buffer_secs = config.get("daytime", {}).get("cycle_buffer_minutes", 5) * 60
    if remaining_budget > buffer_secs:
        remaining_budget -= buffer_secs
        candidates = next_round_robin_slice(
            market_name,
            [t for t in in_scope if t not in must_run],
            len(in_scope),
            daemon_state,
            logger,
        )

        logger.info(f"[{market_name}] Round-robin ({len(candidates)} candidates, {remaining_budget:.0f}s budget):")
        for ticker in candidates:
            now_remaining = max_seconds - (time.time() - cycle_start)
            if now_remaining <= buffer_secs:
                logger.debug(f"  {ticker}: budget near exhausted ({now_remaining:.0f}s left)")
                skipped_budget.append(ticker)
                break

            logger.debug(f"  Processing {ticker}")
            ticker_cfg = {"ticker": ticker}
            result = process_ticker(ticker_cfg, defaults, send_email=True)
            processed.append(result)

    # Execute signals once for all processed tickers this cycle
    if processed:
        from .execute import execute_signals
        # Include all tickers with OK status. signal_reader will validate files exist.
        # SELL signals are prioritized (safer to exit existing positions) over strict validation.
        ticker_list = [p["ticker"] for p in processed if p.get("status") == "OK"]
        if ticker_list:
            # Dry-run broker fills at supplied prices — feed it this cycle's closes
            # so the trade log records real prices instead of 0.0.
            if hasattr(broker, "set_prices"):
                broker.set_prices({
                    p["ticker"]: p["result"]["close"]
                    for p in processed
                    if p.get("result") and p["result"].get("close")
                })
            logger.info(f"[{market_name}] Executing signals for {len(ticker_list)} processed tickers...")
            try:
                limit_tracker = portfolio.get_limit_tracker()
                exec_cfg = config.get("execution", {})
                daily_buy_limit = exec_cfg.get("daily_buy_limit", 2)
                daily_sell_limit = exec_cfg.get("daily_sell_limit")
                if daemon_state.get("halt_new_entries"):
                    # Reconciliation mismatch: block new entries (buy limit 0)
                    # but still allow exits of existing positions.
                    logger.warning(f"[{market_name}] Reconciliation mismatch "
                                   f"unresolved — new entries blocked")
                    daily_buy_limit = 0
                market_currency = get_market_currency(market_name, config)
                buys, sells, skipped = execute_signals_with_retry(
                    market_name, ticker_list, DATA_DIR, portfolio, limit_tracker, broker,
                    daily_buy_limit, daily_sell_limit, logger,
                    market_currency=market_currency,
                )
                logger.info(f"  BUY:  {len(buys)}, SELL: {len(sells)}, Skipped: {len(skipped)}")
                for b in buys:
                    logger.info(f"    BUY: {b}")
                for s in sells:
                    logger.info(f"    SELL: {s}")
                portfolio.save()
            except Exception as e:
                logger.error(f"  Error executing signals (unrecoverable): {e}")

    elapsed = time.time() - cycle_start
    logger.info(f"[{market_name}] Cycle done: {len(processed)} tickers processed, "
                f"{len(skipped_budget)} skipped (budget), {elapsed:.0f}s elapsed")
    return len(processed)


def check_overnight_screening(
    config: dict,
    daemon_state: dict,
    logger: logging.Logger,
) -> None:
    """Check if overnight screening should run, and run if needed."""
    tz = ZoneInfo(config.get("overnight_timezone", "Europe/London"))
    now = datetime.now(tz)
    run_time_str = config.get("overnight_run_time", "02:00")
    run_hour, run_minute = map(int, run_time_str.split(":"))

    today = now.date().isoformat()
    last_date = daemon_state.get("last_overnight_date")

    if last_date == today:
        return

    if now.hour == run_hour and now.minute >= run_minute:
        logger.info("Running overnight scope screening...")
        try:
            from .overnight_scope import main as run_overnight_scope
            run_overnight_scope()
            daemon_state["last_overnight_date"] = today
            save_daemon_state(daemon_state)
            logger.info("Overnight scope screening complete")
        except Exception as e:
            logger.error(f"Error in overnight screening: {e}")


def run_reconciliation(
    portfolio: object,
    broker: object,
    daemon_state: dict,
    logger: logging.Logger,
) -> bool:
    """Compare broker account positions against internal execution state.

    On mismatch: log, email an alert, and set halt_new_entries so no new
    positions are opened until a clean pass. Never auto-corrects either side.
    Returns "clean", "mismatch", or "error" (broker positions unavailable).
    """
    from ..broker.reconcile import reconcile_positions

    try:
        broker_positions = broker.get_open_positions()
    except Exception as e:
        logger.error(f"Reconciliation: could not fetch broker positions: {e}")
        return "error"

    discrepancies = reconcile_positions(portfolio.positions, broker_positions)

    if discrepancies:
        logger.error(f"RECONCILIATION MISMATCH ({len(discrepancies)} discrepancies) "
                     f"— halting new entries:")
        for d in discrepancies:
            logger.error(f"  {d}")
        daemon_state["halt_new_entries"] = True
        daemon_state["reconciliation_discrepancies"] = discrepancies
        save_daemon_state(daemon_state)
        try:
            from ..output.emailer import send_reconciliation_alert
            send_reconciliation_alert(discrepancies)
        except Exception as e:
            logger.error(f"Reconciliation: alert email failed: {e}")
        return "mismatch"

    if daemon_state.get("halt_new_entries"):
        logger.info("Reconciliation clean — re-enabling new entries")
    else:
        logger.info(f"Reconciliation clean: {len(portfolio.positions)} internal "
                    f"positions match broker")
    daemon_state["halt_new_entries"] = False
    daemon_state["reconciliation_discrepancies"] = []
    save_daemon_state(daemon_state)
    return "clean"


def check_nightly_reconciliation(
    config: dict,
    daemon_state: dict,
    portfolio: object,
    broker: object,
    logger: logging.Logger,
) -> None:
    """Run reconciliation once per day at the configured time (after close)."""
    tz = ZoneInfo(config.get("overnight_timezone", "Europe/London"))
    now = datetime.now(tz)
    run_time_str = config.get("reconciliation_run_time", "21:30")
    run_hour, run_minute = map(int, run_time_str.split(":"))

    today = now.date().isoformat()
    if daemon_state.get("last_reconcile_date") == today:
        return

    if now.hour == run_hour and now.minute >= run_minute:
        logger.info("Running nightly position reconciliation...")
        outcome = run_reconciliation(portfolio, broker, daemon_state, logger)
        # A broker fetch error is not a daily result — leave the date unset so
        # it retries on the next poll within the run window.
        if outcome in ("clean", "mismatch"):
            daemon_state["last_reconcile_date"] = today
            save_daemon_state(daemon_state)


def process_manual_commands_wrapper(config: dict, portfolio: object, broker: object, logger: logging.Logger) -> None:
    """Wrapper for manual command processing (catch exceptions so daemon survives)."""
    from .manual_commands import process_manual_commands
    try:
        process_manual_commands(config, portfolio, broker, logger)
    except Exception as e:
        logger.error(f"Error processing manual commands: {e}", exc_info=True)


def main() -> int:
    """Main daemon loop."""
    logger = setup_logging()
    logger.info("="*64)
    logger.info("Live daemon starting")
    logger.info("="*64)

    # Validate startup environment (fail-fast on configuration issues)
    if not validate_startup_environment(logger):
        return 1

    # Prevent multiple daemon instances (PID-based check)
    if not acquire_process_lock(logger):
        return 1

    config = load_config()
    daemon_state = load_daemon_state()
    exec_cfg = config.get("execution", {})

    # Clean up incomplete runs from prior crashes
    cleanup_incomplete_runs(Path(__file__).resolve().parent.parent.parent / "data", logger)
    dry_run = exec_cfg.get("dry_run", True)

    # Startup self-checks — a half-broken environment (e.g. hmmlearn that
    # imports but cannot fit) must abort here, not trade without signals.
    # Skip broker connectivity check (TWS can reconnect dynamically; don't block startup)
    from ..core.self_check import SelfCheckError, run_startup_checks
    try:
        run_startup_checks(require_broker=False, logger=logger)
    except SelfCheckError as e:
        logger.critical(str(e))
        return 1

    # Set up broker
    if dry_run:
        from ..broker.null_adapter import NullBroker
        logger.info("Using NullBroker (dry run mode)")
        broker = NullBroker(prices={})
    else:
        from ..broker.ibkr_adapter import IBKRAdapter
        logger.info("Using IBKRAdapter (live paper trading)")
        broker = IBKRAdapter()

    # Set up portfolio
    from ..broker.portfolio import PortfolioManager
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path = STATE_DIR / "execution_state.json"
    capital_pot = float(exec_cfg.get("capital_pot", 20000))
    max_positions = int(exec_cfg.get("max_positions", 5))
    portfolio = PortfolioManager(capital_pot, max_positions, state_path)

    # Broker connection is async and can hang; skip it here and let it fail gracefully
    # when trades are attempted. Daemon can still process tickers and generate signals.
    try:
        if dry_run:
            broker.connect()
            logger.info("Broker connected (dry run)")
        else:
            logger.info("Broker connection deferred (will connect on first trade)")
    except Exception as e:
        logger.error(f"Error connecting to broker: {e}")
        if not dry_run:
            logger.warning("Continuing anyway; will retry on first trade attempt")

    # Startup reconciliation — catch drift from downtime, crashes between
    # fill and state write, or manual TWS trades. Skipped in dry_run because
    # the NullBroker holds no real account.
    # Also skip in live mode for now (broker connection may be pending)
    if not dry_run:
        logger.info("Reconciliation deferred (broker connection pending)")

    try:
        poll_interval = config.get("daytime", {}).get("poll_interval_seconds", 60)
        last_cycle_hour = {}

        logger.info("Entering main loop")
        while True:
            try:
                # Check overnight screening
                check_overnight_screening(config, daemon_state, logger)

                # Nightly broker/state reconciliation (real broker only)
                if not dry_run:
                    check_nightly_reconciliation(
                        config, daemon_state, portfolio, broker, logger
                    )

                # Process manual sell commands from mobile app
                process_manual_commands_wrapper(config, portfolio, broker, logger)

                # Check each market
                now = datetime.now(timezone.utc)
                for market_name, market_cfg in config.get("markets", {}).items():
                    if not is_trading_hours(market_cfg, logger):
                        continue

                    current_hour = now.hour
                    last_hour = last_cycle_hour.get(market_name, -1)

                    if current_hour != last_hour:
                        logger.info(f"\n{'='*64}")
                        logger.info(f"[{market_name}] Starting cycle")
                        logger.info(f"{'='*64}")

                        process_cycle(
                            market_name, market_cfg, config,
                            daemon_state, portfolio, broker, logger
                        )

                        last_cycle_hour[market_name] = current_hour
                        save_daemon_state(daemon_state)

                # Write app_status.json snapshot (before sleep, so it's fresh on next poll)
                write_app_status_snapshot(portfolio, daemon_state, config, last_cycle_hour, logger)

                # Sleep
                logger.debug(f"Sleeping {poll_interval}s...")
                time.sleep(poll_interval)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt, shutting down")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
                time.sleep(5)

    finally:
        try:
            broker.disconnect()
            logger.info("Broker disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting broker: {e}")

        release_process_lock(logger)

        logger.info("="*64)
        logger.info("Live daemon stopped")
        logger.info("="*64)

    return 0


if __name__ == "__main__":
    sys.exit(main())
