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
from collections.abc import Callable
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


# Handle to the held lock file. Must stay open for the daemon's lifetime:
# the OS releases the lock the instant this process dies, so a held lock
# always means a live daemon — no PID-liveness guessing.
_lock_handle = None

_DAEMON_CMDLINE_MARKERS = ("markov_cli.live_daemon",
                           "markov_cli\\live_daemon.py",
                           "markov_cli/live_daemon.py")


def _is_daemon_cmdline(cmdline: list[str] | None) -> bool:
    joined = " ".join(cmdline or [])
    return any(m in joined for m in _DAEMON_CMDLINE_MARKERS)


def _try_lock(handle) -> bool:
    """Take a non-blocking exclusive OS lock on *handle*. True on success."""
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _read_holder_pid() -> int | None:
    """PID recorded by the current/most recent lock holder, if readable."""
    pid_path = STATE_DIR / "daemon.pid"
    try:
        return int(pid_path.read_text(encoding="utf-8").split("|")[0])
    except Exception:
        return None


def _kill_daemon_process(pid: int, logger: logging.Logger) -> bool:
    """Kill *pid* and its subtree, but only if its cmdline is a live_daemon.

    The cmdline check guards against PID reuse — never kill an unrelated
    process that happens to have inherited a recorded PID.
    """
    if psutil is None:
        logger.error("psutil unavailable — cannot kill other daemon instances")
        return False
    try:
        proc = psutil.Process(pid)
        if not _is_daemon_cmdline(proc.cmdline()):
            logger.warning(f"PID {pid} is not a live_daemon process "
                           f"({proc.name()}); refusing to kill")
            return False
        victims = [proc] + proc.children(recursive=True)
        for p in victims:
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
        gone, alive = psutil.wait_procs(victims, timeout=5)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=5)
        logger.warning(f"Killed daemon instance PID {pid} "
                       f"(+{len(victims) - 1} child processes)")
        return True
    except psutil.NoSuchProcess:
        return True
    except psutil.AccessDenied:
        logger.error(f"Access denied killing PID {pid} — it may be running "
                     f"elevated; kill it from an elevated shell")
        return False
    except Exception as e:
        logger.error(f"Failed to kill PID {pid}: {e}")
        return False


def kill_stray_daemons(logger: logging.Logger) -> int:
    """Kill orphan live_daemon processes that aren't part of this instance.

    An orphan is any python process with a live_daemon cmdline that is not
    this process, an ancestor (the uv/venv shim chain), or a descendant.
    Call only while holding the process lock — the lock holder is the single
    authority allowed to kill others.
    """
    if psutil is None:
        logger.warning("psutil unavailable — skipping orphan daemon sweep")
        return 0

    me = psutil.Process()
    keep = {me.pid}
    keep.update(p.pid for p in me.parents())
    keep.update(p.pid for p in me.children(recursive=True))

    killed = 0
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.pid in keep:
                continue
            if not (proc.info["name"] or "").lower().startswith("python"):
                continue
            if not _is_daemon_cmdline(proc.info["cmdline"]):
                continue
            logger.warning(f"Found stray daemon process PID {proc.pid}")
            if _kill_daemon_process(proc.pid, logger):
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if killed:
        logger.warning(f"Orphan sweep killed {killed} stray daemon instance(s)")
    return killed


def acquire_process_lock(logger: logging.Logger, takeover: bool = False) -> bool:
    """Acquire the exclusive daemon lock to prevent multiple instances.

    Holds an OS-level file lock for the process lifetime; it cannot go stale
    (the OS drops it on process death) and cannot be stolen by a second
    instance while the holder is alive.

    With takeover=True (used by the Task Scheduler start command), a live
    holder is killed and the lock taken over — the supervisor-started
    instance always wins. Without it, this instance backs off.

    Returns True if lock acquired. On False, another daemon is running.
    """
    global _lock_handle
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = STATE_DIR / "daemon.lock"

    tried_takeover = False
    deadline = time.time() + 15
    while True:
        handle = open(lock_path, "a+", encoding="utf-8")
        if _try_lock(handle):
            break
        handle.close()

        holder = _read_holder_pid()
        if not takeover:
            logger.error(f"Daemon already running (PID {holder or 'unknown'}). "
                         f"Exiting; rerun with --takeover to replace it.")
            return False
        if not tried_takeover:
            tried_takeover = True
            logger.warning(f"Lock held by PID {holder or 'unknown'} — taking over")
            if holder is not None and not _kill_daemon_process(holder, logger):
                logger.critical("Takeover failed: could not kill lock holder")
                return False
        if time.time() > deadline:
            logger.critical("Takeover failed: lock still held after kill")
            return False
        # OS may take a moment to release the dead holder's lock
        time.sleep(0.5)

    try:
        pid_path = STATE_DIR / "daemon.pid"
        pid_path.write_text(f"{os.getpid()}|{datetime.now().isoformat()}\n",
                            encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not write daemon.pid: {e}")

    _lock_handle = handle
    logger.info(f"Process lock acquired (PID {os.getpid()})")
    return True


def release_process_lock(logger: logging.Logger) -> None:
    """Release the process lock on shutdown."""
    global _lock_handle
    try:
        if _lock_handle is not None:
            if os.name == "nt":
                import msvcrt
                _lock_handle.seek(0)
                msvcrt.locking(_lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
            _lock_handle.close()
            _lock_handle = None
        (STATE_DIR / "daemon.pid").unlink(missing_ok=True)
        logger.info("Process lock released")
    except Exception as e:
        logger.error(f"Failed to release lock: {e}")


def validate_startup_environment(logger: logging.Logger) -> bool:
    """Validate that startup is possible. Fail loudly if not.

    Checks:
    - Necessary directories exist and are writable
    - SMTP credentials are set (needed for email alerts)

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

    # Check SMTP credentials
    smtp_user = os.environ.get("SMTP_USER", "").strip()
    smtp_password = os.environ.get("SMTP_PASSWORD", "").strip()
    if not smtp_user:
        errors.append("SMTP_USER environment variable not set — email alerts will fail")
    if not smtp_password:
        errors.append("SMTP_PASSWORD environment variable not set — email alerts will fail")

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
        "paused_by_user": daemon_state.get("paused_by_user", False),
        "reconciliation_discrepancies": daemon_state.get("reconciliation_discrepancies", []),
        "last_reconcile_date": daemon_state.get("last_reconcile_date", ""),
        "trades_today": trades_today,
        "markets": markets_status,
        "positions": positions_snapshot,
    }

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    status_path = STATE_DIR / "app_status.json"
    atomic_write_json(status_path, snapshot)


def _write_app_status_snapshot_safe(
    portfolio: object, daemon_state: dict, config: dict, last_cycle_hour: dict, logger: logging.Logger
) -> None:
    """write_app_status_snapshot(), swallowing errors so a snapshot failure
    can't interrupt the ticker-processing loop it's interleaved into."""
    try:
        write_app_status_snapshot(portfolio, daemon_state, config, last_cycle_hour, logger)
    except Exception as e:
        logger.error(f"Failed to write app_status.json: {e}", exc_info=True)


def is_trading_hours(market_cfg: dict, logger: logging.Logger, *, now: datetime | None = None) -> bool:
    """Check if market is currently in trading hours."""
    tz = ZoneInfo(market_cfg["timezone"])
    if now is None:
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


def load_ticker_overrides(market_name: str, logger: logging.Logger) -> dict[str, dict]:
    """Load per-ticker strategy overrides from in_scope_<market>.json."""
    scope_path = STATE_DIR / f"in_scope_{market_name}.json"
    if not scope_path.exists():
        return {}
    try:
        with open(scope_path, encoding="utf-8") as f:
            result = json.load(f)
        return result.get("overrides", {})
    except Exception as e:
        logger.error(f"  Error loading overrides from in_scope_{market_name}.json: {e}")
        return {}


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
    daemon_state: dict | None = None,
    protective_stops: bool = False,
    stop_buffer_pct: float = 1.5,
    *,
    execute_signals: Callable | None = None,
    save_state: Callable | None = None,
    send_interrupt_alert: Callable | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Execute signals with automatic reconnect and retry on socket errors.

    Detects connection failures (socket disconnect, timeout, etc.) and attempts
    to reconnect the broker before retrying signal execution. Non-connection
    errors are raised immediately.

    Returns (buys, sells, skipped) tuple. On connection failure after max_retries,
    returns empty results and continues (doesn't halt daemon).
    """
    from .execute import ExecutionInterrupted
    if execute_signals is None:
        from .execute import execute_signals as _execute_signals
        execute_signals = _execute_signals
    if save_state is None:
        save_state = save_daemon_state

    for attempt in range(max_retries):
        try:
            return execute_signals(
                ticker_list, data_dir, portfolio, limit_tracker, broker,
                daily_buy_limit, daily_sell_limit,
                market_name=market_name,
                market_currency=market_currency,
                protective_stops=protective_stops,
                stop_buffer_pct=stop_buffer_pct,
            )
        except ExecutionInterrupted as exc:
            # exc.unresolved always contains at least the ticker that was being
            # processed when the exception hit — its broker-call outcome is
            # unknown (order may have reached IB before the ack was lost).
            # There is no "nothing happened yet, safe to retry" case here.
            logger.critical(
                f"[{market_name}] Execution interrupted — outcome of {exc.unresolved} "
                f"unknown (broker call may have gone through before the connection "
                f"dropped). {len(exc.buys)} buy(s)/{len(exc.sells)} sell(s) confirmed "
                f"placed before the interrupt. Halting new entries, not retrying. "
                f"Error: {exc.original}"
            )
            if daemon_state is not None:
                daemon_state["halt_new_entries"] = True
                save_state(daemon_state)
            try:
                if send_interrupt_alert is None:
                    from ..output.emailer import send_execution_interrupted_alert
                    send_interrupt_alert = send_execution_interrupted_alert
                send_interrupt_alert(
                    market_name, exc.original, exc.buys, exc.sells, exc.unresolved
                )
            except Exception as email_err:
                logger.error(f"[{market_name}] Execution-interrupted alert email failed: {email_err}")
            return exc.buys, exc.sells, exc.skipped + exc.unresolved
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
    last_cycle_hour: dict | None = None,
    protective_stops: bool = False,
    stop_buffer_pct: float = 1.5,
) -> int:
    """Run one market cycle: prioritize open positions, then round-robin through candidates.

    Returns number of tickers processed.
    """
    from .batch import process_ticker
    from ..output.trade_state import get_open_strategy

    if last_cycle_hour is None:
        last_cycle_hour = {}

    in_scope = load_in_scope_tickers(market_name, logger)
    overrides = load_ticker_overrides(market_name, logger)
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

        # Check for manual sell/pause commands between tickers rather than only
        # once per full market pass — a user-initiated sell shouldn't queue
        # behind an entire round-robin scan (can run ~20+ min).
        process_manual_commands_wrapper(config, portfolio, broker, logger, daemon_state)
        _write_app_status_snapshot_safe(portfolio, daemon_state, config, last_cycle_hour, logger)

        logger.debug(f"  Processing {ticker}")
        ticker_cfg = {"ticker": ticker, **overrides.get(ticker, {})}
        pinned_strategy = get_open_strategy(ticker)
        if pinned_strategy:
            ticker_cfg["strategy"] = pinned_strategy
        result = process_ticker(ticker_cfg, defaults, send_email=True)
        if not str(result.get("status", "")).startswith("OK"):
            logger.warning(f"[{market_name}] {ticker} processing failed: {result.get('status')}")
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

            process_manual_commands_wrapper(config, portfolio, broker, logger, daemon_state)
            _write_app_status_snapshot_safe(portfolio, daemon_state, config, last_cycle_hour, logger)

            logger.debug(f"  Processing {ticker}")
            ticker_cfg = {"ticker": ticker, **overrides.get(ticker, {})}
            result = process_ticker(ticker_cfg, defaults, send_email=True)
            if not str(result.get("status", "")).startswith("OK"):
                logger.warning(f"[{market_name}] {ticker} processing failed: {result.get('status')}")
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
                    logger.warning(f"[{market_name}] Reconciliation mismatch unresolved — new entries blocked")
                    daily_buy_limit = 0
                elif daemon_state.get("paused_by_user"):
                    logger.info(f"[{market_name}] Buying paused by user — new entries blocked")
                    daily_buy_limit = 0
                market_currency = get_market_currency(market_name, config)
                buys, sells, skipped = execute_signals_with_retry(
                    market_name, ticker_list, DATA_DIR, portfolio, limit_tracker, broker,
                    daily_buy_limit, daily_sell_limit, logger,
                    market_currency=market_currency,
                    daemon_state=daemon_state,
                    protective_stops=protective_stops,
                    stop_buffer_pct=stop_buffer_pct,
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


def check_protective_stops(
    portfolio: object,
    broker: object,
    logger: logging.Logger,
    stop_buffer_pct: float = 1.5,
) -> None:
    """Invariant check: every open position has a live protective stop.

    Re-places missing stops, journals vanished-stop-with-fill as stop_loss,
    and cancels orphan stops with no matching position.
    """
    try:
        open_stops = broker.get_open_stop_orders()
    except Exception as e:
        logger.warning(f"check_protective_stops: could not fetch open stops: {e}")
        return

    for ticker, pos in list(portfolio.positions.items()):
        perm_id = pos.get("stop_perm_id")
        if perm_id and perm_id in open_stops:
            continue

        if perm_id:
            try:
                fill = broker.get_stop_fill(perm_id)
                if fill is not None:
                    logger.warning(f"{ticker}: protective stop FILLED @ {fill.fill_price}")
                    portfolio.record_exit(ticker, fill, exit_type="stop_loss")
                    portfolio.clear_stop_order(ticker)
                    portfolio.save()
                    continue
            except Exception as e:
                logger.warning(f"{ticker}: error checking stop fill: {e}")

            logger.warning(f"{ticker}: stop {perm_id} vanished without execution — re-placing")

        from ..broker.types import StopOrderRequest
        try:
            buffered_stop = pos.get("stop_level", 0) * (1 - stop_buffer_pct / 100)
            if not buffered_stop or buffered_stop <= 0:
                buffered_stop = pos.get("stop_price")
            if not buffered_stop or buffered_stop <= 0:
                logger.warning(f"{ticker}: cannot determine stop price, skipping re-place")
                continue

            req = StopOrderRequest(ticker, pos["quantity"], buffered_stop)
            result = broker.place_stop_order(req)
            if result:
                portfolio.set_stop_order(ticker, result.perm_id, result.stop_price)
                portfolio.save()
            else:
                logger.warning(f"{ticker}: stop re-place rejected")
        except Exception as e:
            logger.warning(f"{ticker}: error re-placing stop: {e}")

    for perm_id, info in open_stops.items():
        if info.ticker not in portfolio.positions:
            logger.warning(f"Orphan stop {perm_id} on {info.ticker} — cancelling")
            try:
                broker.cancel_stop_order(perm_id)
            except Exception as e:
                logger.warning(f"Error cancelling orphan stop {perm_id}: {e}")


def run_reconciliation(
    portfolio: object,
    broker: object,
    daemon_state: dict,
    logger: logging.Logger,
    *,
    save_state: Callable | None = None,
    send_alert: Callable | None = None,
) -> bool:
    """Compare broker account positions against internal execution state.

    On mismatch: log, email an alert, and set halt_new_entries so no new
    positions are opened until a clean pass. Never auto-corrects either side.
    Returns "clean", "mismatch", or "error" (broker positions unavailable).
    """
    from ..broker.reconcile import reconcile_positions
    if save_state is None:
        save_state = save_daemon_state

    if not broker.is_connected():
        logger.info("Reconciliation: broker not connected — connecting...")
        try:
            broker.connect()
        except Exception as e:
            logger.error(f"Reconciliation: broker connect failed: {e}")
            return "error"

    logger.debug("Reconciliation: fetching broker positions...")
    try:
        broker_positions = broker.get_open_positions()
    except Exception as e:
        logger.error(f"Reconciliation: could not fetch broker positions: {e}")
        return "error"

    logger.info(f"Reconciliation: comparing {len(portfolio.positions)} internal position(s) against broker...")
    from ..broker.reconcile import check_stop_fills_for_missing_positions
    resolved_stops = check_stop_fills_for_missing_positions(
        portfolio.positions, broker_positions, broker, portfolio
    )
    if resolved_stops:
        portfolio.save()
        for resolution in resolved_stops:
            logger.info(f"  {resolution}")

    discrepancies = reconcile_positions(portfolio.positions, broker_positions)

    if discrepancies:
        logger.error(f"RECONCILIATION MISMATCH ({len(discrepancies)} discrepancies) "
                     f"— halting new entries:")
        for d in discrepancies:
            logger.error(f"  {d}")
        last_discrepancies = daemon_state.get("reconciliation_discrepancies", [])
        already_alerted = daemon_state.get("reconciliation_mismatch_alerted", False)

        daemon_state["halt_new_entries"] = True
        daemon_state["reconciliation_discrepancies"] = discrepancies

        if discrepancies == last_discrepancies and already_alerted:
            logger.info("Mismatch unchanged since last alert — halt stays set, email suppressed")
        else:
            logger.warning(f"About to send reconciliation mismatch alert ({len(discrepancies)} discrepancies)...")
            try:
                if send_alert is None:
                    from ..output.emailer import send_reconciliation_alert
                    send_alert = send_reconciliation_alert
                send_alert(discrepancies)
            except Exception as e:
                logger.error(f"Reconciliation: alert email failed: {e}")
            daemon_state["reconciliation_mismatch_alerted"] = True

        save_state(daemon_state)
        return "mismatch"

    if daemon_state.get("halt_new_entries"):
        logger.info("Reconciliation clean — re-enabling new entries")
    else:
        logger.info(f"Reconciliation clean: {len(portfolio.positions)} internal "
                    f"positions match broker")
    daemon_state["halt_new_entries"] = False
    daemon_state["reconciliation_discrepancies"] = []
    daemon_state["reconciliation_mismatch_alerted"] = False
    save_state(daemon_state)
    return "clean"


def run_startup_reconciliation(
    daemon_state: dict,
    portfolio: object,
    broker: object,
    logger: logging.Logger,
    *,
    run_recon: Callable | None = None,
    save_state: Callable | None = None,
    send_interrupt_alert: Callable | None = None,
    marker_path: Path | None = None,
) -> bool:
    """Reconciliation pass on daemon startup.

    Forces halt_new_entries = True immediately (fail-safe), reads in-flight
    marker if any, delegates to run_reconciliation for the actual comparison,
    and escalates to immediate alert if broker unreachable with marker present.
    Returns True only when reconciliation completes (clean or mismatch), False
    if broker unreachable (caller retries).
    """
    from ..broker.in_flight_marker import read_marker

    if run_recon is None:
        run_recon = run_reconciliation
    if save_state is None:
        save_state = save_daemon_state
    if marker_path is None:
        marker_path = STATE_DIR / "order_in_flight.json"

    daemon_state["halt_new_entries"] = True

    marker = read_marker(marker_path)
    logger.info(f"About to run startup reconciliation (in-flight marker present: {marker is not None})...")

    outcome = run_recon(portfolio, broker, daemon_state, logger,
                        save_state=save_state)

    if outcome == "error":
        logger.warning("Startup reconciliation could not reach broker — will retry next poll")
        if marker is not None:
            logger.critical("About to send startup in-flight alert (broker unreachable, marker present)...")
            try:
                if send_interrupt_alert is None:
                    from ..output.emailer import send_execution_interrupted_alert
                    send_interrupt_alert = send_execution_interrupted_alert
                send_interrupt_alert(
                    "startup", RuntimeError("broker unreachable at startup with an in-flight order marker present"),
                    [], [], [marker["ticker"]]
                )
            except Exception as e:
                logger.error(f"Startup reconciliation: in-flight alert email failed: {e}")
        return False

    logger.info(f"Startup reconciliation resolved: {outcome} — halt {'remains set' if outcome == 'mismatch' else 'cleared'}")

    if marker is not None:
        # A market order can be accepted by IBKR moments before the client's
        # socket drops — the fill confirmation never arrives, but the order
        # itself is still live server-side and can complete any time after
        # this reconciliation pass. Position comparison alone can't catch
        # that (it hasn't filled yet), so check the broker's still-working
        # orders for this exact ticker before trusting the marker away.
        stale_order = None
        try:
            for order in broker.get_open_orders():
                if order["ticker"] == marker["ticker"]:
                    stale_order = order
                    break
        except Exception as e:
            logger.error(f"Startup reconciliation: could not check open orders for in-flight marker: {e}")
            daemon_state["halt_new_entries"] = True
            return True

        if stale_order is not None:
            logger.critical(
                f"In-flight marker for {marker['ticker']} still has a live order at the "
                f"broker (status={stale_order['status']}) — halt stays set, marker kept "
                f"for manual resolution."
            )
            daemon_state["halt_new_entries"] = True
            try:
                if send_interrupt_alert is None:
                    from ..output.emailer import send_execution_interrupted_alert
                    send_interrupt_alert = send_execution_interrupted_alert
                send_interrupt_alert(
                    "startup",
                    RuntimeError(
                        f"in-flight order for {marker['ticker']} is still live at the broker "
                        f"(status={stale_order['status']}) after a client disconnect"
                    ),
                    [], [], [marker["ticker"]]
                )
            except Exception as e:
                logger.error(f"Startup reconciliation: stale-order alert email failed: {e}")
            return True

        logger.info("In-flight marker cleared after reconciliation completed")
        marker_path.unlink(missing_ok=True)

    return True


def check_nightly_reconciliation(
    config: dict,
    daemon_state: dict,
    portfolio: object,
    broker: object,
    logger: logging.Logger,
    protective_stops: bool = False,
    stop_buffer_pct: float = 1.5,
    *,
    run_recon: Callable | None = None,
    save_state: Callable | None = None,
    send_alert: Callable | None = None,
) -> None:
    """Run reconciliation once per day at the configured time (after close)."""
    if run_recon is None:
        run_recon = run_reconciliation
    if save_state is None:
        save_state = save_daemon_state
    tz = ZoneInfo(config.get("overnight_timezone", "Europe/London"))
    now = datetime.now(tz)
    run_time_str = config.get("reconciliation_run_time", "21:30")
    run_hour, run_minute = map(int, run_time_str.split(":"))

    today = now.date().isoformat()
    if daemon_state.get("last_reconcile_date") == today:
        return

    if now.hour == run_hour and now.minute >= run_minute:
        logger.info("Running nightly position reconciliation...")
        outcome = run_recon(portfolio, broker, daemon_state, logger)
        if protective_stops:
            check_protective_stops(portfolio, broker, logger, stop_buffer_pct)
        # A broker fetch error is not a daily result — leave the date unset so
        # it retries on the next poll within the run window.
        if outcome in ("clean", "mismatch"):
            daemon_state["last_reconcile_date"] = today
            daemon_state["reconciliation_consecutive_error_days"] = 0
            daemon_state["reconciliation_alert_sent"] = False
            save_state(daemon_state)
        elif outcome == "error":
            # Count at most one error per calendar day — the run window can
            # retry every ~60s for 30+ minutes, and that retry storm must not
            # itself look like an escalating multi-day outage.
            if daemon_state.get("reconciliation_error_date") != today:
                daemon_state["reconciliation_error_date"] = today
                daemon_state["reconciliation_consecutive_error_days"] = (
                    daemon_state.get("reconciliation_consecutive_error_days", 0) + 1
                )
                save_state(daemon_state)
            if (daemon_state.get("reconciliation_consecutive_error_days", 0) >= 2
                    and not daemon_state.get("reconciliation_alert_sent")):
                logger.critical(
                    "Reconciliation has failed to run for "
                    f"{daemon_state['reconciliation_consecutive_error_days']} consecutive days "
                    "— broker connection may be unreachable at the scheduled run time."
                )
                try:
                    if send_alert is None:
                        from ..output.emailer import send_reconciliation_alert
                        send_alert = send_reconciliation_alert
                    send_alert(
                        [f"Reconciliation has not completed successfully for "
                         f"{daemon_state['reconciliation_consecutive_error_days']} consecutive days "
                         "(broker unreachable at run time)"]
                    )
                except Exception as e:
                    logger.error(f"Reconciliation: escalation alert email failed: {e}")
                daemon_state["reconciliation_alert_sent"] = True
                save_state(daemon_state)


def process_manual_commands_wrapper(config: dict, portfolio: object, broker: object, logger: logging.Logger, daemon_state: dict) -> None:
    """Wrapper for manual command processing (catch exceptions so daemon survives)."""
    from .manual_commands import process_manual_commands
    try:
        process_manual_commands(config, portfolio, broker, logger, daemon_state=daemon_state)
    except Exception as e:
        logger.error(f"Error processing manual commands: {e}", exc_info=True)


def main(argv: list[str] | None = None) -> int:
    """Main daemon loop."""
    import argparse
    parser = argparse.ArgumentParser(description="Live trading daemon")
    parser.add_argument(
        "--takeover", action="store_true",
        help="Kill any running daemon instance and take over the lock "
             "(used by the Task Scheduler start command)")
    parser.add_argument(
        "--protective-stops", action="store_true", default=False,
        help="Enable protective stop orders (default: off)")
    parser.add_argument(
        "--stop-buffer-pct", type=float, default=1.5,
        help="Stop buffer percentage above strategy stop (default: 1.5)")
    args = parser.parse_args(argv)

    logger = setup_logging()
    logger.info("="*64)
    logger.info("Live daemon starting")
    logger.info("="*64)

    # Validate startup environment (fail-fast on configuration issues)
    if not validate_startup_environment(logger):
        return 1

    # Prevent multiple daemon instances (held OS file lock)
    # Exit 0 when yielding to a live instance: a single daemon is the desired
    # state, and a nonzero exit would make Task Scheduler restart-loop us.
    if not acquire_process_lock(logger, takeover=args.takeover):
        return 0

    # Lock held — we are the authority; remove any orphan instances that
    # survived a partial kill (e.g. Task Scheduler "End" only kills cmd.exe)
    kill_stray_daemons(logger)

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
        release_process_lock(logger)
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

    startup_reconciliation_done = False
    if not dry_run:
        logger.warning("New entries halted pending startup reconciliation")

    try:
        poll_interval = config.get("daytime", {}).get("poll_interval_seconds", 60)
        last_cycle_hour = {}

        logger.info("Entering main loop")
        shutting_down = False
        while True:
            had_error = False
            try:
                # Check overnight screening
                check_overnight_screening(config, daemon_state, logger)

                # Startup reconciliation — run on every poll until resolved (live mode only)
                if not dry_run and not startup_reconciliation_done:
                    if run_startup_reconciliation(daemon_state, portfolio, broker, logger):
                        startup_reconciliation_done = True
                        logger.info("Startup reconciliation complete — resuming normal entry evaluation")
                        if args.protective_stops:
                            check_protective_stops(portfolio, broker, logger, args.stop_buffer_pct)

                # Nightly broker/state reconciliation (real broker only)
                if not dry_run:
                    check_nightly_reconciliation(
                        config, daemon_state, portfolio, broker, logger,
                        protective_stops=args.protective_stops,
                        stop_buffer_pct=args.stop_buffer_pct,
                    )

                # Check protective stops (before ticker processing)
                if args.protective_stops and not dry_run and startup_reconciliation_done:
                    check_protective_stops(portfolio, broker, logger, args.stop_buffer_pct)

                # Process manual sell commands from mobile app
                process_manual_commands_wrapper(config, portfolio, broker, logger, daemon_state)

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
                            daemon_state, portfolio, broker, logger,
                            last_cycle_hour=last_cycle_hour,
                            protective_stops=args.protective_stops,
                            stop_buffer_pct=args.stop_buffer_pct,
                        )

                        last_cycle_hour[market_name] = current_hour
                        save_daemon_state(daemon_state)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt, shutting down")
                shutting_down = True
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
                had_error = True
            finally:
                # Write app_status.json snapshot ALWAYS, even on error — app needs fresh heartbeat
                _write_app_status_snapshot_safe(portfolio, daemon_state, config, last_cycle_hour, logger)

                # Sleep before next iteration (5s on error, normal interval on
                # success); skip entirely on shutdown so Ctrl+C exits promptly
                if not shutting_down:
                    sleep_duration = 5 if had_error else poll_interval
                    logger.debug(f"Sleeping {sleep_duration}s...")
                    time.sleep(sleep_duration)

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
