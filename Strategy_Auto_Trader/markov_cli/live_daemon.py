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


class _DailyFileHandler(logging.Handler):
    """FileHandler that swaps to a fresh daemon_<date>_<pid>.log at local midnight.

    Daemon runs for weeks between restarts (Task Scheduler auto-restart), so a
    date computed once at setup_logging() call time goes stale — every line
    after midnight would land in yesterday's file. Date is rechecked per emit
    instead of relying on rollover-on-interval like TimedRotatingFileHandler,
    so the naming stays daemon_<date>_<pid>.log rather than a numbered/suffixed
    file. PID is fixed for the process lifetime, so a restart always starts a
    clean file and it's obvious from the filename which process wrote it.
    """

    def __init__(self):
        super().__init__()
        self._pid = os.getpid()
        self._current_date = None
        self._stream = None

    def _ensure_current_file(self):
        today = datetime.now().date().isoformat()
        if today != self._current_date:
            if self._stream is not None:
                self._stream.close()
            log_path = LOGS_DIR / f"daemon_{today}_{self._pid}.log"
            self._stream = open(log_path, "a", encoding="utf-8")
            self._current_date = today

    def emit(self, record):
        try:
            self._ensure_current_file()
            msg = self.format(record)
            self._stream.write(msg + "\n")
            self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        if self._stream is not None:
            self._stream.close()
        super().close()


def setup_logging() -> logging.Logger:
    """Set up daily log that rolls over to a new file at local midnight."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("live_daemon")
    logger.setLevel(logging.DEBUG)

    handler = _DailyFileHandler()
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
    except FileNotFoundError:
        return None
    except Exception as e:
        logging.getLogger("live_daemon").warning("daemon.pid unreadable: %s", e)
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
            # _kill_daemon_process can itself take ~10s (wait_procs timeouts);
            # give the retry loop its own fresh window to observe the lock
            # release instead of inheriting whatever remains of the pre-kill
            # deadline, which could already be exhausted by a slow kill.
            deadline = time.time() + 15
            continue
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


def cleanup_incomplete_runs(data_dir: Path, logger: logging.Logger,
                             min_age_seconds: float = 600) -> int:
    """Remove run directories that only have inputData.csv (incomplete backtests).

    Skips directories younger than min_age_seconds — an orphaned run_single
    child from a crashed prior daemon can still be actively writing to its
    run_dir (inputData.csv written, compositeBacktest.csv not yet); it isn't
    caught by kill_stray_daemons (that only matches live_daemon cmdlines, not
    worker children), so without an age gate a fresh-looking in-progress run
    gets rmtree'd out from under it, and the worker's later to_csv() call
    fails with "non-existent directory".

    Returns count of directories cleaned up.
    """
    if not data_dir.exists():
        return 0

    cleaned = 0
    now = time.time()
    for run_dir in data_dir.glob("*_*"):
        if not run_dir.is_dir():
            continue

        files = set(f.name for f in run_dir.glob("*"))
        has_output = "compositeBacktest.csv" in files or "qualityGate.json" in files

        if "inputData.csv" in files and not has_output:
            age = now - (run_dir / "inputData.csv").stat().st_mtime
            if age < min_age_seconds:
                continue
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
        except Exception as e:
            logging.getLogger("live_daemon").warning(
                "daemon_state.json unreadable (%s) — starting with empty state. "
                "Halt flags and cursors reset.", e
            )
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
        in_trading = is_trading_hours(market_cfg, logger, market_name=market_name, quiet=True)
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
        "halt_top_k_stale": daemon_state.get("halt_top_k_stale", False),
        "paused_by_user": daemon_state.get("paused_by_user", False),
        "reconciliation_discrepancies": daemon_state.get("reconciliation_discrepancies", []),
        "last_reconcile_date": daemon_state.get("last_reconcile_date", ""),
        "trades_today": trades_today,
        "interest_accrued": portfolio.interest_accrued,
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


def is_trading_hours(
    market_cfg: dict,
    logger: logging.Logger,
    *,
    now: datetime | None = None,
    market_name: str | None = None,
    quiet: bool = False,
) -> bool:
    """Check if market is currently in trading hours."""
    label = market_name or market_cfg["timezone"]
    tz = ZoneInfo(market_cfg["timezone"])
    if now is None:
        now = datetime.now(tz)

    weekday = now.weekday()
    if weekday >= 5:
        if not quiet:
            logger.debug(f"  Market {label}: weekend, skipping")
        return False

    start_str = market_cfg["trading_start"]
    end_str = market_cfg["trading_end"]
    start_time = datetime.strptime(start_str, "%H:%M").time()
    end_time = datetime.strptime(end_str, "%H:%M").time()

    is_open = start_time <= now.time() <= end_time
    if not is_open and not quiet:
        logger.debug(f"  Market {label}: outside hours "
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


def get_open_positions(market_name: str, logger: logging.Logger) -> list[str]:
    """Get tickers with open positions in this market, scoped by each
    position's own recorded "market" field — not by watchlist membership,
    which breaks silently if a ticker is dropped from its watchlist while a
    position on it is still open (a position missing the field defaults to
    matching the current market, the permissive/safe default for legacy data)."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path = STATE_DIR / "execution_state.json"
    if not state_path.exists():
        return []
    try:
        with open(state_path, encoding="utf-8") as f:
            exec_state = json.load(f)
        positions = exec_state.get("positions", {})
        open_in_market = [
            t for t, p in positions.items()
            if p.get("market", market_name) == market_name
        ]
        return sorted(open_in_market)
    except Exception as e:
        logger.error(f"  Error loading execution_state.json: {e}")
        return []


def next_round_robin_slice(
    market_name: str,
    in_scope: list[str],
    daemon_state: dict,
    logger: logging.Logger,
) -> tuple[list[str], int]:
    """Get remaining round-robin candidates starting from the persisted cursor.

    Does NOT advance the cursor — the caller may not get through the whole
    slice within its time budget, so the cursor is only advanced afterwards
    via advance_round_robin_cursor(), by however many tickers were actually
    attempted. Returns (candidates, cursor_start).
    """
    if not in_scope:
        return [], 0

    cursors = daemon_state.setdefault("cursors", {})
    today = datetime.now().date().isoformat()
    key = f"{market_name}:{today}"

    cursor = cursors.get(key, 0)
    if cursor >= len(in_scope):
        cursor = 0

    return in_scope[cursor:], cursor


def advance_round_robin_cursor(
    market_name: str,
    in_scope_len: int,
    cursor_start: int,
    n_attempted: int,
    daemon_state: dict,
    logger: logging.Logger,
) -> None:
    """Persist how far the round-robin actually got this cycle, so the next
    cycle resumes there instead of restarting from the top of the list.
    """
    cursors = daemon_state.setdefault("cursors", {})
    today = datetime.now().date().isoformat()
    key = f"{market_name}:{today}"

    new_cursor = cursor_start + n_attempted
    if new_cursor >= in_scope_len:
        new_cursor = 0
        logger.debug(f"  {market_name}: round-robin wrapped")
    cursors[key] = new_cursor


def execute_signals_with_retry(
    market_name: str,
    ticker_list: list[str],
    data_dir: Path,
    portfolio: object,
    limit_tracker: object,
    broker: object,
    allow_new_entries: bool,
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
                allow_new_entries=allow_new_entries,
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
                # A mid-run interrupt leaves an in-flight marker exactly like a
                # startup crash does. Flag it so the main loop re-enters the
                # startup-reconciliation retry branch (which verifies against
                # the broker's open orders before clearing) on its own, instead
                # of requiring a manual daemon restart to recover.
                daemon_state["needs_reconciliation"] = True
                # Remember which tickers never got a resolved outcome so they
                # can be re-evaluated fresh (not replayed at a stale price) the
                # moment reconciliation clears, instead of waiting for next
                # hour's cycle. A second interrupt in the same market before
                # the first is retried just overwrites — the newest unresolved
                # set is the authoritative one.
                daemon_state.setdefault("pending_retry_tickers", {})[market_name] = exc.unresolved
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
                "ib_async" in error_msg
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


def _evaluate_ticker(
    ticker: str,
    overrides: dict[str, dict],
    defaults: dict,
    logger: logging.Logger,
    market_name: str,
    *,
    pin_open_strategy: bool,
) -> dict:
    """Build ticker_cfg and run process_ticker for a single ticker.

    pin_open_strategy=True carries an already-open position's strategy
    forward over a watchlist override (must-run / retry paths); the
    round-robin candidate scan leaves it False since nothing is open yet.
    """
    from .batch import process_ticker

    ticker_cfg = {"ticker": ticker, **overrides.get(ticker, {})}
    if pin_open_strategy:
        from ..output.trade_state import get_open_strategy
        pinned_strategy = get_open_strategy(ticker)
        if pinned_strategy:
            ticker_cfg["strategy"] = pinned_strategy
    # Phase-1 signal alerts fire off trade_state.json's own "did we alert a BUY"
    # memory, not the broker's real fill state — Phase 2 (execute_signals) can
    # decline an admitted BUY on cash/quota/hours, so an alerted SELL can arrive
    # for a position never actually opened. Emails off; state (record_buy/
    # record_sell, needed by pin_open_strategy above) still updates.
    result = process_ticker(ticker_cfg, defaults, send_email=False)
    if not str(result.get("status", "")).startswith("OK"):
        logger.warning(f"[{market_name}] {ticker} processing failed: {result.get('status')}")
    else:
        _log_ticker_snapshot(logger, ticker, result.get("result"))
    return result


def _log_ticker_snapshot(logger: logging.Logger, ticker: str, r: dict | None) -> None:
    """One DEBUG line per ticker with every component that fed the decision."""
    if not r:
        return
    sma_flags = (
        f"sma20={'Y' if r.get('above_sma20') else 'N'}"
        f" sma50={'Y' if r.get('above_sma50') else 'N'}"
        f" sma200={'?' if r.get('above_sma200') is None else ('Y' if r.get('above_sma200') else 'N')}"
    )
    logger.debug(
        f"    {ticker}: flag={r.get('quality_gate') or r.get('current_signal', '?')}"
        f" (raw={r.get('signal_flag', '?')}) score={r.get('score', 0.0):.1f}"
        f" p_bull={r.get('p_bull', 0.0):.2f} p_bull_smooth={r.get('p_bull_smooth', 0.0):.2f}"
        f" hmm_vote={r.get('hmm_vote')} rsi={r.get('rsi', 0.0):.1f} {sma_flags}"
        f" vol_ratio={r.get('volume_ratio', 0.0):.2f} kelly={r.get('kelly_fraction', 0.0):.3f}"
        f" gate_reason={r.get('quality_gate_reason') or '-'}"
    )


def _execute_processed_tickers(
    market_name: str,
    processed: list[dict],
    config: dict,
    daemon_state: dict,
    portfolio: object,
    broker: object,
    logger: logging.Logger,
    protective_stops: bool = False,
    stop_buffer_pct: float = 1.5,
) -> None:
    """Execute signals once for a batch of already-processed tickers."""
    if not processed:
        return
    # Include all tickers with OK status. signal_reader will validate files exist.
    # SELL signals are prioritized (safer to exit existing positions) over strict validation.
    ticker_list = [p["ticker"] for p in processed if p.get("status") == "OK"]
    if not ticker_list:
        return

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
        allow_new_entries = not (
            daemon_state.get("halt_new_entries")
            or daemon_state.get("halt_top_k_stale")
            or daemon_state.get("paused_by_user")
        )
        if daemon_state.get("halt_new_entries"):
            logger.warning(f"[{market_name}] Reconciliation mismatch unresolved — new entries blocked")
        if daemon_state.get("halt_top_k_stale"):
            logger.warning(f"[{market_name}] top_k_screen stale/missing — new entries blocked")
        if daemon_state.get("paused_by_user"):
            logger.info(f"[{market_name}] Buying paused by user — new entries blocked")
        market_currency = get_market_currency(market_name, config)
        buys, sells, skipped = execute_signals_with_retry(
            market_name, ticker_list, DATA_DIR, portfolio, limit_tracker, broker,
            allow_new_entries, logger,
            market_currency=market_currency,
            daemon_state=daemon_state,
            protective_stops=protective_stops,
            stop_buffer_pct=stop_buffer_pct,
        )
        # HOLD entries are bare tickers; rejected-signal entries carry a "(reason)"
        # suffix (see execute.py) — split on that to report HOLD separately from
        # real BUY/SELL signals blocked downstream (capacity, qty, entries-halted).
        hold_count = sum(1 for s in skipped if "(" not in s)
        rejected_count = len(skipped) - hold_count
        logger.info(f"  BUY:  {len(buys)}, SELL: {len(sells)}, HOLD: {hold_count}, REJECTED: {rejected_count}")
        for b in buys:
            logger.info(f"    BUY: {b}")
        for s in sells:
            logger.info(f"    SELL: {s}")
        portfolio.save()
    except Exception as e:
        logger.error(f"  Error executing signals (unrecoverable): {e}", exc_info=True)


def retry_pending_tickers(
    config: dict,
    daemon_state: dict,
    portfolio: object,
    broker: object,
    logger: logging.Logger,
    protective_stops: bool = False,
    stop_buffer_pct: float = 1.5,
) -> None:
    """Re-evaluate and execute tickers left unresolved by an execution interrupt.

    Runs immediately once reconciliation confirms clear, rather than waiting
    for that market's next hourly cycle. Re-evaluates fresh (not a replay of
    the stale signal) so a long outage naturally reflects whatever's true now
    instead of firing at a stale price. One-shot: the pending record is
    popped *before* the attempt (consumed), not after — a fresh interrupt
    during the retry writes its own new pending_retry_tickers entry (via
    execute_signals_with_retry's own interrupt handler), and popping after
    the attempt would clobber that fresh write. This is bounded by requiring
    reconciliation to clear again before the next attempt fires, not a loop.
    """
    pending = daemon_state.get("pending_retry_tickers", {})
    if not pending:
        return

    for market_name in list(pending.keys()):
        market_cfg = config.get("markets", {}).get(market_name)
        if market_cfg is None:
            daemon_state.get("pending_retry_tickers", {}).pop(market_name, None)
            continue
        if not is_trading_hours(market_cfg, logger, market_name=market_name):
            logger.info(f"[{market_name}] Market closed — deferring interrupted-ticker retry to next open")
            continue

        tickers = daemon_state.get("pending_retry_tickers", {}).pop(market_name, [])
        in_scope = load_in_scope_tickers(market_name, logger)
        overrides = load_ticker_overrides(market_name, logger)
        # ibkr_backfill_universe.py completed 2026-08-14 (589/603 tickers,
        # see HANDOFF.md) — back on the local IBKR-backed cache.
        defaults = {"signal_reports_only": True, "source": "ibkr", **market_cfg.get("defaults", {})}
        retry_tickers = [t for t in tickers if t in in_scope]
        if not retry_tickers:
            save_daemon_state(daemon_state)
            continue

        logger.info(f"[{market_name}] Retrying previously-interrupted ticker(s) immediately: {retry_tickers}")
        retried = [
            _evaluate_ticker(ticker, overrides, defaults, logger, market_name, pin_open_strategy=True)
            for ticker in retry_tickers
        ]
        _execute_processed_tickers(
            market_name, retried, config, daemon_state, portfolio, broker, logger,
            protective_stops=protective_stops, stop_buffer_pct=stop_buffer_pct,
        )
        save_daemon_state(daemon_state)


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
    if last_cycle_hour is None:
        last_cycle_hour = {}

    in_scope = load_in_scope_tickers(market_name, logger)
    overrides = load_ticker_overrides(market_name, logger)
    if not in_scope:
        logger.debug(f"  {market_name}: no in-scope tickers")
        return 0

    # Open positions always run first — unconditionally, not filtered by
    # in_scope, which can lag behind execution_state.json if a ticker was
    # dropped from its watchlist since the last overnight_scope run (see
    # get_open_positions()'s docstring).
    open_positions = get_open_positions(market_name, logger)
    must_run = list(open_positions)

    # Remaining budget for candidates
    max_seconds = config.get("daytime", {}).get("max_seconds_per_cycle", 1500)
    cycle_start = time.time()

    # Daemon cycles skip chart/HTML rendering unless a signal fires; a market
    # config can override by setting defaults.signal_reports_only = false.
    # ibkr_backfill_universe.py completed 2026-08-14 (589/603 tickers,
    # see HANDOFF.md) — back on the local IBKR-backed cache.
    defaults = {"signal_reports_only": True, "source": "ibkr", **market_cfg.get("defaults", {})}
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
        result = _evaluate_ticker(ticker, overrides, defaults, logger, market_name, pin_open_strategy=True)
        processed.append(result)

    # Stage 2: candidates (round-robin through rest)
    remaining_budget = max_seconds - (time.time() - cycle_start)
    buffer_secs = config.get("daytime", {}).get("cycle_buffer_minutes", 5) * 60
    if remaining_budget > buffer_secs:
        remaining_budget -= buffer_secs
        round_robin_universe = [t for t in in_scope if t not in must_run]
        candidates, cursor_start = next_round_robin_slice(
            market_name,
            round_robin_universe,
            daemon_state,
            logger,
        )

        logger.info(f"[{market_name}] Round-robin ({len(candidates)} candidates, {remaining_budget:.0f}s budget):")
        n_attempted = 0
        for ticker in candidates:
            now_remaining = max_seconds - (time.time() - cycle_start)
            if now_remaining <= buffer_secs:
                logger.debug(f"  {ticker}: budget near exhausted ({now_remaining:.0f}s left)")
                skipped_budget.append(ticker)
                break

            process_manual_commands_wrapper(config, portfolio, broker, logger, daemon_state)
            _write_app_status_snapshot_safe(portfolio, daemon_state, config, last_cycle_hour, logger)

            logger.debug(f"  Processing {ticker}")
            result = _evaluate_ticker(ticker, overrides, defaults, logger, market_name, pin_open_strategy=False)
            processed.append(result)
            n_attempted += 1

        advance_round_robin_cursor(
            market_name, len(round_robin_universe), cursor_start, n_attempted, daemon_state, logger
        )

    # Execute signals once for all processed tickers this cycle
    _execute_processed_tickers(
        market_name, processed, config, daemon_state, portfolio, broker, logger,
        protective_stops=protective_stops, stop_buffer_pct=stop_buffer_pct,
    )

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
        t0 = time.time()
        try:
            from .overnight_scope import main as run_overnight_scope
            run_overnight_scope()
            daemon_state["last_overnight_date"] = today
            save_daemon_state(daemon_state)
            elapsed = time.time() - t0
            logger.info(f"Overnight scope screening complete ({elapsed:.0f}s)")
            for mkt in config.get("markets", {}):
                scope_path = STATE_DIR / f"in_scope_{mkt}.json"
                if not scope_path.exists():
                    continue
                try:
                    scope = json.loads(scope_path.read_text(encoding="utf-8"))
                    kept = scope.get("kept", [])
                    excluded = scope.get("excluded", [])
                    logger.info(f"  [{mkt}] in scope ({len(kept)}): {', '.join(kept)}")
                    by_reason: dict[str, list[str]] = {}
                    for e in excluded:
                        by_reason.setdefault(e["reason"], []).append(e["ticker"])
                    for reason, tickers in sorted(by_reason.items()):
                        logger.info(f"  [{mkt}] excluded by {reason} ({len(tickers)}): {', '.join(tickers)}")
                except Exception as exc:
                    logger.warning(f"  Could not read scope summary for {mkt}: {exc}")
        except Exception as e:
            logger.error(f"Error in overnight screening: {e}")
        _check_top_k_screen_health(daemon_state, config, logger)
        _check_orphaned_positions(config, logger)


def _run_ibkr_data_reconcile_subprocess(config: dict, cfg: dict) -> None:
    """Invoke ibkr_reconcile.py in a standalone subprocess (mirrors
    compute_global_top_k's rank_universe_cli pattern) so its own IBKR
    connection (client_id=4 by default) never touches this process's ib_async
    event loop or execution connection (client_id=1)."""
    import subprocess

    broker_cfg = config.get("broker", {})
    cmd = [
        sys.executable, "-m", "Strategy_Auto_Trader.markov_cli.ibkr_reconcile",
        "--lookback-days", str(cfg.get("lookback_days", 14)),
        "--host", broker_cfg.get("host", "127.0.0.1"),
        "--port", str(broker_cfg.get("port", 4002)),
        "--client-id", str(cfg.get("client_id", 4)),
    ]
    timeout_seconds = cfg.get("timeout_seconds", 3600)
    result = subprocess.run(cmd, cwd=ROOT, timeout=timeout_seconds,
                             capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ibkr_reconcile exited {result.returncode}: {result.stderr[-500:]}")


def check_ibkr_data_reconciliation(
    config: dict,
    daemon_state: dict,
    logger: logging.Logger,
    *,
    run_reconcile: Callable | None = None,
    save_state: Callable | None = None,
) -> None:
    """Run IBKR trade-correction reconciliation once per day at the
    configured time — see markov_cli/ibkr_reconcile.py's module docstring
    for why this exists (fetch_hourly()'s append-only cache never re-checks
    an already-cached bar, so a post-hoc IBKR trade correction would
    otherwise be silently missed forever).

    Scheduled before overnight_scope's default 02:00 run (default 01:00) so
    any corrected bars feed that night's vol/top-K screening and HMM refits,
    not next night's."""
    if run_reconcile is None:
        run_reconcile = _run_ibkr_data_reconcile_subprocess
    if save_state is None:
        save_state = save_daemon_state

    cfg = config.get("ibkr_data_reconcile", {})
    if not cfg.get("enabled", True):
        return

    tz = ZoneInfo(config.get("overnight_timezone", "Europe/London"))
    now = datetime.now(tz)
    run_time_str = cfg.get("run_time", "01:00")
    run_hour, run_minute = map(int, run_time_str.split(":"))

    today = now.date().isoformat()
    if daemon_state.get("last_ibkr_data_reconcile_date") == today:
        return

    if now.hour == run_hour and now.minute >= run_minute:
        logger.info("Running IBKR data reconciliation...")
        t0 = time.time()
        try:
            run_reconcile(config, cfg)
            daemon_state["last_ibkr_data_reconcile_date"] = today
            save_state(daemon_state)
            logger.info(f"IBKR data reconciliation complete ({time.time() - t0:.0f}s)")
        except Exception as e:
            logger.error(f"Error in IBKR data reconciliation: {e}")


def _check_orphaned_positions(config: dict, logger: logging.Logger) -> None:
    """After overnight_scope runs, check each market's in_scope_<market>.json
    for orphaned_positions (open positions whose ticker fell out of the
    watchlist file — see screen_market()'s docstring) and alert if any are
    present. Non-halting: the position is already force-kept in scope by
    screen_market(), this is visibility only."""
    for market_name in config.get("markets", {}):
        scope_path = STATE_DIR / f"in_scope_{market_name}.json"
        if not scope_path.exists():
            continue
        try:
            with open(scope_path, encoding="utf-8") as f:
                scope = json.load(f)
        except Exception as e:
            logger.warning(f"  Could not read in_scope_{market_name}.json: {e}")
            continue

        orphaned = scope.get("orphaned_positions", [])
        if orphaned:
            logger.warning(f"  {market_name}: {len(orphaned)} open position(s) "
                           f"missing from watchlist: {', '.join(orphaned)}")
            try:
                from ..output.emailer import send_orphaned_position_alert
                send_orphaned_position_alert(market_name, orphaned)
            except Exception as e:
                logger.error(f"  orphaned-position alert email failed: {e}")


def _check_top_k_screen_health(
    daemon_state: dict,
    config: dict,
    logger: logging.Logger,
) -> None:
    """After overnight_scope runs, check state/top_k_universe.json's status.

    Sets daemon_state["halt_top_k_stale"] = True when the ranking is missing,
    stale (>1 day old), or failed — using a separate key from halt_new_entries
    so reconciliation-halt and top-K-halt are independent conditions that can
    clear independently. Clears the flag when a fresh "ok" ranking is found.
    Only halts if top_k_screen is enabled in config; ignores missing files when
    it is disabled (no ranking is expected in that case).
    """
    top_k_enabled = config.get("top_k_screen", {}).get("enabled", False)
    state_path = STATE_DIR / "top_k_universe.json"

    if not state_path.exists():
        if top_k_enabled:
            logger.warning("  top_k_universe.json missing — halting new entries until ranking recovers")
            daemon_state["halt_top_k_stale"] = True
            try:
                from ..output.emailer import send_top_k_screen_alert
                send_top_k_screen_alert("missing", None)
            except Exception as e:
                logger.error(f"  top_k_screen alert email failed: {e}")
        return

    try:
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
    except Exception as e:
        logger.warning(f"  Could not read top_k_universe.json: {e}")
        if top_k_enabled:
            daemon_state["halt_top_k_stale"] = True
        return

    status = state.get("status")
    state_date = state.get("date")
    is_stale = False
    if state_date:
        try:
            age_days = (datetime.now(timezone.utc).date() - datetime.fromisoformat(state_date).date()).days
            is_stale = age_days > 1
        except Exception as e:
            logger.warning(f"  Could not parse top_k_universe.json date {state_date!r}: {e}")

    if status != "ok" or is_stale:
        effective_status = "stale" if is_stale else (status or "unknown")
        logger.warning(f"  top_k_screen status={effective_status} (state date {state_date}) — halting new entries")
        daemon_state["halt_top_k_stale"] = True
        try:
            from ..output.emailer import send_top_k_screen_alert
            send_top_k_screen_alert(effective_status, state_date)
        except Exception as e:
            logger.error(f"  top_k_screen alert email failed: {e}")
    else:
        if daemon_state.get("halt_top_k_stale"):
            logger.info("  top_k_screen healthy — clearing halt_top_k_stale")
        daemon_state["halt_top_k_stale"] = False


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


def check_pending_order_cancels(
    portfolio: object,
    daemon_state: dict,
    broker: object,
    logger: logging.Logger,
    *,
    run_recon: Callable | None = None,
    save_state: Callable | None = None,
) -> None:
    """Resolve entry orders whose cancel request wasn't confirmed in-call.

    Account TIF preset is GTC, so an order the daemon couldn't confirm as
    cancelled won't auto-expire — it can still fill unattended. Runs once per
    cycle (non-blocking) rather than the adapter blocking the trading loop
    waiting for IBKR to confirm. A late fill is a real, previously-untracked
    position, so it triggers an immediate reconciliation pass (same halt +
    alert path as any other broker/portfolio mismatch) instead of waiting
    for the next nightly reconciliation to notice.
    """
    check_fn = getattr(broker, "check_pending_cancels", None)
    if check_fn is None:
        return
    try:
        events = check_fn()
    except Exception as e:
        logger.warning(f"check_pending_order_cancels: error: {e}")
        return

    for event in events:
        if event.outcome == "cancelled":
            logger.info(f"{event.ticker}: pending cancel confirmed — order no longer working")
        elif event.outcome == "filled":
            logger.warning(
                f"{event.ticker}: order filled AFTER cancel was requested "
                f"({event.quantity}x @ {event.fill.fill_price if event.fill else '?'}) "
                f"— running immediate reconciliation"
            )
            (run_recon or run_reconciliation)(
                portfolio, broker, daemon_state, logger, save_state=save_state,
            )
        elif event.outcome == "timeout_alert":
            logger.warning(
                f"{event.ticker}: cancel still unconfirmed after "
                f"{event.elapsed_minutes:.0f} min — order may still be working at IBKR"
            )
            try:
                from ..output.emailer import send_pending_cancel_timeout_alert
                send_pending_cancel_timeout_alert(
                    event.ticker, event.action, event.quantity, event.elapsed_minutes,
                )
            except Exception as e:
                logger.error(f"check_pending_order_cancels: alert email failed: {e}")


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


def _send_nightly_roundup(config: dict, logger: logging.Logger) -> None:
    """Collect day's ticker results and send daily roundup email."""
    from .batch import _collect_results
    from ..output.emailer import send_daily_roundup

    results = []
    failed = []

    # Iterate through all configured markets and their tickers
    for market_name, market_cfg in config.get("markets", {}).items():
        in_scope = load_in_scope_tickers(market_name, logger)
        if not in_scope:
            continue

        for ticker in in_scope:
            try:
                result = _collect_results(ticker)
                if result:
                    results.append(result)
                else:
                    failed.append({"ticker": ticker, "error": "No result data"})
            except Exception as e:
                logger.warning(f"Failed to collect results for {ticker}: {e}")
                failed.append({"ticker": ticker, "error": str(e)})

    if results or failed:
        logger.info(f"Sending nightly roundup: {len(results)} tickers, {len(failed)} failed")
        send_daily_roundup(results, failed)
    else:
        logger.debug("No ticker results to send in nightly roundup")


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
            # Accrue daily interest on uninvested cash
            interest = portfolio.accrue_daily_interest()
            if interest > 0.01:
                logger.info(f"Daily interest accrued: {interest:.2f}")
            portfolio.save()
            daemon_state["last_reconcile_date"] = today
            daemon_state["reconciliation_consecutive_error_days"] = 0
            daemon_state["reconciliation_alert_sent"] = False
            save_state(daemon_state)

            # Send nightly roundup email
            try:
                _send_nightly_roundup(config, logger)
            except Exception as e:
                logger.error(f"Nightly roundup email failed: {e}", exc_info=True)
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
    parser.add_argument(
        "--send-nightly-roundup", action="store_true",
        help="Send nightly roundup email with today's results and exit")
    args = parser.parse_args(argv)

    logger = setup_logging()
    logger.info("="*64)
    logger.info("Live daemon starting")
    logger.info("="*64)

    # Handle --send-nightly-roundup flag (send email and exit)
    if args.send_nightly_roundup:
        logger.info("Sending nightly roundup email...")
        config = load_config()
        try:
            _send_nightly_roundup(config, logger)
            logger.info("Nightly roundup email sent successfully")
        except Exception as e:
            logger.error(f"Failed to send nightly roundup email: {e}", exc_info=True)
            return 1
        return 0

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
        broker_cfg = config.get("broker", {})
        broker = IBKRAdapter(
            host=broker_cfg.get("host", "127.0.0.1"),
            port=broker_cfg.get("port", 7497),
            client_id=broker_cfg.get("client_id", 1),
        )
        logger.info(f"Using IBKRAdapter (live paper trading) at "
                    f"{broker._host}:{broker._port}")

    # Set up portfolio
    from ..broker.portfolio import PortfolioManager
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path = STATE_DIR / "execution_state.json"
    capital_pot = float(exec_cfg.get("capital_pot", 20000))
    # Determine primary market currency (from first market in config)
    primary_market = next(iter(config.get("markets", {}).keys()), "ftse")
    currency = get_market_currency(primary_market, config)
    portfolio = PortfolioManager(capital_pot, state_path, currency=currency)

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
                # Check IBKR data reconciliation (before overnight screening,
                # so any corrected bars feed tonight's vol/top-K screening)
                check_ibkr_data_reconciliation(config, daemon_state, logger)

                # Check overnight screening
                check_overnight_screening(config, daemon_state, logger)

                # Startup reconciliation — run on every poll until resolved (live mode
                # only). Also re-entered whenever a mid-run execution interrupt sets
                # needs_reconciliation, so a dropped socket self-heals within one poll
                # interval instead of requiring a daemon restart.
                if not dry_run and (not startup_reconciliation_done or daemon_state.get("needs_reconciliation")):
                    if run_startup_reconciliation(daemon_state, portfolio, broker, logger):
                        startup_reconciliation_done = True
                        daemon_state["needs_reconciliation"] = False
                        save_daemon_state(daemon_state)
                        logger.info("Startup reconciliation complete — resuming normal entry evaluation")
                        if args.protective_stops:
                            check_protective_stops(portfolio, broker, logger, args.stop_buffer_pct)
                        retry_pending_tickers(
                            config, daemon_state, portfolio, broker, logger,
                            protective_stops=args.protective_stops,
                            stop_buffer_pct=args.stop_buffer_pct,
                        )

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

                # Resolve any entry-order cancels IBKR hasn't confirmed yet
                if not dry_run and startup_reconciliation_done:
                    check_pending_order_cancels(portfolio, daemon_state, broker, logger)

                # Process manual sell commands from mobile app
                process_manual_commands_wrapper(config, portfolio, broker, logger, daemon_state)

                # Check each market
                now = datetime.now(timezone.utc)
                for market_name, market_cfg in config.get("markets", {}).items():
                    if not is_trading_hours(market_cfg, logger, market_name=market_name):
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
