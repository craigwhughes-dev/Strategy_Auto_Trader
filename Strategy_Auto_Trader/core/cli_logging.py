"""Console+file logging for one-shot batch CLIs (live_sim, monte_carlo_live_sim, ...).

Separate from live_daemon.py's daily-rotating logger: a daemon is a single
persistent process that needs one continuous log across restarts, but each
invocation here is a standalone run — it gets its own timestamped logfile so
concurrent runs (e.g. two Monte Carlo pilots) don't interleave into one file.

Configures the ROOT logger (not just the calling module's) so every module's
`logging.getLogger(__name__)` call anywhere in the process — main script or
any library it imports — propagates into the same file, without each module
needing its own handler setup.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_configured = False

# ib_async messages that indicate a transient socket drop the daemon recovers
# from automatically. ib_async logs these at ERROR; we downgrade to WARNING so
# LogSentinel doesn't alert on recoveries.
_IBKR_TRANSIENT_PATTERNS = (
    "WinError 10054",
    "forcibly closed",
    "ConnectionReset",
    "Connection reset",
    "API connection failed",  # EClient pre-connect refusal during IBC restart window
    "Make sure API port",     # ibapi companion hint line, same event
)


class _IbkrTransientFilter(logging.Filter):
    """Downgrade ib_async transient socket errors from ERROR to WARNING.

    Also strips exc_info/exc_text so the formatter does not append the full
    traceback — LogSentinel matches on 'Traceback'/'Exception' in the text
    body, not just the level prefix.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno == logging.ERROR:
            msg = record.getMessage()
            if any(p in msg for p in _IBKR_TRANSIENT_PATTERNS):
                record.levelno = logging.WARNING
                record.levelname = "WARNING"
                record.exc_info = None
                record.exc_text = None
        return True


def install_ibkr_transient_filter(*handlers: logging.Handler) -> None:
    """Add _IbkrTransientFilter to each handler so propagated ib_async records are downgraded.

    Logger-level filters only fire for records originating at that logger, not
    for propagated ones — so the filter must live on the handlers that actually
    emit output.
    """
    f = _IbkrTransientFilter()
    for h in handlers:
        if not any(isinstance(x, _IbkrTransientFilter) for x in h.filters):
            h.addFilter(f)


# Gateway-connectivity signatures. IBC restarts the gateway and Task Scheduler
# restarts the daemon inside the quiet window every night, so these are
# expected there and only produce LogSentinel noise.
_GATEWAY_DOWN_PATTERNS = _IBKR_TRANSIENT_PATTERNS + (
    "WinError 10061",
    "Connect call failed",
    "could not reach broker",
    "broker connect failed",
    "unreachable",
    "IBKR data reconciliation",
    "ibkr_reconcile exited",
    "Auto-bounc",
)

_QUIET_TZ = ZoneInfo("Europe/London")
_QUIET_START_HOUR = 0
_QUIET_END_HOUR = 4  # exclusive


class _OvernightGatewayFilter(logging.Filter):
    """Drop WARNING+ gateway-connectivity records between 00:00 and 04:00 London time.

    Suppressed records are dropped, not downgraded: a WARNING would still be
    written to the file and picked up by LogSentinel. Anything still failing
    after 04:00 logs normally. `now_fn` is injectable so tests need no clock.
    """

    def __init__(self, now_fn: Callable[[], datetime] | None = None):
        super().__init__()
        self._now_fn = now_fn or (lambda: datetime.now(_QUIET_TZ))

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING:
            return True
        if not _QUIET_START_HOUR <= self._now_fn().hour < _QUIET_END_HOUR:
            return True
        msg = record.getMessage()
        return not any(p in msg for p in _GATEWAY_DOWN_PATTERNS)


def install_overnight_gateway_filter(*handlers: logging.Handler) -> None:
    """Add _OvernightGatewayFilter to each handler (same handler-level reasoning as
    install_ibkr_transient_filter: logger-level filters miss propagated records)."""
    f = _OvernightGatewayFilter()
    for h in handlers:
        if not any(isinstance(x, _OvernightGatewayFilter) for x in h.filters):
            h.addFilter(f)


def setup_cli_logger(cli_name: str) -> Path | None:
    """Attach a per-invocation file handler + console handler to the root logger.

    No-op under pytest (avoids creating a logfile per test) and if called
    more than once in the same process (avoids duplicate handlers piling up
    when a CLI's main() is invoked repeatedly in-process, e.g. by tests).
    """
    global _configured
    if _configured or "PYTEST_CURRENT_TEST" in os.environ:
        return None

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = _LOG_DIR / f"{cli_name}_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Suppress ib_async/ib_insync internal Trade/Fill repr spam (INFO-level
    # placeOrder/orderStatus/execDetails callbacks). WARNING+ still propagates
    # so genuine errors (WinError 10054 etc.) remain visible.
    logging.getLogger("ib_async").setLevel(logging.WARNING)
    logging.getLogger("ib_insync").setLevel(logging.WARNING)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console_handler)

    _configured = True
    install_ibkr_transient_filter(file_handler, console_handler)
    logging.getLogger(cli_name).info(f"Logging to {log_path}")
    return log_path
