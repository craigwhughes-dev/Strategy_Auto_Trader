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
from datetime import datetime
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_configured = False


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

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console_handler)

    _configured = True
    logging.getLogger(cli_name).info(f"Logging to {log_path}")
    return log_path
