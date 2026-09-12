"""Atomic file I/O primitives for safe concurrent access."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

_REPLACE_RETRIES = 10
_REPLACE_RETRY_DELAY_SECONDS = 0.1  # linear backoff: 100ms..1000ms ≈ 5.5s total budget


_HOLDER_SCAN_SCRIPT = """
import json, sys
import psutil
target = sys.argv[1]
holders = []
for proc in psutil.process_iter(["pid", "name"]):
    try:
        for f in proc.open_files():
            if f.path == target:
                holders.append(f"{proc.info['name']}(pid={proc.info['pid']})")
                break
    except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
        continue
print(json.dumps(holders))
"""


def _find_file_holders(path: Path) -> list[str]:
    """Best-effort: list processes with an open handle on path, for diagnosing
    a PermissionError on os.replace (WinError 5).

    Runs in a subprocess, not in-process: psutil's Windows open_files() scan
    has been observed to raise a native access violation (not a catchable
    Python exception) on this host — isolating it means a crash there kills
    a short-lived helper process, never the live daemon.
    """
    import subprocess
    import sys
    import json as _json

    try:
        target = str(path.resolve())
    except OSError:
        target = str(path)
    try:
        result = subprocess.run(
            [sys.executable, "-c", _HOLDER_SCAN_SCRIPT, target],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            logger.warning(f"File-holder scan subprocess failed: {result.stderr[-500:]}")
            return []
        return _json.loads(result.stdout.strip() or "[]")
    except Exception as e:
        logger.warning(f"File-holder scan failed: {e}")
        return []


def _atomic_replace(temp_path: str, path: Path) -> None:
    # os.replace is atomic on Windows (unlike os.rename which can fail),
    # but can still raise a transient PermissionError if another process
    # (AV scanner, a reader without FILE_SHARE_DELETE) briefly holds the
    # target file open — retry a few times before giving up.
    for attempt in range(_REPLACE_RETRIES):
        try:
            os.replace(temp_path, path)
            return
        except PermissionError:
            if attempt == _REPLACE_RETRIES - 1:
                holders = _find_file_holders(path)
                logger.warning(
                    f"os.replace blocked on {path} after {_REPLACE_RETRIES} retries, "
                    f"held by: {holders if holders else 'unknown (no psutil match — may be AV/kernel)'}"
                )
                raise
            time.sleep(_REPLACE_RETRY_DELAY_SECONDS * (attempt + 1))


def atomic_write_json(path: Path, obj: dict) -> None:
    """Write JSON to path atomically (write-temp-then-rename on Windows).

    Creates a temporary file in the same directory, writes the JSON, then
    atomically renames it to the target path. This prevents torn reads and
    corruption if the process dies mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Use same directory as target so temp and target are on same filesystem
    # (ensures rename is atomic on Windows too, not a cross-FS copy)
    fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=".tmp_",
        suffix=".json",
        text=False,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        _atomic_replace(temp_path, path)
    except Exception:
        # Clean up temp file on failure
        try:
            Path(temp_path).unlink()
        except Exception:
            pass
        raise


def atomic_write_csv(path: Path, df: "pd.DataFrame") -> None:
    """Write a DataFrame to CSV atomically (write-temp-then-rename on Windows).

    Same crash-safety as atomic_write_json — a partially-written CSV must
    never be observable. Callers like the IBKR hourly cache trust the last
    row's timestamp to compute how much history is still missing; a torn
    write would corrupt that into either a full re-fetch or a permanent gap.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=".tmp_",
        suffix=".csv",
        text=False,
    )

    try:
        os.close(fd)
        df.to_csv(temp_path)
        _atomic_replace(temp_path, path)
    except Exception:
        try:
            Path(temp_path).unlink()
        except Exception:
            pass
        raise
