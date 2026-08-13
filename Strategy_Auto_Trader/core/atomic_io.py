"""Atomic file I/O primitives for safe concurrent access."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

_REPLACE_RETRIES = 3
_REPLACE_RETRY_DELAY_SECONDS = 0.05


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
        # os.replace is atomic on Windows (unlike os.rename which can fail),
        # but can still raise a transient PermissionError if another process
        # (AV scanner, a reader without FILE_SHARE_DELETE) briefly holds the
        # target file open — retry a few times before giving up.
        for attempt in range(_REPLACE_RETRIES):
            try:
                os.replace(temp_path, path)
                break
            except PermissionError:
                if attempt == _REPLACE_RETRIES - 1:
                    raise
                time.sleep(_REPLACE_RETRY_DELAY_SECONDS * (attempt + 1))
    except Exception:
        # Clean up temp file on failure
        try:
            Path(temp_path).unlink()
        except Exception:
            pass
        raise
