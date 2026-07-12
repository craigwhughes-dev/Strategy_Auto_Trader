"""Atomic file I/O primitives for safe concurrent access."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


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
        # os.replace is atomic on Windows (unlike os.rename which can fail)
        os.replace(temp_path, path)
    except Exception:
        # Clean up temp file on failure
        try:
            Path(temp_path).unlink()
        except Exception:
            pass
        raise
