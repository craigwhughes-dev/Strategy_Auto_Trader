"""Windows-side CLI to pause/resume the daemon's BUY orders.
Writes PAUSE_BUYING / RESUME_BUYING commands directly into
state/commands/pending/, identical in shape to what CommandManager.cs
writes. Pure filesystem write, no network/process dependency.
"""
from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
STATE_DIR = ROOT / "state"
COMMANDS_DIR = STATE_DIR / "commands"


def _write_command(action: str, commands_dir: Path | None = None) -> str:
    """Write a pause or resume command to the pending directory."""
    if commands_dir is None:
        commands_dir = COMMANDS_DIR
    pending_dir = commands_dir / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    from ..core.atomic_io import atomic_write_json

    cmd_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    cmd = {
        "Id": cmd_id, "Action": action, "Ticker": None, "Status": "pending",
        "RequestedAtUtc": now.isoformat().replace("+00:00", "Z"),
        "ExpiresAtUtc": (now + timedelta(hours=4)).isoformat().replace("+00:00", "Z"),
        "Source": "windows-cli",
    }
    atomic_write_json(pending_dir / f"{cmd_id}.json", cmd)
    return cmd_id


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="manual_control",
        description="Pause or resume new BUY orders in the live daemon."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pause", help="Stop the daemon placing new BUY orders (SELL still works)")
    sub.add_parser("unpause", help="Resume normal BUY order placement")
    args = parser.parse_args(argv)

    action = "PAUSE_BUYING" if args.command == "pause" else "RESUME_BUYING"
    cmd_id = _write_command(action)
    print(f"{args.command} command queued (id={cmd_id}). Takes effect within ~60s of the daemon's next poll cycle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
