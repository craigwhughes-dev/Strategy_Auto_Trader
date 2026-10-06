"""Windows-side CLI to pause/resume the daemon's BUY orders, or flatten the book.

Writes PAUSE_BUYING / RESUME_BUYING / SELL_ALL commands directly into the
active profile's state/commands/pending/, identical in shape to what
CommandManager.cs writes. Pure filesystem write, no network/process dependency.

`flatten` needs the daemon alive to act on the queued command. When the daemon
itself is down, use markov_cli.panic_flatten, which talks to the profile's broker directly.
"""
from __future__ import annotations

import argparse
import logging
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..core.cli_logging import setup_cli_logger

logger = logging.getLogger(__name__)


from ..core.profiles import resolve_profile

# Must address the same profile as the daemon it is pausing, so the layout
# comes from the shared resolver rather than being rebuilt here.
PROFILE = resolve_profile()
ROOT = PROFILE.root
STATE_DIR = PROFILE.state_dir
COMMANDS_DIR = PROFILE.commands_dir


def _write_command(
    action: str,
    commands_dir: Path | None = None,
    expires_in: timedelta = timedelta(hours=4),
) -> str:
    """Write a command to the pending directory; returns its id.

    `expires_in` is longer for a flatten than for a routine pause: SELL_ALL is
    requeued while any holding's market is closed, so a request made overnight
    must still be valid at the next open rather than expiring unseen.
    """
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
        "ExpiresAtUtc": (now + expires_in).isoformat().replace("+00:00", "Z"),
        "Source": "windows-cli",
    }
    atomic_write_json(pending_dir / f"{cmd_id}.json", cmd)
    return cmd_id


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    setup_cli_logger("manual_control")

    parser = argparse.ArgumentParser(
        prog="manual_control",
        description="Pause or resume new BUY orders, or flatten the book, "
                    "in the live daemon."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pause", help="Stop the daemon placing new BUY orders (SELL still works)")
    sub.add_parser("unpause", help="Resume normal BUY order placement")
    flatten = sub.add_parser(
        "flatten",
        help="Pause buying, then close every open position (queues PAUSE_BUYING + SELL_ALL)")
    flatten.add_argument(
        "--yes", action="store_true",
        help="Required. Confirms that every open position should be sold at market.")
    args = parser.parse_args(argv)

    if args.command == "flatten":
        if not args.yes:
            logger.error(
                "flatten closes EVERY open position at market on profile "
                f"{PROFILE.name!r}. Re-run with --yes to confirm.")
            return 2
        # Pause first: without it the next cycle can re-enter the names that
        # were just sold, and the two commands are processed in write order.
        pause_id = _write_command("PAUSE_BUYING")
        flatten_id = _write_command("SELL_ALL", expires_in=timedelta(hours=24))
        logger.warning(
            f"flatten queued for profile {PROFILE.name!r}: PAUSE_BUYING={pause_id}, "
            f"SELL_ALL={flatten_id}. SELL_ALL is requeued while any holding's market "
            f"is closed and expires in 24h. Buying stays paused until 'unpause'. "
            f"If the daemon is not running, nothing will happen — use "
            f"markov_cli.panic_flatten instead.")
        return 0

    action = "PAUSE_BUYING" if args.command == "pause" else "RESUME_BUYING"
    cmd_id = _write_command(action)
    logger.info(f"{args.command} command queued (id={cmd_id}). Takes effect within ~60s of the daemon's next poll cycle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
