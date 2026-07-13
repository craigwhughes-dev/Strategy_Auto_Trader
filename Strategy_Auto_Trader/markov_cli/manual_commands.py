"""Manual sell-command processing for the live daemon.

Processes sell commands written by the C# mobile API:
- state/commands/pending/{id}.json -> processing/ -> results/ + done/
- Atomically claims, validates, executes, and records results

Designed to run in the daemon's ~60s poll loop, outside the is_trading_hours gate.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from ..broker.types import FillResult, OrderRequest
from ..core.atomic_io import atomic_write_json


def _ensure_command_dirs(commands_dir: Path) -> None:
    """Create command subdirectories if they don't exist."""
    for subdir in ("pending", "processing", "results", "done"):
        (commands_dir / subdir).mkdir(parents=True, exist_ok=True)


def _parse_iso_timestamp(ts_str: str) -> datetime | None:
    """Parse ISO 8601 timestamp (handles 7-digit fractional seconds + Z).

    Examples:
        "2026-07-13T10:00:00.1234567Z"  -> datetime
        "2026-07-13T10:00:00.123456Z"   -> datetime
        Invalid format                   -> None
    """
    if not ts_str:
        return None

    ts_str = ts_str.strip()

    # Replace 'Z' with '+00:00' for UTC
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"

    # Handle fractional seconds: truncate to 6 digits if more
    match = re.match(r"^(.+?)\.(\d+)(\+.*)$", ts_str)
    if match:
        base, frac, tz_part = match.groups()
        # Truncate fractional seconds to 6 digits (max for fromisoformat)
        frac_truncated = frac[:6].ljust(6, "0")
        ts_str = f"{base}.{frac_truncated}{tz_part}"

    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return None


def _claim_pending_command(pending_path: Path, logger: logging.Logger) -> tuple[bool, dict | None]:
    """Atomically claim a pending command file (move to processing).

    Returns (success, command_dict):
    - (True, command_dict): successfully claimed and read
    - (False, None): file disappeared (race condition), couldn't read, or malformed JSON
    """
    processing_path = pending_path.parent.parent / "processing" / pending_path.name

    try:
        os.replace(pending_path, processing_path)
    except FileNotFoundError:
        # File disappeared (cancelled or claimed by another process)
        return False, None
    except Exception:
        # Other OS errors
        return False, None

    try:
        with open(processing_path, encoding="utf-8") as f:
            return True, json.load(f)
    except Exception:
        # File was successfully claimed (moved to processing) but can't be parsed as JSON
        logger.warning(f"Malformed JSON in processing file {processing_path.name}, moving to done.malformed")

        stem = processing_path.stem
        results_dir = processing_path.parent.parent / "results"
        error_result = {"Id": stem, "Status": "error", "ErrorMessage": "malformed command file"}

        try:
            atomic_write_json(results_dir / f"{stem}.json", error_result)
        except Exception as e:
            logger.warning(f"Could not write error result for {stem}: {e}")

        _move_to_done(processing_path, "malformed", logger)
        return False, None


def _validate_command(cmd: dict) -> tuple[bool, str]:
    """Validate command structure and required fields.

    Returns (is_valid, error_message).
    """
    required_keys = ("Id", "Action", "Status", "RequestedAtUtc", "ExpiresAtUtc", "Source")
    for key in required_keys:
        if key not in cmd:
            return False, f"Missing required key: {key}"

    if cmd["Action"] not in ("SELL", "SELL_ALL", "PAUSE_BUYING", "RESUME_BUYING"):
        return False, f"Invalid Action: {cmd['Action']}"

    if cmd["Action"] == "SELL" and not cmd.get("Ticker"):
        return False, "SELL requires Ticker"

    return True, ""


def _is_expired(cmd: dict, logger: logging.Logger) -> bool:
    """Check if command has expired."""
    expires_str = cmd.get("ExpiresAtUtc")
    if not expires_str:
        return False

    expires_dt = _parse_iso_timestamp(expires_str)
    if not expires_dt:
        logger.warning(f"Could not parse expiry time: {expires_str}")
        return False

    now_utc = datetime.now(timezone.utc)
    return now_utc > expires_dt


def _is_market_open(ticker: str, positions: dict, config: dict, logger: logging.Logger) -> bool:
    """Check if the market for a ticker's position is open.

    For SELL: check the position's market field.
    Returns True if market is open or market info unavailable (safe default).
    """
    from .live_daemon import is_trading_hours

    if ticker not in positions:
        return True  # No position, not a market question

    pos = positions[ticker]
    market_name = pos.get("market")

    if not market_name:
        # No market field set; assume open (legacy behavior)
        return True

    markets = config.get("markets", {})
    if market_name not in markets:
        # Market not in config; assume open (shouldn't happen but safe)
        return True

    market_cfg = markets[market_name]
    return is_trading_hours(market_cfg, logger)


def _execute_sell(
    ticker: str,
    portfolio: object,
    broker: object,
    logger: logging.Logger,
) -> tuple[bool, FillResult | None, str]:
    """Execute a SELL order for a single ticker.

    Returns (success, fill_result, error_msg).
    """
    if ticker not in portfolio.positions:
        return False, None, f"No open position for {ticker}"

    try:
        qty = portfolio.positions[ticker]["quantity"]
        fill = broker.place_order(OrderRequest(ticker, "SELL", qty))
        portfolio.record_exit(ticker, fill)
        return True, fill, ""
    except Exception as e:
        return False, None, str(e)


def _execute_sell_all(
    portfolio: object,
    broker: object,
    logger: logging.Logger,
) -> tuple[bool, list[FillResult], str]:
    """Execute SELL orders for all open positions.

    Returns (success, fill_results, summary_or_error_msg).
    """
    if not portfolio.positions:
        return False, [], "No open positions"

    fills: list[FillResult] = []
    errors: list[str] = []

    for ticker in list(portfolio.positions.keys()):
        try:
            qty = portfolio.positions[ticker]["quantity"]
            fill = broker.place_order(OrderRequest(ticker, "SELL", qty))
            portfolio.record_exit(ticker, fill)
            fills.append(fill)
        except Exception as e:
            errors.append(f"{ticker}: {str(e)}")

    if errors and not fills:
        # All sells failed
        return False, [], "; ".join(errors)

    # Build summary (even if some failed)
    summary_parts = []
    for fill in fills:
        summary_parts.append(f"{fill.ticker} x{fill.quantity} @ {fill.fill_price:.2f}")

    summary = "; ".join(summary_parts)
    if errors:
        summary += f"; Errors: {'; '.join(errors)}"

    success = len(errors) == 0
    return success, fills, summary


def _execute_pause_buying(daemon_state: dict, logger: logging.Logger) -> tuple[bool, str]:
    """Pause buying by user.

    Returns (success, message).
    """
    from .live_daemon import save_daemon_state
    daemon_state["paused_by_user"] = True
    try:
        save_daemon_state(daemon_state)
    except Exception as e:
        return False, str(e)
    return True, "Buying paused by user"


def _execute_resume_buying(daemon_state: dict, logger: logging.Logger) -> tuple[bool, str]:
    """Resume buying by user.

    Returns (success, message).
    """
    from .live_daemon import save_daemon_state
    daemon_state["paused_by_user"] = False
    try:
        save_daemon_state(daemon_state)
    except Exception as e:
        return False, str(e)
    return True, "Buying resumed by user"


def _move_to_done(processing_path: Path, subdir_name: str = "", logger: logging.Logger | None = None) -> None:
    """Move processing file to done directory, optionally with a suffix."""
    done_dir = processing_path.parent.parent / "done"
    done_dir.mkdir(parents=True, exist_ok=True)

    if subdir_name:
        done_path = done_dir / f"{processing_path.stem}.{subdir_name}"
    else:
        done_path = done_dir / processing_path.name

    try:
        os.replace(processing_path, done_path)
    except Exception as e:
        if logger:
            logger.warning(f"Failed to move {processing_path} to {done_path}: {e}")


def _move_to_pending(processing_path: Path, logger: logging.Logger | None = None) -> None:
    """Move processing file back to pending directory."""
    pending_dir = processing_path.parent.parent / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    pending_path = pending_dir / processing_path.name

    try:
        os.replace(processing_path, pending_path)
    except Exception as e:
        if logger:
            logger.warning(f"Failed to move {processing_path} to {pending_path}: {e}")


def _write_result(
    cmd: dict,
    status: str,
    results_dir: Path,
    fill_price: float | None = None,
    quantity: int | None = None,
    summary: str = "",
    error_msg: str = "",
) -> Path | None:
    """Write command result to results/{id}.json via atomic write.

    Returns path if successful, None otherwise.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    result = dict(cmd)  # Copy all original fields
    result["Status"] = status
    if fill_price is not None:
        result["FillPrice"] = fill_price
    if quantity is not None:
        result["Quantity"] = quantity
    if error_msg:
        result["ErrorMessage"] = error_msg
    if summary:
        result["Summary"] = summary

    result_path = results_dir / f"{cmd['Id']}.json"

    try:
        atomic_write_json(result_path, result)
        return result_path
    except Exception:
        return None


def process_manual_commands(
    config: dict,
    portfolio: object,
    broker: object,
    logger: logging.Logger,
    commands_dir: Path | None = None,
    daemon_state: dict | None = None,
) -> int:
    """Process all pending sell commands from the mobile app.

    Algorithm per command:
    1. Claim (atomically move pending -> processing)
    2. Validate structure
    3. Check expiry
    4. Check market hours (for SELL only; SELL_ALL requires all markets open)
    5. Execute and record result
    6. Move to done/ + write results/

    Returns count of commands processed.

    This function runs outside is_trading_hours and should not crash the daemon.
    """
    if daemon_state is None:
        daemon_state = {}
    if not commands_dir:
        root = Path(__file__).resolve().parent.parent.parent
        commands_dir = root / "state" / "commands"

    _ensure_command_dirs(commands_dir)
    pending_dir = commands_dir / "pending"

    if not pending_dir.exists():
        return 0

    processed = 0
    traded = False
    pending_files = sorted(pending_dir.glob("*.json"))

    for pending_path in pending_files:
        success, cmd = _claim_pending_command(pending_path, logger)
        if not success:
            # Check if file was handled as malformed (moved to done.malformed)
            cmd_id = pending_path.stem
            if (commands_dir / "done" / f"{cmd_id}.malformed").exists():
                processed += 1
            else:
                logger.debug(f"  Command {pending_path.name}: claim race or read error, skipped")
            continue

        cmd_id = cmd.get("Id", "unknown")
        logger.debug(f"  Processing command {cmd_id}")

        # Validate structure
        is_valid, err = _validate_command(cmd)
        if not is_valid:
            logger.warning(f"  Command {cmd_id}: malformed ({err}), moving to done.malformed")
            processing_path = commands_dir / "processing" / pending_path.name
            _move_to_done(processing_path, "malformed", logger)
            processed += 1
            continue

        # Check expiry
        if _is_expired(cmd, logger):
            logger.info(f"  Command {cmd_id}: expired, writing result")
            processing_path = commands_dir / "processing" / pending_path.name
            _write_result(cmd, "expired", commands_dir / "results")
            _move_to_done(processing_path, "", logger)
            processed += 1
            continue

        # Check market hours
        action = cmd.get("Action")
        ticker = cmd.get("Ticker")

        if action == "SELL":
            # Single ticker: check if its market is open
            if not _is_market_open(ticker, portfolio.positions, config, logger):
                logger.info(f"  Command {cmd_id}: market closed, requeuing")
                processing_path = commands_dir / "processing" / pending_path.name
                cmd_copy = dict(cmd)
                cmd_copy["Status"] = "queued_for_open"
                atomic_write_json(commands_dir / "pending" / pending_path.name, cmd_copy)
                try:
                    processing_path.unlink()
                except Exception:
                    pass
                processed += 1
                continue

            # Execute SELL
            success, fill, error_msg = _execute_sell(ticker, portfolio, broker, logger)
            if success:
                logger.info(f"  Command {cmd_id}: SELL {ticker} x{fill.quantity} @ {fill.fill_price:.2f}")
                _write_result(
                    cmd, "filled",
                    commands_dir / "results",
                    fill_price=fill.fill_price,
                    quantity=fill.quantity,
                )
                processing_path = commands_dir / "processing" / pending_path.name
                _move_to_done(processing_path, "", logger)
                traded = True
            else:
                logger.error(f"  Command {cmd_id}: SELL failed: {error_msg}")
                _write_result(
                    cmd, "error",
                    commands_dir / "results",
                    error_msg=error_msg,
                )
                processing_path = commands_dir / "processing" / pending_path.name
                _move_to_done(processing_path, "", logger)

        elif action == "SELL_ALL":
            # Check all held positions' markets
            all_open = True
            for pos_ticker in portfolio.positions.keys():
                if not _is_market_open(pos_ticker, portfolio.positions, config, logger):
                    all_open = False
                    break

            if not all_open:
                logger.info(f"  Command {cmd_id}: some markets closed, requeuing")
                processing_path = commands_dir / "processing" / pending_path.name
                cmd_copy = dict(cmd)
                cmd_copy["Status"] = "queued_for_open"
                atomic_write_json(commands_dir / "pending" / pending_path.name, cmd_copy)
                try:
                    processing_path.unlink()
                except Exception:
                    pass
                processed += 1
                continue

            # Execute SELL_ALL
            success, fills, summary = _execute_sell_all(portfolio, broker, logger)
            if success:
                logger.info(f"  Command {cmd_id}: SELL_ALL completed, {len(fills)} position(s) closed")
                _write_result(
                    cmd, "filled",
                    commands_dir / "results",
                    summary=summary,
                )
                processing_path = commands_dir / "processing" / pending_path.name
                _move_to_done(processing_path, "", logger)
                traded = True
            else:
                logger.error(f"  Command {cmd_id}: SELL_ALL failed: {summary}")
                _write_result(
                    cmd, "error",
                    commands_dir / "results",
                    error_msg=summary,
                    summary=summary,
                )
                processing_path = commands_dir / "processing" / pending_path.name
                _move_to_done(processing_path, "", logger)

        elif action == "PAUSE_BUYING":
            success, msg = _execute_pause_buying(daemon_state, logger)
            if success:
                logger.info(f"  Command {cmd_id}: PAUSE_BUYING — buying paused")
                _write_result(cmd, "filled", commands_dir / "results", summary=msg)
            else:
                logger.error(f"  Command {cmd_id}: PAUSE_BUYING failed: {msg}")
                _write_result(cmd, "error", commands_dir / "results", error_msg=msg)
            processing_path = commands_dir / "processing" / pending_path.name
            _move_to_done(processing_path, "", logger)

        elif action == "RESUME_BUYING":
            success, msg = _execute_resume_buying(daemon_state, logger)
            if success:
                logger.info(f"  Command {cmd_id}: RESUME_BUYING — buying resumed")
                _write_result(cmd, "filled", commands_dir / "results", summary=msg)
            else:
                logger.error(f"  Command {cmd_id}: RESUME_BUYING failed: {msg}")
                _write_result(cmd, "error", commands_dir / "results", error_msg=msg)
            processing_path = commands_dir / "processing" / pending_path.name
            _move_to_done(processing_path, "", logger)

        processed += 1

    if traded:
        portfolio.save()
        logger.info(f"Manual commands: {processed} command(s) processed, portfolio saved")
    elif processed > 0:
        logger.info(f"Manual commands: {processed} command(s) processed (no trades)")

    return processed
