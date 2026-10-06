"""Close every managed position directly at the broker, without the daemon.

The daemon-mediated route (`manual_control flatten`, which queues SELL_ALL) is
the normal way to flatten, and it keeps the ledger and the broker in step. It
only works while the daemon is alive — which is exactly the condition that
fails when you most want out. This module is the fallback: it talks to IBKR
itself, using the same sell path the daemon would have run.

Two safety properties matter more than convenience here:

* It sells only positions in the profile's own ledger. Holdings the broker
  reports but this system never opened are listed and left alone, the same way
  reconciliation treats them — a panic must not liquidate the account's other
  contents.
* It previews by default. Nothing is sent without `--yes`.

Usage:
    uv run python -m Strategy_Auto_Trader.markov_cli.panic_flatten
    uv run python -m Strategy_Auto_Trader.markov_cli.panic_flatten --yes
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from ..core.cli_logging import setup_cli_logger
from ..core.profiles import resolve_profile

logger = logging.getLogger("panic_flatten")

PROFILE = resolve_profile()


def load_profile_config() -> dict:
    """Read the active profile's overnight_strategy.json."""
    with open(PROFILE.config_path, encoding="utf-8") as f:
        return json.load(f)


def daemon_appears_live() -> int | None:
    """PID of a running daemon for this profile, if one can be detected.

    A daemon left running will keep trading against the account while this
    flattens it, so the operator is told to stop it first.
    """
    from .live_daemon import _pid_is_live_daemon, _read_pid_record
    record = _read_pid_record(PROFILE.pid_path)
    if record and _pid_is_live_daemon(record["pid"]):
        return record["pid"]
    return None


def build_portfolio(config: dict):
    """PortfolioManager bound to this profile's execution state."""
    from ..broker.portfolio import PortfolioManager
    from .live_daemon import get_market_currency
    exec_cfg = config.get("execution", {})
    primary_market = next(iter(config.get("markets", {}).keys()), "ftse")
    return PortfolioManager(
        float(exec_cfg.get("capital_pot", 20000)),
        PROFILE.execution_state_path,
        currency=get_market_currency(primary_market, config),
    )


def build_broker(config: dict):
    """The broker this profile trades through, with the account guard applied."""
    broker_cfg = config.get("broker", {})
    real_money = bool(config.get("execution", {}).get("real_money", False))
    if broker_cfg.get("provider", "ibkr") == "t212":
        return _build_t212_broker(broker_cfg, real_money)
    return _build_ibkr_broker(broker_cfg, real_money)


def _build_t212_broker(broker_cfg: dict, real_money: bool):
    """Same credentials and account guard as the daemon, so a flatten reaches the account the daemon trades."""
    import os
    from ..broker.t212_adapter import T212AccountMismatchError, T212Adapter
    api_key = os.environ.get("T212_API_KEY")
    api_secret = os.environ.get("T212_API_SECRET")
    if not api_key or not api_secret:
        raise SystemExit("broker.provider is 't212' but T212_API_KEY/T212_API_SECRET are not set")
    try:
        return T212Adapter(
            api_key=api_key,
            api_secret=api_secret,
            environment=broker_cfg.get("t212", {}).get("environment", "demo"),
            allow_live_account=real_money,
        )
    except T212AccountMismatchError as e:
        raise SystemExit(str(e))


def _build_ibkr_broker(broker_cfg: dict, real_money: bool):
    from ..broker.ibkr_adapter import IBKRAdapter
    expected_account = broker_cfg.get("expected_account")
    if expected_account is None:
        raise SystemExit(
            "broker.expected_account is not set — refusing to send orders to an "
            "unverified account")
    return IBKRAdapter(
        host=broker_cfg.get("host", "127.0.0.1"),
        port=broker_cfg.get("port", 7497),
        # A distinct client id: the daemon's own id may still be held by a
        # half-dead session, and a clash would drop whichever connects second.
        client_id=int(broker_cfg.get("panic_client_id", 11)),
        expected_account=expected_account,
        allow_live_account=real_money,
    )


def pause_buying_in_daemon_state() -> None:
    """Set paused_by_user in the profile's daemon state, so a restarted daemon will not re-buy the tier.

    Written to the file rather than queued as a command: the daemon is normally down during a panic
    flatten, so nothing would process a queued command.
    """
    from .live_daemon import load_daemon_state, save_daemon_state
    state = load_daemon_state()
    state["paused_by_user"] = True
    save_daemon_state(state)
    logger.warning("Buying paused in daemon state (paused_by_user=true); resume it before buying is wanted again")


def describe_plan(portfolio, broker_positions: dict[str, int] | None, min_hold_gbp: float | None = None) -> str:
    """Human-readable account of what a flatten would and would not touch."""
    lines = [f"Profile: {PROFILE.name}",
             f"Ledger:  {PROFILE.execution_state_path}"]
    if not portfolio.positions:
        lines.append("No managed positions — nothing to sell.")
    else:
        lines.append(f"Would SELL {len(portfolio.positions)} managed position(s) at market:")
        for ticker, pos in sorted(portfolio.positions.items()):
            if pos.get("stop_managed") is False and min_hold_gbp is not None:
                lines.append(f"  {ticker}: {pos.get('quantity')} tier holding, sold down to £{min_hold_gbp:g} "
                             f"minimum hold")
                continue
            lines.append(f"  {ticker}: {pos.get('quantity')} @ entry "
                         f"{pos.get('fill_price')}")
    if broker_positions:
        unmanaged = {t: q for t, q in broker_positions.items()
                     if t not in portfolio.positions}
        if unmanaged:
            lines.append(f"Leaving {len(unmanaged)} unmanaged broker position(s) alone:")
            for ticker, qty in sorted(unmanaged.items()):
                lines.append(f"  {ticker}: {qty}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    setup_cli_logger("panic_flatten")
    parser = argparse.ArgumentParser(
        prog="panic_flatten",
        description="Close every managed position directly at IBKR, bypassing "
                    "the daemon. Previews unless --yes is given.")
    parser.add_argument(
        "--yes", action="store_true",
        help="Send the orders. Without it, the plan is printed and nothing is sent.")
    parser.add_argument(
        "--force", action="store_true",
        help="Proceed even though a daemon for this profile is still running. "
             "Stop the daemon first unless you understand why you are doing this.")
    args = parser.parse_args(argv)

    config = load_profile_config()
    portfolio = build_portfolio(config)

    live_pid = daemon_appears_live()
    if live_pid and not args.force:
        logger.error(
            f"Daemon for profile {PROFILE.name!r} is still running (PID {live_pid}). "
            f"Stop it first, or prefer 'manual_control flatten --yes' which works "
            f"through it. Use --force to override.")
        return 3

    if not args.yes:
        logger.warning("PREVIEW ONLY — no orders will be sent. Re-run with --yes to act.")
        logger.info(describe_plan(portfolio, None))
        logger.info("With --yes, buying is also paused in the daemon state so a restarted daemon will not re-buy.")
        return 0

    pause_buying_in_daemon_state()

    if not portfolio.positions:
        logger.info(f"No managed positions in profile {PROFILE.name!r} — nothing to do.")
        return 0

    broker = build_broker(config)
    broker.connect()
    try:
        try:
            broker_positions = broker.get_open_positions()
        except Exception as e:
            logger.warning(f"Could not read broker positions: {e}")
            broker_positions = None
        min_hold_gbp = (config.get("tier_allocation") or {}).get("min_hold_gbp", 10.0)
        logger.warning(describe_plan(portfolio, broker_positions, min_hold_gbp))

        from .manual_commands import _execute_sell_all
        success, fills, summary = _execute_sell_all(portfolio, broker, logger, min_hold_gbp=min_hold_gbp)
        portfolio.save()
    finally:
        try:
            broker.disconnect()
        except Exception as e:
            logger.debug(f"disconnect suppressed: {e}")

    if success:
        logger.warning(f"Flatten complete: {len(fills)} position(s) closed. {summary}")
        remaining = portfolio.positions
        if remaining:
            logger.error(f"Still holding {len(remaining)} position(s) after flatten: "
                         f"{', '.join(sorted(remaining))} — investigate before restarting "
                         f"the daemon")
            return 1
        return 0

    logger.error(f"Flatten failed: {summary}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
