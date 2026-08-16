"""Batch runner: read a watchlist JSON and run the model for every ticker.

Usage:
    uv run python -m Strategy_Auto_Trader.markov_cli.batch [--watchlist watchlist.json] [--no-email]
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path

import pandas as pd

from .run import main as run_single
from ..core.cli_logging import setup_cli_logger

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Raised when a process exceeds time limit."""
    pass


def _run_in_child(argv: list[str], error_queue, run_fn=run_single) -> None:
    import traceback as _tb
    try:
        run_fn(argv)
    except SystemExit as e:
        if e.code not in (None, 0):
            error_queue.put(_tb.format_exc())
        raise
    except Exception:
        error_queue.put(_tb.format_exc())
        raise


def run_single_with_timeout(argv: list[str], timeout_seconds: int = 300, run_fn=run_single) -> None:
    """Run backtest with timeout protection (5 minutes default).

    Runs run_fn (run_single by default) in a child process so a stalled
    ticker (stuck network fetch, hung HMM fit, etc.) can be forcibly killed
    instead of freezing the whole daemon cycle indefinitely — signal.SIGALRM
    only interrupts a thread, it can't be used on Windows and doesn't unblock
    a stuck C-level I/O call.

    run_fn is passed as a Process arg (not read off the module at import
    time) so it must be picklable by reference (a module-level function) —
    this is what lets tests inject a fake without the child process, which
    re-imports this module fresh under spawn, silently ignoring any
    mock.patch applied in the parent.

    Raises TimeoutError if run_fn exceeds the limit.
    """
    ctx = multiprocessing.get_context("spawn")
    error_queue = ctx.Queue()
    proc = ctx.Process(target=_run_in_child, args=(argv, error_queue, run_fn))
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        raise TimeoutError(f"Backtest exceeded time limit ({timeout_seconds}s)")

    if proc.exitcode != 0:
        tb = ""
        try:
            tb = error_queue.get_nowait()
        except Exception:
            pass
        detail = f"\n{tb.strip()}" if tb else ""
        raise RuntimeError(f"run_single exited with code {proc.exitcode}{detail}")

DEFAULTS_FILE = Path(__file__).resolve().parent.parent.parent / "config" / "watchlist.json"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _build_argv(ticker_cfg: dict, defaults: dict) -> list[str]:
    """Merge per-ticker overrides onto defaults and return a CLI argv list."""
    merged = {**defaults, **ticker_cfg}
    ticker = merged.pop("ticker")

    argv = ["--ticker", ticker]

    flag_map = {
        "years":              ("--years",              int),
        "window":             ("--window",             int),
        "threshold":          ("--threshold",          float),
        "position_mode":      ("--position-mode",      str),
        "buy_threshold":      ("--buy-threshold",      int),
        "sell_threshold":     ("--sell-threshold",     int),
        "in_sell_threshold":  ("--in-sell-threshold",  int),
        "vol_stop_mult":      ("--vol-stop-mult",      float),
        "vol_stop_window":    ("--vol-stop-window",    int),
        "profit_stop_scale":  ("--profit-stop-scale",  float),
        "min_stop":           ("--min-stop",           float),
        "trailing_stop":      ("--trailing-stop",      float),
        "initial_cash":       ("--initial-cash",       float),
        "transaction_cost":   ("--transaction-cost",   float),
        "rr_ratio":           ("--rr-ratio",           float),
        "rr_risk":            ("--rr-risk",            float),
        "max_hold_days":      ("--max-hold-days",      int),
    }

    for key, (flag, typ) in flag_map.items():
        if key in merged:
            argv.extend([flag, str(merged[key])])

    if merged.get("long_only", True):
        argv.append("--long-only")
    else:
        argv.append("--no-long-only")

    if merged.get("sma200", True):
        argv.append("--sma200")
    else:
        argv.append("--no-sma200")

    if merged.get("no_hmm", False):
        argv.append("--no-hmm")

    # Explicit-only (unlike long_only/sma200 above): unlike those two flags,
    # exit_rsi_reversal's strategy-level default varies per --strategy, so
    # silently defaulting to True here would override every strategy's own
    # exit setting once resolve_strategy() overrides are wired to the CLI.
    if "exit_rsi_reversal" in merged:
        if merged["exit_rsi_reversal"]:
            argv.append("--exit-rsi-reversal")
        else:
            argv.append("--no-exit-rsi-reversal")

    if merged.get("exit_macd_cross", False):
        argv.append("--exit-macd-cross")

    if merged.get("exit_consolidation", False):
        argv.append("--exit-consolidation")

    if merged.get("seasonal_volume", False):
        argv.append("--seasonal-volume")

    if merged.get("signal_reports_only", False):
        argv.append("--signal-reports-only")

    # batch/daemon always suppress run.py's own email send — email decisions
    # are made here in batch.py after the subprocess returns, not inside run.py.
    argv.append("--no-email")

    strategy = merged.pop("strategy", None)
    if strategy:
        argv.extend(["--strategy", str(strategy)])

    source = merged.pop("source", None)
    if source:
        argv.extend(["--source", str(source)])

    plugins = merged.pop("plugins", {})
    if isinstance(plugins, dict):
        for key, flag in (("sizer", "--plugin-sizer"), ("gate", "--plugin-gate"), ("adjuster", "--plugin-adjuster")):
            if key in plugins:
                argv.extend([flag, str(plugins[key])])

    return argv


def _collect_results(ticker: str) -> dict | None:
    """Read the latest run directory for a ticker and extract key data."""
    ticker_safe = ticker.replace("/", "-").replace("\\", "-")
    dirs = sorted(DATA_DIR.glob(f"{ticker_safe}_*"), key=lambda p: p.name)
    if not dirs:
        return None
    latest = dirs[-1]

    csv_path = latest / "compositeBacktest.csv"
    if not csv_path.exists():
        return None

    detail = pd.read_csv(csv_path, index_col="date")
    if detail.empty:
        return None

    last_row = detail.iloc[-1]
    strat_eq = float(detail["strategy_equity"].iloc[-1]) if "strategy_equity" in detail else 1.0
    bh_eq = float(detail["bh_equity"].iloc[-1]) if "bh_equity" in detail else 1.0

    quality_gate = _read_quality_gate(latest)
    effective_signal = str(quality_gate.get("flag", last_row.get("trade_event", "?")))

    # Parse YYYYMMDD from dir name (format: TICKER_YYYYMMDDTHHMMSSz)
    run_date = ""
    try:
        run_date = latest.name.rsplit("_", 1)[-1][:8]
    except Exception:
        pass

    return {
        "ticker": ticker,
        "run_dir": str(latest),
        "run_date": run_date,
        "current_signal": effective_signal,
        "trade_event": str(last_row.get("trade_event", "")) if pd.notna(last_row.get("trade_event")) else "",
        "close": float(last_row.get("close", 0)),
        "portfolio_value": float(last_row.get("portfolio_value", 20000)),
        "strategy_return": strat_eq - 1,
        "bh_return": bh_eq - 1,
        "quality_gate": quality_gate.get("flag", ""),
        "quality_gate_reason": quality_gate.get("reason", ""),
        "score": float(last_row.get("signal_score", 0) or 0),
        "signal_score": float(last_row.get("signal_score", 0) or 0),
        "signal_flag": str(last_row.get("signal_flag", "") or ""),
        "regime_signal": float(last_row.get("regime_signal", 0) or 0),
        "p_bull": float(last_row.get("p_bull", 0) or 0),
        "p_bull_smooth": float(last_row.get("p_bull_smooth", 0) or 0),
        "hmm_vote": int(last_row["hmm_vote"]) if pd.notna(last_row.get("hmm_vote")) else None,
        "rsi": float(last_row.get("rsi", 0) or 0),
        "above_sma20": bool(last_row.get("above_sma20", False)),
        "above_sma50": bool(last_row.get("above_sma50", False)),
        "above_sma200": (bool(last_row["above_sma200"]) if pd.notna(last_row.get("above_sma200")) else None),
        "volume_ratio": float(last_row.get("volume_ratio", 0) or 0),
        "kelly_fraction": float(last_row.get("kelly_fraction", 0) or 0),
        "stop_level": float(last_row.get("stop_level", 0) or 0),
        "target_level": float(last_row.get("target_level", 0) or 0),
        "sell_reason": str(last_row.get("sell_reason", "") or ""),
    }


def _read_quality_gate(run_dir: Path) -> dict:
    """Read the optional qualityGate.json output from the daily runner."""
    gate_path = run_dir / "qualityGate.json"
    if not gate_path.exists():
        return {}
    try:
        with open(gate_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _should_send_buy_alert(result: dict) -> bool:
    effective_signal = result.get("quality_gate") or result.get("current_signal", "")
    # Consolidated engine's quality gate already vetoes weak BUY signals internally,
    # so a BUY trade_event here has already passed the quality gate.
    return result["trade_event"] == "BUY" and effective_signal == "BUY"


def _should_send_sell_alert(result: dict, ticker: str) -> bool:
    if result["trade_event"] != "SELL":
        return False
    from ..output.trade_state import has_open_buy

    return has_open_buy(ticker)


def _fast_screen_tickers(
    ticker_configs: list[dict], years: int = 2
) -> tuple[list[dict], list[str]]:
    """Stage-1 fast screen: keep only tickers whose lightweight backtest is profitable or beats B&H.

    Returns (passed_configs, skipped_tickers).  passed_configs preserves all
    original per-ticker config keys so they can be forwarded to run_single unchanged.
    """
    from .screen import _screen_one

    passed: list[dict] = []
    skipped: list[str] = []
    for cfg in ticker_configs:
        ticker = cfg.get("ticker", "")
        result = _screen_one(ticker, years=years)
        if result is None or not (result["profitable"] or result["beats_bh"]):
            skipped.append(ticker)
        else:
            passed.append(cfg)
    return passed, skipped


def _build_portfolio_status(open_buys: dict[str, str], collected: list[dict]) -> list[dict]:
    """Build portfolio position data for all active trades.

    open_buys: {ticker: buy_date_str} from trade_state.json
    collected: results from this batch run (has current close + signal)
    """
    from datetime import date, datetime

    collected_map = {r["ticker"]: r for r in collected}
    positions = []

    for ticker, buy_date_str in open_buys.items():
        buy_date = datetime.strptime(buy_date_str, "%Y-%m-%d").date()
        days_held = (date.today() - buy_date).days

        # Get current price and signal from this batch run if available
        if ticker in collected_map:
            current_price = collected_map[ticker]["close"]
            current_signal = collected_map[ticker]["current_signal"]
        else:
            current_price = None
            current_signal = "?"

        # Get entry price from the compositeBacktest.csv near the buy date
        entry_price = _get_entry_price(ticker, buy_date_str)

        if entry_price is None or current_price is None:
            continue

        pl_pct = (current_price - entry_price) / entry_price * 100

        positions.append({
            "ticker": ticker,
            "buy_date": buy_date_str,
            "entry_price": entry_price,
            "current_price": current_price,
            "pl_pct": pl_pct,
            "current_signal": current_signal,
            "days_held": days_held,
        })

    return positions


def _get_entry_price(ticker: str, buy_date_str: str) -> float | None:
    """Look up the close price on the BUY date from the latest compositeBacktest.csv."""
    ticker_safe = ticker.replace("/", "-").replace("\\", "-")
    dirs = sorted(DATA_DIR.glob(f"{ticker_safe}_*"), key=lambda p: p.name)
    if not dirs:
        return None
    latest = dirs[-1]
    csv_path = latest / "compositeBacktest.csv"
    if not csv_path.exists():
        return None

    detail = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    if "close" not in detail.columns:
        return None

    # Find the row on or just before the buy date
    try:
        target = pd.Timestamp(buy_date_str)
        if target in detail.index:
            return float(detail.loc[target, "close"])
        # Find nearest date on or before
        prior = detail.index[detail.index <= target]
        if len(prior) > 0:
            return float(detail.loc[prior[-1], "close"])
    except Exception:
        pass
    return None


def process_ticker(
    ticker_cfg: dict, defaults: dict, send_email: bool
) -> dict:
    """Run model for a single ticker, collect results, journal trades, email if send_email and signal fires.

    trade_state.json (record_buy/record_sell) is always updated on a signal,
    regardless of send_email — live_daemon's must-run path pins an open
    position's strategy off this state and needs it kept current even with
    per-ticker alert emails disabled.

    Returns dict with keys: ticker, status, time, result (if successful).
    """
    ticker = ticker_cfg.get("ticker", "???")
    argv = _build_argv(ticker_cfg, defaults)
    t0 = time.time()
    try:
        run_single_with_timeout(argv, timeout_seconds=300)
        elapsed = time.time() - t0

        result = _collect_results(ticker)
        if result:
            from ..output.journal import BACKTEST_JOURNAL, append_trades, extract_trades_from_csv

            strategy_name = str({**defaults, **ticker_cfg}.get("strategy", "default"))
            csv_path = Path(result["run_dir"]) / "compositeBacktest.csv"
            trades = extract_trades_from_csv(ticker, csv_path, strategy=strategy_name)
            journal_trade_count = append_trades(BACKTEST_JOURNAL, trades)

            if result["trade_event"] in ("BUY", "SELL"):
                from ..output.trade_state import record_buy, record_sell, has_open_buy

                if _should_send_buy_alert(result):
                    if send_email:
                        try:
                            from ..output.emailer import send_trade_alert
                            send_trade_alert(result)
                            logger.info(f"BUY alert email sent for {ticker}")
                        except Exception as exc:
                            logger.warning(f"BUY alert email failed for {ticker}: {exc}")
                    record_buy(ticker, {
                        "strategy": strategy_name,
                        "signal": result["trade_event"],
                        "score": result["signal_score"],
                        "gate_flag": result["quality_gate"],
                        "price": result["close"],
                        "regime": result["regime_signal"],
                        "rsi": result["rsi"],
                        "volume_ratio": result["volume_ratio"],
                        "kelly_fraction": result["kelly_fraction"],
                        "stop_level": result["stop_level"],
                        "target_level": result["target_level"],
                        "portfolio_value": result["portfolio_value"],
                        "bh_return": result["bh_return"],
                    })
                elif _should_send_sell_alert(result, ticker):
                    if send_email:
                        try:
                            from ..output.emailer import send_trade_alert
                            send_trade_alert(result)
                            logger.info(f"SELL alert email sent for {ticker}")
                        except Exception as exc:
                            logger.warning(f"SELL alert email failed for {ticker}: {exc}")
                    record_sell(ticker, {
                        "price": result["close"],
                        "reason": result["sell_reason"] or result["quality_gate_reason"],
                        "strategy_return": result["strategy_return"],
                        "bh_return": result["bh_return"],
                    })
                elif result["trade_event"] == "SELL":
                    logger.info(f"  SELL skipped (no prior BUY since reference date)")

            return {"ticker": ticker, "status": "OK", "time": elapsed, "result": result}
        else:
            return {"ticker": ticker, "status": "OK", "time": elapsed, "result": None}

    except TimeoutError as exc:
        elapsed = time.time() - t0
        return {"ticker": ticker, "status": f"TIMEOUT: {exc}", "time": elapsed}
    except Exception as exc:
        elapsed = time.time() - t0
        return {"ticker": ticker, "status": f"FAIL: {exc}", "time": elapsed}


def main() -> int:
    setup_cli_logger("batch")

    parser = argparse.ArgumentParser(prog="batch-runner")
    parser.add_argument("--watchlist", type=Path, default=DEFAULTS_FILE,
                        help="Path to watchlist JSON (default: config/watchlist.json)")
    parser.add_argument("--no-email", action="store_true",
                        help="Skip sending emails (run models only)")
    parser.add_argument("--fast-screen", action="store_true",
                        help="Stage-1 fast screen: skip tickers whose lightweight backtest is "
                             "unprofitable and doesn't beat B&H before running the full engine")
    parser.add_argument("--roundup", action="store_true",
                        help="Send a daily roundup email after the batch run. Requires SMTP_PASSWORD and opt-in")
    parser.add_argument("--portfolio-status", action="store_true",
                        help="Send a portfolio status email for open BUY alerts. Requires SMTP_PASSWORD and opt-in")
    args = parser.parse_args()

    watchlist_path = args.watchlist
    if not watchlist_path.exists():
        logger.info(f"Watchlist not found: {watchlist_path}")
        return 1

    with open(watchlist_path, encoding="utf-8") as f:
        config = json.load(f)

    defaults = config.get("defaults", {})
    tickers = config.get("tickers", [])

    if not tickers:
        logger.info("No tickers in watchlist.")
        return 1

    if args.fast_screen:
        logger.info(f"\n  Stage 1: fast-screening {len(tickers)} tickers (no HMM, no chart)...")
        tickers, skipped_screen = _fast_screen_tickers(tickers)
        logger.info(f"  Screen result: {len(tickers)} passed, {len(skipped_screen)} skipped")
        if skipped_screen:
            logger.info(f"  Skipped: {', '.join(skipped_screen)}")
        if not tickers:
            logger.info("  No tickers passed the fast screen — nothing to run.")
            return 0

    send_email = not args.no_email and bool(os.environ.get("SMTP_PASSWORD"))
    if not args.no_email and not os.environ.get("SMTP_PASSWORD"):
        logger.info("  Note: SMTP_PASSWORD not set, skipping emails. Use --no-email to suppress this message.")

    logger.info(f"\n{'='*64}")
    logger.info(f" Batch run: {len(tickers)} tickers from {watchlist_path.name}")
    logger.info(f" Email alerts: {'ON' if send_email else 'OFF'}")
    logger.info(f" Roundup email: {'ON' if args.roundup else 'OFF'}")
    logger.info(f" Portfolio-status email: {'ON' if args.portfolio_status else 'OFF'}")
    logger.info(f"{'='*64}")

    run_results = []    # status tracking
    collected = []      # successfully collected results for email
    failed_list = []    # failed tickers
    journal_trade_count = 0

    for i, ticker_cfg in enumerate(tickers, 1):
        ticker = ticker_cfg.get("ticker", "???")
        logger.info(f"\n{'='*64}")
        logger.info(f" [{i}/{len(tickers)}]  {ticker}")
        logger.info(f"{'='*64}")

        result_dict = process_ticker(ticker_cfg, defaults, send_email)
        run_results.append({"ticker": result_dict["ticker"], "status": result_dict["status"], "time": result_dict["time"]})

        if result_dict["status"] == "OK" and result_dict.get("result"):
            collected.append(result_dict["result"])
        elif result_dict["status"] != "OK":
            failed_list.append({"ticker": ticker, "error": result_dict["status"]})

    # ── print summary ────────────────────────────────────────────────────────
    logger.info(f"\n\n{'='*64}")
    logger.info(f" Batch summary: {len(run_results)} tickers")
    logger.info(f"{'='*64}")
    ok = sum(1 for r in run_results if r["status"] == "OK")
    fail = len(run_results) - ok
    logger.info(f"  Passed: {ok}   Failed: {fail}")
    logger.info(f"\n  {'Ticker':<12s}  {'Status':<10s}  {'Time':>6s}")
    logger.info(f"  {'-'*12}  {'-'*10}  {'-'*6}")
    for r in run_results:
        status = "OK" if r["status"] == "OK" else "FAIL"
        logger.info(f"  {r['ticker']:<12s}  {status:<10s}  {r['time']:5.1f}s")
    if fail:
        logger.info("\n  Failed tickers:")
        for r in run_results:
            if r["status"] != "OK":
                logger.info(f"    {r['ticker']}: {r['status']}")

    total_time = sum(r["time"] for r in run_results)
    logger.info(f"\n  Total time: {total_time:.0f}s")
    logger.info(f"  Journal: {journal_trade_count} new trade(s) logged")

    # ── daily roundup email ──────────────────────────────────────────────────
    if send_email and args.roundup and collected:
        logger.info("\n  Sending daily roundup email...")
        try:
            from ..output.emailer import send_daily_roundup
            send_daily_roundup(collected, failed_list)
        except Exception as exc:
            logger.warning(f"  Roundup email error: {exc}")

    # ── portfolio status email (active trades with P&L since entry) ────────
    if send_email and args.portfolio_status:
        from ..output.trade_state import get_open_positions
        open_buys = get_open_positions()
        if open_buys:
            logger.info(f"\n  Building portfolio status for {len(open_buys)} active trades...")
            positions = _build_portfolio_status(open_buys, collected)
            if positions:
                try:
                    from ..output.emailer import send_portfolio_status
                    send_portfolio_status(positions)
                except Exception as exc:
                    logger.warning(f"  Portfolio email error: {exc}")

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
