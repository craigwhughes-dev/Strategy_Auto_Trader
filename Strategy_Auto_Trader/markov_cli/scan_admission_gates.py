"""Full-universe ablation scan of optimised_new's admission gates
(require_flip_entry, vol_filter_ok) -- the multi-ticker version of
compare_admission_gates.py's 11-ticker basket, per todo.md's "optimised_new
admission-gate audit" Plan1 A/B: an 11-ticker basket was judged too thin to
decide anything, this scans the full S&P500+FTSE100 universe (~603 tickers)
instead.

Task granularity is per-ticker, not per-(ticker,variant): each worker fetches
a ticker's hourly history and resolves its live vol_filter_ok once, then runs
all 3 variants against that same data, so the fetch/vol_screen cost isn't
paid 3x. Mirrors full_scan.py's ProcessPoolExecutor + incremental-CSV-append
pattern so a multi-hour run is resumable (--force to redo) and inspectable
mid-run.

Usage:
    uv run python -m Strategy_Auto_Trader.markov_cli.scan_admission_gates --workers 4
    uv run python -m Strategy_Auto_Trader.markov_cli.scan_admission_gates --tickers AAPL MSFT --workers 1
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SCAN_DIR = ROOT / "reports" / "admission_gate_scan"

VARIANTS = [
    {"name": "baseline (current defaults)", "require_flip_entry": True,  "force_vol_filter_on": False},
    {"name": "require_flip_entry=False",     "require_flip_entry": False, "force_vol_filter_on": False},
    {"name": "vol_filter pre-screen off",    "require_flip_entry": True,  "force_vol_filter_on": True},
]

#: Matches run.py's _HOURLY_ENGINE_PARAMS / _HOURLY_BAR_DEFAULTS exactly, so
#: this backtest reproduces what optimised_new actually runs on hourly bars.
_HOURLY_DEFAULTS = dict(
    min_train_bars=500, hmm_refit_bars=500,
    regime_smooth=24, min_hold_bars=48,
)

_SUMMARY_COLUMNS = [
    "ticker", "vol_filter_ok", "variant", "status",
    "bars_fetched", "bh_return_pct", "pl", "return_pct",
    "win_rate_pct", "wins", "losses", "sharpe", "sortino", "max_dd_pct",
    "n_trades", "scanned_at", "elapsed_s", "note",
]


def _summary_path() -> Path:
    return SCAN_DIR / "summary.csv"


def _append_rows(rows: list[dict]) -> None:
    """Append rows with a fixed column set so partial rows (no_data, error)
    stay aligned with full rows. Mirrors full_scan.py::_append_summary_row."""
    path = _summary_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).reindex(columns=_SUMMARY_COLUMNS)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _already_done(force: bool) -> set[str]:
    """Tickers with all 3 variants already logged -- skipped unless --force."""
    if force:
        return set()
    path = _summary_path()
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return set()
    if df.empty:
        return set()
    counts = df.groupby("ticker")["variant"].nunique()
    return set(counts[counts >= len(VARIANTS)].index)


def _scan_ticker_gates(ticker: str) -> list[dict]:
    """Fetch `ticker` once, resolve vol_filter_ok once, run all 3 variants.
    Returns one row per variant; never raises -- errors become error rows."""
    from ..quant_hmm.consolidated_engine import consolidated_backtest
    from ..strategy.base.registry import resolve_strategy
    from ..strategy.optimised_new import OptimisedNewEntry, OptimisedNewExit
    from .compare_exits import _fetch

    started = time.time()
    scanned_at = datetime.now().isoformat(timespec="seconds")

    df = _fetch(ticker)
    if df is None or len(df) <= 300:
        return [{
            "ticker": ticker, "vol_filter_ok": None, "variant": v["name"],
            "status": "no_data", "bars_fetched": 0 if df is None else len(df),
            "scanned_at": scanned_at, "elapsed_s": round(time.time() - started, 1),
            "note": "insufficient hourly history",
        } for v in VARIANTS]

    try:
        template_entry, _ = resolve_strategy("optimised_new", ticker=ticker)
        vol_filter_ok = template_entry._vol_filter_ok
    except Exception as exc:
        return [{
            "ticker": ticker, "vol_filter_ok": None, "variant": v["name"],
            "status": "error", "bars_fetched": len(df),
            "scanned_at": scanned_at, "elapsed_s": round(time.time() - started, 1),
            "note": f"vol_screen: {type(exc).__name__}: {exc}",
        } for v in VARIANTS]

    rows = []
    for variant in VARIANTS:
        v_started = time.time()
        effective_vol_filter_ok = True if variant["force_vol_filter_on"] else vol_filter_ok
        entry = OptimisedNewEntry(vol_filter_ok=effective_vol_filter_ok)
        entry.require_flip_entry = variant["require_flip_entry"]
        exit_ = OptimisedNewExit()
        try:
            bt = consolidated_backtest(df, entry_strategy=entry, exit_strategy=exit_, **_HOURLY_DEFAULTS)
            detail = bt["detail"]
            sells = detail[detail["trade_event"] == "SELL"]
            buys = detail[detail["trade_event"] == "BUY"]
            buy_prices = buys["close"].tolist()
            sell_prices = sells["close"].tolist()
            trade_pls = [
                (sell_prices[j] - buy_prices[j]) / buy_prices[j]
                for j in range(min(len(buy_prices), len(sell_prices)))
            ]
            wins = sum(1 for p in trade_pls if p > 0)
            losses = sum(1 for p in trade_pls if p < 0)
            rows.append({
                "ticker": ticker, "vol_filter_ok": vol_filter_ok, "variant": variant["name"],
                "status": "ok", "bars_fetched": len(df),
                "bh_return_pct": round(bt["total_return_bh"] * 100, 2),
                "pl": round(bt["total_pl"], 2),
                "return_pct": round(bt["total_return_strategy"] * 100, 2),
                "win_rate_pct": round(wins / (wins + losses) * 100, 1) if (wins + losses) else 0.0,
                "wins": wins, "losses": losses,
                "sharpe": round(bt["sharpe_strategy"], 3) if pd.notna(bt["sharpe_strategy"]) else None,
                "sortino": round(bt.get("sortino_strategy", float("nan")), 3)
                    if pd.notna(bt.get("sortino_strategy", float("nan"))) else None,
                "max_dd_pct": round(bt["max_drawdown_strategy"] * 100, 2)
                    if pd.notna(bt["max_drawdown_strategy"]) else None,
                "n_trades": bt["n_buys"],
                "scanned_at": scanned_at, "elapsed_s": round(time.time() - v_started, 1),
                "note": "",
            })
        except Exception as exc:
            rows.append({
                "ticker": ticker, "vol_filter_ok": vol_filter_ok, "variant": variant["name"],
                "status": "error", "bars_fetched": len(df),
                "scanned_at": scanned_at, "elapsed_s": round(time.time() - v_started, 1),
                "note": f"{type(exc).__name__}: {exc}",
            })
    return rows


def _scan_ticker_gates_worker(ticker: str) -> dict:
    """Top-level module worker for ProcessPoolExecutor (must catch everything --
    an uncaught exception here kills that worker's future, not the pool)."""
    try:
        return {"rows": _scan_ticker_gates(ticker), "traceback": None}
    except Exception:
        return {
            "rows": [{
                "ticker": ticker, "vol_filter_ok": None, "variant": v["name"],
                "status": "error", "scanned_at": datetime.now().isoformat(timespec="seconds"),
                "note": "worker_error",
            } for v in VARIANTS],
            "traceback": traceback.format_exc(),
        }


def main(argv: list[str] | None = None) -> int:
    from .full_scan import load_sp_ftse_universe

    parser = argparse.ArgumentParser(prog="scan-admission-gates", description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Explicit ticker list (default: full S&P500+FTSE100 universe)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Worker processes (1 = sequential, default: 4)")
    parser.add_argument("--force", action="store_true",
                        help="Re-scan tickers whose 3 variants are already logged")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N tickers (0 = all)")
    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    tickers = args.tickers if args.tickers else load_sp_ftse_universe()
    if args.limit:
        tickers = tickers[:args.limit]

    done_tickers = _already_done(args.force)
    tasks = [t for t in tickers if t not in done_tickers]
    skipped = len(tickers) - len(tasks)

    print(f"Admission-gate scan: {len(tickers)} tickers, {len(VARIANTS)} variants each, "
          f"{args.workers} worker(s)")
    print(f"  output: {_summary_path()}")
    print(f"  {skipped} already done (skipped), {len(tasks)} to run\n", flush=True)

    if not tasks:
        print("Nothing to do.")
        return 0

    done = failed = 0
    t0 = time.time()

    if args.workers == 1:
        for i, ticker in enumerate(tasks, 1):
            print(f"[{i}/{len(tasks)}] {ticker} ...", flush=True)
            payload = _scan_ticker_gates_worker(ticker)
            if payload["traceback"]:
                print(payload["traceback"], flush=True)
            _append_rows(payload["rows"])
            if payload["rows"][0]["status"] == "ok":
                done += 1
            else:
                failed += 1
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        try:
            futures = {executor.submit(_scan_ticker_gates_worker, t): t for t in tasks}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    payload = future.result()
                except BrokenProcessPool:
                    in_flight = [futures[f] for f in futures if not f.done()]
                    print(f"\nBroken worker pool; in-flight tickers: {in_flight}")
                    executor.shutdown(wait=False, cancel_futures=True)
                    return 1

                if payload["traceback"]:
                    print(payload["traceback"], flush=True)
                _append_rows(payload["rows"])
                if payload["rows"][0]["status"] == "ok":
                    done += 1
                else:
                    failed += 1
                completed = done + failed
                elapsed = time.time() - t0
                eta = (elapsed / completed) * (len(tasks) - completed) if completed else 0
                print(f"[{completed}/{len(tasks)}] {ticker}: {payload['rows'][0]['status']} "
                      f"({elapsed:.0f}s elapsed, ~{eta/60:.0f}min remaining)", flush=True)
        finally:
            executor.shutdown(wait=True)

    elapsed = time.time() - t0
    print(f"\nScan finished: {done} ok, {failed} failed/no-data, {skipped} already done. "
          f"{elapsed/60:.1f} min.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
