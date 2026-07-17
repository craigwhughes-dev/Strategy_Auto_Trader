"""Orchestrator: run full_scan.py for every registered strategy, over the
S&P 500 + FTSE 100 universe only (not the wider watchlist-augmented union
that full_scan.py uses by default).

Sequential, resumable (full_scan.py's own per-strategy skip-if-exists
check applies, since output paths are strategy-namespaced), safe to
stop/restart. One strategy crashing outright does not stop the others.

Sentiment columns (options IV/VIX/insider/short-interest) are
ticker-level, not strategy-level, so refetching them once per strategy is
pure waste — always runs with --no-sentiment.

Usage:
    uv run python -m Strategy_Auto_Trader.markov_cli.full_scan_all_strategies \
        [--build-universe] [--strategies default conservative ...] \
        [--force] [--limit N]
"""

from __future__ import annotations

import argparse
from datetime import date

from . import full_scan
from ..strategy.base.registry import STRATEGY_REGISTRY

STRATEGIES = sorted(STRATEGY_REGISTRY)


def main(argv: list[str] | None = None, scan_main=None) -> int:
    """Run full_scan once per strategy.

    scan_main is a DI seam (default: full_scan.main)."""
    run_scan = scan_main if scan_main is not None else full_scan.main
    parser = argparse.ArgumentParser(prog="full-scan-all-strategies", description=__doc__)
    parser.add_argument("--strategies", nargs="+", default=None,
                        help=f"Strategies to run (default: all {len(STRATEGIES)} registered: "
                             f"{', '.join(STRATEGIES)})")
    parser.add_argument("--build-universe", action="store_true",
                        help="Refresh config/universe_sp_ftse.json from Wikipedia")
    parser.add_argument("--force", action="store_true",
                        help="Re-scan tickers whose output already exists")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop each strategy after N tickers (0 = all)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Worker processes for parallel ticker scans per strategy (default: 2)")
    parser.add_argument("--data-cutoff", default="today", metavar="YYYY-MM-DD|today|none",
                        help="Freeze data across strategy passes: drop bars dated on or after "
                             "this date (exchange-local). Default 'today' — the current session "
                             "is excluded so every strategy sees identical history even while a "
                             "market is open (strategy passes run minutes apart and refetch). "
                             "'none' disables the cutoff.")
    args = parser.parse_args(argv)

    if args.data_cutoff.lower() == "none":
        data_cutoff = None
    elif args.data_cutoff.lower() == "today":
        # Resolved once here, not per pass — a sweep crossing midnight must
        # still hand every strategy the same cutoff.
        data_cutoff = date.today().isoformat()
    else:
        try:
            data_cutoff = date.fromisoformat(args.data_cutoff).isoformat()
        except ValueError:
            parser.error(f"--data-cutoff must be YYYY-MM-DD, 'today' or 'none', got {args.data_cutoff!r}")

    strategies = args.strategies if args.strategies else STRATEGIES
    unknown = [s for s in strategies if s not in STRATEGY_REGISTRY]
    if unknown:
        print(f"Unknown strategy name(s): {unknown}. Available: {STRATEGIES}")
        return 1

    if args.build_universe or not full_scan.SP_FTSE_UNIVERSE_FILE.exists():
        print("Building S&P 500 + FTSE 100 universe...")
        full_scan.build_sp_ftse_universe()

    tickers = full_scan.load_sp_ftse_universe()

    print(f"{'='*64}\n"
          f" Full sweep: {len(strategies)} strategies x {len(tickers)} tickers\n"
          f" strategies: {', '.join(strategies)}\n"
          f" data cutoff: {data_cutoff or 'none (live data — cross-strategy diffs may drift)'}\n"
          f"{'='*64}\n", flush=True)

    failed_strategies = []
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{'='*64}\n [{i}/{len(strategies)}] strategy: {strategy}\n{'='*64}", flush=True)
        scan_argv = ["--tickers", *tickers, "--strategy", strategy, "--no-sentiment",
                     "--workers", str(args.workers)]
        if data_cutoff:
            scan_argv.extend(["--data-cutoff", data_cutoff])
        if args.force:
            scan_argv.append("--force")
        if args.limit:
            scan_argv.extend(["--limit", str(args.limit)])
        try:
            run_scan(scan_argv)
        except Exception as exc:
            print(f"  strategy {strategy} crashed: {type(exc).__name__}: {exc}; continuing to next")
            failed_strategies.append(strategy)

    print(f"\n{'='*64}\n"
          f" All strategies done. {len(strategies) - len(failed_strategies)}/{len(strategies)} completed cleanly.\n"
          + (f" Crashed: {', '.join(failed_strategies)}\n" if failed_strategies else "")
          + f"{'='*64}")
    return 1 if failed_strategies else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
