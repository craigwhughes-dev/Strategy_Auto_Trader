"""Thin CLI wrapper around quant_hmm.ticker_ranking.rank_universe().

Runs as a short-lived, standalone process — this is what overnight_scope.py
subprocesses into for its nightly top-K ranking stage, deliberately isolated
from the daemon's own long-running process (which holds a live IBKR session)
so the ProcessPoolExecutor pool it spawns internally never touches that
connection, and a hang can be bounded by the caller's subprocess timeout
rather than relying on in-process cooperation.

Usage:
    uv run python -m Strategy_Auto_Trader.markov_cli.rank_universe_cli \\
        --strategy optimised --vol-weight 0.7 --win-rate-weight 0.3 \\
        --lookback-days 60 --workers 4 --output state/top_k_scores.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import full_scan
from ..quant_hmm.ticker_ranking import rank_universe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rank-universe")
    parser.add_argument("--strategy", default="optimised")
    parser.add_argument("--vol-weight", type=float, default=0.7)
    parser.add_argument("--win-rate-weight", type=float, default=0.3)
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True, help="Path to write {ticker: score} JSON.")
    parser.add_argument(
        "--tickers",
        help="Comma-separated pre-filtered ticker list. If omitted, loads the full S&P500+FTSE universe.",
    )
    parser.add_argument("--seasonal-volume", dest="seasonal_volume", action="store_true",
                        default=False,
                        help="Normalise volume ratio by same-hour-of-day trailing mean.")
    args = parser.parse_args(argv)

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = full_scan.load_sp_ftse_universe()
    scores = rank_universe(
        tickers,
        args.strategy,
        vol_weight=args.vol_weight,
        win_rate_weight=args.win_rate_weight,
        lookback_days=args.lookback_days,
        workers=args.workers,
        use_seasonal_volume=args.seasonal_volume,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(f"Ranked {len(tickers)} tickers, {len(scores)} with candidates, "
          f"scores written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
