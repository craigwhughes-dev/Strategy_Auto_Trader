"""Monte Carlo synthetic-data stress test — portfolio, capital-arbitrated, Track B.

Runs a capital-arbitrated live_sim stress test across many synthetic price paths.
The ticker universe is fixed once from real data (one real generate_candidates
run to select top-K tickers), then synthetic paths are generated from each
ticker's own fitted HMM and fed through the existing generate_candidates →
vol-gate → top-K → arbitrate() pipeline unchanged.

Key invariants (same as monte_carlo.py Track A):
- regime_model=None for all synthetic-path backtests (no cache pollution)
- use_persistent_cache=False in generate_candidates() (same reason)
- position_sizer=None: engine builds fresh KellySizer per call

Cross-ticker correlation: default --market-coupling 0.0 = independent per-ticker
HMMs (previous behaviour). Use --market-coupling 0.3 to fit a shared market HMM
on SPY and bias each ticker's state sequence toward the shared market state per
path — generates realistic co-crash scenarios that independent HMMs cannot produce.

workers=1 inside generate_candidates() per path: parallelism happens at the
path level instead (nested ProcessPoolExecutor not supported by concurrent.futures).

Usage:
    uv run python -m Strategy_Auto_Trader.markov_cli.monte_carlo_live_sim \\
        --tickers SPY QQQ AAPL ... \\
        --strategies optimised_new --n-paths 50 --pot-sizes 25000 50000

Output:
    data/monte_carlo/<universe-label>_<strategy>_portfolio_<timestamp>/mc_summary.json
    data/monte_carlo/<universe-label>_<strategy>_portfolio_<timestamp>/mc_paths.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import full_scan
from .live_sim import _max_drawdown, arbitrate
from ..core.cli_logging import setup_cli_logger
from ..plugins.costs import COST_MODEL_CHOICES
from ..quant_hmm.data_cache import fetch_hourly_cached
from ..quant_hmm.quant_engine import fetch_daily
from ..quant_hmm.synthetic_data import (
    fit_generating_hmm,
    generate_synthetic_df,
    label_hidden_states,
    sample_daily_tiled_states,
    sample_synthetic_path,
)
from ..quant_hmm.ticker_ranking import (
    _filter_candidates_by_daily_trend_quality,
    filter_candidates_by_top_tickers,
    generate_candidates,
)
from ..strategy.base.registry import wants_low_trend_quality

_MC_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "monte_carlo"

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="monte-carlo-live-sim")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Fixed ticker set for all synthetic paths. "
                             "Mutually exclusive with --universe.")
    parser.add_argument("--universe", action="store_true",
                        help="Use the full S&P500+FTSE100 universe and auto-select via --top-k.")
    parser.add_argument("--strategies", nargs="+", default=["optimised_new"])
    parser.add_argument("--start-date", default="2000-01-01",
                        help="Discard candidates whose entry is before this date (default: 2000-01-01)")
    parser.add_argument("--initial-cash", type=float, default=25_000.0,
                        help="Pot size per strategy. Ignored if --pot-sizes given.")
    parser.add_argument("--pot-sizes", type=float, nargs="+", default=None,
                        help="Sweep multiple pot sizes per strategy (default: --initial-cash only)")
    parser.add_argument("--trade-cost", type=float, default=1.0)
    parser.add_argument("--cost-model", default="ibkr_tiered_spread",
                        choices=COST_MODEL_CHOICES)
    parser.add_argument("--top-k", type=int, default=70,
                        help="Retain only candidates from the top-K tickers (default: 70). "
                             "Applied once on real data to fix the universe, then re-applied "
                             "dynamically per synthetic path within that fixed set.")
    parser.add_argument("--vol-weight", type=float, default=0.7)
    parser.add_argument("--win-rate-weight", type=float, default=0.3)
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--min-trend-quality", type=float, default=0.3)
    parser.add_argument("--source", choices=["yfinance", "ibkr"], default="yfinance",
                        help="Real-data source for the one-time ticker-universe bootstrap. "
                             "Synthetic paths never touch this source. Default: yfinance "
                             "(use --source ibkr only after ibkr_backfill_universe.py completes).")
    parser.add_argument("--seasonal-volume", action="store_true", default=False)
    parser.add_argument("--n-paths", type=int, default=50,
                        help="Number of synthetic paths (default: 50)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed; path i uses seed+i (default: 0)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Worker processes at the path level (default: 2). "
                             "generate_candidates() always uses workers=1 per path "
                             "to avoid nested process pools.")
    parser.add_argument("--save-sample-paths", type=int, default=0, metavar="N",
                        help="Save the first N synthetic df_by_ticker dicts as CSV per ticker "
                             "to sample_paths/<path_i>/ (default: 0)")
    parser.add_argument("--block-size", type=int, default=24,
                        help="Contiguous-block length for return bootstrap (default: 24 = "
                             "1 trading day). 1 = iid Gaussian draws (legacy).")
    parser.add_argument("--transmat-noise", type=float, default=0.0,
                        help="Dirichlet noise on HMM transition matrix per path (default: 0.0 = "
                             "off). Stresses parameter uncertainty. Try 0.05–0.2.")
    parser.add_argument("--market-coupling", type=float, default=0.0,
                        help="Cross-ticker panic coupling strength [0.0–1.0] (default: 0.0 = off). "
                             "Fits a market HMM on SPY and biases each ticker's state sequence "
                             "toward the shared market state per path. 0.3 is a realistic starting "
                             "point; 1.0 = fully correlated. Stresses simultaneous co-crash risk "
                             "that independent per-ticker HMMs cannot generate.")
    parser.add_argument("--daily-hmm", action="store_true", default=False,
                        help="Fit per-ticker generating HMMs on long daily history (20+ yr). "
                             "Regime sequences use multi-cycle transition probabilities; block "
                             "bootstrap still draws from each ticker's 2yr real hourly pool.")
    return parser


def _run_one_path(
    df_by_ticker: dict[str, pd.DataFrame],
    fixed_tickers: list[str],
    strategy_name: str,
    pot_sizes: list[float],
    start_date: str,
    top_k: int,
    vol_weight: float,
    win_rate_weight: float,
    lookback_days: int,
    min_trend_quality: float,
    trade_cost: float,
    cost_model_name: str,
    seasonal_volume: bool,
) -> dict:
    """Run one synthetic path: generate_candidates → vol-gate → top-K → arbitrate.

    Top-level so ProcessPoolExecutor can pickle it. generate_candidates uses
    workers=1 (no nested pools). use_persistent_cache=False prevents any
    HMM or IBKR cache writes for synthetic data.
    """
    candidates, price_by_ticker, trend_quality_by_ticker = generate_candidates(
        tickers=fixed_tickers,
        strategy_name=strategy_name,
        vol_filter_tag="daily-rescreened",
        vol_filter_ok=True,
        workers=1,
        use_seasonal_volume=seasonal_volume,
        source="yfinance",  # moot — df_by_ticker bypasses fetch entirely
        df_by_ticker=df_by_ticker,
        use_persistent_cache=False,
    )

    cutoff = pd.Timestamp(start_date)
    candidates = [c for c in candidates if c.date_opened.tz_localize(None) >= cutoff]

    wants_low = wants_low_trend_quality(strategy_name)
    candidates = _filter_candidates_by_daily_trend_quality(
        candidates, trend_quality_by_ticker, min_trend_quality, wants_low,
    )

    if top_k > 0:
        candidates, _ = filter_candidates_by_top_tickers(
            candidates, trend_quality_by_ticker, top_k,
            vol_weight=vol_weight, win_rate_weight=win_rate_weight,
            lookback_days=lookback_days,
        )

    results_by_pot: dict[float, dict] = {}
    for pot_size in pot_sizes:
        result = arbitrate(
            candidates,
            initial_cash=pot_size,
            trade_cost=trade_cost,
            cost_model_name=cost_model_name,
            currency="GBP",
            price_by_ticker=price_by_ticker,
        )
        equity = [row["portfolio_value"] for row in result["equity_curve"]]
        final_val = equity[-1] if equity else pot_size
        results_by_pot[pot_size] = {
            "total_return": (final_val - pot_size) / pot_size if pot_size > 0 else float("nan"),
            "final_portfolio": final_val,
            "max_drawdown": _max_drawdown(equity),
            "n_admitted": result["n_admitted"],
            "n_candidates": result["n_candidates"],
        }
    return results_by_pot


def main(argv: list[str] | None = None) -> int:
    setup_cli_logger("monte_carlo_live_sim")

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if bool(args.tickers) == bool(args.universe):
        parser.error("exactly one of --tickers or --universe is required")

    if args.universe:
        all_universe = full_scan.load_sp_ftse_universe()
    else:
        all_universe = list(args.tickers)

    pot_sizes = args.pot_sizes if args.pot_sizes else [args.initial_cash]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for strategy_name in args.strategies:
        logger.info(f"\n{'='*64}\n Strategy: {strategy_name}\n{'='*64}")
        logger.info(f"  universe={len(all_universe)} tickers, n_paths={args.n_paths}, "
              f"pot_sizes={pot_sizes}, source={args.source}")

        # --- One-time real-data setup ---
        logger.info("  generating real candidates to fix ticker universe...")
        real_candidates, real_price_by_ticker, real_tq_by_ticker = generate_candidates(
            tickers=all_universe,
            strategy_name=strategy_name,
            vol_filter_tag="daily-rescreened",
            vol_filter_ok=True,
            workers=1,
            use_seasonal_volume=args.seasonal_volume,
            source=args.source,
        )

        wants_low = wants_low_trend_quality(strategy_name)
        real_candidates = _filter_candidates_by_daily_trend_quality(
            real_candidates, real_tq_by_ticker, args.min_trend_quality, wants_low,
        )

        if args.top_k > 0:
            real_candidates, ticker_scores = filter_candidates_by_top_tickers(
                real_candidates, real_tq_by_ticker, args.top_k,
                vol_weight=args.vol_weight, win_rate_weight=args.win_rate_weight,
                lookback_days=args.lookback_days,
            )
            fixed_tickers = sorted(
                {c.ticker for c in real_candidates},
                key=lambda t: -ticker_scores.get(t, 0),
            )[:args.top_k]
        else:
            fixed_tickers = sorted({c.ticker for c in real_candidates})

        logger.info(f"  fixed ticker set: {len(fixed_tickers)} tickers")

        # Fetch full OHLC for fixed tickers (cached from generate_candidates pass above)
        logger.info("  fetching full OHLC and fitting generating HMMs...")
        real_dfs: dict[str, pd.DataFrame] = {}
        hmm_models: dict[str, tuple] = {}  # ticker -> (model, order)
        historical_returns: dict[str, np.ndarray] = {}
        historical_labels: dict[str, np.ndarray] = {}

        skipped = []
        for ticker in fixed_tickers:
            df = fetch_hourly_cached(ticker, period="730d", source=args.source)
            if df is None or df.empty:
                skipped.append(ticker)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            try:
                model, order = fit_generating_hmm(df["Close"])
            except ValueError as e:
                logger.info(f"    {ticker}: HMM fit failed ({e}), skipping")
                skipped.append(ticker)
                continue
            log_ret = np.log(df["Close"].values[1:] / df["Close"].values[:-1])
            labels = label_hidden_states(model, order, log_ret)
            real_dfs[ticker] = df
            hmm_models[ticker] = (model, order)
            historical_returns[ticker] = log_ret
            historical_labels[ticker] = labels

        if skipped:
            logger.info(f"  skipped {len(skipped)} tickers (no data / HMM failure): {skipped[:5]}...")
        final_tickers = [t for t in fixed_tickers if t in real_dfs]
        logger.info(f"  {len(final_tickers)} tickers with fitted HMMs")

        # --- Daily long-history HMMs (optional) ---
        daily_hmm_models: dict[str, tuple] = {}
        if args.daily_hmm:
            logger.info("  fitting daily long-history HMMs per ticker (20+ yr)...")
            daily_ok = 0
            for ticker in final_tickers:
                d_df = fetch_daily(ticker)
                if d_df is None or len(d_df) < 252 * 5:
                    continue
                try:
                    dm, do = fit_generating_hmm(d_df["Close"])
                    daily_hmm_models[ticker] = (dm, do)
                    daily_ok += 1
                except ValueError:
                    pass  # fall back to hourly HMM for this ticker
            logger.info(f"  daily HMMs fitted for {daily_ok}/{len(final_tickers)} tickers")

        # --- Market HMM for cross-ticker coupling (finding C) ---
        market_model = market_order = None
        market_max_n = 0
        if args.market_coupling > 0.0:
            logger.info("  fitting market HMM on SPY for cross-ticker panic coupling...")
            spy_df = fetch_hourly_cached("SPY", period="730d", source=args.source)
            if spy_df is not None and not spy_df.empty:
                if isinstance(spy_df.columns, pd.MultiIndex):
                    spy_df.columns = spy_df.columns.get_level_values(0)
                try:
                    market_model, market_order = fit_generating_hmm(spy_df["Close"])
                    market_max_n = max(len(df) for df in real_dfs.values())
                    logger.info(f"  market HMM fitted (SPY {len(spy_df)} bars), coupling={args.market_coupling}")
                except ValueError as e:
                    logger.info(f"  WARNING: SPY HMM fit failed ({e}), market_coupling disabled")
                    args.market_coupling = 0.0
            else:
                logger.info("  WARNING: SPY data unavailable, market_coupling disabled")
                args.market_coupling = 0.0

        # --- Synthetic paths ---
        logger.info(f"  generating synthetic df_by_ticker for each path...")

        def _make_df_by_ticker(path_idx: int) -> dict[str, pd.DataFrame]:
            # Sample one shared market state sequence per path (if coupling enabled).
            # Same length for all tickers; sliced to each ticker's n_bars below.
            market_state_seq = None
            if market_model is not None and args.market_coupling > 0.0:
                market_seed = (args.seed + path_idx) ^ hash("__market_spy__") & 0xFFFFFFFF
                _, market_state_seq = sample_synthetic_path(
                    market_model, market_order, n_returns=market_max_n, seed=market_seed,
                )

            result = {}
            for ticker in final_tickers:
                df = real_dfs[ticker]
                model, order = hmm_models[ticker]
                log_ret = historical_returns[ticker]
                labels = historical_labels[ticker]
                # Derive per-ticker seed from (path, ticker) for reproducibility
                ticker_seed = (args.seed + path_idx) ^ hash(ticker) & 0xFFFFFFFF
                # Slice market states to this ticker's path length (all tickers share
                # the same market state at each synthetic bar index).
                mstates = market_state_seq[:len(df)] if market_state_seq is not None else None
                precomputed = None
                if ticker in daily_hmm_models:
                    dm, do = daily_hmm_models[ticker]
                    precomputed = sample_daily_tiled_states(
                        dm, do, len(df), seed=ticker_seed,
                        transmat_noise=args.transmat_noise,
                        market_states=mstates,
                        market_coupling=args.market_coupling,
                    )
                result[ticker] = generate_synthetic_df(
                    df, model, order, log_ret, labels,
                    n_bars=len(df), seed=ticker_seed,
                    block_size=args.block_size,
                    transmat_noise=args.transmat_noise,
                    market_states=mstates,
                    market_coupling=args.market_coupling,
                    precomputed_state_labels=precomputed,
                )
            return result

        logger.info(f"  running {args.n_paths} synthetic paths (workers={args.workers})...")

        all_results: list[dict] = []

        if args.workers > 1:
            path_dfs = [_make_df_by_ticker(i) for i in range(args.n_paths)]

            if args.save_sample_paths > 0:
                label = strategy_name
                out_dir = _MC_DIR / f"{label}_portfolio_{timestamp}"
                for i, pdf in enumerate(path_dfs[:args.save_sample_paths]):
                    sample_dir = out_dir / "sample_paths" / f"path_{i:04d}"
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    for ticker, sdf in pdf.items():
                        sdf.to_csv(sample_dir / f"{ticker.replace('/', '-')}.csv")

            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        _run_one_path,
                        path_dfs[i], final_tickers, strategy_name, pot_sizes,
                        args.start_date, args.top_k,
                        args.vol_weight, args.win_rate_weight, args.lookback_days,
                        args.min_trend_quality, args.trade_cost, args.cost_model,
                        args.seasonal_volume,
                    ): i
                    for i in range(args.n_paths)
                }
                for future in as_completed(futures):
                    all_results.append(future.result())
                    done = len(all_results)
                    if done % max(1, args.n_paths // 5) == 0:
                        logger.info(f"    {done}/{args.n_paths} paths done")
        else:
            for i in range(args.n_paths):
                pdf = _make_df_by_ticker(i)
                all_results.append(
                    _run_one_path(
                        pdf, final_tickers, strategy_name, pot_sizes,
                        args.start_date, args.top_k,
                        args.vol_weight, args.win_rate_weight, args.lookback_days,
                        args.min_trend_quality, args.trade_cost, args.cost_model,
                        args.seasonal_volume,
                    )
                )
                done = i + 1
                if done % max(1, args.n_paths // 5) == 0:
                    logger.info(f"    {done}/{args.n_paths} paths done")

        # --- Aggregate ---
        label = strategy_name
        out_dir = _MC_DIR / f"{label}_portfolio_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        percentiles = [5, 25, 50, 75, 95]
        summary: dict = {
            "strategy": strategy_name,
            "n_paths": args.n_paths,
            "n_tickers": len(final_tickers),
        }

        paths_rows = []
        for pot_size in pot_sizes:
            key = str(pot_size)
            summary[key] = {}
            for metric in ["total_return", "final_portfolio", "max_drawdown", "n_admitted"]:
                vals = [r[pot_size][metric] for r in all_results
                        if pot_size in r and np.isfinite(r[pot_size][metric])]
                if vals:
                    summary[key][metric] = {
                        f"p{p}": float(np.percentile(vals, p)) for p in percentiles
                    }
                    summary[key][metric]["mean"] = float(np.mean(vals))
            losses = sum(1 for r in all_results
                         if pot_size in r and r[pot_size]["total_return"] < 0)
            summary[key]["prob_of_loss"] = losses / len(all_results) if all_results else float("nan")

        for i, r in enumerate(all_results):
            for pot_size in pot_sizes:
                row = {"path": i, "pot_size": pot_size}
                if pot_size in r:
                    row.update(r[pot_size])
                paths_rows.append(row)

        (out_dir / "mc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        pd.DataFrame(paths_rows).to_csv(out_dir / "mc_paths.csv", index=False)

        logger.info(f"\n  Results ({args.n_paths} paths, {len(final_tickers)} tickers):")
        for pot_size in pot_sizes:
            key = str(pot_size)
            if key in summary:
                tr = summary[key].get("total_return", {})
                dd = summary[key].get("max_drawdown", {})
                pol = summary[key].get("prob_of_loss", float("nan"))
                logger.info(f"  pot={pot_size:,.0f}: "
                      f"return p5={tr.get('p5', float('nan')):+.1%}  "
                      f"p50={tr.get('p50', float('nan')):+.1%}  "
                      f"p95={tr.get('p95', float('nan')):+.1%}  "
                      f"dd_p50={dd.get('p50', float('nan')):.1%}  "
                      f"prob_loss={pol:.1%}")
        logger.info(f"  Outputs: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
