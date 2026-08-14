"""Monte Carlo synthetic-data stress test — single-ticker, Track A.

Samples N synthetic price paths from a Gaussian HMM fitted on real data,
runs each path through consolidated_backtest unchanged, and reports the
distribution of strategy outcomes.

regime_model is always None (never PersistentHMMRegimeModel) to avoid
corrupting the real ticker's on-disk HMM cache. KellySizer is built fresh
per path by the engine (position_sizer=None) so trade history never leaks.

Usage:
    uv run python -m Strategy_Auto_Trader.markov_cli.monte_carlo \\
        --ticker SPY --strategy default --n-paths 300

Output:
    data/monte_carlo/<ticker>_<strategy>_<timestamp>/mc_summary.json
    data/monte_carlo/<ticker>_<strategy>_<timestamp>/mc_paths.csv
    data/monte_carlo/<ticker>_<strategy>_<timestamp>/sample_paths/  (--save-sample-paths N)
"""

from __future__ import annotations

import logging

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .run import (
    _HOURLY_ENGINE_PARAMS,
    _backfill_tunable_defaults,
    _build_arg_parser as _build_base_parser,
    _build_strategy_overrides,
    _resolve_interval_defaults,
)
from ..plugins.context_adjuster import SentimentAdjuster
from ..plugins.costs import make_cost_model
from ..quant_hmm.consolidated_engine import consolidated_backtest
from ..quant_hmm.quant_engine import _HOURS_PER_YEAR, fetch_hourly
from ..quant_hmm.synthetic_data import (
    fit_generating_hmm,
    generate_synthetic_df,
    label_hidden_states,
)
from ..quant_hmm.vol_screen import volatility_profile
from ..strategy.base.registry import resolve_strategy
from ..core.cli_logging import setup_cli_logger

logger = logging.getLogger(__name__)


_MC_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "monte_carlo"


def _build_arg_parser():
    parser = _build_base_parser()
    parser.prog = "monte-carlo"
    mc = parser.add_argument_group("Monte Carlo")
    mc.add_argument("--n-paths", type=int, default=300,
                    help="Number of synthetic paths to simulate (default: 300)")
    group = mc.add_mutually_exclusive_group()
    group.add_argument("--path-bars", type=int, default=None,
                       help="Synthetic path length in bars. Overrides --path-years.")
    group.add_argument("--path-years", type=float, default=3.0,
                       help="Synthetic path length in years (default: 3.0)")
    mc.add_argument("--seed", type=int, default=0,
                    help="Base random seed; path i uses seed+i (default: 0)")
    mc.add_argument("--workers", type=int, default=2,
                    help="Worker processes for parallel path simulation (default: 2)")
    mc.add_argument("--save-sample-paths", type=int, default=0, metavar="N",
                    help="Save the first N synthetic OHLCV frames to sample_paths/ "
                         "(default: 0)")
    mc.add_argument("--block-size", type=int, default=24,
                    help="Contiguous-block length for return bootstrap (default: 24 = "
                         "1 trading day). 1 = iid Gaussian draws (legacy, 0 trades).")
    mc.add_argument("--transmat-noise", type=float, default=0.0,
                    help="Dirichlet noise on HMM transition matrix per path (default: 0.0 = "
                         "off). Stresses parameter uncertainty. Try 0.05–0.2.")
    return parser


def _run_one_path(
    synth_df: pd.DataFrame,
    strategy_name: str,
    vol_filter_ok: bool,
    entry_overrides: dict,
    exit_overrides: dict,
    cost_model_name: str,
    ticker: str,
    backtest_kwargs: dict,
) -> dict:
    """Run one synthetic path. Top-level so ProcessPoolExecutor can pickle it.

    regime_model=None: never use PersistentHMMRegimeModel for synthetic paths
    (would corrupt the real ticker's on-disk HMM cache).
    position_sizer=None: engine builds fresh KellySizer per call, so trade
    history never leaks between paths.
    """
    entry_s, exit_s = resolve_strategy(
        strategy_name, vol_filter_ok=vol_filter_ok,
        entry_overrides=entry_overrides, exit_overrides=exit_overrides,
    )
    cost_model = make_cost_model(cost_model_name, ticker, backtest_kwargs["trade_cost"])
    bt = consolidated_backtest(
        synth_df,
        regime_model=None,
        position_sizer=None,
        entry_strategy=entry_s,
        exit_strategy=exit_s,
        context_adjuster=SentimentAdjuster(),
        cost_model=cost_model,
        **backtest_kwargs,
    )
    return {
        "sharpe_strategy":       bt.get("sharpe_strategy", float("nan")),
        "sortino_strategy":      bt.get("sortino_strategy", float("nan")),
        "max_drawdown_strategy": bt.get("max_drawdown_strategy", float("nan")),
        "total_return_strategy": bt.get("total_return_strategy", float("nan")),
        "final_portfolio":       bt.get("final_portfolio", float("nan")),
        "n_buys":                bt.get("n_buys", 0),
    }


def main(argv: list[str] | None = None) -> int:
    setup_cli_logger("monte_carlo")

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    args.interval = "1h"  # MC is hourly only

    entry_overrides, exit_overrides = _build_strategy_overrides(args)
    _backfill_tunable_defaults(args)
    _resolve_interval_defaults(args)

    n_bars = args.path_bars if args.path_bars else int(round(args.path_years * _HOURS_PER_YEAR))
    engine_params = _HOURLY_ENGINE_PARAMS

    logger.info(f"\nmonte-carlo — ticker={args.ticker}, strategy={args.strategy}, "
          f"n_paths={args.n_paths}, path_bars={n_bars}")

    logger.info(f"  fetching {args.ticker} hourly data (source={args.source})...")
    real_df = fetch_hourly(args.ticker, period="730d", source=args.source)
    if real_df is None or real_df.empty:
        logger.info(f"  ERROR: could not fetch data for {args.ticker}")
        return 1
    if isinstance(real_df.columns, pd.MultiIndex):
        real_df.columns = real_df.columns.get_level_values(0)
    logger.info(f"  {len(real_df)} bars | {real_df.index.min()} -> {real_df.index.max()}")

    profile = volatility_profile(args.ticker, source=args.source)
    tq = profile.get("trend_quality") if profile else None
    real_vol_filter_ok = (tq is None or tq >= 0.0)
    # Synthetic paths always use vol_filter_ok=True: the vol-screen is a real-data
    # admission gate for live trading, not a stress-test gate. Forcing True lets
    # the strategy signals drive entries so the MC produces meaningful results even
    # for tickers the vol-screen would otherwise exclude.
    logger.info(f"  real trend_quality={tq} (vol_filter_ok={real_vol_filter_ok}); "
          f"synthetic paths use vol_filter_ok=True")

    logger.info("  fitting generating HMM on real returns...")
    model, order = fit_generating_hmm(real_df["Close"])

    log_returns = np.log(real_df["Close"].values[1:] / real_df["Close"].values[:-1])
    historical_state_labels = label_hidden_states(model, order, log_returns)

    backtest_kwargs = dict(
        entry_prob=args.entry_prob,
        exit_prob=args.exit_prob,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        volume_min_ratio=args.volume_min_ratio,
        initial_cash=args.initial_cash,
        trade_cost=args.transaction_cost,
        use_kelly=args.use_kelly,
        regime_smooth=args.regime_smooth,
        min_hold_bars=args.min_hold_bars,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        trailing_stop=args.trailing_stop,
        vol_stop_mult=args.vol_stop_mult,
        vol_stop_window=args.vol_stop_window,
        profit_stop_scale=args.profit_stop_scale,
        min_stop_pct=args.min_stop,
        max_hold_days=args.max_hold_days,
        exit_on_rsi_reversal=args.exit_rsi,
        exit_on_macd_cross=args.exit_macd,
        exit_on_consolidation=args.exit_consol,
        use_sar_stop=args.sar_stop,
        sar_af_start=args.sar_af_start,
        sar_af_step=args.sar_af_step,
        sar_af_max=args.sar_af_max,
        skip_unused_indicators=args.skip_unused_indicators,
        min_train_bars=engine_params["min_train_bars"],
        hmm_refit_bars=engine_params["hmm_refit_bars"],
        bars_per_year=engine_params["bars_per_year"],
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_ticker = args.ticker.replace("/", "-").replace("\\", "-")
    out_dir = _MC_DIR / f"{safe_ticker}_{args.strategy}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  generating {args.n_paths} synthetic paths ({n_bars} bars each, "
          f"block_size={args.block_size})...")
    synth_dfs = [
        generate_synthetic_df(
            real_df, model, order, log_returns, historical_state_labels,
            n_bars, seed=args.seed + i, block_size=args.block_size,
            transmat_noise=args.transmat_noise,
        )
        for i in range(args.n_paths)
    ]

    if args.save_sample_paths > 0:
        sample_dir = out_dir / "sample_paths"
        sample_dir.mkdir(exist_ok=True)
        for i, sdf in enumerate(synth_dfs[:args.save_sample_paths]):
            sdf.to_csv(sample_dir / f"path_{i:04d}.csv")
        logger.info(f"  saved {min(args.save_sample_paths, args.n_paths)} sample paths to {sample_dir}")

    logger.info(f"  running backtests (workers={args.workers})...")
    results: list[dict] = []

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _run_one_path, sdf, args.strategy, True,
                    entry_overrides, exit_overrides,
                    args.cost_model, args.ticker, backtest_kwargs,
                ): i
                for i, sdf in enumerate(synth_dfs)
            }
            for future in as_completed(futures):
                results.append(future.result())
                done = len(results)
                if done % max(1, args.n_paths // 10) == 0:
                    logger.info(f"    {done}/{args.n_paths} paths done")
    else:
        for i, sdf in enumerate(synth_dfs):
            results.append(
                _run_one_path(sdf, args.strategy, True,
                              entry_overrides, exit_overrides,
                              args.cost_model, args.ticker, backtest_kwargs)
            )
            done = i + 1
            if done % max(1, args.n_paths // 10) == 0:
                logger.info(f"    {done}/{args.n_paths} paths done")

    metrics = ["sharpe_strategy", "sortino_strategy", "max_drawdown_strategy",
               "total_return_strategy", "final_portfolio"]
    percentiles = [5, 25, 50, 75, 95]

    summary: dict = {
        "ticker": args.ticker, "strategy": args.strategy,
        "n_paths": args.n_paths, "path_bars": n_bars,
    }
    for m in metrics:
        vals = [r[m] for r in results if np.isfinite(r[m])]
        if vals:
            summary[m] = {f"p{p}": float(np.percentile(vals, p)) for p in percentiles}
            summary[m]["mean"] = float(np.mean(vals))

    losses = sum(1 for r in results if r.get("total_return_strategy", 0) < 0)
    summary["prob_of_loss"] = losses / len(results) if results else float("nan")

    paths_rows = [{"path": i, **r} for i, r in enumerate(results)]
    (out_dir / "mc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(paths_rows).to_csv(out_dir / "mc_paths.csv", index=False)

    logger.info(f"\nResults ({args.n_paths} paths, {n_bars} bars each):")
    for m in metrics:
        if m in summary:
            s = summary[m]
            logger.info(f"  {m:35s}  p5={s['p5']:+.3f}  p50={s['p50']:+.3f}  p95={s['p95']:+.3f}")
    logger.info(f"  {'prob_of_loss':35s}  {summary['prob_of_loss']:.1%}")
    logger.info(f"\nOutputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
