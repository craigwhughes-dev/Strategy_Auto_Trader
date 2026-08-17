"""CLI entry point for the consolidated hourly HMM + composite-signal engine.

Usage:
    uv run python -m Strategy_Auto_Trader.markov_cli.run --ticker SPY
"""

from __future__ import annotations

import logging

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..quant_hmm.quant_engine import fetch_daily, fetch_hourly
from ..quant_hmm.consolidated_engine import consolidated_backtest
from ..quant_hmm.ticker_ranking import _HMM_CACHE_DIR as HMM_CACHE_DIR
from ..core.quality_gate import _apply_quality_gate  # noqa: F401 — available for custom gate logic
from ..core.momentum import composite_signal, exit_indicators, momentum_signals
from ..output.charting import plot_backtest
from ..output.report import write_daily_summary
from ..plugins.costs import COST_MODEL_CHOICES, make_cost_model
from ..plugins.kelly_sizer import FixedSizer, KellySizer
from ..plugins.context_adjuster import NullAdjuster, SentimentAdjuster
from ..strategy.base.registry import STRATEGY_REGISTRY, resolve_strategy
from ..core.cli_logging import setup_cli_logger

logger = logging.getLogger(__name__)


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _make_run_dir(ticker: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_ticker = ticker.replace("/", "-").replace("\\", "-")
    run_dir = DATA_DIR / f"{safe_ticker}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _fetch_company_info(ticker: str) -> tuple[str, str]:
    import yfinance as yf
    from ..core.net_retry import call_with_timeout_retry

    info = call_with_timeout_retry(lambda: yf.Ticker(ticker).info)
    if not info:
        return ticker, ""
    name = info.get("longName") or info.get("shortName") or ticker
    sector = info.get("sector") or info.get("industry") or ""
    return name, sector


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="strategy-auto-trader")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--source", choices=["yfinance", "ibkr"], default="yfinance",
                        help="Hourly data source: yfinance (default) or the local "
                             "incremental IBKR-backed cache (default: yfinance)")

    # HMM + composite-signal thresholds
    parser.add_argument("--entry-prob", type=float, default=0.65,
                        help="P(Bull) threshold to consider entry (default: 0.65)")
    parser.add_argument("--exit-prob", type=float, default=0.40,
                        help="P(Bull) threshold for regime exit after min-hold (default: 0.40)")
    parser.add_argument("--buy-threshold", type=float, default=argparse.SUPPRESS,
                        help="Composite score needed to trigger BUY (default: the "
                             "selected strategy's own default; 3.0 for 'default')")
    parser.add_argument("--sell-threshold", type=float, default=argparse.SUPPRESS,
                        help="Composite score triggering SELL (default: the "
                             "selected strategy's own default; -3.0 for 'default')")
    parser.add_argument("--volume-min-ratio", type=float, default=0.8,
                        help="Volume / 100-bar average minimum for entry (default: 0.8)")
    parser.add_argument("--interval", default="1h", choices=["1h", "1d"],
                        help="Bar interval: '1h' hourly ~730d history (default), "
                             "'1d' daily max history (~20–30yr for multi-cycle research). "
                             "Daily mode uses different defaults: regime-smooth=5, "
                             "min-hold-bars=5, min-train-bars=252, bars-per-year=252.")
    parser.add_argument("--regime-smooth", type=int, default=None,
                        help="P(Bull) smoothing window in bars "
                             "(default: 24 for 1h, 5 for 1d)")
    parser.add_argument("--min-hold-bars", type=int, default=None,
                        help="Minimum bars held before regime-exit or signal-SELL "
                             "(default: 48 for 1h, 5 for 1d)")

    # Stop-loss / take-profit — default: the selected strategy's own value.
    parser.add_argument("--stop-loss-pct", type=float, default=argparse.SUPPRESS,
                        help="Hard stop-loss fraction from entry (default: the "
                             "selected strategy's own default; 0.05 = 5%% for 'default')")
    parser.add_argument("--take-profit-pct", type=float, default=argparse.SUPPRESS,
                        help="Hard take-profit fraction from entry (default: the "
                             "selected strategy's own default; 0.15 = 15%% for 'default')")

    # Trailing / vol-stop — default: the selected strategy's own value.
    parser.add_argument("--trailing-stop", type=float, default=argparse.SUPPRESS,
                        help="Fixed trailing stop: fraction drop from peak. 0=off")
    parser.add_argument("--vol-stop-mult", type=float, default=argparse.SUPPRESS,
                        help="Vol-scaled trailing stop multiplier. 0=off")
    parser.add_argument("--vol-stop-window", type=int, default=argparse.SUPPRESS,
                        help="Lookback window in bars for realised vol (default: 20)")
    parser.add_argument("--profit-stop-scale", type=float, default=argparse.SUPPRESS,
                        help="Profit-scaled stop tightening per 1%% of gain. 0=off")
    parser.add_argument("--min-stop", type=float, default=argparse.SUPPRESS,
                        help="Floor on profit-adjusted stop (default: the "
                             "selected strategy's own default)")

    # Capital / costs
    parser.add_argument("--initial-cash", type=float, default=20_000.0)
    parser.add_argument("--transaction-cost", type=float, default=10.0)
    parser.add_argument("--cost-model", default="ibkr_tiered_spread", choices=COST_MODEL_CHOICES,
                        help="Transaction cost model: 'flat' = --transaction-cost/side "
                             "(historical, ~10x real fees at typical stake sizes); "
                             "'ibkr_tiered' = IBKR UK tiered commission + SDRT on .L buys; "
                             "'ibkr_tiered_spread' adds a half-spread estimate per side. "
                             "Default: ibkr_tiered_spread.")

    # Kelly
    parser.add_argument("--no-kelly", dest="use_kelly", action="store_false", default=True,
                        help="Use fixed 10%% allocation instead of Kelly sizing")

    # Volume seasonality
    parser.add_argument("--seasonal-volume", dest="seasonal_volume", action="store_true",
                        default=False,
                        help="Normalise volume ratio by same-hour-of-day trailing mean "
                             "instead of flat rolling-20 average (opt-in, unvalidated — "
                             "run with and without to compare before enabling live)")

    # Exit indicators — default: the selected strategy's own value.
    parser.add_argument("--exit-rsi-reversal", dest="exit_rsi", action="store_true",
                        default=argparse.SUPPRESS)
    parser.add_argument("--no-exit-rsi-reversal", dest="exit_rsi", action="store_false",
                        default=argparse.SUPPRESS)
    parser.add_argument("--exit-macd-cross", dest="exit_macd", action="store_true",
                        default=argparse.SUPPRESS)
    parser.add_argument("--exit-consolidation", dest="exit_consol", action="store_true",
                        default=argparse.SUPPRESS)
    parser.add_argument("--sar-stop", dest="sar_stop", action="store_true",
                        default=argparse.SUPPRESS)
    parser.add_argument("--sar-af-start", type=float, default=0.02)
    parser.add_argument("--sar-af-step", type=float, default=0.02)
    parser.add_argument("--sar-af-max", type=float, default=0.20)
    parser.add_argument("--max-hold-days", type=int, default=argparse.SUPPRESS,
                        help="Force exit after N bars. 0=off (default: the "
                             "selected strategy's own default)")

    # Indicator computation
    parser.add_argument("--no-skip-unused-indicators", dest="skip_unused_indicators",
                        action="store_false", default=True,
                        help="Compute all indicators even when the strategy weights them "
                             "at zero (useful when developing a new strategy)")
    parser.add_argument("--no-hmm-cache", dest="hmm_cache", action="store_false", default=True,
                        help="Recompute the HMM from scratch instead of continuing from the "
                             "persisted per-ticker filter state in data/cache/hmm_cache/")

    # Output
    parser.add_argument("--signal-reports-only", action="store_true", default=False,
                        help="Only render the chart, HTML daily summary and company lookup "
                             "when the current signal is BUY or SELL (used by the live "
                             "daemon to keep hourly cycles fast)")
    parser.add_argument("--no-email", dest="no_email", action="store_true", default=False,
                        help="Suppress trade alert emails even when SMTP_PASSWORD is set "
                             "(used by batch/daemon which handle email decisions externally)")

    # Strategy selection (entry + exit as a named pair; supersedes --plugin-gate)
    parser.add_argument("--strategy", default="default",
                        choices=sorted(STRATEGY_REGISTRY),
                        help="Named entry+exit strategy (default: 'default'). "
                             "Supersedes --plugin-gate when set to a non-default value.")

    # Plugin selection (orthogonal to --strategy)
    parser.add_argument("--plugin-sizer", default="kelly", choices=["kelly", "fixed"],
                        help="Position sizer: 'kelly' (default) or 'fixed' (flat 10%% allocation)")
    parser.add_argument("--plugin-gate", default=argparse.SUPPRESS, choices=["quality", "none"],
                        help="Quality gate override for the selected strategy's "
                             "quality_gate_enabled: 'quality' forces on, 'none' forces "
                             "off. Default: the selected strategy's own setting.")
    parser.add_argument("--gate-sensitivity", type=int, default=argparse.SUPPRESS,
                        help="Number of weak-context/adverse-exit conditions (of 5) "
                             "needed to fire the quality gate (default: the selected "
                             "strategy's own default; 2 for all strategies today)")
    parser.add_argument("--plugin-adjuster", default="sentiment", choices=["sentiment", "none"],
                        help="Context adjuster: 'sentiment' (default) or 'none' (identity)")

    # Compatibility stubs (silently accepted so watchlist.json defaults don't break)
    for stub in ("--years", "--window", "--threshold", "--no-hmm", "--rr-ratio", "--rr-risk"):
        parser.add_argument(stub, default=argparse.SUPPRESS)
    for stub in ("--long-only", "--no-long-only", "--sma200", "--no-sma200",
                 "--fractional", "--no-fractional"):
        parser.add_argument(stub, action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--in-sell-threshold", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--position-mode", default=argparse.SUPPRESS)

    return parser


#: CLI dest -> the value that flag used to hard-default to before it became
#: strategy-owned. Used both to detect "was this flag explicit" (via
#: hasattr, since these dests are now argparse.SUPPRESS) and to backfill the
#: attribute afterwards so every other `args.<dest>` reference in this module
#: keeps working unchanged whether or not the user passed the flag.
_TUNABLE_DEFAULTS: dict[str, object] = {
    "buy_threshold": 3.0,
    "sell_threshold": -3.0,
    "stop_loss_pct": 0.05,
    "take_profit_pct": 0.15,
    "trailing_stop": 0.0,
    "vol_stop_mult": 0.0,
    "vol_stop_window": 20,
    "profit_stop_scale": 0.0,
    "min_stop": 0.05,
    "max_hold_days": 0,
    "exit_rsi": False,
    "exit_macd": False,
    "exit_consol": False,
    "sar_stop": False,
    "plugin_gate": "quality",
    "gate_sensitivity": 2,
}

#: CLI dest -> the matching Entry/Exit constructor kwarg name, where they differ.
_EXIT_OVERRIDE_MAP: dict[str, str] = {
    "stop_loss_pct": "stop_loss_pct",
    "take_profit_pct": "take_profit_pct",
    "trailing_stop": "trailing_stop",
    "vol_stop_mult": "vol_stop_mult",
    "vol_stop_window": "vol_stop_window",
    "profit_stop_scale": "profit_stop_scale",
    "min_stop": "min_stop_pct",
    "max_hold_days": "max_hold_days",
    "exit_rsi": "exit_on_rsi_reversal",
    "exit_macd": "exit_on_macd_cross",
    "exit_consol": "exit_on_consolidation",
    "sar_stop": "use_sar_stop",
}


def _build_strategy_overrides(args: argparse.Namespace) -> tuple[dict, dict]:
    """Build (entry_overrides, exit_overrides) from only the CLI flags the
    user explicitly passed. Must run before any backfill of _TUNABLE_DEFAULTS
    onto `args` — relies on hasattr() being False for an omitted SUPPRESS'd
    flag. An omitted flag leaves the selected strategy's own default alone;
    an explicit flag overrides it uniformly, whatever --strategy is set to.
    """
    explicit = {k for k in _TUNABLE_DEFAULTS if hasattr(args, k)}

    entry_overrides: dict = {}
    if "buy_threshold" in explicit:
        entry_overrides["buy_threshold"] = args.buy_threshold
    if "sell_threshold" in explicit:
        entry_overrides["sell_threshold"] = args.sell_threshold
    if "plugin_gate" in explicit:
        entry_overrides["quality_gate_enabled"] = (args.plugin_gate == "quality")
    if "gate_sensitivity" in explicit:
        entry_overrides["gate_sensitivity"] = args.gate_sensitivity

    exit_overrides = {
        ctor_key: getattr(args, arg_key)
        for arg_key, ctor_key in _EXIT_OVERRIDE_MAP.items()
        if arg_key in explicit
    }
    return entry_overrides, exit_overrides


def _backfill_tunable_defaults(args: argparse.Namespace) -> None:
    """Fill in any _TUNABLE_DEFAULTS attribute the user didn't pass, so every
    `args.<dest>` reference elsewhere in this module keeps working exactly as
    before regardless of --strategy — must run after _build_strategy_overrides
    so it doesn't erase the hasattr()-based explicit/implicit distinction."""
    for key, value in _TUNABLE_DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, value)


_DAILY_BAR_DEFAULTS = {
    "regime_smooth": 5,    # 1 week vs 24 hours
    "min_hold_bars": 5,    # 1 week vs 2 days
}
_HOURLY_BAR_DEFAULTS = {
    "regime_smooth": 24,
    "min_hold_bars": 48,
}
_DAILY_ENGINE_PARAMS = {
    "min_train_bars": 252,   # 1 year of daily bars
    "hmm_refit_bars": 100,   # ~4 months between refits
    "bars_per_year": 252,
}
_HOURLY_ENGINE_PARAMS = {
    "min_train_bars": 500,
    "hmm_refit_bars": 500,
    "bars_per_year": 1700,
}


def _resolve_interval_defaults(args: argparse.Namespace) -> None:
    """Fill None-defaulted params with interval-appropriate values."""
    bar_defaults = _DAILY_BAR_DEFAULTS if args.interval == "1d" else _HOURLY_BAR_DEFAULTS
    for key, val in bar_defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)


def _write_quality_gate(run_dir: Path, flag: str, reason: str = "") -> None:
    payload = {"flag": flag, "reason": reason}
    (run_dir / "qualityGate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_backtest_summary(bt: dict) -> None:
    def _pct(v: float) -> str:
        return f"{v*100:.2f}%" if np.isfinite(v) else "NaN"
    def _f(v: float) -> str:
        return f"{v:.3f}" if np.isfinite(v) else "NaN"

    logger.info(f"  {'':30s}  {'Strategy':>10s}  {'Buy & Hold':>10s}")
    logger.info(f"  {'Sharpe (annualised)':30s}  {_f(bt['sharpe_strategy']):>10s}  {_f(bt['sharpe_bh']):>10s}")
    logger.info(f"  {'Sortino (annualised)':30s}  {_f(bt.get('sortino_strategy', float('nan'))):>10s}  {_f(bt.get('sortino_bh', float('nan'))):>10s}")
    logger.info(f"  {'Calmar':30s}  {_f(bt.get('calmar_strategy', float('nan'))):>10s}  {_f(bt.get('calmar_bh', float('nan'))):>10s}")
    logger.info(f"  {'Max drawdown':30s}  {_pct(bt['max_drawdown_strategy']):>10s}  {_pct(bt['max_drawdown_bh']):>10s}")
    logger.info(f"  {'Total return':30s}  {_pct(bt['total_return_strategy']):>10s}  {_pct(bt['total_return_bh']):>10s}")
    logger.info(f"  {'Bars in market':30s}  {bt['n_bars']:>10d}")

    ic  = bt["initial_cash"]
    fp  = bt["final_portfolio"]
    pl  = bt["total_pl"]
    pct = pl / ic * 100
    sign = "+" if pl >= 0 else ""
    cur = "GBP"
    bh_tr = bt["total_return_bh"]
    bh_fp = ic * (1 + bh_tr) if np.isfinite(bh_tr) else float("nan")
    bh_pl = bh_fp - ic if np.isfinite(bh_fp) else float("nan")
    detail = bt.get("detail", pd.DataFrame())
    if not detail.empty:
        bt_start = str(detail.index[0])[:10]
        bt_end   = str(detail.index[-1])[:10]
    else:
        bt_start = bt_end = "N/A"
    logger.info(f"\n  -- P&L simulation ({bt_start} to {bt_end} | {cur}{ic:,.0f} initial, {cur}{bt.get('trade_cost',10):.0f}/trade) --")
    logger.info(f"  {'Trades (buys + sells)':30s}  {bt['n_buys']} + {bt['n_sells']} = {bt['n_buys']+bt['n_sells']}")
    logger.info(f"  {'Strategy final portfolio':30s}  {cur}{fp:,.2f}")
    logger.info(f"  {'Strategy P&L':30s}  {sign}{cur}{pl:,.2f}  ({sign}{pct:.1f}%)")
    if np.isfinite(bh_pl):
        bh_sign = "+" if bh_pl >= 0 else ""
        logger.info(f"  {'Buy & Hold final':30s}  {cur}{bh_fp:,.2f}")
        logger.info(f"  {'Buy & Hold P&L':30s}  {bh_sign}{cur}{bh_pl:,.2f}  ({bh_sign}{bh_tr*100:.1f}%)")
    logger.info(f"  {'Kelly fraction (final)':30s}  {bt['final_kelly']*100:.1f}%")

    if not detail.empty and "strategy_return" in detail.columns and "bar_return" in detail.columns:
        logger.info(f"\n  -- Yearly P&L --")
        logger.info(f"  {'Year':6s}  {'Strategy':>10s}  {'Buy&Hold':>10s}  {'Trades':>7s}")
        for year, grp in detail.groupby(detail.index.year):
            yr_strat = float((1 + grp["strategy_return"]).prod() - 1)
            yr_bh = float((1 + grp["bar_return"]).prod() - 1)
            n_yr_trades = int((grp["trade_event"] == "BUY").sum())
            logger.info(f"  {year:6d}  {_pct(yr_strat):>10s}  {_pct(yr_bh):>10s}  {n_yr_trades:>7d}")


def main(argv: list[str] | None = None) -> int:
    setup_cli_logger("run")

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    entry_overrides, exit_overrides = _build_strategy_overrides(args)
    _backfill_tunable_defaults(args)
    _resolve_interval_defaults(args)

    engine_params = _DAILY_ENGINE_PARAMS if args.interval == "1d" else _HOURLY_ENGINE_PARAMS

    logger.info(f"\nstrategy-auto-trader (consolidated engine) — ticker={args.ticker}, interval={args.interval}")

    run_dir = _make_run_dir(args.ticker)
    logger.info(f"  run output directory: {run_dir}")

    t0 = time.time()
    if args.interval == "1d":
        logger.info(f"  fetching {args.ticker} daily data from Yahoo Finance (max history)...")
        df = fetch_daily(args.ticker)
        if df is None or df.empty:
            logger.info(f"  ERROR: could not fetch daily data for {args.ticker}")
            _write_quality_gate(run_dir, "HOLD", "no data")
            return 1
    else:
        logger.info(f"  fetching {args.ticker} hourly data (source={args.source})...")
        df = fetch_hourly(args.ticker, period="730d", source=args.source)
        if df is None or df.empty:
            logger.info(f"  ERROR: could not fetch hourly data for {args.ticker}")
            _write_quality_gate(run_dir, "HOLD", "no data")
            return 1

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    logger.info(f"  fetched {len(df)} {args.interval} bars | {df.index.min()} -> {df.index.max()}")

    df.to_csv(run_dir / "inputData.csv")

    _ADJ_MAP = {"sentiment": SentimentAdjuster, "none": NullAdjuster}
    # None lets the engine derive use_kelly/kelly_lookback from exit_strategy
    # (see ExitStrategyProtocol) — only build an explicit sizer here when the
    # CLI wants something other than the strategy's own declared default.
    position_sizer = None
    if args.plugin_sizer == "fixed":
        position_sizer = FixedSizer()
    elif not args.use_kelly:
        position_sizer = KellySizer(use_kelly=False, lookback=20)
    context_adjuster = _ADJ_MAP[args.plugin_adjuster]()

    # Strategy: resolve named entry+exit pair, uniformly for every --strategy.
    # entry_overrides/exit_overrides (built above, before SUPPRESS backfill)
    # carry only the CLI flags the user explicitly passed — everything else
    # falls through to the selected strategy's own defaults.
    # Daily mode: vol screen is calibrated on hourly data — hourly TQ is
    # irrelevant for daily-bar research, so bypass it unconditionally.
    vol_filter_ok = True if args.interval == "1d" else None
    entry_s, exit_s = resolve_strategy(
        args.strategy, ticker=args.ticker,
        vol_filter_ok=vol_filter_ok,
        entry_overrides=entry_overrides, exit_overrides=exit_overrides,
        source=args.source,
    )

    regime_model = None
    if args.hmm_cache and args.interval == "1h":
        from ..plugins.persistent_hmm import PersistentHMMRegimeModel
        safe_ticker = args.ticker.replace("/", "-").replace("\\", "-")
        regime_model = PersistentHMMRegimeModel(
            HMM_CACHE_DIR / f"{safe_ticker}.pkl",
            dates=df.index,
            closes=df["Close"].values,
            regime_smooth=args.regime_smooth,
            bull_edge=args.entry_prob,
            bear_edge=args.exit_prob,
        )

    interval_label = "daily multi-cycle" if args.interval == "1d" else "consolidated walk-forward"
    logger.info(f"\nRunning {interval_label} backtest...")
    bt = consolidated_backtest(
        df,
        regime_model=regime_model,
        entry_prob=args.entry_prob,
        exit_prob=args.exit_prob,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        volume_min_ratio=args.volume_min_ratio,
        initial_cash=args.initial_cash,
        trade_cost=args.transaction_cost,
        cost_model=make_cost_model(args.cost_model, args.ticker, args.transaction_cost),
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
        position_sizer=position_sizer,
        context_adjuster=context_adjuster,
        entry_strategy=entry_s,
        exit_strategy=exit_s,
        skip_unused_indicators=args.skip_unused_indicators,
        min_train_bars=engine_params["min_train_bars"],
        hmm_refit_bars=engine_params["hmm_refit_bars"],
        bars_per_year=engine_params["bars_per_year"],
        use_seasonal_volume=args.seasonal_volume,
    )

    if regime_model is not None:
        regime_model.save()
        logger.info(f"  HMM: {regime_model.cache_hits} bars from cache, "
              f"{regime_model.computed_steps} computed")

    if bt["n_bars"] == 0:
        logger.info("  Insufficient data for consolidated backtest (need >= min_train_bars bars).")
        _write_quality_gate(run_dir, "HOLD", "insufficient data")
        return 0

    _print_backtest_summary(bt)

    detail = bt["detail"]
    detail.to_csv(run_dir / "compositeBacktest.csv")

    # Current signal from last bar
    last_row = detail.iloc[-1]
    last_event  = str(last_row.get("trade_event", ""))
    last_pos    = float(last_row.get("position", 0.0))
    last_p_bull = float(last_row.get("p_bull_smooth", 0.5))
    last_regime = float(last_row.get("regime_signal", 0.0) or 0.0)

    if last_event == "BUY":
        cur_flag = "BUY"
        cur_reason = "entry: composite signal + quality gate passed"
    elif last_event == "SELL":
        cur_flag = "SELL"
        cur_reason = str(last_row.get("sell_reason", "signal"))
    elif last_pos > 0:
        cur_flag = "HOLD"
        cur_reason = "no clear buy or sell signal"
    else:
        cur_flag = "HOLD"
        cur_reason = "flat"

    _write_quality_gate(run_dir, cur_flag, cur_reason)

    logger.info(f"\n  >>> {cur_flag}  (p_bull_smooth={last_p_bull:.2f}, regime_signal={last_regime:.2f}) <<<")
    if last_event:
        logger.info(f"  last bar: {last_event}  reason={last_row.get('sell_reason','')}")

    # Chart + HTML report are for humans; on daemon cycles they are only
    # rendered when there is a signal event worth reviewing.
    make_reports = not args.signal_reports_only or cur_flag in ("BUY", "SELL")
    close_series = pd.Series(df["Close"].values, index=df.index, name="Close")

    if not make_reports:
        logger.info("  Chart/report skipped (no signal event)")

    # Chart (close series from hourly df, uses quant_engine detail format)
    if make_reports:
        try:
            plot_backtest(close_series, bt, ticker=args.ticker,
                          out_path=run_dir / "backtest_chart.png")
        except Exception as exc:
            logger.info(f"  Chart skipped: {exc}")

    # Generate detailed HTML daily summary report
    try:
        from datetime import date
        detail = bt.get("detail", pd.DataFrame()).copy()

        # Rename/add missing columns expected by report.py
        if "signal_score" in detail.columns and "score" not in detail.columns:
            detail["score"] = detail["signal_score"]
        if "effective_stop" not in detail.columns:
            detail["effective_stop"] = detail.get("stop_level", pd.NA)

        # Extract HMM info if available
        hmm_info = bt.get("hmm", None)

        # Get signal and momentum data from last bar
        if make_reports and not detail.empty:
            company_name, company_sector = _fetch_company_info(args.ticker)
            if company_name != args.ticker:
                logger.info(f"  {company_name}  [{company_sector}]" if company_sector else f"  {company_name}")

            last_row = detail.iloc[-1]

            # Extract values safely
            close_val = float(last_row.get("close", 0)) if "close" in last_row.index else 0.0
            regime_signal_val = float(last_row.get("regime_signal", 0) or 0) if "regime_signal" in last_row.index else 0.0
            p_bull = float(last_row.get("p_bull_smooth", 0.5)) if "p_bull_smooth" in last_row.index else 0.5

            # Derive regime from p_bull_smooth
            if p_bull > 0.6:
                regime_val = "Bull"
                hmm_state = 2
            elif p_bull < 0.4:
                regime_val = "Bear"
                hmm_state = 0
            else:
                regime_val = "Sideways"
                hmm_state = 1

            # Compute momentum signals (RSI, SMAs, volume)
            mom = momentum_signals(close_series)

            # Compute ALL signal votes using the composite_signal function
            sig_result = composite_signal(
                markov_signal=regime_signal_val,
                mom=mom,
                sell_threshold=args.sell_threshold,
                hmm_state=hmm_state,
                buy_threshold=args.buy_threshold,
            )

            sig = {
                "flag": cur_flag,
                "score": sig_result["score"],
                "max_score": sig_result["max_score"],
                "votes": sig_result["votes"],  # All votes: markov, rsi, trend, sma200, volume, hmm
            }

            # Compute exit indicators (MACD, RSI reversals, Bollinger, ATR, consolidation)
            exit_ind = exit_indicators(close_series, rsi=None)

            # Stationary distribution and transition matrix from HMM
            if hmm_info:
                stationary = hmm_info.get("stationary_distribution", {})
                transition_matrix = hmm_info.get("transition_matrix", None)
                state_names = hmm_info.get("regime_names", ["Bear", "Sideways", "Bull"])
            else:
                # Fallback if HMM info is missing
                stationary = {"Bear": 0.2, "Sideways": 0.3, "Bull": 0.5}
                transition_matrix = np.array([[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]])
                state_names = ["Bear", "Sideways", "Bull"]

            # Update bt with detail that has the expected columns
            bt_for_report = dict(bt)
            bt_for_report["detail"] = detail

            write_daily_summary(
                ticker=args.ticker,
                company_name=company_name,
                company_sector=company_sector,
                run_date=date.today(),
                close=close_series,
                current_state_name=regime_val,
                markov_sig=regime_signal_val,
                sig=sig,
                mom=mom,
                bt=bt_for_report,
                stationary=stationary,
                transition_matrix=transition_matrix,
                state_names=state_names,
                eff_stop_today=args.vol_stop_mult * 0.05,
                vol_stop_mult=args.vol_stop_mult,
                vol_stop_window=args.vol_stop_window,
                trailing_stop=args.trailing_stop,
                hmm=hmm_info,
                exit_ind=exit_ind,
                out_path=run_dir / "daily_summary.html",
            )
    except Exception as exc:
        logger.info(f"  Daily summary report skipped: {exc}")

    # Send email if BUY or SELL signal (suppressed when --no-email, e.g. daemon/batch callers)
    try:
        if cur_flag in ("BUY", "SELL") and not args.no_email:
            import os
            if os.environ.get("SMTP_PASSWORD"):
                from ..output.emailer import send_trade_alert
                send_trade_alert({
                    "ticker": args.ticker,
                    "current_signal": cur_flag,
                    "close": float(last_row.get("close", 0) or 0) if not detail.empty else 0,
                    "score": float(last_row.get("signal_score", 0) or 0) if not detail.empty else 0,
                    "run_dir": str(run_dir),
                })
    except Exception as exc:
        logger.info(f"  Email alert skipped: {exc}")

    elapsed = time.time() - t0
    logger.info(f"\nIntermediate data written to: {run_dir}  ({elapsed:.0f}s)")
    logger.info("\n----------------------------------------------------------------")
    logger.info(" Framework: Roan (@RohOnChain).")
    logger.info(" Backtests are historical, not forward-looking.")
    logger.info("----------------------------------------------------------------\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
