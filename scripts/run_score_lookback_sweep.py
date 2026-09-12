#!/usr/bin/env python3
"""
score_lookback_days sweep: how much recent candidate history to use for top-K scoring.

Candidates generated ONCE per window; only filter_candidates_by_top_tickers() is
re-run per lookback value (~instant), then arbitrate() re-runs per combo.

Windows:
  - 2023 window (synthetic 2022-01-01 to 2024-12-31): tests whether recent-window
    scoring admits AI/momentum names (NVDA, META) that drove the 2023 rally.
  - Real window (ibkr, 2024-01-01 to present): checks production impact.

Usage:
    uv run python scripts/run_score_lookback_sweep.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Strategy_Auto_Trader.core.cli_logging import setup_cli_logger
from Strategy_Auto_Trader.markov_cli import full_scan
from Strategy_Auto_Trader.markov_cli.live_sim import (
    arbitrate,
    resolve_vix_entry_gate_threshold,
)
from Strategy_Auto_Trader.output.journal import append_trades
from Strategy_Auto_Trader.quant_hmm.quant_engine import fetch_vix_ibkr
from Strategy_Auto_Trader.quant_hmm.ticker_ranking import (
    _filter_candidates_by_daily_trend_quality,
    filter_candidates_by_top_tickers,
    generate_candidates,
)
from Strategy_Auto_Trader.strategy.base.registry import wants_low_trend_quality
from Strategy_Auto_Trader.synthetic_backtest_data.generate import (
    SYNTHETIC_HMM_CACHE_DIR,
    load_synthetic_hourly,
)
from Strategy_Auto_Trader.synthetic_backtest_data.stooq_daily import load_stooq_daily

STRATEGY_NAME = "optimised_new"
JOURNAL_DIR = Path("data/journals")
WORKERS = 4
TOP_K = 70
INITIAL_CASH = 100_000.0
TRADE_COST = 1.0
COST_MODEL = "ibkr_tiered_spread"

# None = all-time (current behavior / baseline)
SCORE_LOOKBACK_VALUES = [None, 252, 504, 756, 1008, 1260]


def _max_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    max_dd = 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            max_dd = min(max_dd, (v - peak) / peak)
    return max_dd


def _build_raw_candidates(tickers, source, start_date, df_by_ticker=None, hmm_cache_dir=None):
    logger = logging.getLogger(__name__)
    logger.info(f"  Generating candidates for {len(tickers)} tickers (source={source})...")
    candidates, price_by_ticker, trend_quality_by_ticker = generate_candidates(
        tickers=list(tickers),
        strategy_name=STRATEGY_NAME,
        vol_filter_tag="daily-rescreened",
        vol_filter_ok=True,
        workers=WORKERS,
        use_seasonal_volume=True,
        source=source,
        df_by_ticker=df_by_ticker,
        use_persistent_cache=True,
        hmm_cache_dir=hmm_cache_dir,
        historical_only=True,
        vol_window=252,
    )

    cutoff = pd.Timestamp(start_date)
    candidates = [c for c in candidates if c.date_opened.tz_localize(None) >= cutoff]

    wants_low = wants_low_trend_quality(STRATEGY_NAME)
    n_before = len(candidates)
    candidates = _filter_candidates_by_daily_trend_quality(
        candidates, trend_quality_by_ticker, 0.0, wants_low,
    )
    logger.info(f"  daily vol gate: {len(candidates)}/{n_before} candidates survive")

    return candidates, price_by_ticker, trend_quality_by_ticker


def _run_combo(label, candidates_all, trend_quality_by_ticker, price_by_ticker,
               vix_series, vix_threshold, score_lookback_days):
    logger = logging.getLogger(__name__)

    n_before = len(candidates_all)
    candidates, ticker_scores = filter_candidates_by_top_tickers(
        candidates_all, trend_quality_by_ticker, TOP_K,
        vol_weight=0.7, win_rate_weight=0.3, lookback_days=60,
        score_lookback_days=score_lookback_days,
    )
    top_tickers = sorted(ticker_scores, key=lambda t: -ticker_scores[t])[:TOP_K]
    logger.info(f"  top-{TOP_K} filter: {len(candidates)}/{n_before} candidates "
                f"from {len(top_tickers)} tickers")

    result = arbitrate(
        candidates,
        initial_cash=INITIAL_CASH,
        trade_cost=TRADE_COST,
        cost_model_name=COST_MODEL,
        currency="GBP",
        price_by_ticker=price_by_ticker,
        same_day_deployment_cap_pct=None,
        vix_series=vix_series,
        vix_entry_gate_threshold=vix_threshold,
        daily_returns_by_ticker=None,
        max_correlation_to_admitted_today=None,
        vix_recovery_window_days=None,
        vix_recovery_kelly_mult=0.5,
        vix_gate_allow_reentry=False,
    )

    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    append_trades(JOURNAL_DIR / f"{label}.csv", result["executed"])

    total_pnl = sum(r.pnl_usd for r in result["executed"])
    peak_deployed = max((row["deployed"] for row in result["equity_curve"]), default=0.0)
    max_dd = _max_drawdown([row["portfolio_value"] for row in result["equity_curve"]])

    rows = [{"strategy": STRATEGY_NAME, "pot_size": INITIAL_CASH, **row}
            for row in result["equity_curve"]]
    rows.append({
        "strategy": STRATEGY_NAME, "pot_size": INITIAL_CASH, "date": "SUMMARY",
        "cash": result["final_cash"], "deployed": peak_deployed, "n_open": 0,
        "portfolio_value": result["final_cash"],
        "realized_pnl_cum": total_pnl, "interest_cum": result["total_interest"],
        "n_candidates": result["n_candidates"], "n_admitted": result["n_admitted"],
        "n_rejected_cash": result["n_rejected_cash"],
        "n_rejected_kelly": result["n_rejected_kelly"],
        "n_rejected_concentration": result["n_rejected_concentration"],
        "n_rejected_vix": result["n_rejected_vix"],
        "n_rejected_correlation": result["n_rejected_correlation"],
        "max_drawdown": max_dd,
        "score_lookback_days": score_lookback_days,
        "top_tickers_sample": ",".join(top_tickers[:10]),
    })
    equity_path = JOURNAL_DIR / f"{label}_equity.csv"
    equity_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(equity_path, index=False)

    logger.info(f"  {label}: {len(result['executed'])}/{result['n_candidates']} admitted, "
                f"final £{result['final_cash']:,.2f} (P&L £{total_pnl:+,.2f}), "
                f"max_dd={max_dd:.1%}")

    return {
        "label": label,
        "score_lookback_days": score_lookback_days,
        "n_admitted": result["n_admitted"],
        "n_candidates": result["n_candidates"],
        "total_pnl": total_pnl,
        "final_cash": result["final_cash"],
        "return_pct": (result["final_cash"] / INITIAL_CASH - 1) * 100,
        "max_dd": max_dd,
        "top_tickers": top_tickers[:20],
    }


def _load_vix_stooq() -> pd.Series | None:
    vix_df = load_stooq_daily("^VIX")
    if vix_df is None or "Close" not in vix_df.columns:
        return None
    s = vix_df["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def _load_vix_ibkr() -> pd.Series | None:
    vix_df = fetch_vix_ibkr(historical_only=True)
    if vix_df is None or "Close" not in vix_df.columns:
        return None
    s = vix_df["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def main():
    setup_cli_logger("score_lookback_sweep")
    logger = logging.getLogger(__name__)

    all_tickers = full_scan.load_sp_ftse_universe()
    vix_threshold = resolve_vix_entry_gate_threshold(STRATEGY_NAME)  # 20.0

    results = []

    # ------------------------------------------------------------------ #
    #  2023 WINDOW  (synthetic 2022-01-01 to 2024-12-31)                  #
    # ------------------------------------------------------------------ #
    logger.info("\n=== 2023 WINDOW (synthetic 2022-01-01 to 2024-12-31) ===")
    synthetic_dir = Path("data_synthetic/hourly")
    synthetic_start = "2022-01-01"
    synthetic_end = "2024-12-31"

    df_by_ticker: dict = {}
    dropped = []
    for ticker in all_tickers:
        df = load_synthetic_hourly(ticker, hourly_dir=synthetic_dir)
        if df is None:
            dropped.append(ticker)
            continue
        window = df.loc[synthetic_start:synthetic_end]
        if window.empty:
            dropped.append(ticker)
            continue
        df_by_ticker[ticker] = window
    synth_tickers = list(df_by_ticker.keys())
    logger.info(f"  {len(synth_tickers)}/{len(all_tickers)} tickers have synthetic data in window")

    synth_cands_raw, synth_prices, synth_tq = _build_raw_candidates(
        synth_tickers, source="ibkr", start_date=synthetic_start,
        df_by_ticker=df_by_ticker, hmm_cache_dir=SYNTHETIC_HMM_CACHE_DIR,
    )

    synth_vix = _load_vix_stooq() if vix_threshold is not None else None
    if synth_vix is None and vix_threshold is not None:
        logger.warning("  VIX gate: ^VIX stooq load failed — gate disabled for 2023 window")

    for sld in SCORE_LOOKBACK_VALUES:
        tag = f"sld{sld}" if sld is not None else "sld_alltime"
        label = f"score_lookback_{tag}_2023"
        logger.info(f"\n  --- score_lookback_days={sld} (2023 window) ---")
        r = _run_combo(label, synth_cands_raw, synth_tq, synth_prices,
                       synth_vix, vix_threshold, sld)
        r["window"] = "2023"
        results.append(r)

    # ------------------------------------------------------------------ #
    #  REAL WINDOW  (ibkr, 2024-01-01 to present)                         #
    # ------------------------------------------------------------------ #
    logger.info("\n=== REAL WINDOW (ibkr, start-date 2024-01-01) ===")
    real_start = "2024-01-01"

    real_cands_raw, real_prices, real_tq = _build_raw_candidates(
        all_tickers, source="ibkr", start_date=real_start,
    )

    real_vix = _load_vix_ibkr() if vix_threshold is not None else None
    if real_vix is None and vix_threshold is not None:
        logger.warning("  VIX gate: ^VIX ibkr fetch failed — gate disabled for real window")

    for sld in SCORE_LOOKBACK_VALUES:
        tag = f"sld{sld}" if sld is not None else "sld_alltime"
        label = f"score_lookback_{tag}_real"
        logger.info(f"\n  --- score_lookback_days={sld} (real window) ---")
        r = _run_combo(label, real_cands_raw, real_tq, real_prices,
                       real_vix, vix_threshold, sld)
        r["window"] = "real"
        results.append(r)

    # ------------------------------------------------------------------ #
    #  Summary                                                             #
    # ------------------------------------------------------------------ #
    logger.info("\n\n=== SCORE LOOKBACK SWEEP SUMMARY ===")
    logger.info(f"{'score_lookback':>15} | {'window':>8} | {'return%':>8} | {'max_dd':>7} | {'admitted':>8} | top-10 tickers (sample)")
    logger.info("-" * 110)
    for r in results:
        sld_str = str(r["score_lookback_days"]) if r["score_lookback_days"] else "all-time"
        top10 = ", ".join(r["top_tickers"][:10])
        logger.info(f"{sld_str:>15} | {r['window']:>8} | {r['return_pct']:>8.1f} | "
                    f"{r['max_dd']:>7.1%} | {r['n_admitted']:>8} | {top10}")

    # Save summary CSV
    summary_rows = [{k: v for k, v in r.items() if k != "top_tickers"} for r in results]
    for i, r in enumerate(results):
        summary_rows[i]["top_tickers_20"] = ",".join(r["top_tickers"])
    summary_path = JOURNAL_DIR / "score_lookback_sweep_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    logger.info(f"\nSummary written to {summary_path}")
    logger.info("Run: uv run python scripts/analyze_score_lookback_sweep.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
