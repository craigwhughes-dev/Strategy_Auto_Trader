"""Ablation-test optimised_new's engine-level admission gates (the ones outside
the weighted composite score) against a fixed ticker basket, using the
strategy's own entry/exit classes so weights, thresholds, vetoes, and the
ratchet exit are exactly what's live. See
strategy/optimised_new.description.md Sec.04 for what each gate does.

Gates covered (todo.md "optimised_new admission-gate audit" Plan1 A/B):
  - require_flip_entry: no CLI flag, no strategy-level override today (see
    strategy/base/protocols.py) -- only ever read via
    getattr(entry, "require_flip_entry", True) in consolidated_engine.py.
    The off variant sets that attribute directly on a fresh entry instance.
  - vol_filter_ok (the trend_quality pre-screen from resolve_strategy()): the
    off variant forces the instance's private `_vol_filter_ok` to True,
    bypassing the veto without touching quant_hmm/vol_screen.py.

Each variant isolates ONE gate against the current-default baseline (the
other gate stays at its live setting) so effects don't get tangled together.
Reports each ticker's live vol_filter_ok status up front -- per the
plan-review-panel note, a basket that's mostly already-vetoed makes the
vol_filter ablation uninformative, so check that before trusting the result.

Usage:
    uv run python -m Strategy_Auto_Trader.markov_cli.compare_admission_gates
"""

from __future__ import annotations

import logging

import sys
import time

import numpy as np
import pandas as pd

from .compare_exits import TEST_TICKERS, _fetch
from ..core.cli_logging import setup_cli_logger

logger = logging.getLogger(__name__)


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


def _resolve_vol_filter_ok(ticker: str) -> bool:
    """Real trend_quality pre-screen result for `ticker` -- computed once per
    ticker and reused across variants so we don't hit vol_screen's yfinance
    fetch three times per ticker for the same answer."""
    from ..strategy.base.registry import resolve_strategy
    entry, _ = resolve_strategy("optimised_new", ticker=ticker)
    return entry._vol_filter_ok


def _build_variant(vol_filter_ok: bool, variant: dict):
    from ..strategy.optimised_new import OptimisedNewEntry, OptimisedNewExit
    effective_vol_filter_ok = True if variant["force_vol_filter_on"] else vol_filter_ok
    entry = OptimisedNewEntry(vol_filter_ok=effective_vol_filter_ok)
    entry.require_flip_entry = variant["require_flip_entry"]
    return entry, OptimisedNewExit()


def main() -> int:
    setup_cli_logger("compare_admission_gates")

    from ..quant_hmm.consolidated_engine import consolidated_backtest

    logger.info(f"Ablating optimised_new admission gates across {len(TEST_TICKERS)} "
          f"tickers (~2.9yr hourly history)\n")

    price_data = {}
    vol_filter_by_ticker = {}
    for ticker in TEST_TICKERS:
        df = _fetch(ticker)
        if df is not None and len(df) > 300:
            price_data[ticker] = df
            vol_filter_by_ticker[ticker] = _resolve_vol_filter_ok(ticker)
            logger.info(f"  Fetched {ticker}: {len(df)} bars, vol_filter_ok={vol_filter_by_ticker[ticker]}")
        else:
            logger.info(f"  Skipped {ticker}: insufficient data")

    n_vetoed = sum(1 for v in vol_filter_by_ticker.values() if not v)
    if n_vetoed:
        logger.info(f"\n  Note: {n_vetoed}/{len(vol_filter_by_ticker)} tickers are vol_filter-vetoed "
              f"today -- baseline and require_flip_entry=False will show 0 trades for those; "
              f"only the 'vol_filter pre-screen off' variant will trade them. Weigh the "
              f"vol_filter ablation result accordingly if this is most of the basket.")

    all_results = []
    t0 = time.time()
    total = len(price_data) * len(VARIANTS)
    done = 0

    for ticker, df in price_data.items():
        vol_filter_ok = vol_filter_by_ticker[ticker]
        for variant in VARIANTS:
            done += 1
            entry, exit_ = _build_variant(vol_filter_ok, variant)
            try:
                bt = consolidated_backtest(
                    df, entry_strategy=entry, exit_strategy=exit_, **_HOURLY_DEFAULTS,
                )
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

                all_results.append({
                    "ticker": ticker,
                    "variant": variant["name"],
                    "sharpe": bt["sharpe_strategy"],
                    "sortino": bt.get("sortino_strategy", float("nan")),
                    "total_return": bt["total_return_strategy"],
                    "max_dd": bt["max_drawdown_strategy"],
                    "pl": bt["total_pl"],
                    "bh_return": bt["total_return_bh"],
                    "n_trades": bt["n_buys"],
                    "wins": wins,
                    "losses": losses,
                    "win_rate": wins / (wins + losses) * 100 if (wins + losses) else 0,
                })
            except Exception:
                all_results.append({
                    "ticker": ticker, "variant": variant["name"],
                    "sharpe": float("nan"), "sortino": float("nan"),
                    "total_return": float("nan"), "max_dd": float("nan"),
                    "pl": float("nan"), "bh_return": float("nan"),
                    "n_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                })

        if done % len(VARIANTS) == 0:
            elapsed = time.time() - t0
            logger.info(f"  [{done}/{total}] {ticker} done ({elapsed:.0f}s)")

    df_res = pd.DataFrame(all_results)
    elapsed = time.time() - t0

    logger.info(f"\n{'='*100}")
    logger.info(f" Results by ticker ({elapsed:.0f}s)")
    logger.info(f"{'='*100}")

    first_variant = VARIANTS[0]["name"]
    for ticker in price_data:
        sub = df_res[df_res["ticker"] == ticker].copy()
        baseline = sub[sub["variant"] == first_variant]
        bh_ret = baseline["bh_return"].iloc[0] * 100 if len(baseline) else 0

        logger.info(f"\n  {ticker}  (B&H: {bh_ret:+.1f}%, vol_filter_ok={vol_filter_by_ticker[ticker]})")
        logger.info(f"  {'Variant':<32s} {'P&L':>10s} {'Return':>8s} {'Win%':>6s} {'W/L':>7s} {'Sharpe':>7s} {'Trades':>7s}")
        logger.info(f"  {'-'*32} {'-'*10} {'-'*8} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")

        for _, row in sub.iterrows():
            ret = row["total_return"] * 100 if np.isfinite(row["total_return"]) else float("nan")
            logger.info(f"  {row['variant']:<32s} {row['pl']:>+9,.0f} {ret:>+7.1f}% {row['win_rate']:>5.0f}% "
                  f"{row['wins']:>3d}/{row['losses']:<3d} {row['sharpe']:>7.3f} {row['n_trades']:>7.0f}")

    logger.info(f"\n{'='*100}")
    logger.info(f" Aggregate (average across {len(price_data)} tickers)")
    logger.info(f"{'='*100}")

    agg = df_res.groupby("variant").agg({
        "total_return": "mean",
        "sharpe": "mean",
        "sortino": "mean",
        "max_dd": "mean",
        "pl": "mean",
        "n_trades": "mean",
        "win_rate": "mean",
    })
    variant_order = [v["name"] for v in VARIANTS]
    agg = agg.loc[[v for v in variant_order if v in agg.index]]

    logger.info(f"\n  {'Variant':<32s} {'Avg P&L':>10s} {'Avg Return':>10s} {'Win%':>6s} {'Sharpe':>7s} {'Sortino':>8s} {'MaxDD':>7s} {'Trades':>7s}")
    logger.info(f"  {'-'*32} {'-'*10} {'-'*10} {'-'*6} {'-'*7} {'-'*8} {'-'*7} {'-'*7}")
    for variant_name, row in agg.iterrows():
        logger.info(f"  {variant_name:<32s} {row['pl']:>+9,.0f} {row['total_return']*100:>+9.1f}% "
              f"{row['win_rate']:>5.0f}% {row['sharpe']:>7.3f} {row['sortino']:>8.3f} "
              f"{row['max_dd']*100:>6.1f}% {row['n_trades']:>6.0f}")

    logger.info(f"\n  {'Variant':<32s} {'Profitable':>11s} {'Profitable %':>13s}")
    logger.info(f"  {'-'*32} {'-'*11} {'-'*13}")
    for variant in VARIANTS:
        vals = df_res[df_res["variant"] == variant["name"]].set_index("ticker")["pl"]
        n_profit = sum(1 for pl in vals if np.isfinite(pl) and pl > 0)
        n_total = sum(1 for pl in vals if np.isfinite(pl))
        logger.info(f"  {variant['name']:<32s} {n_profit:>7d}/{n_total:<3d} {n_profit/n_total*100 if n_total else 0:>12.0f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
