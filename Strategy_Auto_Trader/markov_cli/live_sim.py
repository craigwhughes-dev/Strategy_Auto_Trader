"""Live-style multi-ticker portfolio simulation.

Simulates running a strategy "live" across a basket of tickers sharing a
single capital pool, starting from a given date. Unlike batch.py (which runs
each ticker's backtest independently with its own capital), this walks a
strategy's BUY/SELL signal stream per ticker but arbitrates entries across
tickers against one shared cash pot: priority by signal strength when
multiple tickers want to enter the same day, and position sizing from the
strategy's own Kelly fraction against currently available cash — a candidate
with kelly_fraction <= 0 is rejected outright. No daily-admission cap and no
position-count cap, matching the live daemon (see .claude/rules/cli.md).

Exit timing is NOT re-simulated — each candidate trade's exit date/price is
taken as-is from the strategy's own (single-ticker, unconstrained) backtest,
since exit logic is technical/regime-driven and doesn't depend on shared
capital. Only entry admission (yes/no, and how much capital) is arbitrated.

Each strategy passed via --strategies gets its OWN independent pot (not one
pot shared across strategies) — --initial-cash/--pot-sizes is the size of
that per-strategy pot, re-initialized fresh for every strategy.

Usage:
    uv run python -m Strategy_Auto_Trader.markov_cli.live_sim \\
        --tickers SHEL.L BP.L HSBA.L ULVR.L GSK.L RIO.L DGE.L LSEG.L BATS.L VOD.L \\
        --strategies default conservative trend \\
        --start-date 2026-01-12 \\
        --initial-cash 10000 --trade-cost 1

    # Full S&P500+FTSE100 universe, capital-sweep:
    uv run python -m Strategy_Auto_Trader.markov_cli.live_sim \\
        --universe --strategies conservative default trend optimised \\
        --pot-sizes 25000 50000 100000 200000 --workers 4

    # Synthetic-data stress test (e.g. a historical window real hourly data
    # can't reach) — see synthetic_backtest_data/generate.py:
    uv run python -m Strategy_Auto_Trader.markov_cli.live_sim \\
        --universe --strategies optimised_new --start-date 2008-01-01 \\
        --synthetic-data-dir data_synthetic/hourly --synthetic-end-date 2009-07-31 \\
        --initial-cash 100000 --top-k 70 --workers 4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import full_scan
from ..broker.symbols import sizing_price
from ..output.journal import LIVE_JOURNAL, TradeRecord, append_trades
from ..plugins.costs import COST_MODEL_CHOICES, make_cost_model
from ..plugins.interest import IbkrTieredInterest
from ..quant_hmm.ticker_ranking import (
    Candidate,
    _filter_candidates_by_daily_trend_quality,
    fetch_and_extract,
    filter_candidates_by_top_tickers,
    generate_candidates,
    trend_quality_asof,
)
from ..core.cli_logging import setup_cli_logger
from ..strategy.base.registry import STRATEGY_REGISTRY, wants_low_trend_quality
from ..quant_hmm.quant_engine import fetch_daily
from ..synthetic_backtest_data.generate import SYNTHETIC_HMM_CACHE_DIR, load_synthetic_hourly
from ..synthetic_backtest_data.stooq_daily import load_stooq_daily

_SYNTHETIC_JOURNAL_DIR = Path(__file__).resolve().parent.parent.parent / "data_synthetic" / "journals"

#: Trailing-day window for the correlation admission gate (see
#: max_correlation_to_admitted_today). Fixed, not a strategy-owned knob —
#: nothing in the diagnosis motivates varying it, see BACKTEST_LOG.md
#: correlation-cap entry.
_CORRELATION_LOOKBACK_DAYS = 60

logger = logging.getLogger(__name__)


def _position_value(pos: dict, day: pd.Timestamp, price_by_ticker: dict[str, pd.Series] | None) -> float:
    """Mark one open position to market as of `day` (last known close at/before
    day), falling back to its cost basis (alloc) if no price series or entry
    price is available. price_by_ticker is optional — arbitrate() can be called
    without it (e.g. from tests, or simulate_strategy's backward-compatible
    path), in which case every position is valued at cost basis."""
    entry_price = pos.get("entry_price") or 0.0
    series = price_by_ticker.get(pos["ticker"]) if price_by_ticker else None
    if series is None or series.empty or not entry_price:
        return pos["alloc"]

    lookup_day = day
    idx_tz = getattr(series.index, "tz", None)
    if idx_tz is not None and lookup_day.tzinfo is None:
        lookup_day = lookup_day.tz_localize(idx_tz)
    elif idx_tz is None and lookup_day.tzinfo is not None:
        lookup_day = lookup_day.tz_localize(None)

    try:
        price_now = series.asof(lookup_day)
    except Exception:
        return pos["alloc"]
    if price_now is None or (isinstance(price_now, float) and np.isnan(price_now)):
        return pos["alloc"]
    return pos["alloc"] * (float(price_now) / entry_price)


def _max_drawdown(values: list[float]) -> float:
    """Max peak-to-trough drawdown (negative fraction) of a portfolio-value sequence."""
    if not values:
        return 0.0
    peak = values[0]
    max_dd = 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            max_dd = min(max_dd, (v - peak) / peak)
    return max_dd


def resolve_same_day_deployment_cap(strategy_name: str) -> float | None:
    """Strategy-owned same_day_deployment_cap_pct, read off the strategy's
    registered Entry class (default None for every strategy that doesn't
    declare it, or isn't registered at all) — see .claude/rules/strategy.md.
    Not a CLI flag. An unregistered strategy_name resolves to None here
    rather than raising — resolving a bad name is generate_candidates()'s
    job, not this lookup's."""
    entry_cls = STRATEGY_REGISTRY.get(strategy_name, {}).get("entry")
    return getattr(entry_cls, "same_day_deployment_cap_pct", None)


def resolve_vix_entry_gate_threshold(strategy_name: str) -> float | None:
    """Strategy-owned vix_entry_gate_threshold, read off the strategy's
    registered Entry class (default None = gate disabled). When set, arbitrate()
    blocks all new entries on days where ^VIX daily close >= this level.
    Not a CLI flag — see .claude/rules/strategy.md Strategy-Owned Admission
    Attributes section."""
    entry_cls = STRATEGY_REGISTRY.get(strategy_name, {}).get("entry")
    return getattr(entry_cls, "vix_entry_gate_threshold", None)


def resolve_correlation_cap(strategy_name: str) -> float | None:
    """Strategy-owned max_correlation_to_admitted_today, read off the
    strategy's registered Entry class (default None = gate disabled). When
    set, arbitrate() rejects a candidate whose trailing-return correlation to
    any ticker already admitted that day is >= this threshold. Not a CLI
    flag — see .claude/rules/strategy.md Strategy-Owned Admission Attributes
    section."""
    entry_cls = STRATEGY_REGISTRY.get(strategy_name, {}).get("entry")
    return getattr(entry_cls, "max_correlation_to_admitted_today", None)


def _build_daily_returns_cache(tickers: list[str]) -> dict[str, pd.Series]:
    """Daily pct-change return series per ticker, from the local Stooq daily
    dump (see synthetic_backtest_data/stooq_daily.py — same data already used
    by the synthetic-backtest pipeline, no new fetch). Tickers with no Stooq
    coverage are simply absent from the returned dict; the correlation gate
    treats a missing series as gate-not-triggered for that ticker, same
    fallback contract as the VIX gate's NaN handling."""
    cache: dict[str, pd.Series] = {}
    for ticker in tickers:
        df = load_stooq_daily(ticker)
        if df is None or df.empty:
            continue
        closes = df["Close"]
        if getattr(closes.index, "tz", None) is not None:
            closes = closes.tz_localize(None)
        cache[ticker] = closes.pct_change().dropna()
    return cache


def _pairwise_correlation(
    returns_a: pd.Series, returns_b: pd.Series, as_of: pd.Timestamp,
) -> float | None:
    """Pearson correlation of two tickers' daily returns over the trailing
    _CORRELATION_LOOKBACK_DAYS trading-day window strictly before as_of (no
    lookahead — as_of is the candidate's entry day). Returns None if fewer
    than 20 overlapping observations are available (insufficient history to
    trust a correlation estimate), in which case the gate doesn't trigger."""
    a = returns_a[returns_a.index < as_of].tail(_CORRELATION_LOOKBACK_DAYS)
    b = returns_b[returns_b.index < as_of].tail(_CORRELATION_LOOKBACK_DAYS)
    joined = pd.concat([a, b], axis=1, join="inner")
    if len(joined) < 20:
        return None
    corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
    return float(corr) if pd.notna(corr) else None


def arbitrate(
    candidates: list[Candidate],
    initial_cash: float,
    trade_cost: float,
    cost_model_name: str = "flat",
    currency: str = "GBP",
    price_by_ticker: dict[str, pd.Series] | None = None,
    same_day_deployment_cap_pct: float | None = None,
    vix_series: pd.Series | None = None,
    vix_entry_gate_threshold: float | None = None,
    daily_returns_by_ticker: dict[str, pd.Series] | None = None,
    max_correlation_to_admitted_today: float | None = None,
) -> dict:
    """Walk candidates day-by-day (event days only: opens/closes), arbitrating
    entries against one shared, mutating cash pot. No position-count cap —
    cash and a positive Kelly fraction are the only mandatory gates, matching
    the live daemon (broker/portfolio.py has no max_positions or daily-trade-
    count check either; see .claude/rules/cli.md capital-arbitration section).

    same_day_deployment_cap_pct is an OPTIONAL third gate: caps total new
    capital admitted across a single calendar day at cap_pct * initial_cash.
    It is strategy-owned, not a CLI/engine knob — the caller resolves it from
    the strategy's registered Entry class (see main()) and passes it in here;
    arbitrate() itself stays generic and doesn't know about STRATEGY_REGISTRY.
    Default None means every strategy that doesn't declare the attribute sees
    unchanged behavior. This is NOT the daily_buy_limit/daily_sell_limit
    trade-count cap removed 2026-08-11 (that was centrally hardcoded and
    never enforced) — this targets $-deployment concentration on days when
    many candidates enter/exit together (correlated regime-driven risk), and
    narrows the entry_score-sorted admission from the bottom rather than
    reordering or delaying anyone.

    vix_series / vix_entry_gate_threshold are an OPTIONAL fourth gate:
    blocks ALL new entries on days where the ^VIX daily close (looked up via
    Series.asof(day)) is >= vix_entry_gate_threshold. Strategy-owned —
    resolved from the Entry class attribute and passed in by main(). Uses real
    historical VIX even in synthetic-data mode (the macro signal is real; only
    per-ticker prices are synthetic). Days with no VIX observation (NaN from
    asof) are treated as gate-not-triggered (entries allowed). Cash release for
    closing positions and equity-curve recording still happen on a VIX-blocked
    day — only new entries are suppressed.

    daily_returns_by_ticker / max_correlation_to_admitted_today are an OPTIONAL
    fifth gate: rejects a candidate whose trailing _CORRELATION_LOOKBACK_DAYS
    daily-return correlation to ANY ticker already admitted that same day is
    >= max_correlation_to_admitted_today. Strategy-owned — resolved from the
    Entry class attribute and passed in by main(). daily_returns_by_ticker
    comes from the local Stooq daily dump (_build_daily_returns_cache), built
    once per run and reused across strategies/pot-sizes. A candidate whose
    ticker (or an already-admitted ticker) has no/insufficient return history
    is never rejected on this gate — missing data means gate-not-triggered,
    same fallback contract as the VIX gate's NaN handling. Like every other
    gate here, this narrows admission from the bottom of the entry_score sort
    and never reorders it.

    A candidate with kelly_fraction <= 0 is rejected outright, not sized via
    a flat fallback — this matches live's PortfolioManager.compute_quantity(),
    which returns 0 for kelly_fraction <= 0 and never places the order.

    Position sizing floors to whole shares (IBKR does not reliably support
    fractional-share orders via the API), mirroring compute_quantity()
    exactly: qty = max(1, floor(cash * kelly_fraction / price)) once at least
    1 share is affordable, else the candidate is rejected. A high per-share
    price can reject a candidate outright even with cash available and a
    positive Kelly fraction — this is a real admission gate, not backtest
    noise.

    Returns a dict: executed (list[TradeRecord]), equity_curve (list[dict],
    one row per event day — cash, deployed capital mark-to-market if
    price_by_ticker is given else cost-basis, portfolio_value, cumulative
    realized P&L, cumulative interest), total_interest, final_cash,
    n_candidates, n_admitted, n_rejected_cash (candidates that couldn't be
    sized due to insufficient cash), n_rejected_kelly (candidates with
    kelly_fraction <= 0), n_rejected_concentration (candidates that would
    have exceeded same_day_deployment_cap_pct), n_rejected_vix (candidates
    blocked because VIX >= vix_entry_gate_threshold on their entry day),
    n_rejected_correlation (candidates too correlated to something already
    admitted that day).

    Note: equity_curve is sampled at trade-event days only, not every calendar
    day — for strategies with infrequent trades this is a sparse series. Do
    not treat it as daily-resolution; it's sufficient for max drawdown
    (well-defined regardless of sampling) but not for an annualized Sharpe,
    which is why this function doesn't compute one.
    """
    candidates = sorted(candidates, key=lambda c: c.date_opened)
    n_candidates = len(candidates)

    if not candidates:
        return {
            "executed": [], "equity_curve": [], "total_interest": 0.0,
            "final_cash": initial_cash, "n_candidates": 0, "n_admitted": 0,
            "n_rejected_cash": 0, "n_rejected_kelly": 0,
            "n_rejected_concentration": 0, "n_rejected_vix": 0,
            "n_rejected_correlation": 0,
        }

    by_day: dict[pd.Timestamp, list[Candidate]] = {}
    for c in candidates:
        day = c.date_opened.tz_localize(None).normalize()
        by_day.setdefault(day, []).append(c)

    all_days = sorted(set(
        list(by_day.keys()) +
        [c.date_closed.tz_localize(None).normalize() for c in candidates]
    ))

    cash = initial_cash
    open_positions: list[dict] = []  # {ticker, entry_price, date_closed, exit_proceeds, alloc}
    executed: list[TradeRecord] = []
    equity_curve: list[dict] = []
    interest_model = IbkrTieredInterest(currency)
    total_interest = 0.0
    n_admitted = 0
    n_rejected_cash = 0
    n_rejected_kelly = 0
    n_rejected_concentration = 0
    n_rejected_vix = 0
    n_rejected_correlation = 0
    prev_day = None

    for day in all_days:
        if prev_day is not None:
            days_elapsed = (day - prev_day).days
            if days_elapsed > 0:
                interest = interest_model.daily_accrual(cash) * days_elapsed
                cash += interest
                total_interest += interest
        prev_day = day

        # 1. release cash for positions closing on/before this day
        still_open = []
        for pos in open_positions:
            if pos["date_closed"] <= day:
                cash += pos["exit_proceeds"]
            else:
                still_open.append(pos)
        open_positions = still_open

        # 2. admit new entries for this day, highest score first
        day_candidates = sorted(by_day.get(day, []), key=lambda c: -c.entry_score)
        taken = 0
        deployed_today = 0.0
        admitted_today: list[str] = []
        daily_cap = (same_day_deployment_cap_pct * initial_cash
                     if same_day_deployment_cap_pct else None)

        # VIX gate: block all new entries on high-volatility days
        if vix_entry_gate_threshold is not None and vix_series is not None:
            vix_val = vix_series.asof(day)
            if not pd.isna(vix_val) and float(vix_val) >= vix_entry_gate_threshold:
                n_rejected_vix += len(day_candidates)
                day_candidates = []

        for cand in day_candidates:
            if not cand.kelly_fraction or cand.kelly_fraction <= 0:
                n_rejected_kelly += 1
                continue

            if (max_correlation_to_admitted_today is not None
                    and daily_returns_by_ticker and admitted_today):
                cand_returns = daily_returns_by_ticker.get(cand.ticker)
                if cand_returns is not None:
                    max_corr = 0.0
                    for other_ticker in admitted_today:
                        other_returns = daily_returns_by_ticker.get(other_ticker)
                        if other_returns is None:
                            continue
                        c = _pairwise_correlation(cand_returns, other_returns, day)
                        if c is not None:
                            max_corr = max(max_corr, c)
                    if max_corr >= max_correlation_to_admitted_today:
                        n_rejected_correlation += 1
                        continue

            price = sizing_price(cand.ticker, cand.record.entry_price)
            if price <= 0 or cash < price or cash <= trade_cost:
                n_rejected_cash += 1
                continue

            qty = max(1, int(cash * cand.kelly_fraction / price))
            alloc = qty * price

            if cost_model_name == "flat":
                entry_fee = exit_fee = trade_cost
            else:
                model = make_cost_model(cost_model_name, cand.record.ticker, trade_cost)
                entry_fee = model.cost(alloc, True)
                exit_fee = model.cost(alloc * (1 + cand.return_pct), False)

            # Whole shares can't be trimmed to fit like continuous alloc could —
            # drop a share at a time until the fee-inclusive cost fits cash.
            while qty > 0 and alloc + entry_fee > cash:
                qty -= 1
                alloc = qty * price
                if cost_model_name != "flat":
                    entry_fee = model.cost(alloc, True)
                    exit_fee = model.cost(alloc * (1 + cand.return_pct), False)
            if qty <= 0:
                n_rejected_cash += 1
                continue

            if daily_cap is not None and deployed_today + alloc > daily_cap:
                n_rejected_concentration += 1
                continue

            cash -= (alloc + entry_fee)
            exit_proceeds = alloc * (1 + cand.return_pct) - exit_fee
            open_positions.append({
                "ticker": cand.ticker,
                "entry_price": cand.record.entry_price,
                "date_closed": cand.date_closed.tz_localize(None).normalize(),
                "exit_proceeds": exit_proceeds,
                "alloc": alloc,
            })

            rec = dataclasses.replace(cand.record)
            rec.pnl_usd = exit_proceeds - alloc - entry_fee
            rec.position_size_gbp = alloc
            executed.append(rec)
            taken += 1
            n_admitted += 1
            deployed_today += alloc
            admitted_today.append(cand.ticker)

        skipped = len(day_candidates) - taken
        if taken or skipped:
            logger.info(f"    {day.date()}: took {taken}, skipped {skipped}  (cash={cash:,.2f})")

        deployed = sum(_position_value(pos, day, price_by_ticker) for pos in open_positions)
        equity_curve.append({
            "date": day,
            "cash": cash,
            "deployed": deployed,
            "n_open": len(open_positions),
            "portfolio_value": cash + deployed,
            "realized_pnl_cum": sum(r.pnl_usd for r in executed),
            "interest_cum": total_interest,
        })

    final_cash = cash + sum(p["exit_proceeds"] for p in open_positions)

    return {
        "executed": executed,
        "equity_curve": equity_curve,
        "total_interest": total_interest,
        "final_cash": final_cash,
        "n_candidates": n_candidates,
        "n_admitted": n_admitted,
        "n_rejected_cash": n_rejected_cash,
        "n_rejected_kelly": n_rejected_kelly,
        "n_rejected_concentration": n_rejected_concentration,
        "n_rejected_vix": n_rejected_vix,
        "n_rejected_correlation": n_rejected_correlation,
    }


def simulate_strategy(
    tickers: list[str],
    strategy_name: str,
    start_date: str,
    initial_cash: float,
    trade_cost: float,
    vol_filter_tag: str = "suitable",
    vol_filter_ok: bool = True,
    cost_model_name: str = "flat",
    currency: str = "GBP",
    vix_series: pd.Series | None = None,
    vix_entry_gate_threshold: float | None = None,
    daily_returns_by_ticker: dict[str, pd.Series] | None = None,
) -> list[TradeRecord]:
    """Run one strategy across all tickers with a shared capital pool. Returns executed TradeRecords.

    Kept as a simple, single-pot, sequential entry point (backward compatible)
    — delegates the actual day-by-day arbitration to arbitrate(). For a
    parallelized, multi-pot-size, mark-to-market run see generate_candidates()
    + arbitrate() directly (used by main() for --universe/--pot-sizes runs).
    """
    logger.info(f"\n{'='*64}\n Strategy: {strategy_name}  (vol_filter={vol_filter_tag})\n{'='*64}")

    all_candidates: list[Candidate] = []
    for ticker in tickers:
        logger.info(f"  fetching + backtesting {ticker}...")
        # historical_only=True: live_sim.py is a backtest/simulation tool,
        # never the live daemon's signal path — no need for today's newest
        # bar, and skipping the live gap-fill avoids competing with the
        # daemon's own IBKR polling for pacing-limit headroom.
        cands = fetch_and_extract(ticker, strategy_name, vol_filter_tag, vol_filter_ok,
                                   historical_only=True)
        cutoff = pd.Timestamp(start_date)
        # Normalize to naive timestamps for comparison (hourly data is tz-aware)
        cands = [c for c in cands if c.date_opened.tz_localize(None) >= cutoff]
        logger.info(f"    {len(cands)} candidate trade(s) on/after {start_date}")
        all_candidates.extend(cands)

    result = arbitrate(
        all_candidates,
        initial_cash=initial_cash,
        trade_cost=trade_cost,
        cost_model_name=cost_model_name,
        currency=currency,
        price_by_ticker=None,
        same_day_deployment_cap_pct=resolve_same_day_deployment_cap(strategy_name),
        vix_series=vix_series,
        vix_entry_gate_threshold=vix_entry_gate_threshold,
        daily_returns_by_ticker=daily_returns_by_ticker,
        max_correlation_to_admitted_today=resolve_correlation_cap(strategy_name),
    )

    executed = result["executed"]
    total_pnl = sum(r.pnl_usd for r in executed)
    logger.info(f"\n  {strategy_name}: {len(executed)} trade(s) executed, "
          f"final pot £{result['final_cash']:,.2f} (P&L £{total_pnl:+,.2f} on £{initial_cash:,.0f} start, "
          f"£{result['total_interest']:,.2f} interest on idle cash)")

    return executed



def _write_position_summary(rows: list[dict], path: Path) -> None:
    """Write the per-(strategy, pot_size, date) equity-curve rows to a CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    setup_cli_logger("live_sim")

    parser = argparse.ArgumentParser(prog="live-sim")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Explicit ticker list. Mutually exclusive with --universe.")
    parser.add_argument("--universe", action="store_true",
                        help="Use the full S&P 500 + FTSE 100 universe (config/universe_sp_ftse.json) "
                             "instead of --tickers.")
    parser.add_argument("--strategies", nargs="+", default=["default", "conservative", "trend"])
    parser.add_argument("--start-date", default="2000-01-01")
    parser.add_argument("--initial-cash", type=float, default=10_000.0,
                        help="Pot size per strategy. Ignored if --pot-sizes is given.")
    parser.add_argument("--pot-sizes", type=float, nargs="+", default=None,
                        help="Sweep multiple pot sizes per strategy against the same candidate "
                             "trades (cheap — only the arbitration step re-runs per size, not the "
                             "backtest). Default: just --initial-cash, one pot size.")
    parser.add_argument("--trade-cost", type=float, default=1.0)
    parser.add_argument("--cost-model", default="ibkr_tiered_spread", choices=COST_MODEL_CHOICES,
                        help="Transaction cost model for the capital sim: 'flat' = "
                             "--trade-cost/side (~10x real fees at typical stake sizes); "
                             "'ibkr_tiered' = IBKR UK tiered commission "
                             "+ SDRT on .L buys; 'ibkr_tiered_spread' adds a half-spread "
                             "estimate per side. Default: ibkr_tiered_spread.")
    parser.add_argument("--workers", type=int, default=2,
                        help="Worker processes for parallel per-ticker candidate generation "
                             "(default: 2). 1 = sequential.")
    parser.add_argument("--min-trend-quality", type=float, default=0.0,
                        help="Vol-filter cutoff: exclude tickers below this trend_quality score")
    parser.add_argument("--no-vol-filter", action="store_true",
                        help="Skip the volatility/choppiness pre-screen for every strategy")
    parser.add_argument("--vol-filter-exempt", nargs="+", default=[],
                        help="Strategy names that trade the full ticker list, bypassing the vol filter "
                             "(other strategies in --strategies still get filtered)")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Limit to top-K tickers by hybrid (vol_quality + win_rate) score "
                             "(0 = no limit). Default: 0 (unlimited).")
    parser.add_argument("--vol-weight", type=float, default=0.7,
                        help="Weight for trend_quality in hybrid rank (0-1). Default: 0.7.")
    parser.add_argument("--win-rate-weight", type=float, default=0.3,
                        help="Weight for recent win-rate in hybrid rank (0-1). Default: 0.3.")
    parser.add_argument("--dump-ticker-scores", default=None,
                        help="Write the --top-k hybrid ticker_scores dict to this path as JSON "
                             "(ground truth for diffing against rank_universe_cli.py's live output). "
                             "No effect without --top-k > 0.")
    parser.add_argument("--lookback-days", type=int, default=60,
                        help="Window for computing recent win-rate (days). Default: 60.")
    parser.add_argument("--journal", default=None,
                        help="Journal CSV to append trades to (default: data/journals/live.csv)")
    parser.add_argument("--position-summary", default=None,
                        help="Path to write the per-(strategy,pot_size,date) equity-curve CSV "
                             "(default: data/journals/live_sim_position_summary_<timestamp>.csv). "
                             "Additive output — does not change --journal's default path/format.")
    parser.add_argument("--seasonal-volume", dest="seasonal_volume", action="store_true",
                        default=True,
                        help="Normalise volume ratio by same-hour-of-day trailing mean instead of "
                             "flat rolling-20 mean. On by default — backtested to improve results "
                             "and matches the live daemon's config "
                             "(BACKTEST_LIVE_PARITY_PLAN.md Step 3b, decided 2026-08-28).")
    parser.add_argument("--source", choices=["yfinance", "ibkr"], default="ibkr",
                        help="Hourly data source for every ticker in this run: local incremental "
                             "IBKR-backed cache (default) or yfinance. IBKR falls back to "
                             "yfinance if no cache exists for a ticker.")
    parser.add_argument("--synthetic-data-dir", default=None,
                        help="Use synthetic hourly CSVs from this directory (see "
                             "synthetic_backtest_data/generate.py) instead of fetching real data via "
                             "--source. Requires --synthetic-end-date. Isolated from real data: uses "
                             "SYNTHETIC_HMM_CACHE_DIR instead of the real HMM cache, and defaults "
                             "--journal/--position-summary under data_synthetic/journals/ instead of "
                             "the real journal, unless explicitly overridden.")
    parser.add_argument("--synthetic-end-date", default=None, metavar="YYYY-MM-DD",
                        help="Upper bound for the synthetic data window (--start-date is the lower "
                             "bound). Required together with --synthetic-data-dir — arbitrate() has "
                             "no other end-of-window concept, and the synthetic CSVs span decades.")
    args = parser.parse_args(argv)

    if bool(args.tickers) == bool(args.universe):
        parser.error("exactly one of --tickers or --universe is required")
    if bool(args.synthetic_data_dir) != bool(args.synthetic_end_date):
        parser.error("--synthetic-data-dir and --synthetic-end-date must be given together")
    if args.universe:
        args.tickers = full_scan.load_sp_ftse_universe()

    df_by_ticker: dict[str, pd.DataFrame] | None = None
    if args.synthetic_data_dir:
        synthetic_dir = Path(args.synthetic_data_dir)
        df_by_ticker = {}
        dropped: list[tuple[str, str]] = []
        for ticker in args.tickers:
            df = load_synthetic_hourly(ticker, hourly_dir=synthetic_dir)
            if df is None:
                dropped.append((ticker, "no synthetic file"))
                continue
            window = df.loc[args.start_date:args.synthetic_end_date]
            if window.empty:
                dropped.append((ticker, "no bars in window"))
                continue
            df_by_ticker[ticker] = window
        args.tickers = list(df_by_ticker.keys())
        logger.info(
            f"  synthetic mode: {len(args.tickers)}/{len(args.tickers) + len(dropped)} tickers have "
            f"data in [{args.start_date}, {args.synthetic_end_date}], hmm_cache_dir={SYNTHETIC_HMM_CACHE_DIR}"
        )
        if dropped:
            logger.info(f"  dropped ({len(dropped)}): {', '.join(t for t, _ in dropped)}")
        if args.journal is None:
            args.journal = str(_SYNTHETIC_JOURNAL_DIR / "live_sim_synthetic.csv")
        if args.position_summary is None:
            ts = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
            args.position_summary = str(_SYNTHETIC_JOURNAL_DIR / f"live_sim_synthetic_position_summary_{ts}.csv")

    pot_sizes = args.pot_sizes if args.pot_sizes else [args.initial_cash]

    logger.info(f"Live simulation: {len(args.tickers)} tickers x {len(args.strategies)} strategies "
          f"x {len(pot_sizes)} pot size(s)")
    logger.info(f"  start={args.start_date}  pot_sizes={pot_sizes}  "
          f"trade_cost=£{args.trade_cost:.2f}  workers={args.workers}  source={args.source}")
    if args.vol_filter_exempt:
        logger.info(f"  vol-filter-exempt strategies: {', '.join(args.vol_filter_exempt)}")

    all_executed: list[TradeRecord] = []
    summary_rows: list[dict] = []

    _vix_series_cache: pd.Series | None = None  # fetched once, reused across strategies
    _daily_returns_cache: dict[str, pd.Series] | None = None  # built once, reused across strategies

    for strategy_name in args.strategies:
        exempt = args.no_vol_filter or strategy_name in args.vol_filter_exempt
        vol_filter_tag = "disabled" if args.no_vol_filter else ("exempt" if exempt else "daily-rescreened")

        logger.info(f"\n{'='*64}\n Strategy: {strategy_name}  (vol_filter={vol_filter_tag})\n{'='*64}")
        # Full ticker list always — the vol veto no longer decides which
        # tickers get backtested at all (that baked in today's snapshot for
        # the whole history); it's applied per-candidate below instead, using
        # each candidate's own entry-day trend_quality (see
        # _filter_candidates_by_daily_trend_quality's docstring for why).
        candidates, price_by_ticker, trend_quality_by_ticker = generate_candidates(
            tickers=list(args.tickers),
            strategy_name=strategy_name,
            vol_filter_tag=vol_filter_tag,
            # Always True: the strategy's own built-in single-snapshot veto
            # must stay off so it doesn't double-gate on top of the daily
            # rescreen below.
            vol_filter_ok=True,
            workers=args.workers,
            use_seasonal_volume=args.seasonal_volume,
            source=args.source,
            df_by_ticker=df_by_ticker,
            use_persistent_cache=True,
            hmm_cache_dir=SYNTHETIC_HMM_CACHE_DIR if args.synthetic_data_dir else None,
            # Backtest tool, not the live daemon's signal path — skip the
            # live IBKR gap-fill and serve straight from cache (see
            # generate_candidates' docstring).
            historical_only=True,
        )

        cutoff = pd.Timestamp(args.start_date)
        candidates = [c for c in candidates if c.date_opened.tz_localize(None) >= cutoff]

        if not exempt:
            wants_low = wants_low_trend_quality(strategy_name)
            n_before = len(candidates)
            candidates = _filter_candidates_by_daily_trend_quality(
                candidates, trend_quality_by_ticker, args.min_trend_quality, wants_low,
            )
            logger.info(f"  daily vol gate: {len(candidates)}/{n_before} candidates survive "
                  f"(min_trend_quality={args.min_trend_quality}, "
                  f"{'inverted (wants choppy)' if wants_low else 'standard'})")

        if args.top_k > 0:
            n_before = len(candidates)
            candidates, ticker_scores = filter_candidates_by_top_tickers(
                candidates, trend_quality_by_ticker, args.top_k,
                vol_weight=args.vol_weight, win_rate_weight=args.win_rate_weight,
                lookback_days=args.lookback_days,
            )
            top_tickers_list = sorted(ticker_scores.keys(),
                                      key=lambda t: -ticker_scores[t])[:args.top_k]
            logger.info(f"  top-k filter: {len(candidates)}/{n_before} candidates from {args.top_k} best tickers")
            logger.info(f"    selected: {', '.join(top_tickers_list)}")
            for t in top_tickers_list[:5]:
                logger.info(f"      {t}: score={ticker_scores.get(t, 0):.3f}")
            if len(top_tickers_list) > 5:
                logger.info(f"      ... and {len(top_tickers_list) - 5} more")

            if args.dump_ticker_scores:
                dump_path = Path(args.dump_ticker_scores)
                dump_path.parent.mkdir(parents=True, exist_ok=True)
                dump_path.write_text(json.dumps(ticker_scores, indent=2), encoding="utf-8")
                logger.info(f"  ticker_scores dumped to {dump_path}")

        if candidates:
            earliest = min(c.date_opened for c in candidates)
            logger.info(f"  {len(candidates)} candidate trade(s) on/after {args.start_date} "
                  f"(earliest actual: {earliest.date()})")

        same_day_cap = resolve_same_day_deployment_cap(strategy_name)
        if same_day_cap:
            logger.info(f"  same_day_deployment_cap_pct={same_day_cap} (strategy-owned, "
                        f"{strategy_name}.same_day_deployment_cap_pct)")

        vix_threshold = resolve_vix_entry_gate_threshold(strategy_name)
        vix_series: pd.Series | None = None
        if vix_threshold is not None:
            if _vix_series_cache is None:
                vix_df = fetch_daily("^VIX")
                if vix_df is not None and "Close" in vix_df.columns:
                    s = vix_df["Close"].dropna()
                    s.index = pd.to_datetime(s.index).tz_localize(None)
                    _vix_series_cache = s
            vix_series = _vix_series_cache
            if vix_series is None or vix_series.empty:
                logger.warning("  VIX gate: ^VIX fetch failed — gate disabled for this run")
                vix_threshold = None
            else:
                logger.info(f"  vix_entry_gate_threshold={vix_threshold} (strategy-owned), "
                            f"latest VIX={vix_series.iloc[-1]:.1f}")

        corr_cap = resolve_correlation_cap(strategy_name)
        if corr_cap is not None:
            if _daily_returns_cache is None:
                _daily_returns_cache = _build_daily_returns_cache(list(args.tickers))
                logger.info(f"  correlation gate: daily-return cache built for "
                            f"{len(_daily_returns_cache)}/{len(args.tickers)} tickers (Stooq coverage)")
            logger.info(f"  max_correlation_to_admitted_today={corr_cap} (strategy-owned, "
                        f"{strategy_name}.max_correlation_to_admitted_today)")

        for pot_size in pot_sizes:
            result = arbitrate(
                candidates,
                initial_cash=pot_size,
                trade_cost=args.trade_cost,
                cost_model_name=args.cost_model,
                currency="GBP",
                price_by_ticker=price_by_ticker,
                same_day_deployment_cap_pct=same_day_cap,
                vix_series=vix_series,
                vix_entry_gate_threshold=vix_threshold,
                daily_returns_by_ticker=_daily_returns_cache,
                max_correlation_to_admitted_today=corr_cap,
            )
            all_executed.extend(result["executed"])

            total_pnl = sum(r.pnl_usd for r in result["executed"])
            peak_deployed = max((row["deployed"] for row in result["equity_curve"]), default=0.0)
            max_dd = _max_drawdown([row["portfolio_value"] for row in result["equity_curve"]])
            vix_rej_str = (f", {result['n_rejected_vix']} rejected for VIX gate"
                           if result['n_rejected_vix'] else "")
            corr_rej_str = (f", {result['n_rejected_correlation']} rejected for correlation cap"
                            if result['n_rejected_correlation'] else "")
            logger.info(f"  pot £{pot_size:,.0f}: {len(result['executed'])}/{result['n_candidates']} admitted "
                  f"({result['n_rejected_cash']} rejected for cash, "
                  f"{result['n_rejected_kelly']} rejected for kelly<=0, "
                  f"{result['n_rejected_concentration']} rejected for concentration cap"
                  f"{vix_rej_str}{corr_rej_str}), "
                  f"final £{result['final_cash']:,.2f} (P&L £{total_pnl:+,.2f}, "
                  f"peak deployed £{peak_deployed:,.2f}, max drawdown {max_dd*100:.1f}%)")

            for row in result["equity_curve"]:
                summary_rows.append({"strategy": strategy_name, "pot_size": pot_size, **row})
            summary_rows.append({
                "strategy": strategy_name, "pot_size": pot_size, "date": "SUMMARY",
                "cash": result["final_cash"], "deployed": peak_deployed, "n_open": 0,
                "portfolio_value": result["final_cash"], "realized_pnl_cum": total_pnl,
                "interest_cum": result["total_interest"],
                "n_candidates": result["n_candidates"], "n_admitted": result["n_admitted"],
                "n_rejected_cash": result["n_rejected_cash"],
                "n_rejected_kelly": result["n_rejected_kelly"],
                "n_rejected_concentration": result["n_rejected_concentration"],
                "n_rejected_vix": result["n_rejected_vix"],
                "n_rejected_correlation": result["n_rejected_correlation"],
                "max_drawdown": max_dd,
            })

    journal_path = Path(args.journal) if args.journal else LIVE_JOURNAL
    n_logged = append_trades(journal_path, all_executed)
    logger.info(f"\n{'='*64}\n {n_logged} trade(s) logged to {journal_path}\n{'='*64}")

    if summary_rows:
        if args.position_summary:
            summary_path = Path(args.position_summary)
        else:
            ts = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
            summary_path = LIVE_JOURNAL.parent / f"live_sim_position_summary_{ts}.csv"
        _write_position_summary(summary_rows, summary_path)
        logger.info(f" position summary written to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
