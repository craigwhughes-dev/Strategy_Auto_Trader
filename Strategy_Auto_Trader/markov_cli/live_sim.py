"""Live-style multi-ticker portfolio simulation.

Simulates running a strategy "live" across a basket of tickers sharing a
single capital pool, starting from a given date. Unlike batch.py (which runs
each ticker's backtest independently with its own capital), this walks a
strategy's BUY/SELL signal stream per ticker but arbitrates entries across
tickers against one shared cash pot: a daily cap on new positions, priority
by signal strength when multiple tickers want to enter the same day, and
position sizing from the strategy's own Kelly fraction (or a fixed fallback).

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
        --initial-cash 10000 --trade-cost 1 --kelly-fallback 100 \\
        --max-trades-per-day 1

    # Full S&P500+FTSE100 universe, capital-sweep, unlimited daily admissions:
    uv run python -m Strategy_Auto_Trader.markov_cli.live_sim \\
        --universe --strategies conservative default trend optimised \\
        --pot-sizes 25000 50000 100000 200000 \\
        --max-trades-per-day 0 --workers 4
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import full_scan
from ..output.journal import LIVE_JOURNAL, TradeRecord, append_trades, extract_trades_from_detail
from ..plugins.context_adjuster import SentimentAdjuster
from ..plugins.costs import COST_MODEL_CHOICES, make_cost_model
from ..plugins.interest import IbkrTieredInterest
from ..plugins.persistent_hmm import PersistentHMMRegimeModel
from ..quant_hmm.consolidated_engine import consolidated_backtest
from ..quant_hmm.data_cache import fetch_hourly_cached, fetch_hourly_stooq
from ..quant_hmm.vol_screen import rolling_trend_quality
from ..strategy.base.registry import resolve_strategy, wants_low_trend_quality

_HMM_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "state" / "hmm_cache"


@dataclass
class _Candidate:
    """A single strategy's round-trip trade, before shared-capital arbitration."""
    ticker: str
    date_opened: pd.Timestamp
    date_closed: pd.Timestamp
    entry_score: float
    kelly_fraction: float
    return_pct: float
    record: TradeRecord


def _recent_win_rate(candidates: list[_Candidate], ticker: str, lookback_days: int = 60) -> float:
    """Win-rate for a ticker over lookback_days window (measured from latest candidate).

    Returns fraction of trades that were profitable (return_pct > 0), or 0.5 if
    no trades in window."""
    ticker_cands = [c for c in candidates if c.ticker == ticker]
    if not ticker_cands:
        return 0.5

    if lookback_days:
        latest = max(c.date_opened for c in ticker_cands)
        cutoff = latest - pd.Timedelta(days=lookback_days)
        ticker_cands = [c for c in ticker_cands if c.date_opened >= cutoff]

    if not ticker_cands:
        return 0.5

    wins = sum(1 for c in ticker_cands if c.return_pct > 0)
    return wins / len(ticker_cands)


def _trend_quality_asof(
    ticker: str, when: pd.Timestamp, trend_quality_by_ticker: dict[str, pd.Series]
) -> float | None:
    """Look up trend_quality as of `when` (last known score at/before that moment).

    Returns None if there's no series or insufficient data."""
    series = trend_quality_by_ticker.get(ticker)
    if series is None or series.empty:
        return None

    lookup = when
    idx_tz = getattr(series.index, "tz", None)
    if idx_tz is not None and lookup.tzinfo is None:
        lookup = lookup.tz_localize(idx_tz)
    elif idx_tz is None and lookup.tzinfo is not None:
        lookup = lookup.tz_localize(None)

    try:
        score = series.asof(lookup)
    except Exception:
        return None
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return None
    return float(score)


def _ticker_ranking_score(
    ticker: str,
    candidates: list[_Candidate],
    trend_quality_by_ticker: dict[str, pd.Series],
    as_of_date: pd.Timestamp,
    vol_weight: float = 0.7,
    win_rate_weight: float = 0.3,
    lookback_days: int = 60,
) -> float:
    """Hybrid score: vol_weight * trend_quality_normalized + win_rate_weight * win_rate.

    trend_quality is clipped to [0,1]; win_rate is [0,1]. Returns [0,1] combined score."""
    tq = _trend_quality_asof(ticker, as_of_date, trend_quality_by_ticker)
    if tq is None:
        tq = 0.5
    tq_normalized = max(0.0, min(1.0, tq))

    wr = _recent_win_rate(candidates, ticker, lookback_days)
    return vol_weight * tq_normalized + win_rate_weight * wr


def _filter_candidates_by_top_tickers(
    candidates: list[_Candidate],
    trend_quality_by_ticker: dict[str, pd.Series],
    top_k: int,
    vol_weight: float = 0.7,
    win_rate_weight: float = 0.3,
    lookback_days: int = 60,
) -> tuple[list[_Candidate], dict[str, float]]:
    """Keep only candidates from top-K tickers by hybrid (vol + win-rate) ranking.

    Each ticker is scored as median of all its candidate entry-day scores, then
    top-K are selected. Returns (filtered_candidates, ticker_scores_dict for diagnostics)."""
    if top_k <= 0:
        return candidates, {}

    ticker_scores_by_date: dict[str, list[float]] = {}
    for c in candidates:
        score = _ticker_ranking_score(
            c.ticker, candidates, trend_quality_by_ticker, c.date_opened,
            vol_weight, win_rate_weight, lookback_days
        )
        ticker_scores_by_date.setdefault(c.ticker, []).append(score)

    ticker_median_score = {
        ticker: float(np.median(scores))
        for ticker, scores in ticker_scores_by_date.items()
    }

    top_tickers = set(sorted(ticker_median_score.keys(),
                             key=lambda t: -ticker_median_score[t])[:top_k])

    filtered = [c for c in candidates if c.ticker in top_tickers]
    return filtered, ticker_median_score


def _run_ticker_backtest(
    ticker: str, strategy_name: str, vol_filter_ok: bool = True,
    data_source: str = "yfinance"
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Fetch data and run one ticker's full-history backtest.

    Shared by _fetch_and_extract (candidates only) and _fetch_extract_and_prices
    (candidates + close-price series + rolling trend_quality, for mark-to-market
    valuation and daily vol-filter rescreening) so the fetch + consolidated_backtest
    call isn't duplicated between the two.

    Returns (detail_df, hourly_ohlc_df), or (None, None) on missing/insufficient
    data. The full OHLC frame (not just Close) is returned so callers needing
    High/Low (e.g. for rolling_trend_quality) don't have to re-fetch.
    """
    if data_source == "stooq":
        df = fetch_hourly_stooq(ticker, period="730d")
    else:
        df = fetch_hourly_cached(ticker, period="730d")
    if df is None or df.empty:
        return None, None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    entry_s, exit_s = resolve_strategy(strategy_name, vol_filter_ok=vol_filter_ok)

    safe_ticker = ticker.replace("/", "-").replace("\\", "-")
    regime_model = PersistentHMMRegimeModel(
        _HMM_CACHE_DIR / f"{safe_ticker}.pkl",
        dates=df.index,
        closes=df["Close"].values,
    )

    bt = consolidated_backtest(
        df,
        regime_model=regime_model,
        context_adjuster=SentimentAdjuster(),
        entry_strategy=entry_s,
        exit_strategy=exit_s,
    )
    regime_model.save()
    detail = bt.get("detail", pd.DataFrame())
    if detail.empty:
        return None, None
    return detail, df


def _daily_ohlc_from_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample hourly OHLC bars to daily (High=max, Low=min, Close=last),
    dropping non-trading days — input for rolling_trend_quality(), reusing
    data already fetched for the backtest rather than a separate daily fetch."""
    daily = df.resample("1D").agg({"High": "max", "Low": "min", "Close": "last"})
    return daily.dropna(subset=["Close"])


def _candidates_from_detail(
    ticker: str, detail: pd.DataFrame, strategy_name: str, vol_filter_tag: str
) -> list[_Candidate]:
    """Extract round-trip trades from a backtest detail frame into _Candidate objects."""
    trades = extract_trades_from_detail(
        ticker, detail.reset_index(), strategy=strategy_name, vol_filter=vol_filter_tag
    )
    candidates = []
    for t in trades:
        try:
            opened = pd.Timestamp(t.date_opened)
            closed = pd.Timestamp(t.date_closed)
        except Exception:
            continue
        candidates.append(_Candidate(
            ticker=ticker,
            date_opened=opened,
            date_closed=closed,
            entry_score=t.entry_score,
            kelly_fraction=t.kelly_fraction,
            return_pct=t.return_pct,
            record=t,
        ))
    return candidates


def _fetch_and_extract(
    ticker: str, strategy_name: str, vol_filter_tag: str, vol_filter_ok: bool = True,
    data_source: str = "yfinance"
) -> list[_Candidate]:
    """Run one ticker's full-history backtest and extract its round-trip trades.

    vol_filter_ok is passed straight into the strategy's own built-in veto
    (baked into every Entry class) — this is a bool, not a re-lookup, since
    the caller has usually already screened the ticker once (efficiency).
    """
    detail, _df = _run_ticker_backtest(ticker, strategy_name, vol_filter_ok, data_source)
    if detail is None:
        print(f"  {ticker}: no data or insufficient data, skipping")
        return []
    return _candidates_from_detail(ticker, detail, strategy_name, vol_filter_tag)


def _fetch_extract_and_prices(
    ticker: str, strategy_name: str, vol_filter_tag: str, vol_filter_ok: bool = True,
    data_source: str = "yfinance"
) -> tuple[list[_Candidate], pd.Series | None, pd.Series | None]:
    """Like _fetch_and_extract, but also returns the ticker's close-price series
    (for mark-to-market valuation) and its rolling trend_quality series (for
    daily vol-filter rescreening — see rolling_trend_quality()'s docstring for
    why a single "as of today" snapshot can't be used across a historical
    backtest). Top-level function so it's picklable for ProcessPoolExecutor."""
    detail, df = _run_ticker_backtest(ticker, strategy_name, vol_filter_ok, data_source)
    if detail is None:
        return [], None, None
    candidates = _candidates_from_detail(ticker, detail, strategy_name, vol_filter_tag)
    trend_quality = rolling_trend_quality(_daily_ohlc_from_hourly(df))
    return candidates, df["Close"], trend_quality


def generate_candidates(
    tickers: list[str],
    strategy_name: str,
    vol_filter_tag: str = "suitable",
    vol_filter_ok: bool = True,
    workers: int = 1,
    data_source: str = "yfinance",
) -> tuple[list[_Candidate], dict[str, pd.Series], dict[str, pd.Series]]:
    """Generate one strategy's candidate trades across a ticker list, optionally
    in parallel, retaining each ticker's close-price series (mark-to-market)
    and rolling trend_quality series (daily vol-filter rescreening — see
    rolling_trend_quality()). Does not arbitrate against any capital pot, nor
    apply the vol filter itself — see arbitrate() for capital arbitration
    (re-runnable cheaply per pot size against the same candidates) and
    main()'s per-candidate filtering step for the vol gate."""
    all_candidates: list[_Candidate] = []
    price_by_ticker: dict[str, pd.Series] = {}
    trend_quality_by_ticker: dict[str, pd.Series] = {}

    if workers > 1 and len(tickers) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_fetch_extract_and_prices, t, strategy_name, vol_filter_tag, vol_filter_ok, data_source): t
                for t in tickers
            }
            for future in as_completed(futures):
                ticker = futures[future]
                cands, close, trend_quality = future.result()
                all_candidates.extend(cands)
                if close is not None:
                    price_by_ticker[ticker] = close
                if trend_quality is not None:
                    trend_quality_by_ticker[ticker] = trend_quality
    else:
        for ticker in tickers:
            cands, close, trend_quality = _fetch_extract_and_prices(ticker, strategy_name, vol_filter_tag, vol_filter_ok, data_source)
            all_candidates.extend(cands)
            if close is not None:
                price_by_ticker[ticker] = close
            if trend_quality is not None:
                trend_quality_by_ticker[ticker] = trend_quality

    all_candidates.sort(key=lambda c: c.date_opened)
    return all_candidates, price_by_ticker, trend_quality_by_ticker


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


def arbitrate(
    candidates: list[_Candidate],
    initial_cash: float,
    trade_cost: float,
    kelly_fallback: float,
    max_trades_per_day: int,
    cost_model_name: str = "flat",
    currency: str = "GBP",
    price_by_ticker: dict[str, pd.Series] | None = None,
) -> dict:
    """Walk candidates day-by-day (event days only: opens/closes), arbitrating
    entries against one shared, mutating cash pot. max_trades_per_day <= 0
    means unlimited (cash alone gates admission).

    Returns a dict: executed (list[TradeRecord]), equity_curve (list[dict],
    one row per event day — cash, deployed capital mark-to-market if
    price_by_ticker is given else cost-basis, portfolio_value, cumulative
    realized P&L, cumulative interest), total_interest, final_cash,
    n_candidates, n_admitted, n_rejected_cash (candidates that couldn't be
    sized due to insufficient cash — the diagnostic that actually supports a
    "capital wasn't the constraint" conclusion, not peak-vs-pot alone).

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
            "n_rejected_cash": 0,
        }

    by_day: dict[pd.Timestamp, list[_Candidate]] = {}
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

        # 2. admit new entries for this day, highest score first, up to the cap
        day_candidates = sorted(by_day.get(day, []), key=lambda c: -c.entry_score)
        taken = 0
        for cand in day_candidates:
            if max_trades_per_day > 0 and taken >= max_trades_per_day:
                break
            if cash <= trade_cost:
                n_rejected_cash += 1
                continue

            if cand.kelly_fraction and cand.kelly_fraction > 0:
                alloc = cand.kelly_fraction * cash
            else:
                alloc = min(kelly_fallback, cash)

            if cost_model_name == "flat":
                entry_fee = exit_fee = trade_cost
            else:
                model = make_cost_model(cost_model_name, cand.record.ticker, trade_cost)
                entry_fee = model.cost(alloc, True)
                exit_fee = model.cost(alloc * (1 + cand.return_pct), False)
            alloc = min(alloc, cash - entry_fee)
            if alloc <= 0:
                n_rejected_cash += 1
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

            rec = cand.record
            rec.pnl_usd = exit_proceeds - alloc - entry_fee
            executed.append(rec)
            taken += 1
            n_admitted += 1

        skipped = len(day_candidates) - taken
        if taken or skipped:
            print(f"    {day.date()}: took {taken}, skipped {skipped}  (cash={cash:,.2f})")

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
    }


def simulate_strategy(
    tickers: list[str],
    strategy_name: str,
    start_date: str,
    initial_cash: float,
    trade_cost: float,
    kelly_fallback: float,
    max_trades_per_day: int,
    vol_filter_tag: str = "suitable",
    vol_filter_ok: bool = True,
    cost_model_name: str = "flat",
    currency: str = "GBP",
) -> list[TradeRecord]:
    """Run one strategy across all tickers with a shared capital pool. Returns executed TradeRecords.

    Kept as a simple, single-pot, sequential entry point (backward compatible)
    — delegates the actual day-by-day arbitration to arbitrate(). For a
    parallelized, multi-pot-size, mark-to-market run see generate_candidates()
    + arbitrate() directly (used by main() for --universe/--pot-sizes runs).
    """
    print(f"\n{'='*64}\n Strategy: {strategy_name}  (vol_filter={vol_filter_tag})\n{'='*64}")

    all_candidates: list[_Candidate] = []
    for ticker in tickers:
        print(f"  fetching + backtesting {ticker}...")
        cands = _fetch_and_extract(ticker, strategy_name, vol_filter_tag, vol_filter_ok)
        cutoff = pd.Timestamp(start_date)
        # Normalize to naive timestamps for comparison (hourly data is tz-aware)
        cands = [c for c in cands if c.date_opened.tz_localize(None) >= cutoff]
        print(f"    {len(cands)} candidate trade(s) on/after {start_date}")
        all_candidates.extend(cands)

    result = arbitrate(
        all_candidates,
        initial_cash=initial_cash,
        trade_cost=trade_cost,
        kelly_fallback=kelly_fallback,
        max_trades_per_day=max_trades_per_day,
        cost_model_name=cost_model_name,
        currency=currency,
        price_by_ticker=None,
    )

    executed = result["executed"]
    total_pnl = sum(r.pnl_usd for r in executed)
    print(f"\n  {strategy_name}: {len(executed)} trade(s) executed, "
          f"final pot £{result['final_cash']:,.2f} (P&L £{total_pnl:+,.2f} on £{initial_cash:,.0f} start, "
          f"£{result['total_interest']:,.2f} interest on idle cash)")

    return executed


def _filter_candidates_by_daily_trend_quality(
    candidates: list[_Candidate],
    trend_quality_by_ticker: dict[str, pd.Series],
    min_trend_quality: float,
    wants_low: bool,
) -> list[_Candidate]:
    """Keep only candidates whose ticker's trend_quality, as of their own
    entry day, passes the threshold — the daily-rescreen equivalent of
    overnight_scope.py's stage-1 vol screen (screen_market(), which re-runs
    every night in real trading), applied per candidate instead of once per
    ticker up front. wants_low inverts the direction for choppy-seeking
    strategies (wants_low_trend_quality()), mirroring
    overnight_scope.py:106-135's stage-1 logic exactly. No score yet
    (insufficient trailing history) is permissive, matching
    resolve_strategy()'s documented default with no ticker context.
    """
    kept = []
    for c in candidates:
        score = _trend_quality_asof(c.ticker, c.date_opened, trend_quality_by_ticker)
        if score is None:
            kept.append(c)
            continue
        passes = (score < min_trend_quality) if wants_low else (score >= min_trend_quality)
        if passes:
            kept.append(c)
    return kept


def _write_position_summary(rows: list[dict], path: Path) -> None:
    """Write the per-(strategy, pot_size, date) equity-curve rows to a CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="live-sim")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Explicit ticker list. Mutually exclusive with --universe.")
    parser.add_argument("--universe", action="store_true",
                        help="Use the full S&P 500 + FTSE 100 universe (config/universe_sp_ftse.json) "
                             "instead of --tickers.")
    parser.add_argument("--strategies", nargs="+", default=["default", "conservative", "trend"])
    parser.add_argument("--start-date", default="2026-01-12")
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
    parser.add_argument("--kelly-fallback", type=float, default=100.0)
    parser.add_argument("--max-trades-per-day", type=int, default=1,
                        help="Daily cap on new position admissions. 0 (or negative) = unlimited "
                             "(cash alone gates admission).")
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
    parser.add_argument("--lookback-days", type=int, default=60,
                        help="Window for computing recent win-rate (days). Default: 60.")
    parser.add_argument("--data-source", choices=["yfinance", "stooq"], default="yfinance",
                        help="Data source: 'yfinance' (default, 2.9yr history) or 'stooq' "
                             "(10+ yr from local data/stooq_raw/, hard-fails if missing)")
    parser.add_argument("--journal", default=None,
                        help="Journal CSV to append trades to (default: data/journals/live.csv)")
    parser.add_argument("--position-summary", default=None,
                        help="Path to write the per-(strategy,pot_size,date) equity-curve CSV "
                             "(default: data/journals/live_sim_position_summary_<timestamp>.csv). "
                             "Additive output — does not change --journal's default path/format.")
    args = parser.parse_args(argv)

    if bool(args.tickers) == bool(args.universe):
        parser.error("exactly one of --tickers or --universe is required")
    if args.universe:
        args.tickers = full_scan.load_sp_ftse_universe()

    pot_sizes = args.pot_sizes if args.pot_sizes else [args.initial_cash]

    print(f"Live simulation: {len(args.tickers)} tickers x {len(args.strategies)} strategies "
          f"x {len(pot_sizes)} pot size(s)")
    print(f"  start={args.start_date}  pot_sizes={pot_sizes}  "
          f"trade_cost=£{args.trade_cost:.2f}  kelly_fallback=£{args.kelly_fallback:.0f}  "
          f"max_trades/day={'unlimited' if args.max_trades_per_day <= 0 else args.max_trades_per_day}  "
          f"workers={args.workers}")
    if args.vol_filter_exempt:
        print(f"  vol-filter-exempt strategies: {', '.join(args.vol_filter_exempt)}")

    all_executed: list[TradeRecord] = []
    summary_rows: list[dict] = []

    for strategy_name in args.strategies:
        exempt = args.no_vol_filter or strategy_name in args.vol_filter_exempt
        vol_filter_tag = "disabled" if args.no_vol_filter else ("exempt" if exempt else "daily-rescreened")

        print(f"\n{'='*64}\n Strategy: {strategy_name}  (vol_filter={vol_filter_tag})\n{'='*64}")
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
            data_source=args.data_source,
        )

        cutoff = pd.Timestamp(args.start_date)
        candidates = [c for c in candidates if c.date_opened.tz_localize(None) >= cutoff]

        if not exempt:
            wants_low = wants_low_trend_quality(strategy_name)
            n_before = len(candidates)
            candidates = _filter_candidates_by_daily_trend_quality(
                candidates, trend_quality_by_ticker, args.min_trend_quality, wants_low,
            )
            print(f"  daily vol gate: {len(candidates)}/{n_before} candidates survive "
                  f"(min_trend_quality={args.min_trend_quality}, "
                  f"{'inverted (wants choppy)' if wants_low else 'standard'})")

        if args.top_k > 0:
            n_before = len(candidates)
            candidates, ticker_scores = _filter_candidates_by_top_tickers(
                candidates, trend_quality_by_ticker, args.top_k,
                vol_weight=args.vol_weight, win_rate_weight=args.win_rate_weight,
                lookback_days=args.lookback_days,
            )
            top_tickers_list = sorted(ticker_scores.keys(),
                                      key=lambda t: -ticker_scores[t])[:args.top_k]
            print(f"  top-k filter: {len(candidates)}/{n_before} candidates from {args.top_k} best tickers")
            print(f"    selected: {', '.join(top_tickers_list)}")
            for t in top_tickers_list[:5]:
                print(f"      {t}: score={ticker_scores.get(t, 0):.3f}")
            if len(top_tickers_list) > 5:
                print(f"      ... and {len(top_tickers_list) - 5} more")

        if candidates:
            earliest = min(c.date_opened for c in candidates)
            print(f"  {len(candidates)} candidate trade(s) on/after {args.start_date} "
                  f"(earliest actual: {earliest.date()})")

        for pot_size in pot_sizes:
            result = arbitrate(
                candidates,
                initial_cash=pot_size,
                trade_cost=args.trade_cost,
                kelly_fallback=args.kelly_fallback,
                max_trades_per_day=args.max_trades_per_day,
                cost_model_name=args.cost_model,
                currency="GBP",
                price_by_ticker=price_by_ticker,
            )
            all_executed.extend(result["executed"])

            total_pnl = sum(r.pnl_usd for r in result["executed"])
            peak_deployed = max((row["deployed"] for row in result["equity_curve"]), default=0.0)
            max_dd = _max_drawdown([row["portfolio_value"] for row in result["equity_curve"]])
            print(f"  pot £{pot_size:,.0f}: {len(result['executed'])}/{result['n_candidates']} admitted "
                  f"({result['n_rejected_cash']} rejected for cash), "
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
                "n_rejected_cash": result["n_rejected_cash"], "max_drawdown": max_dd,
            })

    journal_path = Path(args.journal) if args.journal else LIVE_JOURNAL
    n_logged = append_trades(journal_path, all_executed)
    print(f"\n{'='*64}\n {n_logged} trade(s) logged to {journal_path}\n{'='*64}")

    if summary_rows:
        if args.position_summary:
            summary_path = Path(args.position_summary)
        else:
            ts = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
            summary_path = LIVE_JOURNAL.parent / f"live_sim_position_summary_{ts}.csv"
        _write_position_summary(summary_rows, summary_path)
        print(f" position summary written to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
