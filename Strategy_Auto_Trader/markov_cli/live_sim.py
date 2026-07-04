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

Usage:
    uv run python -m Strategy_Auto_Trader.markov_cli.live_sim \\
        --tickers SHEL.L BP.L HSBA.L ULVR.L GSK.L RIO.L DGE.L LSEG.L BATS.L VOD.L \\
        --strategies default conservative trend \\
        --start-date 2026-01-12 \\
        --initial-cash 10000 --trade-cost 1 --kelly-fallback 100 \\
        --max-trades-per-day 1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd

from ..output.journal import LIVE_JOURNAL, TradeRecord, append_trades, extract_trades_from_detail
from ..plugins.context_adjuster import SentimentAdjuster
from ..plugins.kelly_sizer import KellySizer
from ..quant_hmm.consolidated_engine import consolidated_backtest
from ..quant_hmm.quant_engine import fetch_hourly
from ..quant_hmm.vol_screen import screen_tickers
from ..strategy.base.registry import resolve_strategy


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


def _fetch_and_extract(
    ticker: str, strategy_name: str, vol_filter_tag: str, vol_filter_ok: bool = True
) -> list[_Candidate]:
    """Run one ticker's full-history backtest and extract its round-trip trades.

    vol_filter_ok is passed straight into the strategy's own built-in veto
    (baked into every Entry class) — this is a bool, not a re-lookup, since
    the caller has usually already screened the ticker once (efficiency).
    """
    df = fetch_hourly(ticker, period="730d")
    if df is None or df.empty:
        print(f"  {ticker}: no data, skipping")
        return []
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    entry_s, exit_s = resolve_strategy(strategy_name, vol_filter_ok=vol_filter_ok)
    position_sizer = KellySizer(use_kelly=True, lookback=20)

    from pathlib import Path
    from ..plugins.persistent_hmm import PersistentHMMRegimeModel
    safe_ticker = ticker.replace("/", "-").replace("\\", "-")
    cache_dir = Path(__file__).resolve().parent.parent.parent / "state" / "hmm_cache"
    regime_model = PersistentHMMRegimeModel(
        cache_dir / f"{safe_ticker}.pkl",
        dates=df.index,
        closes=df["Close"].values,
    )

    bt = consolidated_backtest(
        df,
        regime_model=regime_model,
        position_sizer=position_sizer,
        context_adjuster=SentimentAdjuster(),
        entry_strategy=entry_s,
        exit_strategy=exit_s,
    )
    regime_model.save()
    detail = bt.get("detail", pd.DataFrame())
    if detail.empty:
        print(f"  {ticker}: insufficient data, skipping")
        return []

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
) -> list[TradeRecord]:
    """Run one strategy across all tickers with a shared capital pool. Returns executed TradeRecords."""
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

    all_candidates.sort(key=lambda c: c.date_opened)

    # Group candidate opens by calendar day
    by_day: dict[pd.Timestamp, list[_Candidate]] = {}
    for c in all_candidates:
        day = c.date_opened.tz_localize(None).normalize()
        by_day.setdefault(day, []).append(c)

    all_days = sorted(set(
        list(by_day.keys()) +
        [c.date_closed.tz_localize(None).normalize() for c in all_candidates]
    ))

    cash = initial_cash
    open_positions: list[dict] = []  # {date_closed, exit_proceeds}
    executed: list[TradeRecord] = []

    for day in all_days:
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
            if taken >= max_trades_per_day:
                break
            if cash <= trade_cost:
                continue

            if cand.kelly_fraction and cand.kelly_fraction > 0:
                alloc = cand.kelly_fraction * cash
            else:
                alloc = min(kelly_fallback, cash)
            alloc = min(alloc, cash - trade_cost)
            if alloc <= 0:
                continue

            cash -= (alloc + trade_cost)
            exit_proceeds = alloc * (1 + cand.return_pct) - trade_cost
            open_positions.append({
                "date_closed": cand.date_closed.tz_localize(None).normalize(),
                "exit_proceeds": exit_proceeds,
            })

            rec = cand.record
            rec.pnl_usd = exit_proceeds - alloc - trade_cost
            executed.append(rec)
            taken += 1

        skipped = len(day_candidates) - taken
        if taken or skipped:
            print(f"    {day.date()}: took {taken}, skipped {skipped}  (cash={cash:,.2f})")

    # Any positions still open at the end of the window: add back to cash as unrealised (not sold)
    final_cash = cash + sum(p["exit_proceeds"] for p in open_positions)

    total_pnl = sum(r.pnl_usd for r in executed)
    print(f"\n  {strategy_name}: {len(executed)} trade(s) executed, "
          f"final pot £{final_cash:,.2f} (P&L £{total_pnl:+,.2f} on £{initial_cash:,.0f} start)")

    return executed


def _vol_filter_tickers(
    tickers: list[str], min_trend_quality: float
) -> tuple[list[str], list[str], dict[str, float]]:
    """Screen tickers for trend-friendliness. Returns (suitable, unsuitable, trend_quality_by_ticker)."""
    print(f"\n{'='*64}\n Volatility filter (min_trend_quality={min_trend_quality})\n{'='*64}")
    kept, profiles = screen_tickers(tickers, min_trend_quality=min_trend_quality, verbose=False)
    scores = {p["ticker"]: p["trend_quality"] for p in profiles}
    unsuitable = [t for t in tickers if t not in kept]
    for t in tickers:
        verdict = "suitable" if t in kept else "UNSUITABLE"
        print(f"  {t:10s}  trend_quality={scores.get(t, float('nan')):>6.2f}  {verdict}")
    return kept, unsuitable, scores


def _skip_records(unsuitable: list[str], strategy: str, start_date: str) -> list[TradeRecord]:
    """One placeholder journal row per unsuitable ticker, for a single strategy."""
    return [
        TradeRecord(date_opened=start_date, ticker=ticker, strategy=strategy, vol_filter="unsuitable")
        for ticker in unsuitable
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="live-sim")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--strategies", nargs="+", default=["default", "conservative", "trend"])
    parser.add_argument("--start-date", default="2026-01-12")
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--trade-cost", type=float, default=1.0)
    parser.add_argument("--kelly-fallback", type=float, default=100.0)
    parser.add_argument("--max-trades-per-day", type=int, default=1)
    parser.add_argument("--min-trend-quality", type=float, default=0.0,
                        help="Vol-filter cutoff: exclude tickers below this trend_quality score")
    parser.add_argument("--no-vol-filter", action="store_true",
                        help="Skip the volatility/choppiness pre-screen for every strategy")
    parser.add_argument("--vol-filter-exempt", nargs="+", default=[],
                        help="Strategy names that trade the full ticker list, bypassing the vol filter "
                             "(other strategies in --strategies still get filtered)")
    args = parser.parse_args(argv)

    print(f"Live simulation: {len(args.tickers)} tickers x {len(args.strategies)} strategies")
    print(f"  start={args.start_date}  cash/strategy=£{args.initial_cash:,.0f}  "
          f"trade_cost=£{args.trade_cost:.2f}  kelly_fallback=£{args.kelly_fallback:.0f}  "
          f"max_trades/day={args.max_trades_per_day}")
    if args.vol_filter_exempt:
        print(f"  vol-filter-exempt strategies: {', '.join(args.vol_filter_exempt)}")

    if args.no_vol_filter:
        suitable, unsuitable = list(args.tickers), []
    else:
        suitable, unsuitable, _ = _vol_filter_tickers(args.tickers, args.min_trend_quality)
        if unsuitable:
            print(f"\n  Excluded {len(unsuitable)}/{len(args.tickers)} ticker(s) as unsuitable: "
                  f"{', '.join(unsuitable)}")

    all_executed: list[TradeRecord] = []

    for strategy_name in args.strategies:
        exempt = args.no_vol_filter or strategy_name in args.vol_filter_exempt
        if exempt:
            tickers_for_strategy = list(args.tickers)
            vol_filter_tag = "disabled" if args.no_vol_filter else "exempt"
        else:
            tickers_for_strategy = suitable
            vol_filter_tag = "suitable"
            all_executed.extend(_skip_records(unsuitable, strategy_name, args.start_date))

        executed = simulate_strategy(
            tickers=tickers_for_strategy,
            strategy_name=strategy_name,
            start_date=args.start_date,
            initial_cash=args.initial_cash,
            trade_cost=args.trade_cost,
            kelly_fallback=args.kelly_fallback,
            max_trades_per_day=args.max_trades_per_day,
            vol_filter_tag=vol_filter_tag,
            # tickers_for_strategy is already screened (or the strategy is exempt) —
            # pass True explicitly so the strategy's own veto doesn't re-check.
            vol_filter_ok=True,
        )
        all_executed.extend(executed)

    n_logged = append_trades(LIVE_JOURNAL, all_executed)
    print(f"\n{'='*64}\n {n_logged} trade(s) logged to {LIVE_JOURNAL}\n{'='*64}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
