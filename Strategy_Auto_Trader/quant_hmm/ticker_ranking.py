"""Ticker candidate-generation and hybrid ranking (trend_quality + win-rate).

Shared by live_sim.py (backtest-side --top-k filtering, retrospective ranking
over a full historical candidate list) and overnight_scope.py / rank_universe_cli.py
(daemon-side nightly ranking, "today" score per ticker via rank_universe()) —
extracted into one module so the live daemon's ticker selection can never
silently drift from what live_sim.py's --top-k sweeps actually validated.
"""

from __future__ import annotations

import logging
import multiprocessing

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..output.journal import TradeRecord, extract_trades_from_detail
from ..plugins.context_adjuster import SentimentAdjuster
from ..plugins.persistent_hmm import PersistentHMMRegimeModel
from .consolidated_engine import consolidated_backtest
from .data_cache import fetch_hourly_cached
from .vol_screen import rolling_trend_quality
from ..strategy.base.registry import resolve_strategy

logger = logging.getLogger(__name__)


_HMM_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "hmm_cache"

# Parallel worker client IDs start at 20 — clear of reserved IDs
# (1=execution daemon, 2=data-fetch default, 4=reconcile, 7/9=tests).
_WORKER_CLIENT_ID_BASE = 20
_worker_client_id: int = 2  # overwritten per worker process by _init_worker_client_id


def _init_worker_client_id(q: "multiprocessing.Queue[int]") -> None:
    global _worker_client_id
    _worker_client_id = q.get()


@dataclass
class Candidate:
    """A single strategy's round-trip trade, before shared-capital arbitration."""
    ticker: str
    date_opened: pd.Timestamp
    date_closed: pd.Timestamp
    entry_score: float
    kelly_fraction: float
    return_pct: float
    record: TradeRecord


def recent_win_rate(candidates: list[Candidate], ticker: str, lookback_days: int = 60) -> float:
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


def trend_quality_asof(
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


def ticker_ranking_score(
    ticker: str,
    candidates: list[Candidate],
    trend_quality_by_ticker: dict[str, pd.Series],
    as_of_date: pd.Timestamp,
    vol_weight: float = 0.7,
    win_rate_weight: float = 0.3,
    lookback_days: int = 60,
) -> float:
    """Hybrid score: vol_weight * trend_quality_normalized + win_rate_weight * win_rate.

    trend_quality is clipped to [0,1]; win_rate is [0,1]. Returns [0,1] combined score."""
    tq = trend_quality_asof(ticker, as_of_date, trend_quality_by_ticker)
    if tq is None:
        tq = 0.5
    tq_normalized = max(0.0, min(1.0, tq))

    wr = recent_win_rate(candidates, ticker, lookback_days)
    return vol_weight * tq_normalized + win_rate_weight * wr


def filter_candidates_by_top_tickers(
    candidates: list[Candidate],
    trend_quality_by_ticker: dict[str, pd.Series],
    top_k: int,
    vol_weight: float = 0.7,
    win_rate_weight: float = 0.3,
    lookback_days: int = 60,
) -> tuple[list[Candidate], dict[str, float]]:
    """Keep only candidates from top-K tickers by hybrid (vol + win-rate) ranking.

    Each ticker is scored as median of all its candidate entry-day scores, then
    top-K are selected. Returns (filtered_candidates, ticker_scores_dict for diagnostics)."""
    if top_k <= 0:
        return candidates, {}

    ticker_scores_by_date: dict[str, list[float]] = {}
    for c in candidates:
        score = ticker_ranking_score(
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


def _filter_candidates_by_daily_trend_quality(
    candidates: list[Candidate],
    trend_quality_by_ticker: dict[str, pd.Series],
    min_trend_quality: float,
    wants_low: bool,
) -> list[Candidate]:
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
        score = trend_quality_asof(c.ticker, c.date_opened, trend_quality_by_ticker)
        if score is None:
            kept.append(c)
            continue
        passes = (score < min_trend_quality) if wants_low else (score >= min_trend_quality)
        if passes:
            kept.append(c)
    return kept


def run_ticker_backtest(
    ticker: str, strategy_name: str, vol_filter_ok: bool = True,
    use_seasonal_volume: bool = False, source: str = "ibkr",
    df: pd.DataFrame | None = None,
    use_persistent_cache: bool = True,
    client_id: int = 2,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Fetch data and run one ticker's full-history backtest.

    Shared by fetch_and_extract (candidates only) and fetch_extract_and_prices
    (candidates + close-price series + rolling trend_quality, for mark-to-market
    valuation and daily vol-filter rescreening) so the fetch + consolidated_backtest
    call isn't duplicated between the two.

    source="ibkr" opts into the local incremental IBKR-backed cache instead of
    yfinance — default stays yfinance for day-to-day research sweeps (see
    quant_engine.fetch_hourly's docstring); full-universe live_sim runs can
    pass source="ibkr" to revalidate against the same data the live daemon
    actually trades on.

    df: when provided, skip fetch_hourly_cached entirely and use this frame
    instead. source is moot when df is supplied.

    use_persistent_cache: when False, pass regime_model=None to
    consolidated_backtest and skip regime_model.save() — required for Monte
    Carlo synthetic paths to avoid corrupting the real ticker's on-disk HMM
    cache. position_sizer is also kept None (engine builds fresh per-call) so
    KellySizer state never leaks across synthetic paths.

    Returns (detail_df, hourly_ohlc_df), or (None, None) on missing/insufficient
    data. The full OHLC frame (not just Close) is returned so callers needing
    High/Low (e.g. for rolling_trend_quality) don't have to re-fetch.
    """
    if df is None:
        df = fetch_hourly_cached(ticker, period="730d", source=source, client_id=client_id)
    if df is None or df.empty:
        return None, None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    entry_s, exit_s = resolve_strategy(strategy_name, vol_filter_ok=vol_filter_ok)

    if use_persistent_cache:
        safe_ticker = ticker.replace("/", "-").replace("\\", "-")
        regime_model = PersistentHMMRegimeModel(
            _HMM_CACHE_DIR / f"{safe_ticker}.pkl",
            dates=df.index,
            closes=df["Close"].values,
        )
    else:
        regime_model = None

    bt = consolidated_backtest(
        df,
        regime_model=regime_model,
        context_adjuster=SentimentAdjuster(),
        entry_strategy=entry_s,
        exit_strategy=exit_s,
        use_seasonal_volume=use_seasonal_volume,
    )
    if use_persistent_cache and regime_model is not None:
        regime_model.save()
    detail = bt.get("detail", pd.DataFrame())
    if detail.empty:
        return None, None
    return detail, df


def daily_ohlc_from_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample hourly OHLC bars to daily (High=max, Low=min, Close=last),
    dropping non-trading days — input for rolling_trend_quality(), reusing
    data already fetched for the backtest rather than a separate daily fetch."""
    daily = df.resample("1D").agg({"High": "max", "Low": "min", "Close": "last"})
    return daily.dropna(subset=["Close"])


def candidates_from_detail(
    ticker: str, detail: pd.DataFrame, strategy_name: str, vol_filter_tag: str
) -> list[Candidate]:
    """Extract round-trip trades from a backtest detail frame into Candidate objects."""
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
        candidates.append(Candidate(
            ticker=ticker,
            date_opened=opened,
            date_closed=closed,
            entry_score=t.entry_score,
            kelly_fraction=t.kelly_fraction,
            return_pct=t.return_pct,
            record=t,
        ))
    return candidates


def fetch_and_extract(
    ticker: str, strategy_name: str, vol_filter_tag: str, vol_filter_ok: bool = True,
    use_seasonal_volume: bool = False, source: str = "ibkr",
    df: pd.DataFrame | None = None,
    use_persistent_cache: bool = True,
    client_id: int = 2,
) -> list[Candidate]:
    """Run one ticker's full-history backtest and extract its round-trip trades.

    vol_filter_ok is passed straight into the strategy's own built-in veto
    (baked into every Entry class) — this is a bool, not a re-lookup, since
    the caller has usually already screened the ticker once (efficiency).

    df / use_persistent_cache: see run_ticker_backtest's docstring.
    """
    detail, _df = run_ticker_backtest(ticker, strategy_name, vol_filter_ok,
                                      use_seasonal_volume=use_seasonal_volume, source=source,
                                      df=df, use_persistent_cache=use_persistent_cache,
                                      client_id=client_id)
    if detail is None:
        logger.info(f"  {ticker}: no data or insufficient data, skipping")
        return []
    return candidates_from_detail(ticker, detail, strategy_name, vol_filter_tag)


def fetch_extract_and_prices(
    ticker: str, strategy_name: str, vol_filter_tag: str, vol_filter_ok: bool = True,
    use_seasonal_volume: bool = False, source: str = "ibkr",
    df: pd.DataFrame | None = None,
    use_persistent_cache: bool = True,
) -> tuple[list[Candidate], pd.Series | None, pd.Series | None]:
    """Like fetch_and_extract, but also returns the ticker's close-price series
    (for mark-to-market valuation) and its rolling trend_quality series (for
    daily vol-filter rescreening — see rolling_trend_quality()'s docstring for
    why a single "as of today" snapshot can't be used across a historical
    backtest). Top-level function so it's picklable for ProcessPoolExecutor.

    df / use_persistent_cache: see run_ticker_backtest's docstring.
    """
    detail, df_out = run_ticker_backtest(ticker, strategy_name, vol_filter_ok,
                                         use_seasonal_volume=use_seasonal_volume, source=source,
                                         df=df, use_persistent_cache=use_persistent_cache,
                                         client_id=_worker_client_id)
    if detail is None:
        return [], None, None
    candidates = candidates_from_detail(ticker, detail, strategy_name, vol_filter_tag)
    trend_quality = rolling_trend_quality(daily_ohlc_from_hourly(df_out))
    return candidates, df_out["Close"], trend_quality


def generate_candidates(
    tickers: list[str],
    strategy_name: str,
    vol_filter_tag: str = "suitable",
    vol_filter_ok: bool = True,
    workers: int = 1,
    use_seasonal_volume: bool = False,
    source: str = "ibkr",
    df_by_ticker: dict[str, pd.DataFrame] | None = None,
    use_persistent_cache: bool = True,
) -> tuple[list[Candidate], dict[str, pd.Series], dict[str, pd.Series]]:
    """Generate one strategy's candidate trades across a ticker list, optionally
    in parallel, retaining each ticker's close-price series (mark-to-market)
    and rolling trend_quality series (daily vol-filter rescreening — see
    rolling_trend_quality()). Does not arbitrate against any capital pot, nor
    apply the vol filter itself — see live_sim.py's arbitrate() for capital
    arbitration and main()'s per-candidate filtering step for the vol gate.

    source="ibkr" opts every ticker in this run onto the local incremental
    IBKR-backed cache instead of yfinance — day-to-day research sweeps stay
    on the yfinance default; pass source="ibkr" for a full-universe live_sim
    revalidation against the data the live daemon actually trades on.

    df_by_ticker: when provided, each ticker reads df_by_ticker.get(ticker)
    instead of calling fetch_hourly_cached — required for Monte Carlo synthetic
    paths (plain DataFrames, picklable across ProcessPoolExecutor without
    closure serialisation issues).

    use_persistent_cache: passed through to run_ticker_backtest — set False for
    Monte Carlo paths to avoid corrupting real on-disk HMM caches."""
    all_candidates: list[Candidate] = []
    price_by_ticker: dict[str, pd.Series] = {}
    trend_quality_by_ticker: dict[str, pd.Series] = {}

    if workers > 1 and len(tickers) > 1:
        id_queue: multiprocessing.Queue = multiprocessing.Queue()
        for i in range(workers):
            id_queue.put(_WORKER_CLIENT_ID_BASE + i)
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker_client_id,
            initargs=(id_queue,),
        ) as executor:
            futures = {
                executor.submit(
                    fetch_extract_and_prices, t, strategy_name, vol_filter_tag,
                    vol_filter_ok, use_seasonal_volume, source,
                    df_by_ticker.get(t) if df_by_ticker else None,
                    use_persistent_cache,
                ): t
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
            cands, close, trend_quality = fetch_extract_and_prices(
                ticker, strategy_name, vol_filter_tag, vol_filter_ok, use_seasonal_volume, source,
                df_by_ticker.get(ticker) if df_by_ticker else None,
                use_persistent_cache)
            all_candidates.extend(cands)
            if close is not None:
                price_by_ticker[ticker] = close
            if trend_quality is not None:
                trend_quality_by_ticker[ticker] = trend_quality

    all_candidates.sort(key=lambda c: c.date_opened)
    return all_candidates, price_by_ticker, trend_quality_by_ticker


def rank_universe(
    tickers: list[str],
    strategy_name: str,
    vol_weight: float = 0.7,
    win_rate_weight: float = 0.3,
    lookback_days: int = 60,
    workers: int = 4,
    use_seasonal_volume: bool = False,
    source: str = "ibkr",
) -> dict[str, float]:
    """Backtest every ticker in `tickers` and return {ticker: hybrid_score} "as
    of today" (the median-of-candidate-day score across each ticker's own full
    trade history, via filter_candidates_by_top_tickers's scoring path — the
    same one live_sim.py --top-k uses, so a caller's top-K selection can never
    drift from what was actually backtested).

    Tickers with zero candidates across the full backtest are omitted from the
    result entirely (no trading evidence to rank on) rather than backfilled
    with a trend_quality-only estimate — a ticker the strategy never trades has
    no demonstrated edge, and admitting it into a top-K set on chart-shape
    alone would displace a ticker with real signal.
    """
    candidates, _price_by_ticker, trend_quality_by_ticker = generate_candidates(
        tickers, strategy_name, vol_filter_ok=True, workers=workers,
        use_seasonal_volume=use_seasonal_volume, source=source,
    )
    _, ticker_scores = filter_candidates_by_top_tickers(
        candidates, trend_quality_by_ticker, top_k=len(tickers),
        vol_weight=vol_weight, win_rate_weight=win_rate_weight, lookback_days=lookback_days,
    )
    return ticker_scores
