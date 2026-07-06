"""Full-universe, full-history diagnostic scan for strategy research.

Runs every ticker in the universe (fresh S&P 500 + FTSE 100 constituent
lists, unioned with all existing watchlists) through the consolidated
hourly backtest over the maximum history yfinance offers at 1h interval
(730 days), with NO volatility screening and NO ticker exclusions — the
vol profile is computed and *recorded* instead of used to skip.

Per ticker, written as soon as that ticker finishes (so review can start
while the scan is still running):

  reports/full_scan/hourly/<ticker>.csv   every bar, every engine column
                                          plus raw indicator values
                                          (SMAs, MACD, Bollinger, ATR,
                                          SAR, rolling vol, exit flags)
                                          + the ticker's vol profile
                                          (constant across all rows).
                                          Same column set as daily.csv;
                                          the day_* aggregate columns are
                                          blank here (bar granularity).
  reports/full_scan/daily/<ticker>.csv    one row per calendar day:
                                          end-of-day snapshot of all
                                          columns + intraday aggregates
                                          + the ticker's vol profile
  data/journals/full_scan/<ticker>.csv    per-ticker trade journal: one
                                          row per closed trade (the usual
                                          TradeRecord fields) plus, on
                                          every row, the ticker+strategy
                                          scalar metrics below (repeated
                                          per row, same convention as
                                          trend_quality in daily.csv):
                                            - risk/return: sharpe/sortino/
                                              calmar (strategy & b&h),
                                              information_ratio, up/down
                                              capture, total_return,
                                              max_drawdown, final_kelly
                                            - trade-level aggregates:
                                              win_rate_pct, avg_win_pct,
                                              avg_loss_pct, profit_factor,
                                              avg_pnl_per_trade,
                                              avg_hours_held, total_pnl,
                                              final_portfolio,
                                              transaction_costs_total,
                                              days_in_market_pct
                                            - trend/regime quality (from
                                              vol_screen.volatility_profile)
                                            - sentiment (from
                                              sentiment.composite_sentiment):
                                              options IV/put-call/skew,
                                              VIX regime, insider activity,
                                              short interest, composite score
  reports/full_scan/summary.csv           one stats row per ticker,
                                          appended incrementally

Deliberately NOT included, with reason:
  - A1-A12 journal analyses (analyze_journal.py): these bucket and compare
    trades ACROSS tickers/strategies (score buckets, sector concentration,
    MFE/MAE percentiles, ...) — not meaningful as columns on one ticker's
    own trades. After the scan, run analyze_journal.py --journal on the
    output of combine_journals() (below) to get them for the whole scan.
  - Execution-quality slippage (signal price vs fill price): this backtest
    fills exactly at the signal bar's close, so slippage is structurally
    zero here — it only exists for live IBKR fills, already logged
    separately (state/execution_state.json, live_daemon.py).

Usage:
    uv run python -m Strategy_Auto_Trader.markov_cli.full_scan \
        [--build-universe] [--tickers AAA BBB ...] [--strategy default] \
        [--force] [--limit N] [--no-sentiment]

Resume: a ticker whose daily CSV already exists is skipped unless
--force is given, so the scan can be stopped and restarted freely.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import asdict
from dataclasses import fields as dc_fields
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.momentum import (
    compute_atr,
    compute_bollinger,
    compute_macd,
    compute_parabolic_sar,
    compute_sma,
)
from ..output.journal import TradeRecord, extract_trades_from_detail
from ..plugins.kelly_sizer import KellySizer
from ..plugins.persistent_hmm import PersistentHMMRegimeModel
from ..quant_hmm import sentiment as sentiment_mod
from ..quant_hmm.consolidated_engine import consolidated_backtest
from ..quant_hmm.quant_engine import fetch_hourly
from ..quant_hmm.vol_screen import volatility_profile
from ..strategy.base.registry import resolve_strategy

ROOT = Path(__file__).resolve().parent.parent.parent
UNIVERSE_FILE = ROOT / "config" / "universe_full.json"
SCAN_DIR = ROOT / "reports" / "full_scan"
JOURNAL_DIR = ROOT / "data" / "journals" / "full_scan"
HMM_CACHE_DIR = ROOT / "state" / "hmm_cache"


def _cache_vix_regime_for_process() -> None:
    """VIX is market-wide, not ticker-specific — cache the one fetch across
    a whole scan run instead of re-downloading it per ticker. Only called
    from main() (a real scan process), never at import time, so importing
    this module never mutates the shared sentiment module for callers/tests."""
    if not hasattr(sentiment_mod.vix_regime, "cache_info"):
        sentiment_mod.vix_regime = lru_cache(maxsize=1)(sentiment_mod.vix_regime)

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_FTSE100 = "https://en.wikipedia.org/wiki/FTSE_100_Index"
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) strategy-research/1.0"}

# Stats keys copied into summary.csv (detail/trade_results excluded).
_SUMMARY_STAT_KEYS = [
    "sharpe_strategy", "sharpe_bh", "sortino_strategy", "sortino_bh",
    "calmar_strategy", "calmar_bh", "information_ratio",
    "up_capture", "down_capture",
    "total_return_strategy", "total_return_bh",
    "max_drawdown_strategy", "max_drawdown_bh",
    "final_portfolio", "total_pl", "n_buys", "n_sells",
    "final_kelly", "n_bars",
]
_VOL_PROFILE_KEYS = [
    "ann_vol", "downside_vol", "efficiency_ratio", "autocorr",
    "choppiness_idx", "sign_change_freq", "trend_quality",
]
# Columns that only mean something aggregated over a calendar day; present
# in hourly.csv too (for a matching column set) but left empty there.
_DAILY_ONLY_KEYS = [
    "n_bars", "day_close_high", "day_close_low", "day_return_pct",
    "day_trade_events", "day_strategy_return",
]
# Ticker+strategy scalar metrics repeated onto every trade row of the
# per-ticker journal (same convention as trend_quality in daily.csv).
# total_pl/final_portfolio/n_buys/n_sells already come via _SUMMARY_STAT_KEYS.
_TRADE_AGG_KEYS = [
    "n_trades_total", "win_rate_pct", "avg_win_pct", "avg_loss_pct",
    "profit_factor", "avg_pnl_per_trade", "avg_hours_held",
    "transaction_costs_total", "days_in_market_pct",
]
_SENTIMENT_KEYS = [
    "iv_rank", "iv_current", "iv_signal", "put_call_ratio", "put_call_signal", "skew",
    "vix_current", "vix_sma20", "vix_regime", "vix_signal", "vix_term_structure",
    "insider_buys_90d", "insider_sells_90d", "insider_net", "insider_signal", "insider_total_value",
    "short_pct_float", "short_ratio", "short_signal",
    "sentiment_score", "sentiment_label", "confidence",
]


def _trade_aggregates(trades: list, bt: dict, trade_cost: float, days_covered: int) -> dict:
    """Aggregate stats across one ticker's closed trades (trade-level metrics)."""
    n = len(trades)
    agg = {k: (0.0 if k != "n_trades_total" else 0) for k in _TRADE_AGG_KEYS}
    agg["n_trades_total"] = n
    agg["transaction_costs_total"] = trade_cost * (bt.get("n_buys", 0) + bt.get("n_sells", 0))
    if n == 0:
        return agg

    winners = [t for t in trades if t.pnl_usd > 0]
    losers = [t for t in trades if t.pnl_usd < 0]
    agg["win_rate_pct"] = round(100 * len(winners) / n, 2)
    agg["avg_win_pct"] = round(100 * float(np.mean([t.return_pct for t in winners])), 4) if winners else 0.0
    agg["avg_loss_pct"] = round(100 * float(np.mean([t.return_pct for t in losers])), 4) if losers else 0.0
    gross_win = sum(t.pnl_usd for t in winners)
    gross_loss = abs(sum(t.pnl_usd for t in losers))
    agg["profit_factor"] = round(gross_win / gross_loss, 3) if gross_loss > 0 else float("nan")
    agg["avg_pnl_per_trade"] = round(sum(t.pnl_usd for t in trades) / n, 2)

    hours_held = []
    total_days_held = 0.0
    for t in trades:
        try:
            opened, closed = pd.Timestamp(t.date_opened), pd.Timestamp(t.date_closed)
        except (ValueError, TypeError):
            continue
        if pd.isna(opened) or pd.isna(closed):
            continue
        hours_held.append((closed - opened).total_seconds() / 3600)
        total_days_held += (closed - opened).total_seconds() / 86400
    agg["avg_hours_held"] = round(float(np.mean(hours_held)), 2) if hours_held else 0.0
    agg["days_in_market_pct"] = round(100 * total_days_held / days_covered, 2) if days_covered else 0.0
    return agg


def _ticker_sentiment(ticker: str, enabled: bool) -> dict:
    if not enabled:
        return {k: None for k in _SENTIMENT_KEYS}
    sent = sentiment_mod.composite_sentiment(ticker)
    flat: dict = {}
    for group in ("options", "vix", "insider", "short_interest"):
        flat.update(sent.get(group, {}))
    flat["sentiment_score"] = sent.get("sentiment_score")
    flat["sentiment_label"] = sent.get("sentiment_label")
    flat["confidence"] = sent.get("confidence")
    return {k: flat.get(k) for k in _SENTIMENT_KEYS}


def snapshot_columns(hourly: pd.DataFrame) -> list[str]:
    """Per-bar columns eligible for a point-in-time snapshot.

    Single source of truth for "what does a bar look like" — used both to
    write hourly.csv itself and to pull entry/exit snapshots onto journal
    rows, so the three reports can never drift apart on column names.
    Excludes the ticker-level vol-profile scalars (constant across every
    row, already surfaced once via _VOL_PROFILE_KEYS) and the day-only
    aggregates (meaningless for a single bar).
    """
    skip = set(_VOL_PROFILE_KEYS) | set(_DAILY_ONLY_KEYS)
    return [c for c in hourly.columns if c not in skip]


def bar_snapshot(hourly: pd.DataFrame, ts: str, cols: list[str], prefix: str) -> dict:
    """Nearest hourly bar to `ts`, as a dict of `{prefix}{column}` -> value."""
    try:
        pos = hourly.index.get_indexer([pd.Timestamp(ts)], method="nearest")[0]
    except (KeyError, ValueError, TypeError):
        pos = -1
    if pos == -1:
        return {f"{prefix}{c}": None for c in cols}
    row = hourly.iloc[pos]
    return {f"{prefix}{c}": row.get(c) for c in cols}


def _write_ticker_journal(
    journal_path: Path, trades: list, extra_cols: dict, hourly: pd.DataFrame,
) -> None:
    """Write one ticker's full research journal: per-trade rows + the
    ticker/strategy-level risk-return, trade-aggregate, trend-quality, and
    sentiment columns repeated on every row, plus the full entry-bar and
    exit-bar indicator snapshot (entry_*/exit_*) via snapshot_columns/
    bar_snapshot so the journal always carries everything hourly.csv does.
    Overwrites (ticker-scoped, not an append-only live log, so no dedup
    needed)."""
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    cols = snapshot_columns(hourly)
    if not trades:
        columns = (
            [f.name for f in dc_fields(TradeRecord)] + list(extra_cols.keys())
            + [f"entry_{c}" for c in cols] + [f"exit_{c}" for c in cols]
        )
        pd.DataFrame(columns=columns).to_csv(journal_path, index=False)
        return
    rows = []
    for t in trades:
        entry_snap = bar_snapshot(hourly, t.date_opened, cols, "entry_")
        exit_snap = bar_snapshot(hourly, t.date_closed, cols, "exit_")
        rows.append(asdict(t) | extra_cols | entry_snap | exit_snap)
    pd.DataFrame(rows).to_csv(journal_path, index=False)


def combine_journals(journal_dir: Path = JOURNAL_DIR, out_path: Path | None = None) -> Path:
    """Concatenate every per-ticker journal into one CSV for cross-ticker
    analysis (feed to scripts/analyze_journal.py --journal <out_path> for
    the A1-A12 analyses, which need many trades/tickers pooled)."""
    out_path = out_path or (journal_dir.parent / "full_scan_combined.csv")
    frames = [pd.read_csv(p) for p in sorted(journal_dir.glob("*.csv"))]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Universe building
# ---------------------------------------------------------------------------

def _wiki_tables(url: str) -> list[pd.DataFrame]:
    """Fetch a Wikipedia page and parse its wikitable elements via bs4."""
    import requests
    from bs4 import BeautifulSoup

    resp = requests.get(url, headers=_UA, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    tables = []
    for table in soup.find_all("table"):
        if "wikitable" not in (table.get("class") or []):
            continue
        rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            rows.append([c.get_text(strip=True) for c in cells])
        if len(rows) < 2:
            continue
        header, body = rows[0], [r for r in rows[1:] if len(r) == len(rows[0])]
        if body:
            tables.append(pd.DataFrame(body, columns=header))
    return tables


def _sp500_tickers() -> list[str]:
    """Current S&P 500 constituents, yfinance symbol convention (BRK.B -> BRK-B)."""
    for df in _wiki_tables(WIKI_SP500):
        sym_col = next((c for c in df.columns if c.lower() in ("symbol", "ticker")), None)
        if sym_col and len(df) > 400:
            return sorted({s.replace(".", "-") for s in df[sym_col] if s})
    raise RuntimeError("S&P 500 constituent table not found on Wikipedia page")


def _ftse100_tickers() -> list[str]:
    """Current FTSE 100 constituents mapped to yfinance .L symbols (BT.A -> BT-A.L)."""
    for df in _wiki_tables(WIKI_FTSE100):
        sym_col = next(
            (c for c in df.columns if c.lower() in ("ticker", "epic", "symbol")), None)
        if sym_col and len(df) > 90:
            out = set()
            for s in df[sym_col]:
                if not s:
                    continue
                s = s.replace(".", "-").rstrip("-")
                out.add(f"{s}.L")
            return sorted(out)
    raise RuntimeError("FTSE 100 constituent table not found on Wikipedia page")


def _watchlist_tickers() -> list[str]:
    """Union of tickers across every config/watchlist*.json."""
    out: set[str] = set()
    for path in sorted((ROOT / "config").glob("watchlist*.json")):
        try:
            wl = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for t in wl.get("tickers", []):
            name = t["ticker"] if isinstance(t, dict) else t
            if name:
                out.add(name)
    return sorted(out)


def build_universe(out_path: Path = UNIVERSE_FILE) -> list[str]:
    """Fetch fresh constituent lists, union with watchlists, save to config.

    FTSE tickers are ordered first (current live focus), then US names.
    A source that fails to fetch degrades to the watchlists' coverage of it.
    """
    sources: dict[str, list[str]] = {}
    for name, fn in (("ftse100", _ftse100_tickers), ("sp500", _sp500_tickers)):
        try:
            sources[name] = fn()
            print(f"  {name}: {len(sources[name])} tickers from Wikipedia")
        except Exception as exc:
            print(f"  {name}: fetch failed ({exc}) — relying on watchlists")
            sources[name] = []

    watch = _watchlist_tickers()
    print(f"  watchlists: {len(watch)} tickers")

    fresh = set(sources["ftse100"]) | set(sources["sp500"])
    all_tickers = fresh | set(watch)
    uk = sorted(t for t in all_tickers if t.endswith(".L"))
    us = sorted(t for t in all_tickers if not t.endswith(".L"))
    universe = uk + us

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "built": datetime.now().isoformat(timespec="seconds"),
        "sources": {k: len(v) for k, v in sources.items()} | {"watchlists": len(watch)},
        "tickers": universe,
    }, indent=2), encoding="utf-8")
    print(f"  universe: {len(universe)} tickers ({len(uk)} UK + {len(us)} US) -> {out_path}")
    return universe


def load_universe() -> list[str]:
    data = json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))
    return data["tickers"]


# ---------------------------------------------------------------------------
# Per-ticker indicator augmentation
# ---------------------------------------------------------------------------

def augment_detail(detail: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Join raw indicator values onto the engine's per-bar detail frame.

    The engine logs booleans (above_sma20, ...) but not the underlying
    series; strategy research wants the actual numbers. Everything here is
    causal (rolling/backward-looking only), aligned on the bar timestamp.
    """
    close = df["Close"].astype(float)
    ind = pd.DataFrame(index=df.index)
    ind["sma20"] = compute_sma(close, 20)
    ind["sma50"] = compute_sma(close, 50)
    ind["sma200"] = compute_sma(close, 200)

    log_rets = np.log(close / close.shift(1))
    ind["rolling_vol_20"] = log_rets.rolling(20).std()

    macd_line, macd_sig, macd_hist = compute_macd(close)
    ind["macd"] = macd_line
    ind["macd_signal"] = macd_sig
    ind["macd_hist"] = macd_hist
    ind["macd_bear_cross"] = (macd_line < macd_sig) & (macd_line.shift(1) >= macd_sig.shift(1))

    bb_mid, bb_up, bb_lo = compute_bollinger(close)
    ind["bb_mid"] = bb_mid
    ind["bb_upper"] = bb_up
    ind["bb_lower"] = bb_lo
    bb_w = (bb_up - bb_lo) / bb_mid
    ind["bb_width"] = bb_w
    bb_sq = bb_w < bb_w.rolling(20).mean()

    atr = compute_atr(close, window=14)
    ind["atr14"] = atr
    atr_ratio = atr / atr.rolling(20).mean()
    ind["atr_ratio"] = atr_ratio
    ind["consolidation"] = bb_sq & (atr_ratio < 0.75)

    ind["sar"] = compute_parabolic_sar(close)

    ind["dist_sma20_pct"] = (close - ind["sma20"]) / ind["sma20"] * 100
    ind["dist_sma50_pct"] = (close - ind["sma50"]) / ind["sma50"] * 100
    ind["dist_sma200_pct"] = (close - ind["sma200"]) / ind["sma200"] * 100

    ind["ret_1d_pct"] = close.pct_change(8) * 100      # ~1 trading day of hourly bars
    ind["ret_5d_pct"] = close.pct_change(40) * 100
    ind["ret_20d_pct"] = close.pct_change(160) * 100

    if "Volume" in df.columns:
        ind["volume"] = df["Volume"].astype(float)
    for col in ("Open", "High", "Low"):
        if col in df.columns:
            ind[col.lower()] = df[col].astype(float)

    out = detail.join(ind.round(6), how="left")
    return out


def daily_snapshot(hourly: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-bar rows to one end-of-day row per calendar day.

    Last bar's values for every column, plus intraday aggregates so
    mid-day trade events and ranges are not lost.
    """
    idx = pd.DatetimeIndex(hourly.index)
    day = idx.tz_localize(None).normalize()
    grouped = hourly.groupby(day)

    daily = grouped.last()
    daily.index.name = "date"
    daily["n_bars"] = grouped.size()
    daily["day_close_high"] = grouped["close"].max()
    daily["day_close_low"] = grouped["close"].min()
    daily["day_return_pct"] = ((grouped["close"].last() / grouped["close"].first() - 1) * 100).round(4)

    events = grouped["trade_event"].apply(lambda s: "+".join(x for x in s if x))
    daily["day_trade_events"] = events
    daily["trade_event"] = events          # last-bar value would hide mid-day fills
    reasons = grouped["sell_reason"].apply(lambda s: "; ".join(x for x in s if x))
    daily["sell_reason"] = reasons

    daily["day_strategy_return"] = grouped["strategy_return"].apply(lambda s: float((1 + s).prod() - 1)).round(6)
    return daily


# ---------------------------------------------------------------------------
# Scan loop
# ---------------------------------------------------------------------------

_SUMMARY_COLUMNS = (
    ["ticker", "strategy", "scanned_at", "status", "note",
     "first_bar", "last_bar", "bars_fetched", "days_covered",
     "n_trades_journaled", "elapsed_s"]
    + _VOL_PROFILE_KEYS + _SUMMARY_STAT_KEYS
)


def _append_summary_row(row: dict) -> None:
    """Append one row with a fixed column set so partial rows (no_data,
    error) stay aligned with full rows."""
    summary_path = SCAN_DIR / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row]).reindex(columns=_SUMMARY_COLUMNS)
    frame.to_csv(summary_path, mode="a", header=not summary_path.exists(), index=False)


def scan_ticker(
    ticker: str, strategy_name: str, *, trade_cost: float = 10.0, sentiment: bool = True,
) -> dict:
    """Run one ticker end-to-end; returns its summary row. Never skips on vol."""
    started = time.time()
    row: dict = {
        "ticker": ticker,
        "strategy": strategy_name,
        "scanned_at": datetime.now().isoformat(timespec="seconds"),
        "status": "ok",
        "note": "",
    }

    prof = volatility_profile(ticker)
    for k in _VOL_PROFILE_KEYS:
        row[k] = prof.get(k) if prof else None

    df = fetch_hourly(ticker, period="730d")
    if df is None or df.empty:
        row["status"] = "no_data"
        return row
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"])

    row["first_bar"] = str(df.index[0])
    row["last_bar"] = str(df.index[-1])
    row["bars_fetched"] = len(df)

    # vol_filter_ok=True: research scan, the veto must never suppress signals.
    entry_s, exit_s = resolve_strategy(strategy_name, vol_filter_ok=True)
    safe = ticker.replace("/", "-").replace("\\", "-")
    regime_model = PersistentHMMRegimeModel(
        HMM_CACHE_DIR / f"{safe}.pkl", dates=df.index, closes=df["Close"].values,
    )

    bt = consolidated_backtest(
        df,
        regime_model=regime_model,
        position_sizer=KellySizer(use_kelly=True, lookback=20),
        entry_strategy=entry_s,
        exit_strategy=exit_s,
        trade_cost=trade_cost,
        skip_unused_indicators=False,   # research scan: compute everything
    )
    regime_model.save()

    detail = bt.get("detail", pd.DataFrame())
    if detail.empty:
        row["status"] = "insufficient_data"
        return row

    hourly = augment_detail(detail, df)
    if prof:
        for k in _VOL_PROFILE_KEYS:
            hourly[k] = prof.get(k)
    daily = daily_snapshot(hourly)
    for k in _DAILY_ONLY_KEYS:
        hourly[k] = np.nan

    hourly_path = SCAN_DIR / "hourly" / f"{safe}.csv"
    daily_path = SCAN_DIR / "daily" / f"{safe}.csv"
    hourly_path.parent.mkdir(parents=True, exist_ok=True)
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(hourly_path)
    daily.to_csv(daily_path)

    trades = extract_trades_from_detail(
        ticker, detail.reset_index(), strategy=strategy_name,
        vol_filter="research_scan",
    )
    days_covered = len(daily)

    extra_cols: dict = {}
    for k in _SUMMARY_STAT_KEYS:
        extra_cols[k] = bt.get(k)
    for k in _VOL_PROFILE_KEYS:
        extra_cols[k] = prof.get(k) if prof else None
    extra_cols.update(_trade_aggregates(trades, bt, trade_cost, days_covered))
    extra_cols.update(_ticker_sentiment(ticker, sentiment))

    _write_ticker_journal(JOURNAL_DIR / f"{safe}.csv", trades, extra_cols, hourly)

    for k in _SUMMARY_STAT_KEYS:
        row[k] = bt.get(k)
    row["n_trades_journaled"] = len(trades)
    row["days_covered"] = days_covered
    row["elapsed_s"] = round(time.time() - started, 1)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="full-scan", description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Explicit ticker list (default: config/universe_full.json)")
    parser.add_argument("--strategy", default="default",
                        help="Strategy whose signals/trades to log (default: default)")
    parser.add_argument("--build-universe", action="store_true",
                        help="Refresh config/universe_full.json from Wikipedia + watchlists")
    parser.add_argument("--force", action="store_true",
                        help="Re-scan tickers whose output already exists")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after N tickers (0 = all)")
    parser.add_argument("--trade-cost", type=float, default=10.0,
                        help="Per-trade cost used for transaction_costs_total (default: 10.0)")
    parser.add_argument("--no-sentiment", action="store_true",
                        help="Skip options/VIX/insider/short-interest fetches (faster; "
                             "sentiment columns are mostly empty anyway for non-US tickers)")
    args = parser.parse_args(argv)

    if not args.no_sentiment:
        _cache_vix_regime_for_process()

    if args.build_universe or (args.tickers is None and not UNIVERSE_FILE.exists()):
        print("Building universe...")
        build_universe()

    tickers = args.tickers if args.tickers else load_universe()
    if args.limit:
        tickers = tickers[: args.limit]

    print(f"Full scan: {len(tickers)} tickers, strategy={args.strategy}, "
          f"no vol screening, max hourly history (730d)")
    print(f"  outputs: {SCAN_DIR}  journals: {JOURNAL_DIR}\n", flush=True)

    done = failed = skipped = 0
    for i, ticker in enumerate(tickers, 1):
        safe = ticker.replace("/", "-").replace("\\", "-")
        if not args.force and (SCAN_DIR / "daily" / f"{safe}.csv").exists():
            skipped += 1
            continue
        print(f"[{i}/{len(tickers)}] {ticker} ...", flush=True)
        try:
            row = scan_ticker(
                ticker, args.strategy,
                trade_cost=args.trade_cost, sentiment=not args.no_sentiment,
            )
        except Exception as exc:
            row = {
                "ticker": ticker, "strategy": args.strategy,
                "scanned_at": datetime.now().isoformat(timespec="seconds"),
                "status": "error", "note": f"{type(exc).__name__}: {exc}",
            }
            traceback.print_exc()
        _append_summary_row(row)
        if row["status"] == "ok":
            done += 1
            print(f"    ok: {row['bars_fetched']} bars, {row['days_covered']} days, "
                  f"{row['n_buys']} buys/{row['n_sells']} sells, "
                  f"trend_quality={row.get('trend_quality')}, {row['elapsed_s']}s", flush=True)
        else:
            failed += 1
            print(f"    {row['status']}: {row.get('note', '')}", flush=True)

    print(f"\nScan finished: {done} ok, {failed} failed/no-data, {skipped} already done.")
    print(f"For the A1-A12 cross-ticker journal analyses, combine then analyze:\n"
          f"  uv run python -c \"from Strategy_Auto_Trader.markov_cli.full_scan import combine_journals; "
          f"print(combine_journals())\"\n"
          f"  uv run python scripts/analyze_journal.py --journal <combined path>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
