"""
daily_backtest_compare.py — run live_sim for the most recent trading day
and compare what it would have traded against what the live daemon actually did.

Signal-date window covers T-1 AND T because:
  - Overnight run (02:00) uses T-1 bars → signal date = T-1, order placed T
  - Daytime runs use today's bars as they complete → signal date = T, order placed T
Both cases result in entry_date = T in execution_state.json.

Source is ibkr to match the daemon's data source. Uses clientId 2 (ibkr_data.py
default) which is free at 22:30 when the daemon's daytime workers are idle.
Running manually during trading hours will hit clientId conflicts and fall back
to yfinance automatically.

Writes output to:
  data/journals/daily_compare_<date>.csv   — sim journal (signal window T-1..T)
  logs/daily_compare_<date>.log            — comparison report

Usage:
    uv run python scripts/daily_backtest_compare.py
    uv run python scripts/daily_backtest_compare.py --date 2026-08-28
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"


def _most_recent_trading_day() -> date:
    today = pd.Timestamp.today().normalize()
    while today.weekday() >= 5:
        today -= pd.Timedelta(days=1)
    return today.date()


def _prev_trading_day(d: date) -> date:
    ts = pd.Timestamp(d) - pd.Timedelta(days=1)
    while ts.weekday() >= 5:
        ts -= pd.Timedelta(days=1)
    return ts.date()


def _load_top_k() -> list[str]:
    path = STATE_DIR / "top_k_universe.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["tickers"]


def _load_config() -> dict:
    path = CONFIG_DIR / "overnight_strategy.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_live_sim(tickers: list[str], strategy: str, capital: float, sim_start: date,
                  journal_path: Path, workers: int, source: str) -> bool:
    cmd = [
        "uv", "run", "python", "-m",
        "Strategy_Auto_Trader.markov_cli.live_sim",
        "--tickers", *tickers,
        "--strategies", strategy,
        "--start-date", str(sim_start),
        "--initial-cash", str(capital),
        "--journal", str(journal_path),
        "--position-summary", str(journal_path.with_suffix(".pos.csv")),
        "--source", source,
        "--workers", str(workers),
    ]
    print(f"Running live_sim (signal window from {sim_start}, {len(tickers)} tickers)...")
    result = subprocess.run(cmd, cwd=ROOT, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: live_sim exited {result.returncode}", file=sys.stderr)
        return False
    return True


def _sim_entries(journal_path: Path, dates: set[date]) -> pd.DataFrame:
    """Return sim journal rows whose signal date falls in `dates`."""
    if not journal_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(journal_path)
    if df.empty or "date_opened" not in df.columns:
        return pd.DataFrame()
    df["_date"] = pd.to_datetime(df["date_opened"], utc=True, errors="coerce").dt.date
    return df[df["_date"].isin(dates)].copy()


def _open_position_entry_date(detail: "pd.DataFrame") -> date | None:
    """Return the entry date of the current open trade if position > 0 on last bar, else None.

    Finds the first bar of the trailing contiguous block where position > 0.
    """
    if detail is None or detail.empty or detail.iloc[-1]["position"] <= 0:
        return None
    # walk backwards to find where the current trade started
    for i in range(len(detail) - 2, -1, -1):
        if detail.iloc[i]["position"] <= 0:
            entry_ts = detail.index[i + 1]
            return entry_ts.date()
    # position > 0 for the entire history (unlikely)
    return detail.index[0].date()


def _check_open_positions(
    tickers: list[str], strategy: str, source: str, signal_dates: set[date]
) -> set[str]:
    """Return subset of tickers where the backtest ended with an open position
    whose entry date falls within signal_dates. Uses clientId=3 (clear of all
    reserved IDs) for the inline IBKR data fetch."""
    from Strategy_Auto_Trader.quant_hmm.ticker_ranking import run_ticker_backtest  # noqa: PLC0415

    matched: set[str] = set()
    for ticker in tickers:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            detail, _ = run_ticker_backtest(
                ticker, strategy, vol_filter_ok=True, source=source, client_id=3
            )
        entry_date = _open_position_entry_date(detail)
        if entry_date is not None and entry_date in signal_dates:
            matched.add(ticker)
    return matched


def _daemon_entries(target_date: date) -> list[dict]:
    path = STATE_DIR / "execution_state.json"
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    positions = state.get("positions", {})
    target_str = str(target_date)
    return [
        {"ticker": ticker, **pos}
        for ticker, pos in positions.items()
        if pos.get("entry_date", "") == target_str
    ]


def _build_report(target_date: date, prev_date: date, sim_df: pd.DataFrame,
                  daemon_entries: list[dict], open_matches: set[str]) -> str:
    lines: list[str] = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  Backtest vs Daemon — daemon date: {target_date}")
    lines.append(f"  Signal window: {prev_date} .. {target_date}")
    lines.append(f"{'='*60}\n")

    sim_tickers = set(sim_df["ticker"].tolist()) if not sim_df.empty else set()
    daemon_tickers = {e["ticker"] for e in daemon_entries}

    # --- Sim entries (closed trades) ---
    lines.append(f"Sim entries, closed ({len(sim_tickers)}):")
    if sim_df.empty:
        lines.append("  (none)")
    else:
        score_col = "entry_score" if "entry_score" in sim_df.columns else None
        kelly_col = "kelly_fraction" if "kelly_fraction" in sim_df.columns else None
        for _, row in sim_df.sort_values("entry_score" if score_col else "ticker", ascending=False).iterrows():
            parts = [f"  {row['ticker']:<12}"]
            if score_col:
                parts.append(f"score={row[score_col]:.1f}")
            if kelly_col:
                parts.append(f"kelly={row[kelly_col]:.3f}")
            if "entry_price" in row:
                parts.append(f"price={row['entry_price']:.4g}")
            lines.append("  ".join(parts))

    # --- Sim open positions (entered in window, not yet closed) ---
    lines.append(f"\nSim entries, open/in-progress ({len(open_matches)}):")
    if open_matches:
        for t in sorted(open_matches):
            lines.append(f"  {t}")
    else:
        lines.append("  (none)")

    lines.append("")

    # --- Daemon entries ---
    lines.append(f"Daemon entries ({len(daemon_tickers)}):")
    if not daemon_entries:
        lines.append("  (none)")
    else:
        for e in sorted(daemon_entries, key=lambda x: x["ticker"]):
            lines.append(
                f"  {e['ticker']:<12}  fill={e.get('fill_price', '?'):.4g}"
                f"  qty={e.get('quantity', '?')}"
                f"  kelly={e.get('kelly_fraction', '?'):.3f}"
            )

    lines.append("")

    # --- Diff ---
    all_sim = sim_tickers | open_matches
    both = all_sim & daemon_tickers
    sim_only = all_sim - daemon_tickers
    daemon_only = daemon_tickers - all_sim

    lines.append(f"Match (sim AND daemon): {len(both)}")
    if both:
        for t in sorted(both):
            suffix = " [open]" if t in open_matches else ""
            lines.append(f"  {t}{suffix}")

    lines.append(f"\nSim only — predicted but daemon didn't enter: {len(sim_only)}")
    if sim_only:
        lines.append("  " + ", ".join(sorted(sim_only)))

    lines.append(f"\nDaemon only — entered but sim didn't predict: {len(daemon_only)}")
    if daemon_only:
        lines.append("  " + ", ".join(sorted(daemon_only)))
        lines.append("  *** Investigate: live signal diverged from backtest ***")

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default: most recent trading day)")
    args = parser.parse_args(argv)

    target_date = date.fromisoformat(args.date) if args.date else _most_recent_trading_day()
    prev_date = _prev_trading_day(target_date)
    print(f"Daemon date: {target_date}  Signal window: {prev_date} .. {target_date}")

    tickers = _load_top_k()
    config = _load_config()
    strategy = config["top_k_screen"]["strategy"]
    capital = config["execution"]["capital_pot"]
    workers = config["daytime"].get("workers", 4)
    # ibkr matches daemon's data source. clientId 2 (ibkr_data.py default) is only free
    # after trading hours — this script is scheduled at 22:30 when daemon workers are idle.
    # Running manually during trading hours will get clientId-in-use errors and fall back to yfinance.
    source = "ibkr"

    date_str = str(target_date).replace("-", "")
    journal_path = DATA_DIR / "journals" / f"daily_compare_{date_str}.csv"
    log_path = LOGS_DIR / f"daily_compare_{target_date}.log"

    # Start sim from prev_date so overnight-sourced signals (signal bar = T-1) are included
    ok = _run_live_sim(tickers, strategy, capital, prev_date, journal_path, workers, source)
    if not ok:
        return 1

    sim_df = _sim_entries(journal_path, {prev_date, target_date})
    daemon = _daemon_entries(target_date)

    signal_dates = {prev_date, target_date}
    daemon_tickers = {e["ticker"] for e in daemon}
    sim_tickers = set(sim_df["ticker"].tolist()) if not sim_df.empty else set()
    daemon_only = daemon_tickers - sim_tickers

    # For daemon-only entries, check whether the backtest also opened that position
    # but it's still in-flight (not yet closed) at the end of the data window.
    # These are correct predictions, not divergences — candidates_from_detail only
    # returns closed trades so live entries show up as "missing" until they close.
    open_matches: set[str] = set()
    if daemon_only:
        print(f"Checking {len(daemon_only)} daemon-only ticker(s) for in-progress backtest positions...")
        open_matches = _check_open_positions(list(daemon_only), strategy, source, signal_dates)

    report = _build_report(target_date, prev_date, sim_df, daemon, open_matches)
    print(report)

    LOGS_DIR.mkdir(exist_ok=True)
    log_path.write_text(report, encoding="utf-8")
    print(f"Report written to: {log_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
