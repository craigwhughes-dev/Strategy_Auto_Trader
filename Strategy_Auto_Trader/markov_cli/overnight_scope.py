"""Overnight scope screening — determine in-scope tickers for each market.

Excludes tickers with poor volatility character (via vol_screen) and negative
sentiment (via sentiment) unless they have an open position. Writes audit trail
and generates scoped watchlists for the daemon to use.

Usage:
    uv run python -m Strategy_Auto_Trader.markov_cli.overnight_scope
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = ROOT / "config"
STATE_DIR = ROOT / "state"
DATA_DIR = ROOT / "data"


def load_config() -> dict:
    """Load overnight_strategy.json from config/."""
    config_path = CONFIG_DIR / "overnight_strategy.json"
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def load_watchlist(watchlist_path: str) -> dict:
    """Load a watchlist JSON file."""
    path = CONFIG_DIR / watchlist_path
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_execution_state() -> dict:
    """Load execution_state.json to check for open positions."""
    state_path = STATE_DIR / "execution_state.json"
    if not state_path.exists():
        return {}
    try:
        with open(state_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def screen_market(market_name: str, market_cfg: dict, exec_state: dict) -> dict:
    """Screen tickers for a single market.

    Returns dict with:
      - market: market name
      - date: screening date (ISO format)
      - kept: list of in-scope tickers
      - excluded: list of {ticker, reason} dicts
      - open_positions: tickers with open positions (always kept)
    """
    from ..quant_hmm.vol_screen import screen_tickers
    from ..quant_hmm.sentiment import composite_sentiment

    watchlist = load_watchlist(market_cfg["watchlist"])
    all_tickers = [t["ticker"] if isinstance(t, dict) else t for t in watchlist.get("tickers", [])]

    # Check vol screen config
    vol_cfg = market_cfg.get("vol_screen", {})
    do_vol_screen = vol_cfg.get("enabled", True)
    min_trend_quality = vol_cfg.get("min_trend_quality", 0.0)
    vol_period = vol_cfg.get("period", "2y")

    # Check sentiment config
    sent_cfg = market_cfg.get("sentiment_screen", {})
    do_sentiment_screen = sent_cfg.get("enabled", True)
    min_sentiment_score = sent_cfg.get("min_sentiment_score", -0.3)
    exclude_labels = set(sent_cfg.get("exclude_labels", ["bearish"]))

    # Check exemption rule
    exempt_if_open = market_cfg.get("exempt_if_open_position", True)

    # Open positions in this market (heuristic: if position is open, keep it)
    open_positions = list(exec_state.get("positions", {}).keys()) if exempt_if_open else []

    kept: list[str] = []
    excluded: list[dict] = []

    # Stage 1: volatility screen
    if do_vol_screen:
        print(f"  Vol-screening {len(all_tickers)} tickers for {market_name}...")
        vol_kept, vol_profiles = screen_tickers(
            all_tickers,
            min_trend_quality=min_trend_quality,
            period=vol_period,
            verbose=False
        )
        stage1_tickers = set(vol_kept)
        for ticker in all_tickers:
            if ticker not in stage1_tickers and ticker not in open_positions:
                excluded.append({"ticker": ticker, "reason": "vol_screen"})
    else:
        stage1_tickers = set(all_tickers)

    # Stage 2: sentiment screen
    if do_sentiment_screen:
        print(f"  Sentiment-screening {len(stage1_tickers)} tickers for {market_name}...")
        for ticker in list(stage1_tickers):
            if ticker in open_positions:
                continue
            sent = composite_sentiment(ticker)
            score = sent.get("sentiment_score", 0.0)
            label = sent.get("sentiment_label", "neutral")
            if label in exclude_labels or score < min_sentiment_score:
                stage1_tickers.discard(ticker)
                excluded.append({
                    "ticker": ticker,
                    "reason": f"sentiment:{label}({score:.2f})"
                })

    # Final kept list: stage1 plus any open positions
    kept = sorted(list(stage1_tickers) + [t for t in open_positions if t in all_tickers])

    return {
        "market": market_name,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "kept": kept,
        "excluded": excluded,
        "open_positions": open_positions,
    }


def write_scope_result(market_name: str, result: dict) -> None:
    """Write in_scope_<market>.json audit trail."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / f"in_scope_{market_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def generate_scoped_watchlist(
    market_name: str,
    original_watchlist_path: str,
    in_scope_tickers: list[str],
    execution_cfg: dict,
) -> None:
    """Generate a scoped watchlist with filtered tickers and merged defaults."""
    original = load_watchlist(original_watchlist_path)
    original_defaults = original.get("defaults", {})

    merged_defaults = {**original_defaults}
    merged_defaults.update({
        "capital_pot": execution_cfg.get("capital_pot", 20000),
        "max_positions": execution_cfg.get("max_positions", 5),
        "daily_buy_limit": execution_cfg.get("daily_buy_limit", 2),
        "daily_sell_limit": execution_cfg.get("daily_sell_limit", None),
    })

    scoped_tickers = [
        t if isinstance(t, dict) else {"ticker": t}
        for t in original.get("tickers", [])
        if (t.get("ticker") if isinstance(t, dict) else t) in in_scope_tickers
    ]

    scoped_watchlist = {
        "defaults": merged_defaults,
        "tickers": scoped_tickers,
    }

    gen_dir = CONFIG_DIR / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    path = gen_dir / f"watchlist_{market_name}_scoped.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scoped_watchlist, f, indent=2)


def main() -> int:
    """Run overnight scope screening for all markets."""
    config = load_config()
    exec_state = load_execution_state()

    print(f"\n{'='*64}")
    print(f" Overnight scope screening")
    print(f"{'='*64}\n")

    for market_name, market_cfg in config.get("markets", {}).items():
        print(f" {market_name}")
        result = screen_market(market_name, market_cfg, exec_state)

        write_scope_result(market_name, result)
        generate_scoped_watchlist(
            market_name,
            market_cfg["watchlist"],
            result["kept"],
            config.get("execution", {}),
        )

        print(f"   Kept: {len(result['kept'])} tickers")
        print(f"   Excluded: {len(result['excluded'])} tickers")
        if result['open_positions']:
            print(f"   Open positions (exempt): {', '.join(result['open_positions'])}")
        print()

    print(f"{'='*64}")
    print(f" Overnight scope complete")
    print(f"{'='*64}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
