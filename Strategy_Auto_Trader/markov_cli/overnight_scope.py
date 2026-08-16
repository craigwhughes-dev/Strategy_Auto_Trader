"""Overnight scope screening — determine in-scope tickers for each market.

Excludes tickers with poor volatility character (via vol_screen) unless they
have an open position. Writes audit trail and generates scoped watchlists for
the daemon to use.

Usage:
    uv run python -m Strategy_Auto_Trader.markov_cli.overnight_scope
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..core.cli_logging import setup_cli_logger

logger = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = ROOT / "config"
STATE_DIR = ROOT / "state"
DATA_DIR = ROOT / "data"

#: Ticker-dict keys that are treated as live daemon overrides when carried
#: from a watchlist into in_scope_<market>.json. Whitelisted explicitly (not
#: "any extra key") so unrelated metadata added to a ticker dict later can't
#: silently become a daemon behavior override. Expand if per-ticker tuning
#: beyond strategy assignment is needed.
OVERRIDE_KEYS = {"strategy"}


def load_config() -> dict:
    """Load overnight_strategy.json from config/."""
    config_path = CONFIG_DIR / "overnight_strategy.json"
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def load_watchlist(watchlist_path: str) -> dict:
    """Load a watchlist JSON file (path relative to repo root, e.g. "config/watchlist_ftse.json")."""
    path = ROOT / watchlist_path
    if not path.exists():
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


def _top_k_state_path() -> Path:
    return STATE_DIR / "top_k_universe.json"


def _load_previous_top_k_set() -> set[str] | None:
    """Fallback source when a ranking run fails: reuse the last successful
    night's top-K ticker set rather than leaving markets unscreened by top-K
    entirely. Returns None if no prior state exists (first-ever run, or it
    was never successful)."""
    path = _top_k_state_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data.get("tickers", []))
    except Exception:
        return None


def compute_global_top_k(
    config: dict,
    exec_state: dict,
    vol_kept: set[str] | None = None,
) -> set[str] | None:
    """Rank the combined universe once (in a separate subprocess) and return
    the global top-K ticker set, or None if top_k_screen is
    disabled/unconfigured (callers then apply no top-K filtering — vol/
    sentiment screen alone still applies, matching pre-top-K behavior).

    vol_kept: if provided, rank only these pre-screened tickers instead of
    the full S&P500+FTSE universe — eliminates wasted ranking slots on tickers
    that Stage 1 (vol_screen) would veto anyway.

    Ranking runs in a standalone subprocess (rank_universe_cli.py), not an
    in-process ProcessPoolExecutor pool, so it never touches this process's
    IBKR connection and a hang gets a real OS-level kill via the subprocess
    timeout rather than an after-the-fact log line.

    On any failure (subprocess timeout, non-zero exit, malformed output):
    falls back to the previous night's top-K set; if none exists either,
    returns None (degrades to vol/sentiment-only scope) — a market is never
    left with zero tickers, and this process never blocks indefinitely.
    """
    import subprocess
    import sys
    import tempfile
    import time

    cfg = config.get("top_k_screen", {})
    if not cfg.get("enabled", False):
        return None

    k = cfg.get("k", 70)
    strategy = cfg.get("strategy", "optimised")
    timeout_seconds = cfg.get("timeout_seconds", 18000)
    open_positions = set(exec_state.get("positions", {}).keys())

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "top_k_scores.json"
        cmd = [
            sys.executable, "-m", "Strategy_Auto_Trader.markov_cli.rank_universe_cli",
            "--strategy", strategy,
            "--vol-weight", str(cfg.get("vol_weight", 0.7)),
            "--win-rate-weight", str(cfg.get("win_rate_weight", 0.3)),
            "--lookback-days", str(cfg.get("lookback_days", 60)),
            "--workers", str(cfg.get("workers", 4)),
            "--output", str(output_path),
        ]
        if cfg.get("seasonal_volume", False):
            cmd.append("--seasonal-volume")
        if vol_kept:
            cmd += ["--tickers", ",".join(sorted(vol_kept | open_positions))]
            logger.info(f"  top_k_screen: ranking {len(vol_kept)} TQ-screened tickers "
                  f"(+ {len(open_positions)} open positions)")

        start = time.time()
        try:
            result = subprocess.run(
                cmd, cwd=ROOT, timeout=timeout_seconds,
                capture_output=True, text=True,
            )
            elapsed = time.time() - start
            if result.returncode != 0:
                logger.info(f"  top_k_screen: rank_universe_cli exited {result.returncode} "
                      f"after {elapsed/60:.1f} min, falling back to previous night's set. "
                      f"stderr: {result.stderr[-500:]}")
                return _load_previous_top_k_set()
            scores = json.loads(output_path.read_text(encoding="utf-8"))
        except subprocess.TimeoutExpired:
            logger.info(f"  top_k_screen: ranking exceeded {timeout_seconds/3600:.1f}h timeout, "
                  f"falling back to previous night's set")
            return _load_previous_top_k_set()
        except Exception as e:
            logger.info(f"  top_k_screen: ranking failed ({e}), falling back to previous night's set")
            return _load_previous_top_k_set()

        logger.info(f"  top_k_screen: ranked {len(scores)} tickers in {elapsed/60:.1f} min")

    top_tickers = set(sorted(scores.keys(), key=lambda t: -scores[t])[:k])
    top_tickers |= open_positions  # never drop an open position for falling outside top-K

    from ..core.atomic_io import atomic_write_json
    dest_path = _top_k_state_path()
    atomic_write_json(dest_path, {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "k": k,
        "strategy": strategy,
        "tickers": sorted(top_tickers),
        "scores": scores,
        "status": "ok",
    })
    logger.info(f"  top_k_screen: copied {output_path} -> {dest_path}")
    return top_tickers


def screen_market(market_name: str, market_cfg: dict, exec_state: dict,
                   global_top_k: set[str] | None = None) -> dict:
    """Screen tickers for a single market.

    Returns dict with:
      - market: market name
      - date: screening date (ISO format)
      - kept: list of in-scope tickers
      - excluded: list of {ticker, reason} dicts
      - open_positions: tickers with open positions (always kept)
      - overrides: {ticker: {override_key: value}} for tickers with a watchlist override
    """
    from ..quant_hmm.vol_screen import screen_tickers
    from ..strategy.base.registry import wants_low_trend_quality, wants_vol_screen_disabled

    watchlist = load_watchlist(market_cfg["watchlist"])
    all_tickers = [t["ticker"] if isinstance(t, dict) else t for t in watchlist.get("tickers", [])]

    overrides = {
        entry["ticker"]: {k: v for k, v in entry.items() if k in OVERRIDE_KEYS}
        for entry in watchlist.get("tickers", [])
        if isinstance(entry, dict) and any(k in OVERRIDE_KEYS for k in entry)
    }

    # Check vol screen config
    vol_cfg = market_cfg.get("vol_screen", {})
    min_trend_quality = vol_cfg.get("min_trend_quality", 0.0)
    max_downside_vol = vol_cfg.get("max_downside_vol", None)
    vol_period = vol_cfg.get("period", "2y")

    # Check exemption rule
    exempt_if_open = market_cfg.get("exempt_if_open_position", True)

    # Open positions in this market, scoped by each position's own recorded
    # "market" field (not by watchlist membership — a ticker can be dropped
    # from its watchlist while a position on it is still open; watchlist
    # membership is not a reliable proxy for "which market is this position
    # in"). A position missing the field (legacy data) defaults to matching
    # the current market rather than being silently dropped from consideration.
    open_positions = [
        t for t, p in exec_state.get("positions", {}).items()
        if p.get("market", market_name) == market_name
    ] if exempt_if_open else []

    kept: list[str] = []
    excluded: list[dict] = []

    # Stage 1: volatility screen
    market_strategy = market_cfg.get("defaults", {}).get("strategy")
    wants_choppy = (
        wants_low_trend_quality(market_strategy) if market_strategy else False
    )
    wants_screen_off = (
        wants_vol_screen_disabled(market_strategy) if market_strategy else False
    )
    # Strategy opt-out always wins over vol_screen.enabled config.
    # Per-ticker watchlist strategy overrides not handled — same as wants_choppy.
    do_vol_screen = vol_cfg.get("enabled", True) and not wants_screen_off

    if do_vol_screen:
        logger.info(f"  Vol-screening {len(all_tickers)} tickers for {market_name}...")
        vol_kept, vol_profiles = screen_tickers(
            all_tickers,
            min_trend_quality=min_trend_quality,
            max_downside_vol=max_downside_vol,
            period=vol_period,
            verbose=False
        )
        if wants_choppy:
            # This market's strategy is designed to trade the low-trend-quality
            # names the default screen vetoes (see registry.py resolve_strategy
            # docstring) — keep those instead. Downside-vol cap still applies
            # as a risk safety net.
            profile_by_ticker = {p["ticker"]: p for p in vol_profiles}
            stage1_tickers = {
                t for t in all_tickers
                if t in profile_by_ticker
                and profile_by_ticker[t]["trend_quality"] < min_trend_quality
                and (max_downside_vol is None or profile_by_ticker[t]["downside_vol"] <= max_downside_vol)
            }
            reason = "vol_screen_inverted"
        else:
            stage1_tickers = set(vol_kept)
            reason = "vol_screen"
        for ticker in all_tickers:
            if ticker not in stage1_tickers and ticker not in open_positions:
                excluded.append({"ticker": ticker, "reason": reason})
    else:
        stage1_tickers = set(all_tickers)

    # Stage 2: global top-K ranking intersection (computed once for both
    # markets by main(), not per-market — see compute_global_top_k())
    if global_top_k is not None:
        before = len(stage1_tickers)
        stage1_tickers = {
            t for t in stage1_tickers
            if t in global_top_k or t in open_positions
        }
        already_excluded = {e["ticker"] for e in excluded}
        for ticker in all_tickers:
            if ticker not in stage1_tickers and ticker not in open_positions and ticker not in already_excluded:
                excluded.append({"ticker": ticker, "reason": "top_k_screen"})
        logger.info(f"  top-K filter for {market_name}: {len(stage1_tickers)}/{before} survive "
              f"(global top-{len(global_top_k)} intersected with market watchlist)")

    # Final kept list: stage1 plus every open position, unconditionally — a
    # position dropped from the watchlist file itself must still be kept in
    # scope so the daemon can keep monitoring/exiting it (see orphaned below).
    kept = sorted(set(stage1_tickers) | set(open_positions))

    # Tickers with an open position that's no longer in the watchlist file at
    # all — force-included above, but this is worth surfacing: it means the
    # watchlist has drifted out from under a live position (see todo.md,
    # 2026-07-30 entry). Not halt-worthy on its own, just needs visibility.
    orphaned = sorted(t for t in open_positions if t not in all_tickers)
    if orphaned:
        logger.info(f"  WARNING: {len(orphaned)} open position(s) not in {market_name}'s "
              f"watchlist file, force-kept: {', '.join(orphaned)}")

    return {
        "market": market_name,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "kept": kept,
        "excluded": excluded,
        "open_positions": open_positions,
        "orphaned_positions": orphaned,
        "overrides": overrides,
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
    """Generate a scoped watchlist with filtered tickers and merged defaults.

    Note: This function's output (config/generated/watchlist_<market>_scoped.json)
    is currently unused by the live daemon (which reads in_scope_<market>.json
    instead) — kept for reference / potential future use."""
    original = load_watchlist(original_watchlist_path)
    original_defaults = original.get("defaults", {})

    merged_defaults = {**original_defaults}
    merged_defaults.update({
        "capital_pot": execution_cfg.get("capital_pot", 20000),
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


#: Top-level config keys that screen_market() expects to find on market_cfg
#: (vol_screen/exempt_if_open_position are documented as shared-across-markets
#: defaults, but screen_market() only ever looks at market_cfg — main() must
#: merge them in, or a market_cfg without its own override silently gets the
#: function's hardcoded fallbacks instead of the configured values (e.g.
#: max_downside_vol has no code-level default, so it silently becomes
#: None/unenforced). Market-level keys win if ever set.
_MARKET_MERGE_KEYS = ("vol_screen", "exempt_if_open_position")


def _with_merged_defaults(market_cfg: dict, config: dict) -> dict:
    """Return market_cfg with top-level shared blocks merged in under it
    (market-level values, if present, take precedence)."""
    merged = {key: config[key] for key in _MARKET_MERGE_KEYS if key in config}
    merged.update(market_cfg)
    return merged


def _collect_vol_kept_combined(config: dict) -> set[str]:
    """Run vol_screen Stage 1 across all market watchlists and return the
    combined set of tickers passing min_trend_quality (no downside_vol cap —
    that key is intentionally absent from config). Used to pre-filter the
    ranking universe so rank_universe_cli ranks only quality tickers."""
    from ..quant_hmm.vol_screen import screen_tickers

    vol_cfg = config.get("vol_screen", {})
    if not vol_cfg.get("enabled", True):
        return set()

    min_tq = vol_cfg.get("min_trend_quality", 0.0)
    period = vol_cfg.get("period", "2y")

    combined: set[str] = set()
    for market_cfg in config.get("markets", {}).values():
        watchlist = load_watchlist(market_cfg["watchlist"])
        tickers = [t["ticker"] if isinstance(t, dict) else t for t in watchlist.get("tickers", [])]
        logger.info(f"  Pre-screening {len(tickers)} {market_cfg['watchlist'].split('_')[1].split('.')[0]} "
              f"tickers for ranking (min_trend_quality={min_tq})...")
        kept, _ = screen_tickers(tickers, min_trend_quality=min_tq, max_downside_vol=None,
                                 period=period, verbose=False)
        combined.update(kept)

    logger.info(f"  Pre-screen: {len(combined)} tickers pass TQ≥{min_tq} across all markets")
    return combined


def main() -> int:
    """Run overnight scope screening for all markets."""
    setup_cli_logger("overnight_scope")

    config = load_config()
    exec_state = load_execution_state()

    logger.info(f"\n{'='*64}")
    logger.info(f" Overnight scope screening")
    logger.info(f"{'='*64}\n")

    # Pre-screen watchlists for TQ before ranking so rank_universe_cli ranks
    # only quality tickers (avoids wasting k slots on tickers Stage 1 vetoes).
    vol_kept_combined = _collect_vol_kept_combined(config)
    global_top_k = compute_global_top_k(config, exec_state,
                                         vol_kept=vol_kept_combined or None)

    for market_name, raw_market_cfg in config.get("markets", {}).items():
        logger.info(f" {market_name}")
        market_cfg = _with_merged_defaults(raw_market_cfg, config)
        result = screen_market(market_name, market_cfg, exec_state, global_top_k=global_top_k)

        write_scope_result(market_name, result)
        generate_scoped_watchlist(
            market_name,
            market_cfg["watchlist"],
            result["kept"],
            config.get("execution", {}),
        )

        logger.info(f"   Kept ({len(result['kept'])}): {', '.join(result['kept'])}")
        if result['excluded']:
            by_reason: dict[str, list[str]] = {}
            for e in result['excluded']:
                by_reason.setdefault(e['reason'], []).append(e['ticker'])
            for reason, tickers in sorted(by_reason.items()):
                logger.info(f"   Excluded by {reason} ({len(tickers)}): {', '.join(tickers)}")
        if result['open_positions']:
            logger.info(f"   Open positions (exempt): {', '.join(result['open_positions'])}")
        logger.info("")

    logger.info(f"{'='*64}")
    logger.info(f" Overnight scope complete")
    logger.info(f"{'='*64}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
