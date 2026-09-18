"""4-tier allocation: cost of LSE trading hours vs VIX/VXN moves that happen while LSE is shut.

Three runs over identical dates (tier choice always via MultiTierAllocator4Tier.signal):

  A  look-ahead (existing daily model): tier from the full-day VIX/VXN close, earns that
     SAME day's return. Not executable — the close is unknown until the day is over.
  A' ideal-executable: same full-day-close tier, earns the NEXT day's return. Assumes the
     tier could be traded at the end of the US session. Impossible for EQGB.L/ISF.L/CSH2.L
     (LSE shut), so it is an upper bound, not a strategy.
  B  live-faithful: tier from the latest VIX/VXN bar that has closed by the LSE cutoff
     (ftse trading_end, config/overnight_strategy.json), executed at that day's LSE close,
     earns the NEXT day's return.

A -> A' = pure look-ahead in the existing backtests. A' -> B = cost of the LSE-hours constraint.

Hourly IBKR bars are start-stamped, so a bar's Close only counts once the bar has ended.
Daily asset closes come from load_synthetic_daily (real + Brownian-bridge blend); only
VIX/VXN hourly timing is real wall-clock. No transaction costs modelled in any run.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from .multi_tier_allocator_4tier import MultiTierAllocator4Tier
from .multi_tier_backtest_4tier import load_synthetic_daily

_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "config" / "overnight_strategy.json"
_TIERS = (1, 2, 3, 4)
_DEFENSIVE_TIER = 4


def load_lse_cutoff(config_path: Path = _CONFIG_PATH) -> tuple[str, time]:
    """(timezone, trading_end) of the ftse market — the same source is_trading_hours reads."""
    ftse = json.loads(config_path.read_text())["markets"]["ftse"]
    hh, mm = ftse["trading_end"].split(":")
    return ftse["timezone"], time(int(hh), int(mm))


def bars_by_end_time(df: pd.DataFrame) -> pd.Series:
    """Close series indexed by bar END time (UTC). End = next bar's start; last bar gets +1h."""
    close = df["Close"].astype(float)
    start = close.index
    end = pd.Series(start[1:].append(pd.DatetimeIndex([start[-1] + pd.Timedelta(hours=1)])), index=start)
    out = pd.Series(close.values, index=pd.DatetimeIndex(end.values, tz="UTC"))
    return out[~out.index.duplicated(keep="last")].sort_index()


def _london_bound(day: pd.Timestamp, tz: str, at: time | None) -> pd.Timestamp:
    """UTC instant for `day` at local `at` (None = end of that local day)."""
    local_day = pd.Timestamp(day.date())
    if at is None:
        return (local_day + pd.Timedelta(days=1)).tz_localize(tz).tz_convert("UTC") - pd.Timedelta(seconds=1)
    return (local_day + pd.Timedelta(hours=at.hour, minutes=at.minute)).tz_localize(tz).tz_convert("UTC")


def value_asof(series: pd.Series, day: pd.Timestamp, tz: str, at: time | None) -> float | None:
    """Last known value at local `at` on `day` (stale carried forward, as the daemon would see it)."""
    v = series.asof(_london_bound(day, tz, at))
    return None if pd.isna(v) else float(v)


def tier_series(
    dates: pd.DatetimeIndex,
    vxn_end: pd.Series,
    vix_end: pd.Series,
    tz: str,
    at: time | None,
    allocator: MultiTierAllocator4Tier | None = None,
) -> pd.Series:
    allocator = allocator or MultiTierAllocator4Tier()
    tiers = [
        allocator.signal(d, value_asof(vxn_end, d, tz, at), value_asof(vix_end, d, tz, at)).tier
        for d in dates
    ]
    return pd.Series(tiers, index=dates)


def strategy_returns(tiers: pd.Series, tier_returns: pd.DataFrame, lag: int) -> pd.Series:
    """Return earned on each date by the tier chosen `lag` trading days earlier."""
    held = tiers.shift(lag).reindex(tier_returns.index)
    valid = held.notna()
    cols = held[valid].astype(int).map({t: i for i, t in enumerate(_TIERS)}).values
    vals = tier_returns[valid].values[np.arange(valid.sum()), cols]
    return pd.Series(vals, index=tier_returns.index[valid])


def summarise(returns: pd.Series, initial_cash: float) -> dict:
    final = initial_cash * float(np.prod(1 + returns.values))
    return MultiTierAllocator4Tier._compute_summary(returns.values, initial_cash, final)


def compare(
    tier_returns: pd.DataFrame,
    vxn_hourly: pd.DataFrame,
    vix_hourly: pd.DataFrame,
    tz: str,
    cutoff: time,
    initial_cash: float = 100_000.0,
) -> dict:
    dates = tier_returns.index
    vxn_end, vix_end = bars_by_end_time(vxn_hourly), bars_by_end_time(vix_hourly)
    close_tiers = tier_series(dates, vxn_end, vix_end, tz, None)
    cutoff_tiers = tier_series(dates, vxn_end, vix_end, tz, cutoff)

    runs = {
        "A_lookahead": strategy_returns(close_tiers, tier_returns, lag=0),
        "A'_ideal": strategy_returns(close_tiers, tier_returns, lag=1),
        "B_live": strategy_returns(cutoff_tiers, tier_returns, lag=1),
    }
    common = runs["B_live"].index
    runs = {k: v.loc[common] for k, v in runs.items()}

    differ = (close_tiers != cutoff_tiers).reindex(dates)
    # tier chosen on T earns T+1: attribute the P&L gap to the deciding day T
    gap = (runs["A'_ideal"] - runs["B_live"]).shift(-1).reindex(dates)
    return {
        "summaries": {k: summarise(v, initial_cash) for k, v in runs.items()},
        "n_days": len(common),
        "n_differ": int(differ.sum()),
        "n_missed_riskoff": int(((close_tiers == _DEFENSIVE_TIER) & (cutoff_tiers != _DEFENSIVE_TIER)).sum()),
        "gap_on_differ_days_pct": float(gap[differ].sum() * 100),
        "close_tiers": close_tiers,
        "cutoff_tiers": cutoff_tiers,
    }


def load_hourly_index(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")
    return df.sort_index()


def build_tier_returns(data_dir: Path, start: str, end: str) -> pd.DataFrame:
    files = {1: "EQGB_COMPLETE.csv", 2: "SPY.csv", 3: "ISF.L.csv", 4: "CSH2.L.csv"}
    closes = {}
    for tier, name in files.items():
        daily = load_synthetic_daily(data_dir / name, start, end)
        daily.index = pd.to_datetime(daily.index.date)
        closes[tier] = daily["Close"]
    px = pd.DataFrame(closes).dropna()
    return px.pct_change().dropna()[list(_TIERS)]


def _print_report(result: dict, start: str, end: str) -> None:
    print(f"\n4-TIER LSE-HOURS LAG  {start} .. {end}  ({result['n_days']} days)")
    # _compute_summary's sharpe is mean*252/std (no sqrt): divide by sqrt(252) for the conventional figure
    print("| Run | Sharpe (repo) | Sharpe (annualised) | Sortino (repo) | Return % | Max DD % |")
    print("|---|---|---|---|---|---|")
    for name, s in result["summaries"].items():
        print(
            f"| {name} | {s['sharpe']:.2f} | {s['sharpe'] / np.sqrt(252):.2f} | {s['sortino']:.2f} "
            f"| {s['total_return_pct']:.1f} | {s['max_drawdown_pct']:.2f} |"
        )
    print(f"\nDays cutoff-tier != full-close-tier: {result['n_differ']} of {result['n_days']}")
    print(f"  of which full-close says defensive (CSH2.L) but cutoff did not: {result['n_missed_riskoff']}")
    print(f"  A' minus B summed daily return on those days: {result['gap_on_differ_days_pct']:+.2f}%")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", default="data_synthetic/hourly")
    p.add_argument("--hourly-cache", default="data/cache/ibkr_hourly")
    p.add_argument("--start-date", default="2007-11-20", help="Real VXN hourly coverage starts here")
    p.add_argument("--end-date", default="2026-09-15")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    tz, cutoff = load_lse_cutoff()
    cache = Path(args.hourly_cache)
    vix = load_hourly_index(cache / "INDEX_VIX.csv")
    vxn = load_hourly_index(cache / "INDEX_VXN.csv")
    tier_returns = build_tier_returns(Path(args.data_dir), args.start_date, args.end_date)
    _log.info(f"LSE cutoff {cutoff} {tz}; VIX bars {len(vix)}, VXN bars {len(vxn)}, days {len(tier_returns)}")
    _print_report(compare(tier_returns, vxn, vix, tz, cutoff), args.start_date, args.end_date)


if __name__ == "__main__":
    main()
