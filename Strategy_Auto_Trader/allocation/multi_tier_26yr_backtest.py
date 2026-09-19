"""26-year stress test: all key strategies from 2001-01-23 to 2026-09-15.

Uses daily VIX/VXN closes from the IBKR daily cache (data/cache/ibkr_daily/).
INDEX_VIX.csv is already cached back to 1990. INDEX_VXN.csv is fetched from
the IB Gateway on first run and cached; subsequent runs use the cache offline.

Daily closes are stamped at 21:30 UTC so that value_asof(16:30 London) returns
the *prior* day's close — matching run-B's timing (tier set from last VIX/VXN
available before LSE cutoff, earning next day's return). lag=0 in
strategy_returns compensates for the one-day offset already in the series.

Extends the 18-year run-B backtest (2007-11-20) back to 2001 to include:
  - Dot-com bust / recovery (2001-2003)
  - Pre-GFC bull run (2003-2007)
  - GFC (2008-2009)

Strategies:
  Deployed asym10d    4-tier VXN/VIX rotation, asym10d filter, 13 bps/switch
  Vol-target 5%       EQGB/CSH2 vol-target 5% 20d; continuous weight, no tx cost
  Vol-target 10%      same at 10%
  SMA200+VT 5%        VT5% when EQGB above 200d SMA; 100% CSH2 otherwise
  Static 30/70        30% Nasdaq + 70% CSH2 blend; no cost

Run (gateway must be reachable for first-time VXN download; subsequent runs offline):
  uv run python -m Strategy_Auto_Trader.allocation.multi_tier_26yr_backtest
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .multi_tier_backtest_lse_lag import (
    MultiTierAllocator4Tier,
    build_tier_returns,
    load_lse_cutoff,
    net_of_switch_cost,
    strategy_returns,
    summarise,
    switch_flags,
    tier_series,
    value_asof,
)
from .multi_tier_comparators_lse_lag import (
    sma_vol_target_returns,
    static_blend_returns,
    vol_target_returns,
)
from .tier_filters import apply_asymmetric_hysteresis

_START = "2001-01-23"   # first day VXN is available in yfinance
_END = "2026-09-15"
_MEASURED_BPS = 13.0
_ASYM_DAYS = 10
_SPLIT_2007 = pd.Timestamp("2007-11-20")  # run-B data starts here
_SPLIT_2019 = pd.Timestamp("2019-01-01")

_log = logging.getLogger(__name__)


def _load_ibkr_daily_index(symbol: str, exchange: str = "CBOE", currency: str = "USD") -> pd.Series:
    """Load daily Close for a market index from IBKR daily cache.

    Fetches from IB Gateway (port 4002) if not yet cached; subsequent calls
    use the on-disk cache offline.  Cache: data/cache/ibkr_daily/INDEX_{symbol}.csv
    """
    from ..broker.ibkr_data import IBKRDataClient
    fetcher = IBKRDataClient()
    df = fetcher.fetch_index_daily(symbol, exchange, currency)
    if df is None or df.empty:
        raise RuntimeError(
            f"No daily data for {symbol} from IBKR. Ensure IB Gateway is running on port 4002."
        )
    close = df["Close"].astype(float)
    if close.index.tz is None:
        close.index = close.index.tz_localize("UTC")
    return close.sort_index()


def _load_vxn_extended(start: str) -> pd.Series:
    """VXN daily: IBKR from 2007-11-20; yfinance ^VXN for the pre-2007 gap.

    IBKR does not have VXN history before 2007-11-20. yfinance has ^VXN back to
    2001-01-23. The pre-2007 leg uses yfinance as the only available source.
    Post-2007 uses IBKR (authoritative, already cached from fetch_index_daily).
    """
    import warnings
    # IBKR leg (2007+)
    ibkr = _load_ibkr_daily_index("VXN", "CBOE")

    ibkr_start = ibkr.index[0]
    gap_needed = pd.Timestamp(start, tz="UTC") < ibkr_start

    if not gap_needed:
        return ibkr

    # yfinance leg for pre-IBKR gap
    import yfinance as yf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download("^VXN", start=start, end=ibkr_start.strftime("%Y-%m-%d"),
                          progress=False, auto_adjust=True)
    if raw.empty:
        _log.warning("yfinance VXN fetch returned empty for pre-2007 gap; tier 1 inactive pre-2007")
        return ibkr

    yf_close = raw["Close"].squeeze()
    yf_close.index = pd.DatetimeIndex(yf_close.index).tz_localize("UTC")
    # Merge: yfinance for the gap, IBKR for the rest (no overlap; concat in order)
    combined = pd.concat([yf_close.sort_index(), ibkr])
    return combined[~combined.index.duplicated(keep="last")].sort_index()


def daily_to_end_series(closes: pd.Series) -> pd.Series:
    """Stamp daily closes at 21:30 UTC (after US session close).

    asof(16:30 London) ≈ 15:30-16:30 UTC returns the *prior* day's close,
    mirroring run-B timing: tier is set from the last VIX/VXN bar before LSE cutoff.
    """
    idx = pd.DatetimeIndex([
        pd.Timestamp(t.date()).tz_localize("UTC") + pd.Timedelta(hours=21, minutes=30)
        for t in closes.index
    ])
    return pd.Series(closes.values, index=idx).sort_index()


def _sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")


def _annual_returns(r: pd.Series) -> pd.Series:
    return r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1) * 100


def run(
    data_dir: str = "data_synthetic/hourly",
    start: str = _START,
    end: str = _END,
    cost_bps: float = _MEASURED_BPS,
    asym_days: int = _ASYM_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (summary_df, annual_df)."""
    tz, cutoff = load_lse_cutoff()

    # --- Tier returns (synthetic daily) ---
    tier_returns = build_tier_returns(Path(data_dir), start, end)

    # --- VIX / VXN: daily closes, stamped post-US-close ---
    # VIX: IBKR daily cache, 1990+.
    # VXN: IBKR daily cache from 2007-11-20; yfinance ^VXN fills 2001-2007 gap
    #      (IBKR does not hold VXN history before 2007).
    _log.info("Loading daily VIX from IBKR cache; VXN IBKR+yfinance hybrid …")
    vix_raw = _load_ibkr_daily_index("VIX", "CBOE")
    vxn_raw = _load_vxn_extended(start)
    vix_end = daily_to_end_series(vix_raw)
    vxn_end = daily_to_end_series(vxn_raw)

    dates = tier_returns.index
    allocator = MultiTierAllocator4Tier()

    # Deployed: 4-tier asym10d
    # daily_to_end_series stamps closes at 21:30 UTC; asof(16:30 London≈15:30 UTC) returns
    # the PRIOR day's close → tier[T] already uses T-1 information. Use lag=0 in
    # strategy_returns so we earn T's return from tier[T] (T-1 signal), matching run-B's
    # "T-1 intraday signal → T's return" timing. Using lag=1 here would double the lag.
    raw_tiers = tier_series(dates, vxn_end, vix_end, tz, cutoff, allocator)
    asym_tiers = apply_asymmetric_hysteresis(raw_tiers, offensive_days=asym_days)
    r_dep_raw = strategy_returns(asym_tiers, tier_returns, lag=0)
    sw_dep = switch_flags(asym_tiers, r_dep_raw.index, lag=0)
    r_deployed = net_of_switch_cost(r_dep_raw, sw_dep, cost_bps)

    # Comparators (no VIX/VXN needed)
    r_vt5 = vol_target_returns(tier_returns, 0.05)
    r_vt10 = vol_target_returns(tier_returns, 0.10)
    r_sma_vt5 = sma_vol_target_returns(tier_returns, 0.05)
    r_static = static_blend_returns(tier_returns, 0.30)

    strategies: dict[str, pd.Series] = {
        f"Deployed asym{asym_days}d ({cost_bps:g}bps)": r_deployed,
        "Vol-target 5%": r_vt5,
        "Vol-target 10%": r_vt10,
        "SMA200+VT 5%": r_sma_vt5,
        "Static 30/70": r_static,
    }

    # --- Summary table ---
    rows = []
    for label, r in strategies.items():
        s = summarise(r, 100_000.0)
        pre_07 = r[r.index < _SPLIT_2007]
        mid = r[(r.index >= _SPLIT_2007) & (r.index < _SPLIT_2019)]
        post = r[r.index >= _SPLIT_2019]
        rows.append({
            "strategy": label,
            "sharpe": round(s["sharpe"], 3),
            "sortino": round(s["sortino"], 3),
            "return%": round(s["total_return_pct"], 1),
            "maxDD%": round(s["max_drawdown_pct"], 2),
            "sh_01-07": round(_sharpe(pre_07), 2),
            "sh_07-19": round(_sharpe(mid), 2),
            "sh_19+": round(_sharpe(post), 2),
        })
    summary_df = pd.DataFrame(rows)

    # --- Annual returns ---
    annual_df = pd.DataFrame({label: _annual_returns(r) for label, r in strategies.items()}).round(1)

    return summary_df, annual_df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", default="data_synthetic/hourly")
    p.add_argument("--start-date", default=_START)
    p.add_argument("--end-date", default=_END)
    p.add_argument("--cost-bps", type=float, default=_MEASURED_BPS)
    p.add_argument("--asym-days", type=int, default=_ASYM_DAYS)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    summary_df, annual_df = run(
        args.data_dir, args.start_date, args.end_date, args.cost_bps, args.asym_days
    )

    years = (pd.Timestamp(args.end_date) - pd.Timestamp(args.start_date)).days / 365.25
    print(f"\n26-YEAR STRESS TEST  {args.start_date} .. {args.end_date}  ({years:.1f} years)")
    print(f"Deployed: {args.cost_bps:g} bps/switch, asym{args.asym_days}d filter")
    print(f"Comparators: vol-target/SMA200+VT = continuous daily weight (no tx cost)\n")

    # Summary table
    hdr = (f"{'Strategy':<34} {'Sharpe':>7} {'Sortino':>8} {'Ret%':>8} "
           f"{'MaxDD%':>7} {'sh01-07':>8} {'sh07-19':>8} {'sh19+':>7}")
    print(hdr)
    print("-" * len(hdr))
    for _, row in summary_df.iterrows():
        lbl = str(row["strategy"])[:34]
        print(
            f"{lbl:<34} {row['sharpe']:>7.3f} {row['sortino']:>8.3f} {row['return%']:>8.1f} "
            f"{row['maxDD%']:>7.2f} {row['sh_01-07']:>8.2f} {row['sh_07-19']:>8.2f} {row['sh_19+']:>7.2f}"
        )

    # Annual table (abbreviated column names for width)
    print("\n\nANNUAL RETURNS (%)")
    ann_display = annual_df.rename(columns={
        f"Deployed asym{args.asym_days}d ({args.cost_bps:g}bps)": "Deployed",
        "Vol-target 5%": "VT5%",
        "Vol-target 10%": "VT10%",
        "SMA200+VT 5%": "SMA+VT5%",
        "Static 30/70": "Static30/70",
    })
    print(ann_display.to_string(float_format=lambda x: f"{x:+.1f}"))


if __name__ == "__main__":
    main()
