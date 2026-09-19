"""Parameter-free comparators for the 4-tier tier allocation strategy (run B).

Reproduces the reviewer's comparators from TIER_STRATEGY_REVIEW.md §8 using
identical data, costs and window as the existing sweeps, to settle D1.

Comparators (all net of 13 bps per discrete switch where applicable):

  Static 30/70      30% Nasdaq + 70% CSH2 daily blend; no rebalancing cost
  Vol-target 10%    20d rolling vol; scale Nasdaq to target 10% ann vol; rest in CSH2 (no tx cost)
  Vol-target 15%    same at 15% target
  SMA200            hold Nasdaq when above 200d SMA, else CSH2; 13 bps per switch
  VXN-2tier asym10d VXN<=18 cutoff → Nasdaq, else CSH2; asym10d; 13 bps per switch
  C2 timing         prior-day close signal, execute 1 day later (lag=2); all 4 tiers; 13 bps

Reference rows: deployed 4-tier run B raw + asym10d, and B&H Nasdaq/CSH2/SPY/ISF.L.
Sub-period columns (pre-2019 / 2019+) guard against regime-specific fits.

Run:
  uv run python -m Strategy_Auto_Trader.allocation.multi_tier_comparators_lse_lag
"""

from __future__ import annotations

import argparse
import logging
from datetime import time

import numpy as np
import pandas as pd

from .multi_tier_allocator_4tier import MultiTierAllocator4Tier
from .multi_tier_backtest_lse_lag import (
    _TIERS,
    bars_by_end_time,
    load_inputs,
    net_of_switch_cost,
    strategy_returns,
    summarise,
    switch_flags,
    tier_series,
    value_asof,
)
from .tier_filters import apply_asymmetric_hysteresis

_MEASURED_BPS = 13.0
_SPLIT = pd.Timestamp("2019-01-01")
_VXN_THRESHOLD = 18.0
_VOL_WINDOW = 20
_SMA_WINDOW = 200
_ASYM_DAYS = 10

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal builders
# ---------------------------------------------------------------------------

def static_blend_returns(tier_returns: pd.DataFrame, w_nasdaq: float = 0.30) -> pd.Series:
    """Fixed daily blend of Nasdaq and CSH2; no discrete switches, no cost."""
    return w_nasdaq * tier_returns[1] + (1.0 - w_nasdaq) * tier_returns[4]


def vol_target_returns(
    tier_returns: pd.DataFrame, target_vol: float, window: int = _VOL_WINDOW
) -> pd.Series:
    """Scale Nasdaq to hit `target_vol` annualised; remainder in CSH2.

    Weight is lagged by 1 day to prevent look-ahead (we use yesterday's vol estimate).
    No discrete switch cost: the weight changes continuously so 13 bps/day would be wrong.
    """
    r1 = tier_returns[1]
    ann_vol = r1.rolling(window, min_periods=window).std() * np.sqrt(252)
    w = (target_vol / ann_vol).clip(upper=1.0).shift(1)  # lag 1: yesterday's estimate
    w = w.fillna(target_vol / r1.std() / np.sqrt(252))   # initial fill = full-sample ratio
    w = w.clip(0.0, 1.0)
    return w * r1 + (1.0 - w) * tier_returns[4]


def sma_vol_target_returns(
    tier_returns: pd.DataFrame,
    target_vol: float,
    sma_window: int = _SMA_WINDOW,
    vol_window: int = _VOL_WINDOW,
) -> pd.Series:
    """Vol-target when Nasdaq is above its SMA; 100% CSH2 when below.

    Both the SMA status and vol-target weight are lagged by 1 day to avoid look-ahead.
    No discrete switch cost (continuous weight changes).
    """
    r1 = tier_returns[1]
    r4 = tier_returns[4]

    px = (1.0 + r1).cumprod()
    sma = px.rolling(sma_window, min_periods=sma_window).mean()
    above_sma = (px > sma).shift(1)  # yesterday's SMA status

    ann_vol = r1.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(252)
    w_vt = (target_vol / ann_vol).clip(upper=1.0).shift(1)
    w_vt = w_vt.fillna(target_vol / r1.std() / np.sqrt(252))
    w_vt = w_vt.clip(0.0, 1.0)

    w = w_vt.where(above_sma == True, 0.0)
    return w * r1 + (1.0 - w) * r4


def sma_signal_tiers(tier_returns: pd.DataFrame, window: int = _SMA_WINDOW) -> pd.Series:
    """Daily tier signal (1=Nasdaq, 4=CSH2) based on Nasdaq vs its SMA.

    Signal[T] = tier chosen from T's close (to trade at T+1 → earn T+1 return via lag=1).
    First `window` rows are NaN (no SMA yet); strategy_returns drops them.
    """
    px = (1.0 + tier_returns[1]).cumprod()
    sma = px.rolling(window, min_periods=window).mean()
    above = px > sma
    return above.map({True: 1, False: 4})


def two_tier_vxn_tiers(
    dates: pd.DatetimeIndex,
    vxn_end: pd.Series,
    tz: str,
    at: time | None,
    threshold: float = _VXN_THRESHOLD,
) -> pd.Series:
    """Binary tier: 1 (Nasdaq) when VXN <= threshold at cutoff, else 4 (CSH2)."""
    vals = []
    for d in dates:
        v = value_asof(vxn_end, d, tz, at)
        vals.append(1 if (v is not None and v <= threshold) else 4)
    return pd.Series(vals, index=dates)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _sharpe_of(returns: pd.Series) -> float:
    if len(returns) < 10:
        return float("nan")
    s = summarise(returns, 100_000.0)
    return s["sharpe"]


def _metrics(net_returns: pd.Series, switches: pd.Series | None, years: float) -> dict:
    """Full metrics dict for one strategy, from NET returns (cost already deducted)."""
    s = summarise(net_returns, 100_000.0)
    pre = net_returns[net_returns.index < _SPLIT]
    post = net_returns[net_returns.index >= _SPLIT]
    sw_count = 0.0 if switches is None else float(switches.sum())
    return {
        "sw/yr": sw_count / years,
        "sharpe": s["sharpe"],
        "return%": s["total_return_pct"],
        "maxDD%": s["max_drawdown_pct"],
        "sh_pre2019": _sharpe_of(pre),
        "sh_2019+": _sharpe_of(post),
    }


def _make_row(label: str, raw_returns: pd.Series, switches: pd.Series | None, bps: float) -> dict:
    """Apply cost (if switches given) and compute metrics."""
    if switches is not None:
        net = net_of_switch_cost(raw_returns, switches, bps)
    else:
        net = raw_returns
    years = len(raw_returns) / 252
    m = _metrics(net, switches, years)
    return {"strategy": label, **m}


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run(
    tier_returns: pd.DataFrame,
    vxn_hourly: pd.DataFrame,
    vix_hourly: pd.DataFrame,
    tz: str,
    cutoff: time,
    cost_bps: float = _MEASURED_BPS,
    asym_days: int = _ASYM_DAYS,
) -> pd.DataFrame:
    dates = tier_returns.index
    vxn_end = bars_by_end_time(vxn_hourly)
    vix_end = bars_by_end_time(vix_hourly)

    # 4-tier signals: cutoff (run B) and full-close (for C2)
    cutoff_tiers = tier_series(dates, vxn_end, vix_end, tz, cutoff)
    close_tiers = tier_series(dates, vxn_end, vix_end, tz, None)

    rows = []

    # --- B&H references (no cost, no switch) ---
    for tier_num, label in [(1, "[B&H] Nasdaq (EQGB proxy)"), (2, "[B&H] SPY"), (3, "[B&H] ISF.L"), (4, "[B&H] CSH2.L")]:
        rows.append(_make_row(label, tier_returns[tier_num], None, 0.0))

    # --- Static 30% Nasdaq / 70% CSH2 ---
    r_blend = static_blend_returns(tier_returns, w_nasdaq=0.30)
    rows.append(_make_row("Static 30% Nasdaq / 70% CSH2 (no cost)", r_blend, None, 0.0))

    # --- Vol-target (no discrete switch cost) ---
    for tv in (0.10, 0.15):
        r_vt = vol_target_returns(tier_returns, tv, _VOL_WINDOW)
        rows.append(_make_row(f"Vol-target {int(tv*100)}% 20d (no tx cost)", r_vt, None, 0.0))

    # --- SMA200: Nasdaq vs CSH2, 13 bps per switch ---
    sma_tiers = sma_signal_tiers(tier_returns, _SMA_WINDOW)
    r_sma = strategy_returns(sma_tiers, tier_returns, lag=1)
    sw_sma = switch_flags(sma_tiers, r_sma.index, lag=1)
    rows.append(_make_row(f"Nasdaq SMA{_SMA_WINDOW} else CSH2 ({cost_bps:g}bps/sw)", r_sma, sw_sma, cost_bps))

    # --- 2-tier VXN<=18 Nasdaq/CSH2 with asymmetric hysteresis ---
    vxn2_raw = two_tier_vxn_tiers(dates, vxn_end, tz, cutoff, _VXN_THRESHOLD)
    vxn2_asym = apply_asymmetric_hysteresis(vxn2_raw, offensive_days=asym_days)
    r_vxn2 = strategy_returns(vxn2_asym, tier_returns, lag=1)
    sw_vxn2 = switch_flags(vxn2_asym, r_vxn2.index, lag=1)
    rows.append(
        _make_row(f"VXN<={_VXN_THRESHOLD} 2-tier Nasdaq/CSH2 asym{asym_days}d ({cost_bps:g}bps/sw)", r_vxn2, sw_vxn2, cost_bps)
    )

    # --- C2 timing: prior-day close signal, execute next day close (lag=2) ---
    r_c2 = strategy_returns(close_tiers, tier_returns, lag=2)
    sw_c2 = switch_flags(close_tiers, r_c2.index, lag=2)
    rows.append(_make_row(f"C2: T-1 close signal lag=2 ({cost_bps:g}bps/sw)", r_c2, sw_c2, cost_bps))

    # --- Reference: deployed 4-tier run B raw ---
    r_b_raw = strategy_returns(cutoff_tiers, tier_returns, lag=1)
    sw_b_raw = switch_flags(cutoff_tiers, r_b_raw.index, lag=1)
    rows.append(_make_row(f"[REF] Run B raw ({cost_bps:g}bps/sw)", r_b_raw, sw_b_raw, cost_bps))

    # --- Reference: deployed 4-tier run B + asymmetric hysteresis ---
    asym_tiers = apply_asymmetric_hysteresis(cutoff_tiers, offensive_days=asym_days)
    r_b_asym = strategy_returns(asym_tiers, tier_returns, lag=1)
    sw_b_asym = switch_flags(asym_tiers, r_b_asym.index, lag=1)
    rows.append(_make_row(f"[REF] Run B asym{asym_days}d ({cost_bps:g}bps/sw)", r_b_asym, sw_b_asym, cost_bps))

    return pd.DataFrame(rows)


def _print_report(df: pd.DataFrame, start: str, end: str, cost_bps: float) -> None:
    n_days = int(round(df[df["strategy"] == "[REF] Run B raw (13bps/sw)"]["sw/yr"].item() * len(df) / df.shape[0], 0)) if False else "?"
    print(f"\nCOMPARATOR TABLE  {start} .. {end}  (cost {cost_bps:g} bps/switch where shown)")
    print(f"  split pre/post {_SPLIT.date()}")
    print(f"  B&H rows: no cost, no switching")
    print(f"  Vol-target: continuous, no per-day tx cost applied")
    print()
    cols = ["strategy", "sw/yr", "sharpe", "return%", "maxDD%", "sh_pre2019", "sh_2019+"]
    header = f"{'Strategy':<52} {'sw/yr':>6} {'Sharpe':>7} {'Ret%':>8} {'MaxDD%':>7} {'sh<2019':>8} {'sh>=2019':>9}"
    print(header)
    print("-" * len(header))
    for _, row in df.iterrows():
        sw = f"{row['sw/yr']:.1f}" if not np.isnan(row["sw/yr"]) else "   -"
        pre = f"{row['sh_pre2019']:.2f}" if not np.isnan(row["sh_pre2019"]) else "  nan"
        post = f"{row['sh_2019+']:.2f}" if not np.isnan(row["sh_2019+"]) else "  nan"
        label = str(row["strategy"])
        if len(label) > 52:
            label = label[:49] + "..."
        print(f"{label:<52} {sw:>6} {row['sharpe']:>7.2f} {row['return%']:>8.1f} {row['maxDD%']:>7.2f} {pre:>8} {post:>9}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", default="data_synthetic/hourly")
    p.add_argument("--hourly-cache", default="data/cache/ibkr_hourly")
    p.add_argument("--start-date", default="2007-11-20")
    p.add_argument("--end-date", default="2026-09-15")
    p.add_argument("--cost-bps", type=float, default=_MEASURED_BPS)
    p.add_argument("--asym-days", type=int, default=_ASYM_DAYS)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    tz, cutoff, vix, vxn, tier_returns = load_inputs(
        args.data_dir, args.hourly_cache, args.start_date, args.end_date
    )
    df = run(tier_returns, vxn, vix, tz, cutoff, args.cost_bps, args.asym_days)
    _print_report(df, args.start_date, args.end_date, args.cost_bps)


if __name__ == "__main__":
    main()
