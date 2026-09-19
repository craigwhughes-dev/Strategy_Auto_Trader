"""Block bootstrap confidence intervals and annual returns for key allocation strategies.

Strategies benchmarked (all using run-B LSE-lag data, 2007-11-20..2026-09-15):

  Deployed asym10d      4-tier VXN/VIX with asym10d filter, 13 bps/sw
  Vol-target 5%         EQGB/CSH2 vol-target 5% 20d; continuous weight, no tx cost
  Vol-target 10%        same at 10%
  Static 30/70          30% Nasdaq + 70% CSH2 blend; no cost
  SMA200+VT 5%          Vol-target 5% when Nasdaq above 200d SMA; 100% CSH2 otherwise

Block bootstrap: 21-day blocks, 2000 iterations, seed=42.
Outputs: Sharpe percentile table + annual returns breakdown for SMA200+VT vs VT5% vs deployed.

Run:
  uv run python -m Strategy_Auto_Trader.allocation.multi_tier_bootstrap_lse_lag
"""

from __future__ import annotations

import argparse
import logging
from datetime import time

import numpy as np
import pandas as pd

from .multi_tier_backtest_lse_lag import (
    load_inputs,
    net_of_switch_cost,
    strategy_returns,
    switch_flags,
    tier_series,
    bars_by_end_time,
)
from .multi_tier_comparators_lse_lag import (
    static_blend_returns,
    vol_target_returns,
    sma_vol_target_returns,
)
from .tier_filters import apply_asymmetric_hysteresis

_MEASURED_BPS = 13.0
_ASYM_DAYS = 10
_BLOCK_SIZE = 21
_N_ITER = 2000
_SEED = 42

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------

def block_bootstrap_sharpe(
    returns: pd.Series,
    block_size: int = _BLOCK_SIZE,
    n_iter: int = _N_ITER,
    seed: int = _SEED,
) -> np.ndarray:
    """Return array of annualised Sharpe ratios from block-bootstrap resamples."""
    rng = np.random.default_rng(seed)
    arr = returns.dropna().values
    n = len(arr)
    n_blocks = int(np.ceil(n / block_size))
    sharpes = np.empty(n_iter)
    for i in range(n_iter):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        boot = np.concatenate([arr[s : s + block_size] for s in starts])[:n]
        std = boot.std()
        sharpes[i] = (boot.mean() / std * np.sqrt(252)) if std > 0 else np.nan
    return sharpes


def bootstrap_table(strategies: dict[str, pd.Series], block_size: int, n_iter: int) -> pd.DataFrame:
    rows = []
    for label, rets in strategies.items():
        bs = block_bootstrap_sharpe(rets, block_size, n_iter)
        full_sharpe = rets.mean() / rets.std() * np.sqrt(252)
        rows.append(
            {
                "strategy": label,
                "full_SR": round(full_sharpe, 3),
                "p5": round(float(np.nanpercentile(bs, 5)), 3),
                "p25": round(float(np.nanpercentile(bs, 25)), 3),
                "p50": round(float(np.nanpercentile(bs, 50)), 3),
                "p95": round(float(np.nanpercentile(bs, 95)), 3),
                "prob_SR_gt_0": round(float((bs > 0).mean()), 3),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Annual returns breakdown
# ---------------------------------------------------------------------------

def annual_returns(returns: pd.Series) -> pd.Series:
    """Annualised returns per calendar year from daily returns."""
    return returns.groupby(returns.index.year).apply(lambda r: (1 + r).prod() - 1) * 100


def annual_table(strategies: dict[str, pd.Series]) -> pd.DataFrame:
    dfs = {}
    for label, rets in strategies.items():
        dfs[label] = annual_returns(rets)
    return pd.DataFrame(dfs).round(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    tier_returns: pd.DataFrame,
    vxn_hourly: pd.DataFrame,
    vix_hourly: pd.DataFrame,
    tz: str,
    cutoff: time,
    cost_bps: float = _MEASURED_BPS,
    asym_days: int = _ASYM_DAYS,
    block_size: int = _BLOCK_SIZE,
    n_iter: int = _N_ITER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = tier_returns.index
    vxn_end = bars_by_end_time(vxn_hourly)
    vix_end = bars_by_end_time(vix_hourly)

    # Deployed: 4-tier asym10d, 13 bps
    cutoff_tiers = tier_series(dates, vxn_end, vix_end, tz, cutoff)
    asym_tiers = apply_asymmetric_hysteresis(cutoff_tiers, offensive_days=asym_days)
    r_deployed_raw = strategy_returns(asym_tiers, tier_returns, lag=1)
    sw_deployed = switch_flags(asym_tiers, r_deployed_raw.index, lag=1)
    r_deployed = net_of_switch_cost(r_deployed_raw, sw_deployed, cost_bps)

    # Vol-target 5%
    r_vt5 = vol_target_returns(tier_returns, 0.05)

    # Vol-target 10%
    r_vt10 = vol_target_returns(tier_returns, 0.10)

    # Static 30/70
    r_static = static_blend_returns(tier_returns, 0.30)

    # SMA200 + vol-target 5%
    r_sma_vt5 = sma_vol_target_returns(tier_returns, 0.05)

    # SMA200 + vol-target 7%
    r_sma_vt7 = sma_vol_target_returns(tier_returns, 0.07)

    strategies = {
        f"Deployed asym{asym_days}d ({cost_bps:g}bps)": r_deployed,
        "Vol-target 5%": r_vt5,
        "Vol-target 10%": r_vt10,
        "Static 30/70": r_static,
        "SMA200 + Vol-target 5%": r_sma_vt5,
        "SMA200 + Vol-target 7%": r_sma_vt7,
    }

    bs_df = bootstrap_table(strategies, block_size, n_iter)
    ann_df = annual_table(strategies)
    return bs_df, ann_df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", default="data_synthetic/hourly")
    p.add_argument("--hourly-cache", default="data/cache/ibkr_hourly")
    p.add_argument("--start-date", default="2007-11-20")
    p.add_argument("--end-date", default="2026-09-15")
    p.add_argument("--cost-bps", type=float, default=_MEASURED_BPS)
    p.add_argument("--asym-days", type=int, default=_ASYM_DAYS)
    p.add_argument("--block-size", type=int, default=_BLOCK_SIZE)
    p.add_argument("--n-iter", type=int, default=_N_ITER)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    tz, cutoff, vix, vxn, tier_returns = load_inputs(
        args.data_dir, args.hourly_cache, args.start_date, args.end_date
    )

    print(f"\nBlock bootstrap  block={args.block_size}d  n_iter={args.n_iter}  seed={_SEED}")
    print(f"Period: {args.start_date} .. {args.end_date}  ({len(tier_returns)/252:.1f} years)\n")

    bs_df, ann_df = run(
        tier_returns, vxn, vix, tz, cutoff,
        args.cost_bps, args.asym_days, args.block_size, args.n_iter,
    )

    # --- Bootstrap table ---
    print("SHARPE PERCENTILE TABLE")
    hdr = f"{'Strategy':<36} {'Full SR':>8} {'p5':>7} {'p25':>7} {'p50':>7} {'p95':>7} {'P(SR>0)':>8}"
    print(hdr)
    print("-" * len(hdr))
    for _, row in bs_df.iterrows():
        lbl = str(row["strategy"])[:36]
        print(
            f"{lbl:<36} {row['full_SR']:>8.3f} {row['p5']:>7.3f} {row['p25']:>7.3f}"
            f" {row['p50']:>7.3f} {row['p95']:>7.3f} {row['prob_SR_gt_0']:>8.3f}"
        )

    # --- Annual returns ---
    print("\n\nANNUAL RETURNS (%)")
    # show 5 strategies; truncate labels to 22 chars
    ann_display = ann_df.rename(columns=lambda c: c[:22])
    print(ann_display.to_string(float_format=lambda x: f"{x:+.1f}"))

    # --- Key observations ---
    deployed_bs_row = bs_df[bs_df["strategy"].str.startswith("Deployed")]
    sma_vt5_row = bs_df[bs_df["strategy"] == "SMA200 + Vol-target 5%"]
    vt5_row = bs_df[bs_df["strategy"] == "Vol-target 5%"]

    if not deployed_bs_row.empty and not sma_vt5_row.empty:
        dep_p5 = deployed_bs_row.iloc[0]["p5"]
        sma_p5 = sma_vt5_row.iloc[0]["p5"]
        vt5_p5 = vt5_row.iloc[0]["p5"] if not vt5_row.empty else float("nan")
        print(f"\nKey: SMA200+VT5% p5={sma_p5:.3f}  VT5% p5={vt5_p5:.3f}  Deployed p5={dep_p5:.3f}")
        if not np.isnan(sma_p5) and not np.isnan(dep_p5):
            print(f"  SMA200+VT5% worst-case p5 is {sma_p5/dep_p5:.2f}x deployed worst-case")


if __name__ == "__main__":
    main()
