"""Continuous-weight comparators on the intraday engine (Task C, decision D1).

Runs static blend, vol-target and SMA200+vol-target on the same intraday
data and fill model as the deployed tier rule, so results are directly
comparable on xSharpe (excess-over-cash Sharpe).

Comparators:
  Static 30/70      30% Nasdaq / 70% cash — fixed weights, no cost
  Vol-target 5%     trailing 20d vol; weekly (5d) rebalance; 2% deadband; partial cost
  Vol-target 10%    same at 10% target
  SMA200+VT5%       vol-target 5% when Nasdaq above SMA200, else 100% cash; same cadence

Reference rows: deployed VXN rule (13bps/sw), two-state VXN 23/24, B&H Nasdaq.

Cost model for vol-target:
  Each rebalance trades |delta_w| * pot_gbp. At £20k pot the IBKR min (£1/side)
  dominates — a 5% drift trades £1k, commission = £1/side = 1 bps of NAV.
  Over 30 rebalances/yr: ~30-60 bps/yr drag (small vs xSharpe differences).

Run: uv run python -m Strategy_Auto_Trader.allocation.intraday_comparators
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .intraday_engine import (
    ASSETS,
    DEFAULT_COST_BPS,
    WINDOWS,
    Inputs,
    Run,
    buy_and_hold,
    load_inputs,
    simulate,
    tiers_vxn_deadband,
    window_stats,
)
from .tier_cost_helper import POT_GBP, partial_rebalance_cost_bps, switch_cost_bps

_NASDAQ = 0
_CASH = 3

_VOL_WINDOW = 20
_SMA_WINDOW = 200
_CADENCE_DAYS = 5
_DEADBAND = 0.02

_WINDOWS_REF = [
    ("26yr (incl. bridged)", "y26"),
    ("since VXN data 2007-11-20", "vxn_era"),
    ("recent real 2024-03-25", "recent"),
]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _daily_simple_ret(inp: Inputs) -> np.ndarray:
    """(n_days, 4) daily simple returns per asset, first bar excluded."""
    n_days = len(inp.days)
    out = np.zeros((n_days, len(ASSETS)))
    for i in range(len(ASSETS)):
        col = inp.log_ret[:, i].copy()
        col[0] = 0.0
        out[:, i] = np.expm1(np.bincount(inp.day_codes, weights=col, minlength=n_days))
    return out


def _cash_daily(inp: Inputs) -> np.ndarray:
    col = inp.log_ret[:, _CASH].copy()
    col[0] = 0.0
    return np.expm1(np.bincount(inp.day_codes, weights=col, minlength=len(inp.days)))


def _run_blend(
    r_nasdaq: np.ndarray,
    r_cash: np.ndarray,
    target_w: np.ndarray,
    cadence_days: int,
    deadband: float,
    cost_fn,  # callable(delta_w) -> bps of NAV; None means no cost
    inp: Inputs,
) -> Run:
    """Simulate a continuous-weight Nasdaq/cash strategy with optional periodic rebalancing.

    Args:
        target_w: per-day target weight in Nasdaq, already lagged (no look-ahead)
        cadence_days: check for rebalance every this many days
        deadband: minimum |actual_w - target_w| to trigger a rebalance
        cost_fn: callable(delta_w) -> cost in bps of NAV; None = no cost
    """
    n = len(inp.days)
    daily_ret = np.zeros(n)
    rebal_flags = np.zeros(n)
    current_w = float(target_w[0]) if not np.isnan(target_w[0]) else 0.0

    for i in range(n):
        tw = float(target_w[i]) if not np.isnan(target_w[i]) else current_w
        nr = float(r_nasdaq[i])
        cr = float(r_cash[i])

        if cadence_days > 0 and (i % cadence_days == 0) and (abs(tw - current_w) > deadband):
            delta_w = abs(tw - current_w)
            cost_bps = cost_fn(delta_w) if cost_fn is not None else 0.0
            daily_ret[i] = current_w * nr + (1.0 - current_w) * cr - cost_bps / 1e4
            current_w = tw
            rebal_flags[i] = 1.0
        else:
            daily_ret[i] = current_w * nr + (1.0 - current_w) * cr
            # drift: let weights evolve with returns between rebalances
            nv = current_w * (1.0 + nr)
            cv = (1.0 - current_w) * (1.0 + cr)
            total = nv + cv
            if total > 0:
                current_w = nv / total

    return Run(daily_ret, rebal_flags, inp.days, _cash_daily(inp))


# ---------------------------------------------------------------------------
# Public strategy functions
# ---------------------------------------------------------------------------

def static_blend(inp: Inputs, w_nasdaq: float = 0.30) -> Run:
    """Fixed 30% Nasdaq / 70% cash, never rebalanced — no transaction cost."""
    dr = _daily_simple_ret(inp)
    daily = w_nasdaq * dr[:, _NASDAQ] + (1.0 - w_nasdaq) * dr[:, _CASH]
    return Run(daily, np.zeros(len(inp.days)), inp.days, _cash_daily(inp))


def vol_target(
    inp: Inputs,
    target_vol: float = 0.05,
    vol_window: int = _VOL_WINDOW,
    cadence_days: int = _CADENCE_DAYS,
    deadband: float = _DEADBAND,
    pot_gbp: float = POT_GBP,
    include_spread: bool = False,
) -> Run:
    """Nasdaq/cash blend targeting `target_vol` annualised (20d trailing vol).

    Weight is lagged by 1 day (yesterday's vol estimate); rebalanced weekly
    when drift exceeds `deadband`; cost per rebalance = partial_rebalance_cost_bps.
    """
    dr = _daily_simple_ret(inp)
    r_nasdaq = dr[:, _NASDAQ]
    r_cash = dr[:, _CASH]
    r_n_series = pd.Series(r_nasdaq)

    ann_vol = r_n_series.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(252)
    full_vol = float(r_n_series.std() * np.sqrt(252))
    init_w = min(1.0, target_vol / full_vol) if full_vol > 0 else 0.0

    target_w = (target_vol / ann_vol).clip(upper=1.0).shift(1).fillna(init_w).clip(0.0, 1.0).to_numpy()

    cost_fn = lambda dw: partial_rebalance_cost_bps(dw, pot_gbp, include_spread=include_spread)
    return _run_blend(r_nasdaq, r_cash, target_w, cadence_days, deadband, cost_fn, inp)


def vxn_scaled_blend(
    inp: Inputs,
    enter_at: float = 23.0,
    full_weight_at: float = 15.0,
    cadence_days: int = _CADENCE_DAYS,
    deadband: float = _DEADBAND,
    pot_gbp: float = POT_GBP,
    include_spread: bool = False,
) -> Run:
    """Nasdaq weight scales linearly from 0 at VXN=enter_at to 1 at VXN=full_weight_at.

    Weight formula: clamp((enter_at - VXN) / (enter_at - full_weight_at), 0, 1).
    Uses last intraday VXN reading per day, lagged by 1 day to avoid look-ahead.
    Rebalanced weekly when drift exceeds deadband; cost = partial_rebalance_cost_bps.
    """
    dr = _daily_simple_ret(inp)
    r_nasdaq = dr[:, _NASDAQ]
    r_cash = dr[:, _CASH]

    n_days = len(inp.days)
    daily_vxn = np.full(n_days, np.nan)
    for d in range(n_days):
        mask = inp.day_codes == d
        valid = inp.vxn[mask]
        valid = valid[~np.isnan(valid)]
        if len(valid) > 0:
            daily_vxn[d] = valid[-1]

    raw_w = (enter_at - daily_vxn) / (enter_at - full_weight_at)
    target_w = pd.Series(raw_w).clip(0.0, 1.0).shift(1).fillna(0.0).to_numpy()

    cost_fn = lambda dw: partial_rebalance_cost_bps(dw, pot_gbp, include_spread=include_spread)
    return _run_blend(r_nasdaq, r_cash, target_w, cadence_days, deadband, cost_fn, inp)


def sma_vol_target(
    inp: Inputs,
    target_vol: float = 0.05,
    sma_window: int = _SMA_WINDOW,
    vol_window: int = _VOL_WINDOW,
    cadence_days: int = _CADENCE_DAYS,
    deadband: float = _DEADBAND,
    pot_gbp: float = POT_GBP,
    include_spread: bool = False,
) -> Run:
    """Vol-target 5% when Nasdaq is above its SMA200; 100% cash otherwise.

    Both the SMA signal and vol-target weight are lagged by 1 day (no look-ahead).
    Cost per rebalance = partial_rebalance_cost_bps (same as vol_target).
    """
    dr = _daily_simple_ret(inp)
    r_nasdaq = dr[:, _NASDAQ]
    r_cash = dr[:, _CASH]
    r_n_series = pd.Series(r_nasdaq)

    px = (1.0 + r_n_series).cumprod()
    sma = px.rolling(sma_window, min_periods=sma_window).mean()
    above_sma = (px > sma).shift(1).fillna(False)

    ann_vol = r_n_series.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(252)
    full_vol = float(r_n_series.std() * np.sqrt(252))
    init_w = min(1.0, target_vol / full_vol) if full_vol > 0 else 0.0

    tw_raw = (target_vol / ann_vol).clip(upper=1.0).shift(1).fillna(init_w).clip(0.0, 1.0)
    target_w = tw_raw.where(above_sma, 0.0).to_numpy()

    cost_fn = lambda dw: partial_rebalance_cost_bps(dw, pot_gbp, include_spread=include_spread)
    return _run_blend(r_nasdaq, r_cash, target_w, cadence_days, deadband, cost_fn, inp)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _register_windows() -> None:
    WINDOWS.setdefault("y26", (None, None))
    WINDOWS.setdefault("vxn_era", ("2007-11-20", None))
    WINDOWS.setdefault("recent", ("2024-03-25", None))


def comparator_table(inp: Inputs) -> dict[str, pd.DataFrame]:
    """Build all comparator rows for the three reference windows.

    Returns a dict keyed by window label.
    """
    _register_windows()

    tiers_deployed = tiers_vxn_deadband(inp.vxn, 23.0, 24.0)
    from .intraday_engine import tiers_vxn_vix
    ladder = tiers_vxn_vix(np.full(len(inp.vix), np.nan), inp.vix, 0.0, 15.0, 17.5)
    nasdaq_mask = tiers_vxn_deadband(inp.vxn, 23.0, 24.0) == 0
    tiers_deployed_full = np.where(nasdaq_mask, 0, ladder)

    runs = {
        "Deployed (VXN 23/24 + VIX ladder) 13bps": simulate(inp, tiers_deployed_full, cost_bps=DEFAULT_COST_BPS),
        "Two-state VXN 23/24 (current rule) 13bps": simulate(inp, tiers_deployed, cost_bps=DEFAULT_COST_BPS),
        "Static 30% Nasdaq / 70% cash (no cost)": static_blend(inp, 0.30),
        "Vol-target 5% wkly 2% deadband": vol_target(inp, 0.05),
        "Vol-target 10% wkly 2% deadband": vol_target(inp, 0.10),
        "SMA200 + vol-target 5%": sma_vol_target(inp, 0.05),
        "B&H Nasdaq": buy_and_hold(inp, "NASDAQ"),
    }

    result = {}
    for win_label, win_key in _WINDOWS_REF:
        rows = []
        for label, run in runs.items():
            s = window_stats(run, win_key)
            n_days = s["n_days"]
            years = n_days / 252
            cagr = ((1 + s["ret_pct"] / 100) ** (1 / years) - 1) * 100 if years > 0 else float("nan")
            rows.append({"strategy": label, "cagr_pct": cagr, **s})
        result[win_label] = pd.DataFrame(rows)
    return result


def main() -> None:
    _register_windows()
    print("... loading 26yr dataset ...")
    inp = load_inputs()

    # Cost model info
    print(f"\nCost model (partial rebalance at £{POT_GBP:,.0f} pot):")
    for dw in (0.02, 0.05, 0.10):
        c = partial_rebalance_cost_bps(dw, POT_GBP, include_spread=False)
        print(f"  |dw|={dw:.0%}: {c:.2f} bps NAV per rebalance  (£{abs(dw)*POT_GBP:.0f} traded, ~£{abs(dw)*POT_GBP*c/1e4:.2f} cost)")

    print("\n... running comparators ...")
    tables = comparator_table(inp)

    print(f"\nCOMPARATORS ON INTRADAY ENGINE  (fill=same_bar; xSharpe=excess-over-cash)")
    print(f"  Vol-target cadence: weekly 5d, 2% deadband; SMA window: 200d")
    hdr = f"  {'Strategy':<42} {'Ret%':>7} {'CAGR%':>6} {'xSh':>6} {'maxDD%':>7} {'reb/yr':>7}"
    sep = "  " + "-" * (len(hdr) - 2)

    for win_label, df in tables.items():
        years = df.iloc[0]["n_days"] / 252
        print(f"\n--- {win_label}  ({years:.1f} yrs) ---")
        print(hdr)
        print(sep)
        for _, row in df.iterrows():
            xsh = row["xsharpe"]
            xsh_str = f"{xsh:+.2f}" if not np.isnan(xsh) else "  nan"
            print(
                f"  {row['strategy']:<42} "
                f"{row['ret_pct']:>7.1f} {row['cagr_pct']:>6.1f} "
                f"{xsh_str:>6} {row['max_dd_pct']:>7.1f} "
                f"{row['sw_per_yr']:>7.1f}"
            )


if __name__ == "__main__":
    main()
