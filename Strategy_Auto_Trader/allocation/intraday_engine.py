"""Intraday-faithful engine for the 4-tier VXN/VIX rotation.

Reality being modelled (live daemon, LSE-only funds):
  - decisions happen on the LSE hourly bar grid, from the LATEST VIX/VXN print at that instant
    (a print counts once its bar has ended — see core.trading_sessions);
  - VXN prints only in US hours, so in the London morning it is the prior close, held stale;
    VIX has London-morning prints only from 2016 (real IBKR data), earlier it is stale too;
  - no trades while the LSE is shut.

Fill modes:
  same_bar  trade at the price of the bar that ends when the signal is read (live daemon polls
            every 60s and acts immediately, so this is the closer match)
  next_bar  trade one LSE bar later — conservative order-latency check

Data: data_synthetic/hourly_spliced/ (real IBKR hourly where it exists, correlated-bridge
before that; see synthetic_backtest_data/build_intraday_dataset.py). Tier 2 is IUSA as a
stand-in for a UK-listed S&P 500 UCITS fund.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .multi_tier_allocator_4tier import MultiTierAllocator4Tier

_DATASET = Path(__file__).resolve().parents[2] / "data_synthetic" / "hourly_spliced"
_LONDON = "Europe/London"
ASSETS = ("NASDAQ", "SP500", "FTSE", "CASH")  # tier 1..4
_CASH = 3
DEFAULT_COST_BPS = 13.0

# Eras. Threshold selection uses TRAIN only; bridged years are never used to choose thresholds.
WINDOWS: dict[str, tuple[str | None, str | None]] = {
    "all": (None, None),
    "bridged": (None, "2007-11-19"),
    "real": ("2007-11-20", None),
    "train": ("2007-11-20", "2019-12-31"),
    "test": ("2020-01-01", None),
    "morning_vix": ("2016-01-01", None),
}


@dataclass(frozen=True)
class Inputs:
    grid: pd.DatetimeIndex   # UTC LSE bar ends
    log_ret: np.ndarray      # (n, 4) log return of each asset over the interval ending at grid[k]
    vix: np.ndarray          # latest VIX print at grid[k] (NaN before the first)
    vxn: np.ndarray
    day_codes: np.ndarray    # London calendar-day index of grid[k]
    days: pd.DatetimeIndex   # the London trading days, sorted


@dataclass(frozen=True)
class Run:
    daily: np.ndarray        # simple daily portfolio return
    switches: np.ndarray     # switches per day
    days: pd.DatetimeIndex
    cash_daily: np.ndarray | None = None  # same-day return of the cash asset, for excess-over-cash Sharpe


def _close_by_end(dataset: Path, name: str) -> pd.Series:
    df = pd.read_csv(dataset / f"{name}.csv", index_col=0)
    ends = pd.to_datetime(df["bar_end_utc"], utc=True)
    return pd.Series(df["Close"].to_numpy(), index=pd.DatetimeIndex(ends)).groupby(level=0).last().sort_index()


def load_inputs(dataset: Path = _DATASET) -> Inputs:
    prices = {n: _close_by_end(dataset, n) for n in ASSETS}
    grid = prices["FTSE"].index
    aligned = np.column_stack([np.log(prices[n].reindex(grid, method="ffill").to_numpy()) for n in ASSETS])
    log_ret = np.vstack([np.zeros((1, 4)), np.diff(aligned, axis=0)])
    vix = _close_by_end(dataset, "VIX").reindex(grid, method="ffill").to_numpy()
    vxn = _close_by_end(dataset, "VXN").reindex(grid, method="ffill").to_numpy()
    london_day = grid.tz_convert(_LONDON).tz_localize(None).normalize()
    codes, days = pd.factorize(london_day)
    return Inputs(grid, log_ret, vix, vxn, codes, pd.DatetimeIndex(days))



def tiers_vxn_vix(vxn: np.ndarray, vix: np.ndarray, vxn_cut: float, vix_1: float, vix_2: float) -> np.ndarray:
    """Tier index per bar: 0 Nasdaq if VXN<=cut, else 1 S&P if VIX<=vix_1, 2 FTSE if VIX<=vix_2, else cash. NaN never qualifies."""
    tiers = np.full(len(vix), _CASH)
    tiers[vix <= vix_2] = 2
    tiers[vix <= vix_1] = 1
    tiers[vxn <= vxn_cut] = 0
    return tiers


def tiers_vix_only(vix: np.ndarray, cut_1: float, cut_2: float, cut_3: float) -> np.ndarray:
    """Same ladder driven by VIX alone: Nasdaq<=cut_1<S&P<=cut_2<FTSE<=cut_3<cash."""
    tiers = np.full(len(vix), _CASH)
    tiers[vix <= cut_3] = 2
    tiers[vix <= cut_2] = 1
    tiers[vix <= cut_1] = 0
    return tiers


def tiers_vxn_deadband(vxn: np.ndarray, enter_at: float, exit_above: float) -> np.ndarray:
    """Nasdaq-or-cash with a deadband on VXN: enter when VXN <= enter_at, then hold until VXN > exit_above.

    exit_above == enter_at is the plain single-threshold rule. Between the two the previous state
    persists, so small wobbles around the threshold do not trade. A NaN reading triggers neither
    side and so keeps the current state; before any reading the state is cash.
    """
    if exit_above < enter_at:
        raise ValueError(f"exit_above ({exit_above}) must be >= enter_at ({enter_at})")
    event = np.zeros(len(vxn), dtype=np.int8)
    event[vxn <= enter_at] = 1
    event[vxn > exit_above] = -1
    last_event = np.maximum.accumulate(np.where(event != 0, np.arange(len(vxn)), -1))
    in_market = (last_event >= 0) & (event[np.maximum(last_event, 0)] == 1)
    return np.where(in_market, 0, _CASH)


def _daily_from_steps(inp: Inputs, step: np.ndarray) -> np.ndarray:
    return np.expm1(np.bincount(inp.day_codes, weights=step, minlength=len(inp.days)))


def _cash_daily(inp: Inputs) -> np.ndarray:
    step = inp.log_ret[:, _CASH].copy()
    step[0] = 0.0
    return _daily_from_steps(inp, step)


def simulate(inp: Inputs, tiers: np.ndarray, fill: str = "same_bar", cost_bps: float = DEFAULT_COST_BPS) -> Run:
    n = len(tiers)
    held = tiers.copy()
    if fill == "next_bar":
        held[0] = _CASH
        held[1:] = tiers[:-1]
    elif fill != "same_bar":
        raise ValueError(f"unknown fill mode {fill!r}")
    step = np.zeros(n)
    step[1:] = inp.log_ret[np.arange(1, n), held[:-1]]
    switched = np.zeros(n, dtype=bool)
    switched[1:] = held[1:] != held[:-1]
    step += switched * np.log1p(-cost_bps / 1e4)
    switches = np.bincount(inp.day_codes, weights=switched, minlength=len(inp.days))
    return Run(_daily_from_steps(inp, step), switches, inp.days, _cash_daily(inp))


def buy_and_hold(inp: Inputs, asset: str) -> Run:
    step = inp.log_ret[:, ASSETS.index(asset)].copy()
    step[0] = 0.0
    return Run(_daily_from_steps(inp, step), np.zeros(len(inp.days)), inp.days, _cash_daily(inp))


def _bounds(days: pd.DatetimeIndex, window: str) -> slice:
    lo, hi = WINDOWS[window]
    start = 0 if lo is None else int(days.searchsorted(pd.Timestamp(lo), side="left"))
    stop = len(days) if hi is None else int(days.searchsorted(pd.Timestamp(hi), side="right"))
    return slice(start, stop)


def _excess_sharpe(rets: np.ndarray, cash: np.ndarray | None) -> float:
    """Annualised Sharpe of returns over the cash asset. Raw Sharpe rewards idle cash (income, ~no vol), so
    cash-heavy strategies look better than they are; this removes that."""
    if cash is None:
        return float("nan")
    excess = rets - cash
    std = np.std(excess, ddof=1)
    return float(np.mean(excess) / std * np.sqrt(252)) if std > 0 else float("nan")


def window_stats(run: Run, window: str) -> dict:
    sl = _bounds(run.days, window)
    rets = run.daily[sl]
    if len(rets) < 2:
        return {"n_days": len(rets), "sharpe": np.nan, "xsharpe": np.nan, "sortino": np.nan, "ret_pct": np.nan, "max_dd_pct": np.nan, "sw_per_yr": np.nan}
    summary = MultiTierAllocator4Tier._compute_summary(rets, 1.0, float(np.prod(1 + rets)))
    years = len(rets) / 252
    return {
        "n_days": len(rets),
        "sharpe": summary["sharpe"],
        "xsharpe": _excess_sharpe(rets, run.cash_daily[sl] if run.cash_daily is not None else None),
        "sortino": summary["sortino"],
        "ret_pct": summary["total_return_pct"],
        "max_dd_pct": summary["max_drawdown_pct"],
        "sw_per_yr": float(run.switches[sl].sum() / years),
    }


def annual_returns(run: Run) -> pd.Series:
    daily = pd.Series(run.daily, index=run.days)
    return (daily.groupby(daily.index.year).apply(lambda r: float(np.prod(1 + r) - 1) * 100)).round(1)
