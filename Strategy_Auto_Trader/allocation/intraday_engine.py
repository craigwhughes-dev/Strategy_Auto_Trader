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


def _persistent_state(enter_mask: np.ndarray, exit_mask: np.ndarray, default: bool) -> np.ndarray:
    """Accumulate-based persistent binary state, shared by every VXN/VVIX gate in this module.

    enter_mask[i] True -> state True from bar i; exit_mask[i] True -> state False from bar i;
    neither -> state carries forward unchanged (this is what makes NaN / inside-the-deadband bars
    keep prior state). `default` is the state before any event has ever fired.
    """
    event = np.zeros(len(enter_mask), dtype=np.int8)
    event[enter_mask] = 1
    event[exit_mask] = -1
    idx = np.arange(len(enter_mask))
    last_event = np.maximum.accumulate(np.where(event != 0, idx, -1))
    fired = last_event >= 0
    state_at_last_event = event[np.maximum(last_event, 0)] == 1
    return np.where(fired, state_at_last_event, default)


def tiers_vxn_deadband(vxn: np.ndarray, enter_at: float, exit_above: float) -> np.ndarray:
    """Nasdaq-or-cash with a deadband on VXN: enter when VXN <= enter_at, then hold until VXN > exit_above.

    exit_above == enter_at is the plain single-threshold rule. Between the two the previous state
    persists, so small wobbles around the threshold do not trade. A NaN reading triggers neither
    side and so keeps the current state; before any reading the state is cash.
    """
    if exit_above < enter_at:
        raise ValueError(f"exit_above ({exit_above}) must be >= enter_at ({enter_at})")
    in_market = _persistent_state(vxn <= enter_at, vxn > exit_above, default=False)
    return np.where(in_market, 0, _CASH)


def load_vvix_daily() -> pd.Series:
    """Real Cboe VVIX (vol-of-VIX) daily closes, Yahoo ``^VVIX`` 2007-01-03+.

    FRED does not carry VVIX under any series id (VVIXCLS and other guesses all 404 as of
    2026-09-30) — checked before reaching for yfinance. Uses the project's existing scoped
    yfinance exception (`research/index_history.py`; daily index closes only, never used to
    price a trade) rather than inventing a second fetch path. No pre-2006 bridge: unlike VXN,
    there is no published earlier-methodology VVIX series to anchor a synthetic bridge to, and
    real coverage (2007-01-03) already predates every real/train/test window boundary (2007-11-20+).
    """
    from ..research.index_history import load_daily_closes
    return load_daily_closes("^VVIX")


def vvix_for_grid(grid: pd.DatetimeIndex, daily: pd.Series | None = None) -> np.ndarray:
    """Broadcast daily VVIX closes onto an hourly grid: stamp each close at 21:30 UTC (after the
    US session), then hold stale — same convention as ``daily_to_end_series`` in
    multi_tier_26yr_backtest.py. NaN before 2007-01-03 / before the first print."""
    if daily is None:
        daily = load_vvix_daily()
    stamped = pd.Series(
        daily.to_numpy(),
        index=pd.DatetimeIndex([pd.Timestamp(d.date()).tz_localize("UTC") + pd.Timedelta(hours=21, minutes=30) for d in daily.index]),
    ).sort_index()
    return stamped.reindex(grid, method="ffill").to_numpy()


_VIX_DAILY_1990 = Path("data_synthetic/daily/VIX_1990_2026.csv")

# Fit of real VVIX (Yahoo ^VVIX, 2007-01-03+) against an EWMA-realized-vol-of-VIX proxy over
# their full overlap (2007-01-03..2026-09-30, n=4947 trading days): vvix ~= _PROXY_A*proxy + _PROXY_B,
# proxy = 21.7 * annualized EWMA(halflife=12d) stdev of daily log-returns of VIX itself (i.e. vol-of-vol,
# not vol). Swept rolling windows (10/21/30d, corr ~0.51-0.52, insensitive) and EWMA halflives
# (3-30d, corr peaks 0.609-0.610 at halflife 10-15d) before settling on 12d. R^2=0.37, RMSE=12.8
# (real VVIX std=16.2) - a soft/noisy fit, NOT a reconstruction: real VVIX is CBOE-computed
# implied vol of VIX options (forward-looking), this proxy is realized vol of the VIX cash index
# (backward-looking), and the two only agree 33-38% of the time on which days are in each other's
# top decile. Good enough for coarse pre-2007 regime bucketing; do not use for a tight threshold
# gate expecting real-VVIX-like spike timing.
_PROXY_A = 0.261
_PROXY_B = 63.44
_PROXY_EWMA_HALFLIFE = 12


def load_vvix_proxy() -> pd.Series:
    """VVIX daily series back to 1990: real Cboe ^VVIX where it exists (2007-01-03+), a fitted
    EWMA-vol-of-VIX proxy everywhere before that. See _PROXY_A/_PROXY_B comment above for how the
    proxy segment is constructed and how well it actually tracks the real series (corr ~0.61,
    R^2=0.37 on the overlap) - treat proxy-segment values as directional/regime-level only.
    """
    vix_df = pd.read_csv(_VIX_DAILY_1990, index_col=0, parse_dates=True)
    vix_close = vix_df["Close"].sort_index()
    vix_close.index = vix_close.index.tz_localize(None)
    log_ret = np.log(vix_close).diff()
    ewma_vol = log_ret.pow(2).ewm(halflife=_PROXY_EWMA_HALFLIFE).mean().pow(0.5) * np.sqrt(252) * 100
    proxy = (_PROXY_A * ewma_vol + _PROXY_B).rename("vvix_proxy")

    real = load_vvix_daily()
    real_naive = pd.Series(real.to_numpy(), index=real.index.tz_localize(None))
    return real_naive.combine_first(proxy).rename("vvix_proxy")  # real values win wherever both exist


def tiers_vxn_vvix_deadband(vxn: np.ndarray, vvix: np.ndarray, vxn_enter: float, vxn_exit: float, vvix_threshold: float) -> np.ndarray:
    """VXN deadband (Nasdaq-or-cash) with a persistent VVIX veto: once VVIX >= vvix_threshold,
    force cash until VVIX drops back below it. NaN in VVIX never forces cash and never reopens a
    closed gate on its own — same accumulate-based persistence as tiers_vxn_deadband, not a
    stateless ``np.where`` on the raw comparison (a stateless compare would treat NaN as a breach).
    """
    tiers = tiers_vxn_deadband(vxn, vxn_enter, vxn_exit)
    gate_open = _persistent_state(vvix < vvix_threshold, vvix >= vvix_threshold, default=True)
    return np.where(gate_open, tiers, _CASH)


def tiers_vxn_vvix_banded(
    vxn: np.ndarray,
    vvix: np.ndarray,
    vvix_edges: tuple[float, ...],
    vxn_pairs: tuple[tuple[float, float], ...],
    base_band: int,
) -> np.ndarray:
    """Nasdaq-or-cash VXN deadband whose (enter_at, exit_above) pair is picked per-bar from a small
    ladder of discrete VVIX bands, both legs moving together (calm -> both loosen, stressed -> both
    tighten).

    ``vvix_edges`` (strictly ascending) cuts VVIX into len(vvix_edges)+1 bands:
      VVIX < edges[0]               -> vxn_pairs[0]   (calmest)
      edges[i-1] <= VVIX < edges[i] -> vxn_pairs[i]
      VVIX >= edges[-1]             -> vxn_pairs[-1]  (most stressed)

    ``base_band`` names the index of the "no VVIX effect" pair — NaN VVIX (no reading yet, or a
    stale-hold gap) always resolves to base_band, matching tiers_vxn_deadband's own NaN-keeps-state
    convention (here: NaN keeps today's threshold pair fixed at the base rule, not just position).

    Configuring every entry of vxn_pairs identically reproduces tiers_vxn_deadband(vxn, *that pair)
    exactly, for any vvix_edges/base_band/vvix values (including all-NaN) — the ladder only matters
    when the pairs actually differ. Each pair must satisfy exit_above >= enter_at.
    """
    if len(vxn_pairs) != len(vvix_edges) + 1:
        raise ValueError(f"need {len(vvix_edges) + 1} vxn_pairs for {len(vvix_edges)} vvix_edges, got {len(vxn_pairs)}")
    if not 0 <= base_band < len(vxn_pairs):
        raise ValueError(f"base_band {base_band} out of range for {len(vxn_pairs)} bands")
    for enter_at, exit_above in vxn_pairs:
        if exit_above < enter_at:
            raise ValueError(f"exit_above ({exit_above}) must be >= enter_at ({enter_at})")

    nan = np.isnan(vvix)
    band = np.searchsorted(np.asarray(vvix_edges), np.where(nan, 0.0, vvix), side="right")
    band[nan] = base_band

    enter_at_arr = np.asarray([p[0] for p in vxn_pairs])[band]
    exit_above_arr = np.asarray([p[1] for p in vxn_pairs])[band]
    in_market = _persistent_state(vxn <= enter_at_arr, vxn > exit_above_arr, default=False)
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
