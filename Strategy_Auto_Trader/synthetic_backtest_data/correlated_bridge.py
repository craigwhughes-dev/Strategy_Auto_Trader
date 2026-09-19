"""Cross-asset correlated Brownian bridge: intraday paths between real daily closes.

`bridge.py` draws each series' intraday path independently, so a VIX spike and the fund it
should hit are unrelated. For a backtest that reacts to intraday VIX moves that destroys the
one relationship being tested. Here one correlated shock vector is drawn per shared bar-end
instant and each series' bridge is built from its own column. Correlation and per-bar
volatility are measured from real hourly data, not assumed.

What is real and what is assumed:
  - real: every day's close (each path is pinned to it), the cross-correlation and the
    intraday/daily vol ratio (estimated from real hourly bars).
  - assumed: the shape of the day. No overnight gap is modelled (it is spread through the
    session, as in `bridge.py`), and correlation is only applied where two series' bars end at
    the same instant — otherwise the draws are independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.trading_sessions import Session, bar_end, session_bars
from .vol import rolling_daily_vol

_HOUR = pd.Timedelta(hours=1)


@dataclass(frozen=True)
class BridgeParams:
    names: tuple[str, ...]
    corr: np.ndarray
    scale: dict[str, float]


def trailing_sigma(daily_close: pd.Series, window: int = 21) -> pd.Series:
    """Per-day sigma from the PRIOR `window` days. The warmup rows are back-filled."""
    return rolling_daily_vol(daily_close, window).shift(1).bfill()


def _nearest_psd_corr(m: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh((m + m.T) / 2)
    fixed = vecs @ np.diag(np.clip(vals, 1e-6, None)) @ vecs.T
    d = np.sqrt(np.diag(fixed))
    return fixed / np.outer(d, d)


def _local_dates(idx: pd.DatetimeIndex, session: Session) -> pd.DatetimeIndex:
    return idx.tz_convert(session.tz).tz_localize(None).normalize()


def _core_hourly_returns(close: pd.Series, session: Session) -> pd.DataFrame:
    """Full-hour, in-session, non-first-bar log returns with the day's trailing sigma attached."""
    local = close.index.tz_convert(session.tz)
    frame = pd.DataFrame({
        "lc": np.log(close.astype(float).to_numpy()),
        "date": _local_dates(close.index, session),
        "hhmm": local.strftime("%H:%M"),
        "end": bar_end(close.index, session),
    }, index=close.index)
    frame["dur"] = (frame["end"] - frame.index) / _HOUR
    frame["r"] = frame.groupby("date")["lc"].diff()
    contiguous = frame.groupby("date")["end"].shift(1) == frame.index
    daily = close.groupby(frame["date"]).last()
    frame["sigma"] = trailing_sigma(daily).reindex(frame["date"]).to_numpy()
    keep = contiguous & (frame["dur"] == 1.0) & frame["hhmm"].isin(session.starts[1:]) & frame["r"].notna() & (frame["sigma"] > 0)
    return frame[keep]


def estimate_params(
    real_close: dict[str, pd.Series],
    sessions: dict[str, Session],
    since: str,
    min_pairs: int = 200,
) -> BridgeParams:
    """Correlation and intraday/daily vol scale from real hourly bars on/after `since`."""
    names = tuple(real_close)
    rets: dict[str, pd.Series] = {}
    scale: dict[str, float] = {}
    for name in names:
        close = real_close[name]
        core = _core_hourly_returns(close[close.index >= pd.Timestamp(since, tz="UTC")], sessions[name])
        expected = (core["sigma"] ** 2) * core["dur"] / sessions[name].total_hours
        scale[name] = float(np.sqrt((core["r"] ** 2).sum() / expected.sum()))
        rets[name] = pd.Series(core["r"].to_numpy(), index=core["end"]).groupby(level=0).last()
    corr = pd.DataFrame(rets).corr(min_periods=min_pairs).reindex(index=names, columns=names).fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(corr, 1.0)
    return BridgeParams(names, _nearest_psd_corr(corr), scale)


def bridge_path(
    prev_close: float,
    next_close: float,
    sigma_day: float,
    durations: np.ndarray,
    z: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Log-space bridge from prev_close to next_close over bars of the given durations (hours).

    Step variance is proportional to bar duration, so a half-hour bar moves less than a full one.
    The last close equals next_close exactly. sigma_day=0 gives exact log-linear accrual.
    """
    dur = np.asarray(durations, dtype=float)
    total = dur.sum()
    frac = np.cumsum(dur) / total
    walk = np.cumsum(scale * sigma_day * np.sqrt(dur / total) * z)
    l0, l1 = np.log(prev_close), np.log(next_close)
    path = np.exp(l0 + (l1 - l0) * frac + (walk - frac * walk[-1]))
    path[-1] = next_close
    return path


def _prior_close(daily: pd.Series) -> pd.Series:
    return daily.shift(1)


def _assemble(starts: list, ends: list, closes: list) -> pd.DataFrame:
    if not starts:
        return pd.DataFrame(columns=["Close", "bar_end"])
    return pd.DataFrame(
        {"Close": np.concatenate(closes), "bar_end": pd.DatetimeIndex(np.concatenate(ends)).tz_localize("UTC")},
        index=pd.DatetimeIndex(np.concatenate(starts)).tz_localize("UTC"),
    ).sort_index()


def bridge_dataset(
    daily: dict[str, pd.Series],
    sessions: dict[str, Session],
    params: BridgeParams,
    rng: np.random.Generator,
    since: pd.Timestamp,
    until: dict[str, pd.Timestamp],
) -> dict[str, pd.DataFrame]:
    """Correlated hourly bars for each series on its trading days in [since, until[name]).

    `daily` holds tz-naive-date-indexed real (or scaled proxy) daily closes; a trading day
    needs a prior daily close in the same series to bridge from.
    """
    names = params.names
    chol = np.linalg.cholesky(params.corr)
    sigma = {n: trailing_sigma(daily[n]) for n in names}
    prior = {n: _prior_close(daily[n]) for n in names}
    active = {n: daily[n].index[(daily[n].index >= since) & (daily[n].index < until[n])] for n in names}
    out = {n: ([], [], []) for n in names}

    for day in sorted(set().union(*map(set, active.values()))):
        bars = {n: session_bars(day, sessions[n]) for n in names if day in active[n] and pd.notna(prior[n][day])}
        if not bars:
            continue
        ends = sorted(set().union(*[set(b["end"]) for b in bars.values()]))
        shocks = pd.DataFrame(rng.standard_normal((len(ends), len(names))) @ chol.T, index=pd.DatetimeIndex(ends), columns=names)
        for n, b in bars.items():
            path = bridge_path(prior[n][day], daily[n][day], sigma[n][day], ((b["end"] - b["start"]) / _HOUR).to_numpy(),
                               shocks.loc[b["end"], n].to_numpy(), params.scale[n])
            starts, end_l, closes = out[n]
            starts.append(b["start"].dt.tz_convert(None).to_numpy())
            end_l.append(b["end"].dt.tz_convert(None).to_numpy())
            closes.append(path)
    return {n: _assemble(*out[n]) for n in names}


def calibrate_params(
    target: BridgeParams,
    daily: dict[str, pd.Series],
    sessions: dict[str, Session],
    since: pd.Timestamp,
    until: dict[str, pd.Timestamp],
    seed: int,
    iterations: int = 4,
    tol: float = 0.02,
) -> BridgeParams:
    """Inputs to `bridge_dataset` whose OUTPUT reproduces the target correlation and vol scale.

    Pinning each path to its daily close mixes every bar's shock into the others, so bridged
    correlation and intraday vol come out below the requested values (correlation only exists on
    the few shared bars). Bridge a real era from its own daily closes, re-measure with the same
    estimator used for `target`, and nudge the inputs until the two agree.
    """
    names = target.names
    corr_in, scale_in = target.corr.copy(), dict(target.scale)
    for _ in range(iterations):
        trial = BridgeParams(names, corr_in, scale_in)
        out = bridge_dataset(daily, sessions, trial, np.random.default_rng(seed), since, until)
        got = estimate_params({n: out[n]["Close"] for n in names}, sessions, since=str(since.date()))
        gap = target.corr - got.corr
        if np.abs(gap).max() < tol and all(abs(target.scale[n] / got.scale[n] - 1) < tol for n in names):
            break
        corr_in = np.clip(corr_in + gap, -0.98, 0.98)
        np.fill_diagonal(corr_in, 1.0)
        corr_in = _nearest_psd_corr(corr_in)
        scale_in = {n: scale_in[n] * target.scale[n] / got.scale[n] for n in names}
    return BridgeParams(names, corr_in, scale_in)


def accrue_deterministic(
    daily: pd.Series,
    session: Session,
    since: pd.Timestamp,
    until: pd.Timestamp,
) -> pd.DataFrame:
    """Log-linear intraday accrual between daily closes (cash-like series: no intraday shape to invent)."""
    prior = _prior_close(daily)
    starts, ends, closes = [], [], []
    for day in daily.index[(daily.index >= since) & (daily.index < until)]:
        if pd.isna(prior[day]):
            continue
        b = session_bars(day, session)
        dur = ((b["end"] - b["start"]) / _HOUR).to_numpy()
        closes.append(bridge_path(prior[day], daily[day], 0.0, dur, np.zeros(len(dur)), 1.0))
        starts.append(b["start"].dt.tz_convert(None).to_numpy())
        ends.append(b["end"].dt.tz_convert(None).to_numpy())
    return _assemble(starts, ends, closes)
