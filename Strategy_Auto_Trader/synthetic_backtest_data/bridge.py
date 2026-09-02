"""Brownian-bridge synthesis of intraday hourly bars between two real daily
closes.

The path between `prev_close` and `next_close` is random (driven by daily
vol), but is constructed so it lands exactly on `next_close` at the last
bar — real daily closes stay ground truth, only the shape of the day is
synthetic.

Volume: when the caller supplies the day's real total (from Stooq/IBKR daily
data), it is distributed across the day's synthetic bars proportional to
each bar's |Close - Open| — a bigger synthetic intraday move gets a bigger
share of the real daily volume, so volume-ratio gates see something that
tracks price action instead of a flat number every bar. This is a proxy,
not real intraday volume — actual trading has its own volume shape (e.g. a
U-curve around the open/close auctions) unrelated to price-move size within
the day. Falls back to a flat placeholder (`_PLACEHOLDER_VOLUME`) when no
real daily volume is supplied, or when it's missing/zero (real daily data
can have zero-volume days, e.g. thin historical UK listings) — a constant
in that case makes any volume-ratio gate see a neutral ratio of 1.0 rather
than fabricating a directional signal that has no basis at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_PLACEHOLDER_VOLUME = 100_000
_INTRABAR_NOISE_FRACTION = 0.25  # fraction of sigma used to widen High/Low past Open/Close


def generate_bridge_path(
    prev_close: float,
    next_close: float,
    sigma: float,
    n_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return `n_steps` synthetic closes bridging prev_close -> next_close
    in log-space. Index 0 of the result is the first synthetic hourly bar
    (prev_close itself is not re-emitted); the last element always equals
    `next_close` exactly.
    """
    l0, l1 = np.log(prev_close), np.log(next_close)

    increments = rng.normal(loc=0.0, scale=sigma, size=n_steps)
    w = np.cumsum(increments)
    bridge = w - (np.arange(1, n_steps + 1) / n_steps) * w[-1]

    t = np.arange(1, n_steps + 1) / n_steps
    log_path = l0 + (l1 - l0) * t + bridge
    path = np.exp(log_path)
    path[-1] = next_close
    return path


def build_hourly_ohlcv_for_day(
    prev_close: float,
    next_close: float,
    sigma: float,
    n_bars: int,
    rng: np.random.Generator,
    daily_volume: float | None = None,
) -> pd.DataFrame:
    """Synthesize one trading day's hourly OHLCV rows. Caller supplies
    `sigma` (must be finite — filter out NaN/inf rolling-vol warmup rows
    before calling) and sets the returned frame's index.

    daily_volume: the day's real total volume, distributed across bars
    proportional to |Close - Open| (see module docstring). None, NaN, or
    <= 0 falls back to a flat _PLACEHOLDER_VOLUME for every bar.
    """
    close = generate_bridge_path(prev_close, next_close, sigma, n_bars, rng)
    open_ = np.empty(n_bars)
    open_[0] = prev_close
    open_[1:] = close[:-1]

    noise = np.abs(rng.normal(loc=0.0, scale=sigma * _INTRABAR_NOISE_FRACTION, size=n_bars))
    bar_max = np.maximum(open_, close)
    bar_min = np.minimum(open_, close)
    high = bar_max * (1 + noise)
    low = bar_min * (1 - noise)

    if daily_volume is not None and np.isfinite(daily_volume) and daily_volume > 0:
        moves = np.abs(close - open_)
        move_total = moves.sum()
        # A near-zero total (e.g. sigma=0 / prev_close==next_close) is
        # floating-point noise from the log/exp round-trip in
        # generate_bridge_path, not real signal — dividing by it would
        # amplify that noise into a fake volume split. 1e-9 is far below
        # any genuine price move at typical equity price scales.
        weights = moves / move_total if move_total > 1e-9 else np.full(n_bars, 1.0 / n_bars)
        volume = weights * daily_volume
    else:
        volume = np.full(n_bars, _PLACEHOLDER_VOLUME)

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })
