"""Brownian-bridge synthesis of intraday hourly bars between two real daily
closes.

The path between `prev_close` and `next_close` is random (driven by daily
vol), but is constructed so it lands exactly on `next_close` at the last
bar — real daily closes stay ground truth, only the shape of the day is
synthetic.

Volume is a flat placeholder (`_PLACEHOLDER_VOLUME`), not modeled — a
constant value makes any volume-ratio gate see a neutral ratio of 1.0 rather
than fabricating a directional volume signal that has no real basis here.
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
) -> pd.DataFrame:
    """Synthesize one trading day's hourly OHLCV rows. Caller supplies
    `sigma` (must be finite — filter out NaN/inf rolling-vol warmup rows
    before calling) and sets the returned frame's index."""
    close = generate_bridge_path(prev_close, next_close, sigma, n_bars, rng)
    open_ = np.empty(n_bars)
    open_[0] = prev_close
    open_[1:] = close[:-1]

    noise = np.abs(rng.normal(loc=0.0, scale=sigma * _INTRABAR_NOISE_FRACTION, size=n_bars))
    bar_max = np.maximum(open_, close)
    bar_min = np.minimum(open_, close)
    high = bar_max * (1 + noise)
    low = bar_min * (1 - noise)

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": _PLACEHOLDER_VOLUME,
    })
