"""P1 of PLAN_CRASH_PHASES.md: mechanical episode detection and phase labelling.

An *episode* is a peak-to-trough drawdown of at least `min_depth` on a daily close
series, plus the following `horizon` trading days (12 months). It runs from the
running high before the fall to `horizon` bars after the trough, and the next episode
is then sought from a fresh running high.

Closing an episode on *recovery to the prior peak* instead — the first thing tried —
is unusable: on the Nasdaq the March 2000 peak was not regained until November 2014,
so 2008 and 2009 were swallowed inside one 83% episode and disappeared from the
register. Episodes are therefore closed by elapsed time after the trough, which is
what the plan's own wording specifies. `recovered` is kept as a descriptive flag only
and plays no part in closing an episode.

Within an episode three phase markers are placed:

    capitulation  a window of `capit_days` trading days centred on the trough
    stabilization trough -> first close regaining `stab_retrace` of the peak-to-trough range
    confirmation  first close regaining `confirm_retrace` of that range

These labels use hindsight and are meant to. They are the ground truth that
*causal* indicators are scored against in P5; the indicators themselves may only
use data available at the time. Nothing here is a trading signal.

Two selection biases are inherent to the definition and are not defects to be
fixed, only reported (Copilot panel finding, 2026-09-28):
  - Only completed drawdowns qualify. A dip that never reached `min_depth` is not an
    episode, so the register cannot answer "is this dip the start of a crash".
  - An episode still open at the end of the series has no confirmed trough. Such an
    episode is returned with `recovered=False` and must be excluded from scoring
    unless `post_trough_days` is large enough to make the trough safe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Owner-ratified 2026-09-28. Briefly raised to 0.25 to target deep episodes, then reverted: depth here is
# measured on EQQQ in GBP (unhedged), and sterling's ~10% fall in March 2020 compresses COVID to -21.6% against
# the USD index's ~-30%, so a 0.25 floor excluded the single best recovery event for a currency reason.
MIN_DEPTH = 0.20
HORIZON = 252
CAPIT_DAYS = 10
STAB_RETRACE = 0.20
CONFIRM_RETRACE = 0.50


@dataclass(frozen=True)
class Episode:
    """One peak-to-trough-to-recovery drawdown, as positional indices into the close series."""

    peak_idx: int
    trough_idx: int
    end_idx: int                 # last bar of the episode: trough + horizon, clamped to the series
    recovery_idx: int | None     # first close back at or above the peak; None if it never happens. Descriptive only
    peak_close: float
    trough_close: float
    recovered: bool

    @property
    def depth_pct(self) -> float:
        """Peak-to-trough return, negative (e.g. -0.35 for a 35% drawdown)."""
        return self.trough_close / self.peak_close - 1.0

    @property
    def rng(self) -> float:
        """Peak-to-trough price range, positive."""
        return self.peak_close - self.trough_close


@dataclass(frozen=True)
class Phases:
    """Phase markers for one episode, as positional indices into the same close series.

    `stabilization_idx` / `confirmation_idx` are None when the episode's data ends before
    that retracement level is regained.
    """

    capitulation_start: int
    capitulation_end: int
    stabilization_idx: int | None
    confirmation_idx: int | None


def find_episodes(
    close: pd.Series,
    min_depth: float = MIN_DEPTH,
    horizon: int = HORIZON,
    post_trough_days: int = 0,
) -> list[Episode]:
    """Every drawdown of at least `min_depth` in `close`, non-overlapping, in time order.

    An episode opens when the close falls `min_depth` below the running high, and closes
    `horizon` bars after its trough. The trough keeps moving while the price makes new
    lows, so a deeper leg inside an open drawdown extends that episode instead of opening
    a second one. Scanning then resumes from a fresh running high, which is what lets a
    later crash from a lower high be found at all.

    `post_trough_days` drops episodes whose trough sits closer than that many bars to the
    end of the series, where the trough is not yet established.
    """
    if min_depth <= 0.0:
        raise ValueError(f"min_depth must be positive, got {min_depth}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    px = close.to_numpy(dtype=float)
    n = len(px)
    out: list[Episode] = []
    i = 0
    while i < n:
        peak_idx = i
        t = i + 1
        while t < n:
            if px[t] >= px[peak_idx]:
                peak_idx = t
            elif px[t] <= (1.0 - min_depth) * px[peak_idx]:
                break
            t += 1
        if t >= n:
            break
        trough_idx = t
        j = t + 1
        while j < n and j <= trough_idx + horizon:
            if px[j] < px[trough_idx]:
                trough_idx = j
            j += 1
        end_idx = min(trough_idx + horizon, n - 1)
        after = np.flatnonzero(px[trough_idx:] >= px[peak_idx])
        recovery_idx = int(trough_idx + after[0]) if after.size else None
        out.append(
            Episode(
                peak_idx=peak_idx,
                trough_idx=trough_idx,
                end_idx=end_idx,
                recovery_idx=recovery_idx,
                peak_close=float(px[peak_idx]),
                trough_close=float(px[trough_idx]),
                recovered=recovery_idx is not None,
            )
        )
        i = end_idx + 1
    if post_trough_days > 0:
        out = [e for e in out if e.trough_idx + post_trough_days < n]
    return out


def label_phases(
    episode: Episode,
    close: pd.Series,
    capit_days: int = CAPIT_DAYS,
    stab_retrace: float = STAB_RETRACE,
    confirm_retrace: float = CONFIRM_RETRACE,
) -> Phases:
    """Phase markers for one episode. Retracement levels are measured off the trough.

    A marker is only placed if its level is regained *inside* the episode; a recovery that
    takes longer than the episode horizon is reported as None rather than reaching forward
    into the next episode's data.
    """
    if capit_days < 1:
        raise ValueError(f"capit_days must be >= 1, got {capit_days}")
    if not 0.0 < stab_retrace < confirm_retrace < 1.0:
        raise ValueError(f"need 0 < stab_retrace < confirm_retrace < 1, got {stab_retrace}, {confirm_retrace}")
    px = close.to_numpy(dtype=float)
    n = len(px)
    half = capit_days // 2
    lo = max(0, episode.trough_idx - half)
    hi = min(n - 1, episode.trough_idx + (capit_days - 1 - half))
    return Phases(
        capitulation_start=lo,
        capitulation_end=hi,
        stabilization_idx=_first_at_or_above(px, episode, episode.trough_close + stab_retrace * episode.rng),
        confirmation_idx=_first_at_or_above(px, episode, episode.trough_close + confirm_retrace * episode.rng),
    )


def _first_at_or_above(px: np.ndarray, episode: Episode, level: float) -> int | None:
    start, stop = episode.trough_idx, min(episode.end_idx, len(px) - 1)
    hits = np.flatnonzero(px[start : stop + 1] >= level)
    return int(start + hits[0]) if hits.size else None


def episode_register(close: pd.Series, market: str, **kwargs) -> pd.DataFrame:
    """One row per episode with dates rather than indices — the P4 event-register format."""
    eps = find_episodes(close, **kwargs)
    idx = close.index
    rows = []
    for e in eps:
        ph = label_phases(e, close)
        rows.append(
            {
                "market": market,
                "peak_date": idx[e.peak_idx],
                "trough_date": idx[e.trough_idx],
                "end_date": idx[e.end_idx],
                "recovery_date": idx[e.recovery_idx] if e.recovery_idx is not None else pd.NaT,
                "depth_pct": round(e.depth_pct * 100.0, 2),
                "recovered": e.recovered,
                "trough_to_recovery_days": (e.recovery_idx - e.trough_idx) if e.recovery_idx is not None else np.nan,
                "stabilization_date": idx[ph.stabilization_idx] if ph.stabilization_idx is not None else pd.NaT,
                "confirmation_date": idx[ph.confirmation_idx] if ph.confirmation_idx is not None else pd.NaT,
            }
        )
    return pd.DataFrame(rows)


SHOCK_WINDOW_DAYS = 182


def group_into_shocks(register: pd.DataFrame, window_days: int = SHOCK_WINDOW_DAYS) -> pd.DataFrame:
    """Add a `shock_id` collapsing episodes in different markets that are one global event.

    Equity markets co-move at rho ~0.7-0.9 in crashes, so 2008 in Tokyo and 2008 in New York are
    one shock observed twice and must not be counted as two independent observations (the panel's
    central objection to the cross-market design, 2026-09-28). Two episodes join the same shock
    when their troughs fall within `window_days` calendar days; grouping is transitive via
    single-linkage over the sorted trough dates, so a shock whose troughs are spread across
    markets over several months still collapses to one id.

    `window_days` is pre-registered, not tuned: 182 days (six months) is wide enough to hold the
    2008 spread (Oct 2008 to Mar 2009 depending on market) as a single event.
    """
    if register.empty:
        return register.assign(shock_id=pd.Series(dtype="int64"))
    out = register.sort_values("trough_date").reset_index(drop=True).copy()
    troughs = pd.to_datetime(out["trough_date"], utc=True)
    gap = troughs.diff().dt.days.fillna(0)
    out["shock_id"] = (gap > window_days).cumsum().astype("int64")
    return out


def count_independent_shocks(register: pd.DataFrame, window_days: int = SHOCK_WINDOW_DAYS) -> int:
    """Number of distinct global shocks in a register — the figure P4's stop-gate is measured on."""
    return int(group_into_shocks(register, window_days)["shock_id"].nunique())
