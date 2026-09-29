"""Scoring for PLAN_CRASH_PHASES.md's first cut: does a confirmation indicator mark recoveries
without also marking the way down?

The statistic is deliberately coarse, because the oracle-lag curve (BACKTEST_LOG 2026-09-28 late)
showed sub-monthly precision earns nothing: at 15 trading days late an oracle still keeps 70-91%
of its value, and the curves are non-monotonic below a month. So this does not measure distance
to a boundary. It asks two yes/no questions per episode:

    hit          did the indicator fire in [trough, trough + window] trading days?
    false alarm  did it fire earlier in the episode, at a point from which the market
                 still fell at least `fa_decline` before reaching the trough?

A false alarm so defined is unambiguous: the indicator said "turn" and the market went on to lose
another tenth of its value. This is exactly where T1's 50-day-SMA trigger failed, and it is the
clause that makes the test hard.

**Shocks, not episodes, are the unit of observation.** Equity markets co-move at rho ~0.7-0.9 in
crashes, so 2008 in twelve markets is one observation twelve times over. Per-market results are
averaged within a shock, then across shocks, so every shock carries equal weight however many
markets happen to record it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from Strategy_Auto_Trader.research.crash_phases import Episode

HIT_WINDOW = 42
FA_DECLINE = 0.10
N_PERMUTATIONS = 10_000


@dataclass(frozen=True)
class EpisodeScore:
    hit: bool
    false_alarm: bool
    n_firings: int


@dataclass(frozen=True)
class Score:
    hit_rate: float
    false_alarm_rate: float
    n_shocks: int

    @property
    def edge(self) -> float:
        """The primary statistic: hit rate minus false-alarm rate."""
        return self.hit_rate - self.false_alarm_rate


def score_episode(
    firings: np.ndarray,
    close: pd.Series,
    episode: Episode,
    hit_window: int = HIT_WINDOW,
    fa_decline: float = FA_DECLINE,
) -> EpisodeScore:
    """Hit and false-alarm flags for one indicator on one episode."""
    if hit_window < 1:
        raise ValueError(f"hit_window must be >= 1, got {hit_window}")
    px = close.to_numpy(dtype=float)
    t, end = episode.trough_idx, episode.end_idx
    hi = min(t + hit_window, end)
    hit = bool(firings[t : hi + 1].any())

    pre = np.flatnonzero(firings[episode.peak_idx : t]) + episode.peak_idx
    # A firing is a false alarm only if a further fall of at least fa_decline followed it.
    false_alarm = bool(any(episode.trough_close <= (1.0 - fa_decline) * px[i] for i in pre))
    return EpisodeScore(hit=hit, false_alarm=false_alarm, n_firings=int(firings[episode.peak_idx : end + 1].sum()))


def aggregate(rows: pd.DataFrame) -> Score:
    """Average per-market flags within each shock, then across shocks (equal weight per shock)."""
    if rows.empty:
        return Score(hit_rate=float("nan"), false_alarm_rate=float("nan"), n_shocks=0)
    per_shock = rows.groupby("shock_id")[["hit", "false_alarm"]].mean()
    return Score(
        hit_rate=float(per_shock["hit"].mean()),
        false_alarm_rate=float(per_shock["false_alarm"].mean()),
        n_shocks=int(len(per_shock)),
    )


def rotate(firings: np.ndarray, lo: int, hi: int, shift: int) -> np.ndarray:
    """Circularly rotate the firing pattern within [lo, hi], leaving the rest untouched.

    Rotation is the null this test needs: it preserves how many times the indicator fires and how
    those firings are spaced, destroying only *where* in the episode they sit. A uniform random
    draw would not — any volatility-clustered indicator beats uniform noise trivially, which would
    make a significant result meaningless.
    """
    out = firings.copy()
    seg = firings[lo : hi + 1]
    if seg.size:
        out[lo : hi + 1] = np.roll(seg, shift % seg.size)
    return out


def rotation_tables(
    firings: np.ndarray,
    close: pd.Series,
    episode: Episode,
    hit_window: int = HIT_WINDOW,
    fa_decline: float = FA_DECLINE,
) -> tuple[np.ndarray, np.ndarray]:
    """Hit and false-alarm outcomes for *every* possible rotation of one episode's firings.

    Returns two boolean arrays of length `span`, indexed by rotation shift, so the permutation test
    is a lookup rather than a rescore. Shift 0 reproduces the observed result exactly. This is an
    optimisation only: rescoring 10,000 rotations of 123 episodes directly takes hours, and the
    identity in `test_rotation_table_shift_zero_matches_direct_scoring` pins the two together.
    """
    px = close.to_numpy(dtype=float)
    lo, t, end = episode.peak_idx, episode.trough_idx, episode.end_idx
    span = end - lo + 1
    offsets = np.flatnonzero(firings[lo : end + 1])
    if span <= 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)

    positions = np.arange(span)
    hit_mask = (positions >= t - lo) & (positions <= min(t + hit_window, end) - lo)
    # A pre-trough firing is a false alarm only if at least fa_decline of further fall followed.
    fa_mask = (positions < t - lo) & (episode.trough_close <= (1.0 - fa_decline) * px[lo : end + 1])

    if offsets.size == 0:
        return np.zeros(span, dtype=bool), np.zeros(span, dtype=bool)
    rotated = (offsets[:, None] + positions[None, :]) % span
    return hit_mask[rotated].any(axis=0), fa_mask[rotated].any(axis=0)


def permutation_p_value(
    observed_edge: float,
    per_episode: list[tuple[np.ndarray, pd.Series, Episode, int]],
    n_permutations: int = N_PERMUTATIONS,
    seed: int = 20260928,
    hit_window: int = HIT_WINDOW,
    fa_decline: float = FA_DECLINE,
) -> tuple[float, np.ndarray]:
    """One-sided p-value for `observed_edge` under the rotation null, plus the null distribution.

    `per_episode` carries (firings, close, episode, shock_id) so each draw is rescored with the
    same shock weighting as the observed statistic.
    """
    if not per_episode:
        return float("nan"), np.zeros(0)
    rng = np.random.default_rng(seed)
    shock_ids = np.array([shock for _, _, _, shock in per_episode])
    uniq, inverse = np.unique(shock_ids, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(uniq))

    tables = [rotation_tables(f, c, e, hit_window, fa_decline) for f, c, e, _ in per_episode]
    spans = np.array([len(h) for h, _ in tables])
    hits = np.zeros((len(per_episode), n_permutations), dtype=bool)
    fas = np.zeros((len(per_episode), n_permutations), dtype=bool)
    for i, (h, fa) in enumerate(tables):
        if spans[i] == 0:
            continue
        draws = rng.integers(spans[i], size=n_permutations)
        hits[i], fas[i] = h[draws], fa[draws]

    # Mean within each shock, then across shocks — the weighting `aggregate` applies.
    per_shock_hit = np.zeros((len(uniq), n_permutations))
    per_shock_fa = np.zeros((len(uniq), n_permutations))
    np.add.at(per_shock_hit, inverse, hits.astype(float))
    np.add.at(per_shock_fa, inverse, fas.astype(float))
    per_shock_hit /= counts[:, None]
    per_shock_fa /= counts[:, None]
    null = per_shock_hit.mean(axis=0) - per_shock_fa.mean(axis=0)

    # +1 in numerator and denominator: an observed value can never be reported as p = 0.
    p = float((np.sum(null >= observed_edge) + 1) / (n_permutations + 1))
    return p, null
