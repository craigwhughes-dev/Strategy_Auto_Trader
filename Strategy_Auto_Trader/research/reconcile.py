"""X0b of PLAN_EXIT_GENERALISATION.md: reconcile the event register's peak/trough closes.

The register (``data/cache/yahoo_index/event_register.csv``) was built once from Yahoo's chart
endpoint and never cross-checked. X2 reads each episode's *peak close* and *trough close* off those
same series to form the ``protection`` metric, whose denominator ``peak - trough`` is small for
shallow episodes — so a single stale or shifted print there swings the score. X0b checks exactly
those two prices per episode before anything downstream trusts them.

Two tracks, because a genuinely independent free daily source back to 1970 exists for almost none
of these markets:

  Track A - independent second source, where one is reachable.
      * ^IXIC  vs FRED ``NASDAQCOM`` - full history from 1971-02-05 (verified 2026-09-29).
      * ^GSPC  vs FRED ``SP500``     - only 2016 onward (FRED caps this series at ~10 years).
      FRED's keyless ``fredgraph.csv`` endpoint serves full history *only when no browser
      User-Agent is sent* - a ``Mozilla`` UA makes it stall, and the ``fred/series/observations``
      API (not used here) is the one that silently truncates to ~3 years. See the data-trap notes
      in PLAN_EXIT_GENERALISATION.md.

  Track B - intrinsic-defect scan, for all 12 indices / all episodes. Catches the failure modes
      this project has already hit without needing a second source:
        * depth-consistency  - recompute depth_pct from the cached closes at the register's
                               peak/trough dates; must match the stored value. Confirms the closes
                               X2 will read are the ones that built the register (no re-fetch drift).
        * unit-shift         - a step change in price magnitude (ISF.L daily was ~100x too large
                               before 2004-04-16); shows as one adjacent-day ratio far from 1.
        * isolated-spike     - a single bar whose move dwarfs its neighbours and reverts.
        * stale-run          - a run of identical consecutive closes (real indices rarely repeat).

An episode is *dropped from this plan* only on a **detected defect** — a Track-A disagreement beyond
tolerance, or a Track-B flag landing on (or adjacent to) its peak/trough date. "No independent source
exists for this market" is **not** a drop; it is recorded as ``intrinsic-only`` reconciliation. This
reinterprets the plan's terser "cannot be reconciled is dropped" wording, which read literally would
drop every non-^IXIC episode and gut the study; the distinction is surfaced in the driver's report.

Research only. Nothing here touches the trading pipeline.
"""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_FREDGRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}&coed={end}"

# Yahoo index symbol -> (FRED series id, earliest date the FRED series is trustworthy).
# ^GSPC's FRED series (SP500) only reaches back ~10 years, so it confirms the recent episodes only.
SECOND_SOURCE: dict[str, tuple[str, str]] = {
    "^IXIC": ("NASDAQCOM", "1971-01-01"),
    "^GSPC": ("SP500", "2016-01-01"),
}


def fetch_fred_daily(series_id: str, start: str = "1900-01-01", end: str = "2100-01-01", timeout: int = 40) -> pd.Series:
    """Full daily history of a FRED series via the keyless ``fredgraph.csv`` endpoint.

    Sends **no** User-Agent on purpose: a ``Mozilla`` UA makes FRED stall, and this endpoint (unlike
    the ``fred/series/observations`` API) returns the full requested range without a key.
    """
    url = _FREDGRAPH.format(sid=series_id, start=start, end=end)
    req = urllib.request.Request(url)  # deliberately no headers
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", "replace")
    df = pd.read_csv(io.StringIO(text))
    df.columns = ["date", "value"]
    df = df[df["value"] != "."].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
    df["value"] = df["value"].astype(float)
    return pd.Series(df["value"].to_numpy(), index=pd.DatetimeIndex(df["date"]), name=series_id).sort_index()


def _close_at(close: pd.Series, when: pd.Timestamp, max_lag: int = 4) -> tuple[float | None, int]:
    """Close on ``when`` if present, else the nearest earlier bar within ``max_lag`` calendar days.

    Returns (value, lag_in_calendar_days). lag 0 is an exact-date hit; a miss returns (None, -1).
    Calendar-day lag (not bar position) so a holiday/weekend peak resolves to the trading day just
    before it, while a genuinely absent stretch (bad data) exceeds the tolerance and misses.
    """
    idx = close.index
    if when in idx:
        return float(close.loc[when]), 0
    prior = idx[idx <= when]
    if len(prior) == 0:
        return None, -1
    nearest = prior[-1]
    lag = (when - nearest).days
    if lag > max_lag:
        return None, -1
    return float(close.loc[nearest]), lag


@dataclass
class EpisodeReconResult:
    market: str
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    stored_depth_pct: float
    recomputed_depth_pct: float | None
    depth_ok: bool
    second_source: str | None          # FRED series id, or None if no independent source
    peak_rel_err: float | None         # |cached-fred|/fred at peak, None if not cross-checked
    trough_rel_err: float | None
    second_source_ok: bool | None      # None => intrinsic-only (no source), True/False => cross-checked
    defect_flags: list[str] = field(default_factory=list)

    @property
    def dropped(self) -> bool:
        """Drop only on a detected defect, never merely for lacking an independent source."""
        if not self.depth_ok:
            return True
        if self.second_source_ok is False:
            return True
        return bool(self.defect_flags)

    @property
    def confirmation(self) -> str:
        if self.second_source_ok is True:
            return "second-source"
        if self.second_source_ok is False:
            return "second-source-FAIL"
        return "intrinsic-only"


def scan_unit_shift(close: pd.Series, ratio: float = 5.0) -> pd.DatetimeIndex:
    """Dates where the close steps by more than ``ratio``x versus the prior bar (a units cutover)."""
    px = close.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = px[1:] / px[:-1]
    hit = np.flatnonzero((rel > ratio) | (rel < 1.0 / ratio))
    return close.index[hit + 1]


def scan_spike(close: pd.Series, window: int = 21, k: float = 10.0, revert_frac: float = 0.6) -> pd.DatetimeIndex:
    """Isolated bad prints: a large log return that **reverts** the next bar.

    A crash trough is a huge single-day move too, but it does not revert - the price stays down and
    recovers gradually. A data error (a single mis-scaled or fat-fingered print) goes out and
    (roughly) straight back, so its return and the next bar's return are both large and opposite in
    sign. Requiring that reversion is what stops this flagging genuine capitulation days such as the
    Hang Seng's -21.75% Tiananmen fall on 1989-06-05, which is real and must not be dropped.

    A bar ``t`` is flagged when |ret(t)| > ``k`` x rolling-median-|ret| AND ret(t+1) is opposite in
    sign and at least ``revert_frac`` of ret(t) in magnitude.
    """
    lr = np.log(close.to_numpy(dtype=float))
    dlr = np.diff(lr)  # dlr[i] is the return into bar i+1
    if len(dlr) < 2:
        return close.index[:0]
    mad = pd.Series(np.abs(dlr)).rolling(window, min_periods=window // 2, center=True).median().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.abs(dlr) / mad
    big = np.nan_to_num(z, nan=0.0) > k
    r_in = dlr[:-1]
    r_out = dlr[1:]
    reverts = (np.sign(r_in) != np.sign(r_out)) & (np.abs(r_out) >= revert_frac * np.abs(r_in))
    hit = np.flatnonzero(big[:-1] & reverts)
    return close.index[hit + 1]  # +1 maps dlr index to the spiking bar


def scan_stale(close: pd.Series, run: int = 5) -> pd.DatetimeIndex:
    """Start dates of runs of ``run`` or more identical consecutive closes."""
    px = close.to_numpy(dtype=float)
    same = np.concatenate([[False], px[1:] == px[:-1]])
    out: list[int] = []
    i = 0
    n = len(px)
    while i < n:
        if same[i]:
            j = i
            while j < n and same[j]:
                j += 1
            if (j - i + 1) >= run:
                out.append(i - 1)
            i = j
        else:
            i += 1
    return close.index[out] if out else close.index[:0]


def _near(target: pd.Timestamp, flagged: pd.DatetimeIndex, bars: int, idx: pd.DatetimeIndex) -> bool:
    """True if ``target`` is within ``bars`` positions of any flagged date on the series index."""
    if len(flagged) == 0 or target not in idx:
        return False
    tpos = idx.get_loc(target)
    for f in flagged:
        if f in idx and abs(idx.get_loc(f) - tpos) <= bars:
            return True
    return False


def reconcile_episode(
    row: pd.Series,
    close: pd.Series,
    fred: pd.Series | None,
    fred_from: pd.Timestamp | None,
    *,
    depth_tol_pct: float = 0.10,
    source_tol: float = 0.01,
    neighbourhood_bars: int = 3,
    precomputed_flags: dict[str, pd.DatetimeIndex] | None = None,
) -> EpisodeReconResult:
    """Reconcile one register row's peak and trough closes. See module docstring for the two tracks."""
    peak_date = pd.to_datetime(row["peak_date"], utc=True).normalize()
    trough_date = pd.to_datetime(row["trough_date"], utc=True).normalize()
    stored_depth = float(row["depth_pct"])

    peak_close, _ = _close_at(close, peak_date)
    trough_close, _ = _close_at(close, trough_date)

    # Track B - depth consistency
    if peak_close and trough_close and peak_close != 0:
        recomputed = round((trough_close / peak_close - 1.0) * 100.0, 2)
        depth_ok = abs(recomputed - stored_depth) <= depth_tol_pct
    else:
        recomputed = None
        depth_ok = False

    # Track B - intrinsic defect neighbourhood
    flags = precomputed_flags or {
        "unit_shift": scan_unit_shift(close),
        "spike": scan_spike(close),
        "stale": scan_stale(close),
    }
    defect_flags: list[str] = []
    for name, dates in flags.items():
        if _near(peak_date, dates, neighbourhood_bars, close.index):
            defect_flags.append(f"{name}@peak")
        if _near(trough_date, dates, neighbourhood_bars, close.index):
            defect_flags.append(f"{name}@trough")

    # Track A - independent second source
    sid = None
    peak_err = trough_err = None
    source_ok: bool | None = None
    if fred is not None and fred_from is not None:
        sid = fred.name
        if peak_date >= fred_from and trough_date >= fred_from:
            fp, _ = _close_at(fred, peak_date)
            ft, _ = _close_at(fred, trough_date)
            if fp and ft and peak_close and trough_close:
                peak_err = abs(peak_close - fp) / fp
                trough_err = abs(trough_close - ft) / ft
                source_ok = (peak_err <= source_tol) and (trough_err <= source_tol)
        # episodes before fred_from stay intrinsic-only (source_ok None)

    return EpisodeReconResult(
        market=str(row["market"]),
        peak_date=peak_date,
        trough_date=trough_date,
        stored_depth_pct=stored_depth,
        recomputed_depth_pct=recomputed,
        depth_ok=depth_ok,
        second_source=sid,
        peak_rel_err=peak_err,
        trough_rel_err=trough_err,
        second_source_ok=source_ok,
        defect_flags=defect_flags,
    )
