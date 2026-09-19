"""Trading-session bar grids and the bar-end rule shared by real and bridged intraday data.

IBKR hourly bars are stamped by bar START. Availability of a bar's Close is its END, so
every consumer must work from `bar_end`, never from the start stamp. Session bars all end on
a whole UTC hour (or at the session close), which lets a VIX bar and an LSE-fund bar that end
at the same instant be treated as simultaneous — the VIX bars start at :30/:15 but end on
the hour, so there is no half-hour look-ahead between signal and fill price.

The end rule is the same for real and bridged bars so the splice is seamless:
    end = min(floor_to_hour(start) + 1h, session close that local day)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Session:
    tz: str
    starts: tuple[str, ...]  # local HH:MM bar starts of the regular session
    close: str               # local HH:MM session close (caps the last bar's end)

    @property
    def total_hours(self) -> float:
        first = pd.Timestamp(f"2000-01-01 {self.starts[0]}")
        return (pd.Timestamp(f"2000-01-01 {self.close}") - first).total_seconds() / 3600


LSE = Session("Europe/London", ("08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00"), "16:30")
# VIX/VXN: cash-index hours. The 09:30 bar ends 10:00; the last bar (16:00) is only the final 15 minutes.
US = Session("America/New_York", ("09:30", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00"), "16:15")


def _local_close(local_idx: pd.DatetimeIndex, session: Session) -> pd.DatetimeIndex:
    hh, mm = session.close.split(":")
    naive = local_idx.tz_localize(None).normalize() + pd.Timedelta(hours=int(hh), minutes=int(mm))
    return naive.tz_localize(session.tz)


def bar_end(start_utc: pd.DatetimeIndex, session: Session) -> pd.DatetimeIndex:
    """UTC bar-end for UTC bar-start stamps (vectorised)."""
    raw_end = start_utc.floor("h") + pd.Timedelta(hours=1)
    close_utc = _local_close(start_utc.tz_convert(session.tz), session).tz_convert("UTC")
    return pd.DatetimeIndex(pd.Series(raw_end).where(raw_end <= close_utc, close_utc))


def session_bars(day: pd.Timestamp, session: Session) -> pd.DataFrame:
    """Bar grid (start/end, UTC) for one local trading date."""
    day = pd.Timestamp(day).tz_localize(None).normalize()
    local_starts = pd.DatetimeIndex([day + pd.Timedelta(hours=int(s[:2]), minutes=int(s[3:])) for s in session.starts])
    starts = local_starts.tz_localize(session.tz).tz_convert("UTC")
    return pd.DataFrame({"start": starts, "end": bar_end(starts, session)})
