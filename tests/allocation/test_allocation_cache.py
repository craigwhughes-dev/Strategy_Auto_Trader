"""MultiTierAllocationManager VIX/VXN reads: delegate to timer-refreshed IndexFeeds.

Replaces the earlier once-a-day-cache tests. That behaviour was a live-trading bug: the tier was
fixed by the first reading of the day and never reacted to intraday VIX/VXN moves.
"""

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd

from Strategy_Auto_Trader.allocation.allocation_manager import MultiTierAllocationManager
from Strategy_Auto_Trader.allocation.index_feed import IndexFeed


class _Clock:
    def __init__(self, start: str):
        self.now = pd.Timestamp(start, tz="UTC")

    def __call__(self) -> pd.Timestamp:
        return self.now


def _bars(*rows: tuple[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"Close": [c for _, c in rows]}, index=pd.DatetimeIndex([t for t, _ in rows], tz="UTC"))


def _manager(clock: _Clock) -> MultiTierAllocationManager:
    mgr = MultiTierAllocationManager()
    mgr._vix_feed = IndexFeed("VIX", clock=clock)
    mgr._vxn_feed = IndexFeed("VXN", clock=clock)
    return mgr


def test_vix_and_vxn_are_read_independently():
    mgr = _manager(_Clock("2026-09-16 15:05"))
    vix = mgr._get_vix_current(Mock(return_value=_bars(("2026-09-16 14:00", 16.0))))
    vxn = mgr._get_vxn_current(Mock(return_value=_bars(("2026-09-16 14:00", 22.0))))
    assert (vix, vxn) == (16.0, 22.0)


def test_repeated_cycles_within_the_refresh_window_do_not_refetch():
    clock = _Clock("2026-09-16 15:05")
    mgr = _manager(clock)
    fetcher = Mock(return_value=_bars(("2026-09-16 14:00", 16.0)))
    for _ in range(5):
        mgr._get_vix_current(fetcher)
    fetcher.assert_called_once()


def test_fetch_failure_on_cold_start_returns_none():
    mgr = _manager(_Clock("2026-09-16 15:05"))
    assert mgr._get_vix_current(Mock(return_value=None)) is None


def test_signal_changes_intraday_when_the_reading_moves():
    clock = _Clock("2026-09-16 09:05")
    mgr = _manager(clock)
    calm = _bars(("2026-09-16 07:15", 14.0), ("2026-09-16 08:00", 14.2))
    stressed = _bars(("2026-09-16 07:15", 14.0), ("2026-09-16 08:00", 14.2), ("2026-09-16 09:00", 24.0))
    fetcher = Mock(side_effect=[calm, stressed])

    morning = mgr.signal(pd.Timestamp("2026-09-16").date(), vxn=None, vix=mgr._get_vix_current(fetcher), verbose=False)
    clock.now += pd.Timedelta(hours=2)
    later = mgr.signal(pd.Timestamp("2026-09-16").date(), vxn=None, vix=mgr._get_vix_current(fetcher), verbose=False)

    assert morning.tier == 2 and later.tier == 4
