"""IndexFeed: latest COMPLETED hourly bar, timer-refreshed, outage-tolerant."""

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest

from Strategy_Auto_Trader.allocation.index_feed import IndexFeed, latest_completed_close


def _bars(*rows: tuple[str, float]) -> pd.DataFrame:
    """Hourly index bars stamped by START (UTC), like the IBKR cache."""
    return pd.DataFrame({"Close": [c for _, c in rows]}, index=pd.DatetimeIndex([t for t, _ in rows], tz="UTC"))


class _Clock:
    def __init__(self, start: str):
        self.now = pd.Timestamp(start, tz="UTC")

    def __call__(self) -> pd.Timestamp:
        return self.now

    def advance(self, **kw) -> None:
        self.now += pd.Timedelta(**kw)


# 2026-09-16 is in US daylight time: 09:30 ET == 13:30 UTC; the 07:15 UTC bar is 03:15 ET (GTH).
DAY = [("2026-09-16 07:15", 17.0), ("2026-09-16 08:00", 17.2), ("2026-09-16 09:00", 17.4), ("2026-09-16 13:30", 16.0)]


class TestLatestCompletedClose:
    def test_bar_counts_only_after_it_has_ended(self):
        df = _bars(*DAY)
        just_before = pd.Timestamp("2026-09-16 09:59:59", tz="UTC")  # 09:00 bar ends at 10:00
        just_after = pd.Timestamp("2026-09-16 10:00:00", tz="UTC")
        assert latest_completed_close(df, just_before)[0] == 17.2
        assert latest_completed_close(df, just_after)[0] == 17.4

    def test_forming_bar_is_ignored(self):
        df = _bars(("2026-09-16 08:00", 17.2), ("2026-09-16 09:00", 99.0))  # 09:00 bar still forming at 09:30
        assert latest_completed_close(df, pd.Timestamp("2026-09-16 09:30", tz="UTC"))[0] == 17.2

    def test_us_open_half_hour_bar_ends_at_the_hour(self):
        df = _bars(("2026-09-16 13:30", 16.0))
        assert latest_completed_close(df, pd.Timestamp("2026-09-16 13:59", tz="UTC")) is None
        assert latest_completed_close(df, pd.Timestamp("2026-09-16 14:00", tz="UTC"))[0] == 16.0

    def test_no_completed_bar_returns_none(self):
        assert latest_completed_close(_bars(("2026-09-16 13:30", 16.0)), pd.Timestamp("2026-09-16 08:00", tz="UTC")) is None

    def test_naive_index_is_treated_as_utc(self):
        df = pd.DataFrame({"Close": [15.0]}, index=pd.DatetimeIndex(["2026-09-16 08:00"]))
        assert latest_completed_close(df, pd.Timestamp("2026-09-16 10:00", tz="UTC"))[0] == 15.0


class TestIndexFeed:
    def test_cold_start_fetches_and_returns_latest_completed(self):
        clock = _Clock("2026-09-16 10:05")
        fetcher = Mock(return_value=_bars(*DAY))
        assert IndexFeed("VIX", clock=clock).current(fetcher) == 17.4
        fetcher.assert_called_once()

    def test_reuses_data_within_refresh_window(self):
        clock = _Clock("2026-09-16 10:05")
        feed = IndexFeed("VIX", refresh_seconds=300, clock=clock)
        fetcher = Mock(return_value=_bars(*DAY))
        feed.current(fetcher)
        clock.advance(seconds=60)
        feed.current(fetcher)
        fetcher.assert_called_once()

    def test_refreshes_after_window_and_picks_up_the_new_bar(self):
        clock = _Clock("2026-09-16 09:55")
        feed = IndexFeed("VIX", refresh_seconds=300, clock=clock)
        old = _bars(*DAY[:3])
        new = _bars(*DAY[:3], ("2026-09-16 10:00", 19.5))
        fetcher = Mock(side_effect=[old, new, new])
        assert feed.current(fetcher) == 17.2  # 09:00 bar not ended yet at 09:55
        clock.advance(minutes=11)  # 10:06 — the 10:00 bar is still forming, so the 09:00 bar's close stands
        assert feed.current(fetcher) == 17.4
        clock.advance(minutes=60)  # 11:06 — the 10:00 bar has ended
        assert feed.current(fetcher) == 19.5
        assert fetcher.call_count == 3

    def test_intraday_spike_is_seen_the_same_day_unlike_a_daily_cache(self):
        clock = _Clock("2026-09-16 08:05")
        feed = IndexFeed("VIX", refresh_seconds=300, clock=clock)
        morning = _bars(("2026-09-16 07:15", 16.0))
        with_spike = _bars(("2026-09-16 07:15", 16.0), ("2026-09-16 08:00", 16.2), ("2026-09-16 09:00", 26.0))
        fetcher = Mock(side_effect=[morning, with_spike])
        assert feed.current(fetcher) == 16.0
        clock.advance(hours=2)
        assert feed.current(fetcher) == 26.0

    def test_failed_refresh_keeps_last_good_value(self):
        clock = _Clock("2026-09-16 10:05")
        feed = IndexFeed("VIX", refresh_seconds=300, clock=clock)
        fetcher = Mock(side_effect=[_bars(*DAY), ConnectionError("gateway down")])
        assert feed.current(fetcher) == 17.4
        clock.advance(minutes=6)
        assert feed.current(fetcher) == 17.4

    def test_outage_beyond_tolerance_returns_none(self):
        clock = _Clock("2026-09-16 10:05")
        feed = IndexFeed("VIX", refresh_seconds=300, max_outage_seconds=3600, clock=clock)
        fetcher = Mock(side_effect=[_bars(*DAY)] + [None] * 20)
        feed.current(fetcher)
        clock.advance(hours=2)
        assert feed.current(fetcher) is None

    def test_failed_fetches_are_spaced_not_retried_every_cycle(self):
        clock = _Clock("2026-09-16 10:05")
        feed = IndexFeed("VIX", refresh_seconds=300, clock=clock)
        fetcher = Mock(return_value=None)
        for _ in range(4):
            feed.current(fetcher)
            clock.advance(seconds=60)
        assert fetcher.call_count == 1

    def test_no_data_ever_returns_none(self):
        assert IndexFeed("VIX", clock=_Clock("2026-09-16 10:05")).current(Mock(return_value=None)) is None

    def test_missing_close_column_returns_none(self):
        broken = pd.DataFrame({"Open": [15.0]}, index=pd.DatetimeIndex(["2026-09-16 08:00"], tz="UTC"))
        assert IndexFeed("VIX", clock=_Clock("2026-09-16 10:05")).current(Mock(return_value=broken)) is None

    @pytest.mark.parametrize("name", ["VIX", "VXN"])
    def test_feeds_are_independent_per_instance(self, name):
        clock = _Clock("2026-09-16 10:05")
        a, b = IndexFeed("VIX", clock=clock), IndexFeed("VXN", clock=clock)
        assert a.current(Mock(return_value=_bars(("2026-09-16 08:00", 17.0)))) == 17.0
        assert b.current(Mock(return_value=_bars(("2026-09-16 08:00", 21.0)))) == 21.0
