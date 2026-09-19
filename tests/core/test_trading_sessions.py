from __future__ import annotations

import pandas as pd
import pytest

from Strategy_Auto_Trader.core.trading_sessions import LSE, US, bar_end, session_bars


def _utc(*stamps: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(stamps, tz="UTC")


class TestBarEnd:
    def test_lse_full_hour_bar_ends_one_hour_later(self):
        # 09:00 London in January == 09:00 UTC
        assert bar_end(_utc("2024-01-10 09:00"), LSE)[0] == pd.Timestamp("2024-01-10 10:00", tz="UTC")

    def test_lse_last_bar_capped_at_session_close_in_winter_and_summer(self):
        winter = bar_end(_utc("2024-01-10 16:00"), LSE)[0]
        summer = bar_end(_utc("2024-07-10 15:00"), LSE)[0]  # 16:00 BST
        assert winter == pd.Timestamp("2024-01-10 16:30", tz="UTC")
        assert summer == pd.Timestamp("2024-07-10 15:30", tz="UTC")

    @pytest.mark.parametrize("start,end", [
        ("2024-01-10 14:30", "2024-01-10 15:00"),  # 09:30 ET, EST: half-hour bar ends on the hour
        ("2024-07-10 13:30", "2024-07-10 14:00"),  # 09:30 ET, EDT
    ])
    def test_us_open_half_bar_ends_on_the_hour(self, start, end):
        assert bar_end(_utc(start), US)[0] == pd.Timestamp(end, tz="UTC")

    def test_us_last_bar_capped_at_1615_et(self):
        assert bar_end(_utc("2024-07-10 20:00"), US)[0] == pd.Timestamp("2024-07-10 20:15", tz="UTC")

    def test_overnight_global_hours_bar_ends_on_next_hour(self):
        # 03:15 ET (07:15 UTC in summer) -> 08:00 UTC
        assert bar_end(_utc("2024-07-10 07:15"), US)[0] == pd.Timestamp("2024-07-10 08:00", tz="UTC")

    def test_vix_and_lse_bars_share_end_instants(self):
        vix = bar_end(_utc("2024-07-10 13:30", "2024-07-10 14:00"), US)
        lse = bar_end(_utc("2024-07-10 13:00", "2024-07-10 14:00"), LSE)  # 14:00 and 15:00 BST bars
        assert vix[0] == lse[0] == pd.Timestamp("2024-07-10 14:00", tz="UTC")


class TestSessionBars:
    @pytest.mark.parametrize("session", [LSE, US])
    @pytest.mark.parametrize("day", ["2024-01-10", "2024-07-10", "2024-03-28"])
    def test_bars_are_contiguous_and_sum_to_session_length(self, session, day):
        bars = session_bars(pd.Timestamp(day), session)
        assert len(bars) == len(session.starts)
        assert (bars["end"] > bars["start"]).all()
        assert bars["end"].is_monotonic_increasing
        assert ((bars["end"] - bars["start"]).sum() / pd.Timedelta(hours=1)) == pytest.approx(session.total_hours)

    def test_dst_shifts_utc_start_of_lse_session(self):
        winter = session_bars(pd.Timestamp("2024-01-10"), LSE)["start"].iloc[0]
        summer = session_bars(pd.Timestamp("2024-07-10"), LSE)["start"].iloc[0]
        assert winter.hour == 8 and summer.hour == 7
