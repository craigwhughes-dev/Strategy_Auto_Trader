"""Data-outage handling: both VIX and VXN lost for longer than the limit forces the allocator to cash."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from Strategy_Auto_Trader.allocation.allocation_manager import MultiTierAllocationManager
from Strategy_Auto_Trader.allocation.index_feed import MAX_OUTAGE_SECONDS, IndexFeed

TODAY = date(2026, 9, 21)
LIMIT = 4 * 3600
CONFIG = Path(__file__).resolve().parents[2] / "config" / "overnight_strategy.json"
PRICES = {"EQGB.L": 10.0, "CSH2.L": 100.0}


class _Clock:
    def __init__(self, start: str):
        self.now = pd.Timestamp(start, tz="UTC")

    def __call__(self) -> pd.Timestamp:
        return self.now

    def advance(self, **kw) -> None:
        self.now += pd.Timedelta(**kw)


def _bars(close: float = 20.0) -> pd.DataFrame:
    return pd.DataFrame({"Close": [close]}, index=pd.DatetimeIndex(["2026-09-16 08:00"], tz="UTC"))


def _plain_manager() -> MultiTierAllocationManager:
    return MultiTierAllocationManager(vxn_threshold=23.0, vxn_exit_threshold=24.0, lower_tiers_enabled=False)


def _manager(clock: _Clock) -> MultiTierAllocationManager:
    mgr = _plain_manager()
    mgr._vix_feed = IndexFeed("VIX", max_outage_seconds=LIMIT, clock=clock)
    mgr._vxn_feed = IndexFeed("VXN", max_outage_seconds=LIMIT, clock=clock)
    return mgr


def _read_both(mgr: MultiTierAllocationManager, vix_fetch, vxn_fetch) -> None:
    mgr._get_vix_current(vix_fetch)
    mgr._get_vxn_current(vxn_fetch)


def test_default_outage_limit_is_four_hours():
    assert MAX_OUTAGE_SECONDS == LIMIT
    assert MultiTierAllocationManager().outage_limit_hours == 4


def test_shipped_config_uses_four_hour_limit():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["tier_allocation"]["index_max_outage_seconds"] == LIMIT


class TestFeedIsLost:
    def test_not_lost_before_first_attempt(self):
        assert IndexFeed("VIX", clock=_Clock("2026-09-16 10:05")).is_lost() is False

    def test_cold_start_with_failing_fetch_is_lost_only_after_the_limit(self):
        clock = _Clock("2026-09-16 10:05")
        feed = IndexFeed("VIX", refresh_seconds=300, max_outage_seconds=LIMIT, clock=clock)
        fetcher = Mock(return_value=None)
        feed.current(fetcher)
        clock.advance(hours=3, minutes=59)
        feed.current(fetcher)
        assert feed.is_lost() is False
        clock.advance(minutes=2)
        feed.current(fetcher)
        assert feed.is_lost() is True

    def test_clock_runs_from_last_success(self):
        clock = _Clock("2026-09-16 10:05")
        feed = IndexFeed("VIX", refresh_seconds=300, max_outage_seconds=LIMIT, clock=clock)
        fetcher = Mock(side_effect=[_bars()] + [None] * 100)
        feed.current(fetcher)
        clock.advance(hours=4, minutes=1)
        assert feed.current(fetcher) is None
        assert feed.is_lost() is True

    def test_recovery_clears_lost(self):
        clock = _Clock("2026-09-16 10:05")
        feed = IndexFeed("VIX", refresh_seconds=300, max_outage_seconds=LIMIT, clock=clock)
        fetcher = Mock(side_effect=[_bars(), None, _bars()])
        feed.current(fetcher)
        clock.advance(hours=5)
        feed.current(fetcher)
        assert feed.is_lost() is True
        clock.advance(minutes=6)
        feed.current(fetcher)
        assert feed.is_lost() is False


class TestManagerOutage:
    def test_both_feeds_must_be_lost(self):
        clock = _Clock("2026-09-16 10:05")
        mgr = _manager(clock)
        ok, bad = Mock(return_value=_bars()), Mock(return_value=None)
        _read_both(mgr, ok, bad)  # VIX fine, VXN failing from the start
        clock.advance(hours=5)
        _read_both(mgr, ok, bad)
        assert mgr.data_outage() == (False, False)

    def test_outage_is_reported_once_then_clears(self):
        clock = _Clock("2026-09-16 10:05")
        mgr = _manager(clock)
        bad = Mock(return_value=None)
        _read_both(mgr, bad, bad)
        assert mgr.data_outage() == (False, False)  # cold start, within the limit
        clock.advance(hours=4, minutes=1)
        _read_both(mgr, bad, bad)
        assert mgr.data_outage() == (True, True)
        assert mgr.data_outage() == (True, False)  # same episode, no repeat alert
        assert mgr.app_status_dict()["data_outage"] is True
        clock.advance(minutes=6)
        _read_both(mgr, Mock(return_value=_bars()), bad)  # VIX recovers
        assert mgr.data_outage() == (False, False)
        assert mgr.app_status_dict()["data_outage"] is False

    def test_second_episode_alerts_again(self):
        clock = _Clock("2026-09-16 10:05")
        mgr = _manager(clock)
        bad, good = Mock(return_value=None), Mock(return_value=_bars())
        _read_both(mgr, bad, bad)
        clock.advance(hours=5)
        _read_both(mgr, bad, bad)
        assert mgr.data_outage()[1] is True
        clock.advance(minutes=6)
        _read_both(mgr, good, good)
        assert mgr.data_outage() == (False, False)
        clock.advance(hours=5)
        _read_both(mgr, bad, bad)
        assert mgr.data_outage() == (True, True)


class TestSignalWithNoReadings:
    def test_no_readings_while_holding_nasdaq_sells_to_cash(self):
        mgr = _plain_manager()
        mgr.current_asset = "EQGB.L"
        orders = mgr.rebalance(TODAY, None, None, PRICES, 0.0, {"EQGB.L": 500})
        assert [(o.action, o.ticker) for o in orders] == [("SELL", "EQGB.L"), ("BUY", "CSH2.L")]
        assert mgr.current_asset == "CSH2.L"

    def test_no_readings_while_in_cash_places_nothing(self):
        mgr = _plain_manager()
        mgr.current_asset = "CSH2.L"
        assert mgr.rebalance(TODAY, None, None, PRICES, 0.0, {"CSH2.L": 50}) == []

    def test_after_forced_cash_reentry_needs_the_entry_level_not_the_exit_level(self):
        mgr = _plain_manager()
        mgr.current_asset = "EQGB.L"
        mgr.rebalance(TODAY, None, None, PRICES, 0.0, {"EQGB.L": 500})
        assert mgr.signal(TODAY, 23.5, 20.0, verbose=False).target_asset == "CSH2.L"
        assert mgr.signal(TODAY, 22.9, 20.0, verbose=False).target_asset == "EQGB.L"

    def test_missing_vix_does_not_break_a_nasdaq_entry(self):
        """The buy reason used to format VIX=None with :.1f and raise."""
        mgr = _plain_manager()
        mgr.current_asset = "CSH2.L"
        orders = mgr.rebalance(TODAY, 20.0, None, PRICES, 20_000.0, {"CSH2.L": 0})
        buy = next(o for o in orders if o.action == "BUY")
        assert buy.ticker == "EQGB.L" and "VIX=n/a" in buy.reason
