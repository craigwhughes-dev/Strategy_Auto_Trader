"""Tests for MultiTierAllocationManager daily cache logic."""

from datetime import date
from unittest.mock import Mock
import pandas as pd
import pytest

from Strategy_Auto_Trader.allocation.allocation_manager import MultiTierAllocationManager


class TestAllocationCache:
    """Test VIX/VXN caching with daily TTL (stale-fetch, fresh-read)."""

    def test_vix_cache_hit_same_day(self):
        """Cache hit: same day returns cached value without fetching."""
        mgr = MultiTierAllocationManager()

        # Pre-populate cache for today
        today = date.today()
        cache_df = pd.DataFrame({
            "Close": [15.0, 15.5, 16.0],
        })
        mgr._vix_cache = cache_df
        mgr._vix_cache_date = today

        # Mock fetcher should NOT be called
        fetcher = Mock(return_value=None)
        result = mgr._get_vix_current(fetcher)

        assert result == 16.0  # Last value
        fetcher.assert_not_called()  # Fetch was skipped

    def test_vix_cache_miss_stale_date(self):
        """Cache miss: stale cache (prior day) triggers fresh fetch."""
        mgr = MultiTierAllocationManager()

        # Pre-populate cache with prior day
        yesterday = date(2026, 1, 1)
        stale_cache_df = pd.DataFrame({"Close": [15.0]})
        mgr._vix_cache = stale_cache_df
        mgr._vix_cache_date = yesterday

        # Mock fetcher returns fresh data
        today = date.today()
        fresh_df = pd.DataFrame({
            "Close": [15.5, 16.0, 16.5],
        })
        fetcher = Mock(return_value=fresh_df)

        result = mgr._get_vix_current(fetcher)

        assert result == 16.5  # Last value of fresh data
        fetcher.assert_called_once()  # Fetch was triggered
        assert mgr._vix_cache_date == today  # Cache updated to today

    def test_vix_cache_cold_start(self):
        """Cold start: no cache, fetches fresh data."""
        mgr = MultiTierAllocationManager()
        assert mgr._vix_cache is None
        assert mgr._vix_cache_date is None

        today = date.today()
        fresh_df = pd.DataFrame({
            "Close": [15.0, 15.5, 16.0],
        })
        fetcher = Mock(return_value=fresh_df)

        result = mgr._get_vix_current(fetcher)

        assert result == 16.0
        fetcher.assert_called_once()
        assert mgr._vix_cache_date == today

    def test_vix_cache_fetch_failure_returns_none(self):
        """Fetch failure: returns None, doesn't update cache."""
        mgr = MultiTierAllocationManager()

        fetcher = Mock(return_value=None)
        result = mgr._get_vix_current(fetcher)

        assert result is None
        assert mgr._vix_cache is None

    def test_vix_cache_extraction_error_handles_gracefully(self):
        """Extraction error (e.g., missing Close column): returns None."""
        mgr = MultiTierAllocationManager()

        today = date.today()
        broken_df = pd.DataFrame({"Open": [15.0]})  # No Close column
        mgr._vix_cache = broken_df
        mgr._vix_cache_date = today

        fetcher = Mock()  # Should not be called
        result = mgr._get_vix_current(fetcher)

        assert result is None
        fetcher.assert_not_called()  # Cache hit attempt before error

    def test_vxn_cache_behaves_identically_to_vix(self):
        """VXN cache has same logic as VIX (hit/miss/error handling)."""
        mgr = MultiTierAllocationManager()

        today = date.today()
        cache_df = pd.DataFrame({"Close": [20.0, 21.0, 22.0]})
        mgr._vxn_cache = cache_df
        mgr._vxn_cache_date = today

        fetcher = Mock(return_value=None)
        result = mgr._get_vxn_current(fetcher)

        assert result == 22.0
        fetcher.assert_not_called()

    def test_separate_cache_for_vix_and_vxn(self):
        """VIX and VXN caches are independent."""
        mgr = MultiTierAllocationManager()

        today = date.today()
        vix_df = pd.DataFrame({"Close": [15.0, 15.5, 16.0]})
        vxn_df = pd.DataFrame({"Close": [20.0, 21.0, 22.0]})

        mgr._vix_cache = vix_df
        mgr._vix_cache_date = today
        mgr._vxn_cache = vxn_df
        mgr._vxn_cache_date = today

        vix_result = mgr._get_vix_current(Mock())
        vxn_result = mgr._get_vxn_current(Mock())

        assert vix_result == 16.0
        assert vxn_result == 22.0

    def test_cache_persistence_across_cycles(self):
        """Cache persists and is reused across multiple calls on same day."""
        mgr = MultiTierAllocationManager()

        today = date.today()
        cache_df = pd.DataFrame({"Close": [15.0, 15.5, 16.0]})
        mgr._vix_cache = cache_df
        mgr._vix_cache_date = today

        fetcher = Mock(return_value=None)

        # First call
        result1 = mgr._get_vix_current(fetcher)
        # Second call on same day
        result2 = mgr._get_vix_current(fetcher)

        assert result1 == result2 == 16.0
        # Fetcher should never have been called
        fetcher.assert_not_called()

    def test_cache_miss_after_midnight_rollover(self):
        """Midnight rollover: cache from prior day is stale, triggers fetch."""
        mgr = MultiTierAllocationManager()

        # Simulate prior day's cache
        yesterday = date(2026, 1, 1)
        old_cache = pd.DataFrame({"Close": [15.0]})
        mgr._vix_cache = old_cache
        mgr._vix_cache_date = yesterday

        # Fresh data for today
        today = date.today()
        new_df = pd.DataFrame({"Close": [16.0, 16.5, 17.0]})
        fetcher = Mock(return_value=new_df)

        # First call of new day
        result = mgr._get_vix_current(fetcher)

        # Should have fetched fresh data
        assert result == 17.0
        fetcher.assert_called_once()
        assert mgr._vix_cache_date == today


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
