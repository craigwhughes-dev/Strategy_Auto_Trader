"""Tests for allocation_manager.py — focused on the VVIX-banded tier 1 gate added 2026-09-30.

Does not attempt a full retrofit of the pre-existing untested signal()/rebalance() logic (no test
file existed for this module before this one) — scoped to the new behavior plus the regression
guard that the "base" band (the default) reproduces the original VXN-only rule exactly.
"""

from __future__ import annotations

from datetime import date

import pytest

from Strategy_Auto_Trader.allocation.allocation_manager import MultiTierAllocationManager


def _mgr(**overrides) -> MultiTierAllocationManager:
    kwargs = dict(
        vxn_threshold=23.0,
        vxn_exit_threshold=24.0,
        lower_tiers_enabled=False,
        vvix_edge_low=76.0,
        vvix_edge_high=122.0,
        vxn_calm_enter=25.0,
        vxn_calm_exit=26.0,
        vxn_stressed_enter=21.0,
        vxn_stressed_exit=22.0,
    )
    kwargs.update(overrides)
    return MultiTierAllocationManager(**kwargs)


class TestNasdaqPair:
    def test_base_band_uses_the_plain_vxn_thresholds(self):
        m = _mgr()
        assert m._nasdaq_pair("base") == (23.0, 24.0)

    def test_calm_band_uses_the_wider_pair(self):
        m = _mgr()
        assert m._nasdaq_pair("calm") == (25.0, 26.0)

    def test_stressed_band_uses_the_narrower_pair(self):
        m = _mgr()
        assert m._nasdaq_pair("stressed") == (21.0, 22.0)

    def test_unknown_band_rejected(self):
        m = _mgr()
        with pytest.raises(ValueError):
            m._nasdaq_pair("panic")

    def test_unconfigured_calm_and_stressed_pairs_default_to_the_base_pair(self):
        """No vxn_calm_*/vxn_stressed_* configured at all -> every band is identical to base,
        i.e. a config with no VVIX keys behaves exactly like before this feature existed."""
        m = MultiTierAllocationManager(vxn_threshold=23.0, vxn_exit_threshold=24.0)
        assert m._nasdaq_pair("calm") == m._nasdaq_pair("base") == m._nasdaq_pair("stressed") == (23.0, 24.0)


class TestSignalBandAware:
    def test_default_vvix_band_is_base_and_matches_pre_vvix_behavior(self):
        m = _mgr()
        sig = m.signal(date(2026, 1, 5), vxn=24.5, vix=None, verbose=False)
        assert sig.vvix_band == "base"
        assert sig.tier == 4  # 24.5 > base enter 23 -> cash, unaffected by VVIX config present

    def test_calm_band_admits_a_vxn_reading_the_base_band_would_reject(self):
        m = _mgr()
        sig = m.signal(date(2026, 1, 5), vxn=24.5, vix=None, vvix_band="calm", verbose=False)
        assert sig.tier == 1
        assert sig.target_asset == "EQGB.L"
        assert sig.vvix_band == "calm"

    def test_stressed_band_rejects_a_vxn_reading_the_base_band_would_admit(self):
        m = _mgr()
        sig = m.signal(date(2026, 1, 5), vxn=22.0, vix=None, vvix_band="stressed", verbose=False)
        assert sig.tier == 4  # base would have admitted (22 <= 23); stressed enter is 21

    def test_deadband_persistence_holds_within_a_band(self):
        """Same mechanic as the plain VXN deadband: once held, stays held until VXN exceeds the
        band's own exit edge, not the entry edge."""
        m = _mgr()
        sig1 = m.signal(date(2026, 1, 5), vxn=20.0, vix=None, vvix_band="calm", verbose=False)
        assert sig1.target_asset == "EQGB.L"
        m.current_asset = "EQGB.L"
        sig2 = m.signal(date(2026, 1, 6), vxn=25.5, vix=None, vvix_band="calm", verbose=False)
        assert sig2.target_asset == "EQGB.L"  # 25.5 <= calm exit 26, still held
        sig3 = m.signal(date(2026, 1, 7), vxn=26.5, vix=None, vvix_band="calm", verbose=False)
        assert sig3.target_asset != "EQGB.L"  # 26.5 > calm exit 26, now exits


class TestFromConfigVvixKeys:
    def test_accepts_the_new_vvix_keys(self):
        m = MultiTierAllocationManager.from_config({
            "vxn_threshold": 23, "vxn_exit_threshold": 24, "lower_tiers_enabled": False,
            "vix_tier1": 15, "vix_tier2": 17.5, "commission_pct": 0.05,
            "index_refresh_seconds": 300, "index_max_outage_seconds": 14400,
            "vvix_edge_low": 76, "vvix_edge_high": 122, "vvix_confirm_days": 3,
            "vxn_calm_enter": 25, "vxn_calm_exit": 26,
            "vxn_stressed_enter": 21, "vxn_stressed_exit": 22,
        })
        assert (m.vvix_edge_low, m.vvix_edge_high, m.vvix_confirm_days) == (76.0, 122.0, 3.0)
        assert m._nasdaq_pair("calm") == (25.0, 26.0)

    def test_still_rejects_a_genuinely_unknown_key(self):
        with pytest.raises(ValueError):
            MultiTierAllocationManager.from_config({"vxn_threshold": 23, "vvix_typo_key": 1})

    def test_inverted_calm_pair_rejected(self):
        with pytest.raises(ValueError):
            MultiTierAllocationManager(vxn_calm_enter=26.0, vxn_calm_exit=25.0)


class TestAppStatusDictBandAware:
    """app_status_dict() must reflect the *active* confirmed band's enter/exit pair, not always
    the base pair — regression guard for a bug where tier 1's enter_gate_value/exit_gate_value
    were hardcoded to vxn_threshold/vxn_exit_threshold regardless of vvix_band."""

    def test_tier1_enter_exit_follow_the_confirmed_band(self):
        m = _mgr()
        m.signal(date(2026, 1, 5), vxn=24.5, vix=None, vvix_band="calm", verbose=False)
        tier1 = m.app_status_dict()["tier_allocation"]["tiers"][0]
        assert (tier1["enter_gate_value"], tier1["exit_gate_value"]) == (25.0, 26.0)

    def test_tier1_enter_exit_default_to_base_pair(self):
        m = _mgr()
        m.signal(date(2026, 1, 5), vxn=24.5, vix=None, verbose=False)
        tier1 = m.app_status_dict()["tier_allocation"]["tiers"][0]
        assert (tier1["enter_gate_value"], tier1["exit_gate_value"]) == (23.0, 24.0)

    def test_vvix_block_exposes_all_three_bands(self):
        m = _mgr()
        m.signal(date(2026, 1, 5), vxn=24.5, vix=None, verbose=False)
        thresholds = m.app_status_dict()["tier_allocation"]["vvix"]["vxn_thresholds"]
        assert thresholds == {
            "calm": {"enter": 25.0, "exit": 26.0},
            "base": {"enter": 23.0, "exit": 24.0},
            "stressed": {"enter": 21.0, "exit": 22.0},
        }

    def test_inverted_stressed_pair_rejected(self):
        with pytest.raises(ValueError):
            MultiTierAllocationManager(vxn_stressed_enter=22.0, vxn_stressed_exit=21.0)

    def test_inverted_vvix_edges_rejected(self):
        with pytest.raises(ValueError):
            MultiTierAllocationManager(vvix_edge_low=122.0, vvix_edge_high=76.0)

    def test_non_positive_confirm_days_rejected(self):
        with pytest.raises(ValueError):
            MultiTierAllocationManager(vvix_confirm_days=0)


class TestAppStatusLatestIndicators:
    """The dashboard follows the latest indicator readings (refreshed hourly outside the decision
    cycle), while the reading the last decision acted on stays under "decision"."""

    def test_display_uses_latest_readings_and_keeps_decision_readings(self):
        m = _mgr()
        m.signal(date(2026, 10, 30), vxn=30.02, vix=21.79, verbose=False)
        m.set_indicator_readings(vix=19.5, vxn=22.5, vvix=106.43)
        status = m.app_status_dict({"vvix_last_value": 98.01})["tier_allocation"]
        assert (status["vix_current"], status["vxn_current"]) == (19.5, 22.5)
        assert status["tiers"][0]["current_value"] == 22.5
        assert status["tiers"][0]["passes"] is True  # 22.5 <= base enter 23
        assert status["decision"] == {"vix": 21.79, "vxn": 30.02, "vvix_band": "base"}
        assert status["vvix"]["current"] == 106.43
        assert status["vvix"]["last_close"] == 98.01
        assert status["action"] == m._last_action  # still the last real decision
        assert status["indicators_as_of"] is not None

    def test_falls_back_to_decision_readings_before_first_refresh(self):
        m = _mgr()
        m.signal(date(2026, 10, 30), vxn=30.02, vix=21.79, verbose=False)
        status = m.app_status_dict({"vvix_last_value": 98.01})["tier_allocation"]
        assert (status["vix_current"], status["vxn_current"]) == (21.79, 30.02)
        assert status["vvix"]["current"] == 98.01

    def test_band_shown_is_the_one_the_next_decision_will_use(self):
        m = _mgr(vvix_confirm_days=3)
        m.signal(date(2026, 10, 30), vxn=24.5, vix=None, verbose=False)
        status = m.app_status_dict({"vvix_calm_streak_days": 0, "vvix_stressed_streak_days": 3})["tier_allocation"]
        assert status["vvix_band"] == "stressed"
        assert status["decision"]["vvix_band"] == "base"
        assert status["tiers"][0]["enter_gate_value"] == 21.0
