"""Deadband on the Nasdaq tier, and construction from the `tier_allocation` config section."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from Strategy_Auto_Trader.allocation.allocation_manager import MultiTierAllocationManager

TODAY = date(2026, 9, 21)
CONFIG = Path(__file__).resolve().parents[2] / "config" / "overnight_strategy.json"


def _mgr(held: str = "CSH2.L", enter: float = 23.0, exit_: float | None = 24.0) -> MultiTierAllocationManager:
    mgr = MultiTierAllocationManager(vxn_threshold=enter, vxn_exit_threshold=exit_, vix_tier1=15.0, vix_tier2=17.5)
    mgr.current_asset = held
    return mgr


def _tier(mgr: MultiTierAllocationManager, vxn: float, vix: float = 30.0) -> int:
    return mgr.signal(TODAY, vxn, vix, verbose=False).tier


class TestDeadband:
    def test_enters_nasdaq_only_at_or_below_the_entry_level(self):
        mgr = _mgr(held="CSH2.L")
        assert [_tier(mgr, v) for v in (22.9, 23.0, 23.1, 24.0)] == [1, 1, 4, 4]

    def test_holds_nasdaq_up_to_the_exit_level(self):
        mgr = _mgr(held="EQGB.L")
        assert [_tier(mgr, v) for v in (23.0, 23.5, 24.0, 24.1)] == [1, 1, 1, 4]

    def test_reading_inside_the_band_keeps_whatever_is_held(self):
        assert _tier(_mgr(held="EQGB.L"), 23.5) == 1
        assert _tier(_mgr(held="CSH2.L"), 23.5) == 4

    def test_no_deadband_when_exit_is_not_set(self):
        for held in ("EQGB.L", "CSH2.L"):
            assert [_tier(_mgr(held=held, exit_=None), v) for v in (23.0, 23.1)] == [1, 4]

    def test_signal_flags_the_exit_only_beyond_the_exit_level(self):
        mgr = _mgr(held="EQGB.L")
        assert mgr.signal(TODAY, 24.0, 30.0, verbose=False).needs_rebalance is False
        assert mgr.signal(TODAY, 24.5, 30.0, verbose=False).needs_rebalance is True

    def test_missing_vxn_never_qualifies_for_nasdaq_even_when_held(self):
        assert _tier(_mgr(held="EQGB.L"), None) == 4

    def test_wobble_around_the_entry_level_does_not_flip_the_position(self):
        mgr = _mgr(held="CSH2.L")
        mgr.current_asset = "EQGB.L" if _tier(mgr, 22.5) == 1 else "CSH2.L"  # enters at 22.5
        moves = 0
        for vxn in (23.4, 22.9, 23.6, 23.2, 23.9, 23.1):  # all inside the band
            new = "EQGB.L" if _tier(mgr, vxn) == 1 else "CSH2.L"
            moves += new != mgr.current_asset
            mgr.current_asset = new
        assert moves == 0

    def test_status_reports_the_effective_and_both_edge_gates(self):
        mgr = _mgr(held="EQGB.L")
        mgr.signal(TODAY, 23.5, 30.0, verbose=False)
        nasdaq = mgr.app_status_dict()["tier_allocation"]["tiers"][0]
        assert (nasdaq["gate_value"], nasdaq["enter_gate_value"], nasdaq["exit_gate_value"], nasdaq["passes"]) == (24.0, 23.0, 24.0, True)


class TestValidation:
    def test_exit_below_entry_rejected(self):
        with pytest.raises(ValueError):
            MultiTierAllocationManager(vxn_threshold=24.0, vxn_exit_threshold=23.0)

    def test_vix_tiers_must_be_ordered(self):
        with pytest.raises(ValueError):
            MultiTierAllocationManager(vix_tier1=17.5, vix_tier2=15.0)


class TestFromConfig:
    def test_reads_every_field(self):
        mgr = MultiTierAllocationManager.from_config({
            "vxn_threshold": 22, "vxn_exit_threshold": 25, "vix_tier1": 14, "vix_tier2": 18,
            "index_refresh_seconds": 120, "index_max_outage_seconds": 7200,
        })
        assert (mgr.vxn_threshold, mgr.vxn_exit_threshold, mgr.vix_tier1, mgr.vix_tier2) == (22.0, 25.0, 14.0, 18.0)
        assert mgr._vix_feed._refresh_seconds == mgr._vxn_feed._refresh_seconds == 120
        assert mgr._vix_feed._max_outage_seconds == 7200

    def test_partial_section_keeps_defaults_for_the_rest(self):
        mgr = MultiTierAllocationManager.from_config({"vxn_threshold": 21})
        assert mgr.vxn_threshold == 21.0 and mgr.vxn_exit_threshold == 21.0 and mgr.vix_tier2 == 17.5

    @pytest.mark.parametrize("section", [None, {}])
    def test_missing_section_falls_back_to_defaults(self, section):
        mgr = MultiTierAllocationManager.from_config(section)
        assert mgr.vxn_threshold == 18.0 and mgr.vxn_exit_threshold == 18.0

    def test_unknown_key_is_rejected_not_ignored(self):
        with pytest.raises(ValueError, match="vxn_treshold"):
            MultiTierAllocationManager.from_config({"vxn_treshold": 23})

    @pytest.mark.parametrize("bad", ["23", None, True, [23]])
    def test_non_numeric_value_rejected(self, bad):
        with pytest.raises(ValueError):
            MultiTierAllocationManager.from_config({"vxn_threshold": bad})

    def test_inconsistent_values_rejected(self):
        with pytest.raises(ValueError):
            MultiTierAllocationManager.from_config({"vxn_threshold": 25, "vxn_exit_threshold": 24})

    def test_summary_states_the_thresholds(self):
        text = MultiTierAllocationManager.from_config({"vxn_threshold": 23, "vxn_exit_threshold": 24}).thresholds_summary()
        assert "VXN<=23" in text and "VXN>24" in text


def test_shipped_config_has_a_valid_tier_allocation_section():
    section = json.loads(CONFIG.read_text(encoding="utf-8")).get("tier_allocation")
    assert section, "config/overnight_strategy.json must define tier_allocation"
    mgr = MultiTierAllocationManager.from_config(section)
    assert mgr.vxn_exit_threshold >= mgr.vxn_threshold


def test_live_signal_reproduces_the_backtest_deadband_on_a_random_path():
    """The manager and intraday_engine.tiers_vxn_deadband must agree bar for bar, or the backtest
    numbers do not describe what the daemon would do."""
    import numpy as np

    from Strategy_Auto_Trader.allocation import intraday_engine as eng

    rng = np.random.default_rng(7)
    shocks = rng.normal(0, 0.7, 3000)
    vxn = np.empty(3000)
    vxn[0] = 23.5
    for i in range(1, 3000):  # mean-reverting around the band so both states are exercised
        vxn[i] = vxn[i - 1] + 0.15 * (23.5 - vxn[i - 1]) + shocks[i]
    backtest_in_nasdaq = eng.tiers_vxn_deadband(vxn, 23.0, 24.0) == 0

    mgr = _mgr(held="CSH2.L", enter=23.0, exit_=24.0)
    live_in_nasdaq = []
    for value in vxn:
        sig = mgr.signal(TODAY, float(value), 30.0, verbose=False)  # VIX 30 -> cash tier, so only Nasdaq or cash
        mgr.current_asset = sig.target_asset
        live_in_nasdaq.append(sig.tier == 1)

    assert np.array_equal(np.array(live_in_nasdaq), backtest_in_nasdaq)
    assert 0.2 < backtest_in_nasdaq.mean() < 0.8  # the path actually exercises both states
    assert (np.diff(backtest_in_nasdaq.astype(int)) != 0).sum() > 20


def _lower(enabled: bool, held: str = "CSH2.L") -> MultiTierAllocationManager:
    mgr = MultiTierAllocationManager(vxn_threshold=23.0, vxn_exit_threshold=24.0, vix_tier1=15.0, vix_tier2=17.5,
                                     lower_tiers_enabled=enabled)
    mgr.current_asset = held
    return mgr


class TestLowerTiersFlag:
    def test_enabled_by_default_keeps_the_sp_and_ftse_tiers(self):
        mgr = MultiTierAllocationManager(vxn_threshold=23.0, vix_tier1=15.0, vix_tier2=17.5)
        assert mgr.lower_tiers_enabled is True
        assert (_tier(mgr, 30.0, 14.0), _tier(mgr, 30.0, 16.0), _tier(mgr, 30.0, 20.0)) == (2, 3, 4)

    def test_disabled_sends_everything_that_is_not_nasdaq_to_cash(self):
        mgr = _lower(False)
        assert (_tier(mgr, 30.0, 14.0), _tier(mgr, 30.0, 16.0), _tier(mgr, 30.0, 20.0)) == (4, 4, 4)

    def test_disabled_does_not_touch_the_nasdaq_tier(self):
        assert [_tier(_lower(False, "CSH2.L"), v, 14.0) for v in (22.0, 23.5)] == [1, 4]
        assert [_tier(_lower(False, "EQGB.L"), v, 14.0) for v in (23.5, 25.0)] == [1, 4]

    def test_a_legacy_lower_tier_holding_is_rotated_out_when_disabled(self):
        sig = _lower(False, held="ISF.L").signal(TODAY, 30.0, 16.0, verbose=False)
        assert (sig.target_asset, sig.needs_rebalance) == ("CSH2.L", True)

    def test_status_marks_the_lower_tiers_disabled(self):
        mgr = _lower(False)
        mgr.signal(TODAY, 30.0, 14.0, verbose=False)
        tiers = mgr.app_status_dict()["tier_allocation"]["tiers"]
        assert [(t["tier_num"], t.get("enabled"), t["passes"]) for t in tiers[1:3]] == [(2, False, False), (3, False, False)]
        assert [t["label"] for t in tiers[1:3]] == ["Balanced (disabled)", "Defensive (disabled)"]  # visible even to a UI that ignores `enabled`
        assert [t["label"] for t in (tiers[0], tiers[3])] == ["Nasdaq", "Money-Market"]

    def test_status_labels_are_unchanged_when_the_lower_tiers_are_on(self):
        mgr = _lower(True)
        mgr.signal(TODAY, 30.0, 14.0, verbose=False)
        tiers = mgr.app_status_dict()["tier_allocation"]["tiers"]
        assert [t["label"] for t in tiers] == ["Nasdaq", "Balanced", "Defensive", "Money-Market"]
        assert tiers[1]["passes"] is True and tiers[1]["enabled"] is True

    def test_summary_says_the_lower_tiers_are_off(self):
        assert "S&P/FTSE tiers OFF" in _lower(False).thresholds_summary()
        assert "S&P VIX<=15" in _lower(True).thresholds_summary()

    @pytest.mark.parametrize("value", [True, False])
    def test_config_accepts_a_boolean(self, value):
        assert MultiTierAllocationManager.from_config({"lower_tiers_enabled": value}).lower_tiers_enabled is value

    @pytest.mark.parametrize("bad", ["false", 0, 1, None])
    def test_config_rejects_a_non_boolean_flag(self, bad):
        with pytest.raises(ValueError, match="true or false"):
            MultiTierAllocationManager.from_config({"lower_tiers_enabled": bad})

    def test_shipped_config_has_the_lower_tiers_off(self):
        section = json.loads(CONFIG.read_text(encoding="utf-8"))["tier_allocation"]
        assert section["lower_tiers_enabled"] is False
        assert MultiTierAllocationManager.from_config(section).lower_tiers_enabled is False


class TestCommission:
    def test_default_is_the_ibkr_uk_tiered_rate_not_double_it(self):
        assert MultiTierAllocationManager().commission_pct == 0.05

    def test_config_sets_the_commission(self):
        assert MultiTierAllocationManager.from_config({"commission_pct": 0.08}).commission_pct == 0.08

    @pytest.mark.parametrize("bad", [-0.1, "0.05", True])
    def test_bad_commission_rejected(self, bad):
        with pytest.raises(ValueError):
            MultiTierAllocationManager.from_config({"commission_pct": bad})

    def test_shipped_config_commission_is_005(self):
        section = json.loads(CONFIG.read_text(encoding="utf-8"))["tier_allocation"]
        assert section["commission_pct"] == 0.05
        assert MultiTierAllocationManager.from_config(section).commission_pct == 0.05

    def test_rebalance_sizing_uses_the_configured_commission(self):
        """At 0.05% a GBP 20,000 rotation buys ~1,999 shares at 10.00; at the old 0.1% it left ~GBP 20 more idle."""
        mgr = MultiTierAllocationManager(vxn_threshold=23.0, vxn_exit_threshold=24.0, lower_tiers_enabled=False)
        mgr.current_asset = "CSH2.L"
        orders = mgr.rebalance(TODAY, 20.0, 30.0, {"EQGB.L": 10.0, "CSH2.L": 100.0}, 20_000.0, {"CSH2.L": 0})
        buy = next(o for o in orders if o.action == "BUY")
        assert buy.ticker == "EQGB.L"
        assert buy.quantity == int(20_000.0 / (10.0 * 1.0005))
        old = int(20_000.0 / (10.0 * 1.001))
        assert buy.quantity > old
