"""Tests for the live wiring of the one-shot crash-recovery override
(allocation/crash_recovery_override.py), added 2026-11-04 per
ONE_SHOT_CRASH_RECOVERY_OVERRIDE.md. Distinct from the pre-existing offline
tests/allocation/test_crash_recovery_override.py, which tests the batch research
mechanism in scripts/tier_analysis/crash_recovery_override_sweep.py unchanged.

decide()/hyg_recovery_confirmed_today()/price_rising_today() are tested directly with
synthetic inputs rather than through the full fetch path, since the one-shot state
machine and the streak/sticky deadband are the actual new logic here — IBKR connectivity
is exercised only by the symbols.py routing test and was hand-verified against the paper
Gateway during planning (see plan/ONE_SHOT_CRASH_RECOVERY_OVERRIDE.md).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from Strategy_Auto_Trader.allocation import crash_recovery_override as cro
from Strategy_Auto_Trader.research.crash_phases import find_episodes

_NOOP_SAVE = lambda daemon_state: None
_SENTINEL_EPISODE = object()  # decide() only checks `episode is None`; peak-date tracking uses episode_peak_date


def _state() -> dict:
    return {}


def _logger():
    import logging
    return logging.getLogger("test_crash_recovery_override_live")


class TestLoadConfig:
    def test_missing_section_is_disabled_with_defaults(self):
        cfg = cro.load_config(None)
        assert cfg["enabled"] is False
        assert cfg["hyg_recovery_threshold"] == 0.90

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError):
            cro.load_config({"bogus": 1})

    def test_overrides_merge_over_defaults(self):
        cfg = cro.load_config({"enabled": True, "rise_days": 7})
        assert cfg["enabled"] is True
        assert cfg["rise_days"] == 7
        assert cfg["min_depth"] == 0.20  # untouched default


class TestDecideOneShotStateMachine:
    """The online translation of the batch script's one_shot_override_tiers."""

    def test_no_episode_is_a_pure_no_op(self):
        state = _state()
        force = cro.decide(state, date(2026, 1, 1), None, None, False, True, True, _logger(), save_state=_NOOP_SAVE)
        assert force is False
        assert state.get("crash_override", {}) == {}

    def test_dormant_while_base_still_holds_nasdaq(self):
        """Episode active, but base never exited to cash yet — nothing to accelerate."""
        state = _state()
        peak = date(2020, 2, 19)
        force = cro.decide(state, date(2020, 2, 20), _SENTINEL_EPISODE, peak,
                            base_would_pick_nasdaq=True, hyg_confirmed=True, nasdaq_rising=True,
                            logger=_logger(), save_state=_NOOP_SAVE)
        assert force is False
        assert state["crash_override"]["exited_to_cash"] is False

    def test_fires_once_exited_and_both_gates_confirm(self):
        state = _state()
        peak = date(2020, 2, 19)
        # Day 1: base exits to cash, signal not yet confirmed.
        force1 = cro.decide(state, date(2020, 3, 23), _SENTINEL_EPISODE, peak,
                             base_would_pick_nasdaq=False, hyg_confirmed=False, nasdaq_rising=False,
                             logger=_logger(), save_state=_NOOP_SAVE)
        assert force1 is False
        assert state["crash_override"]["exited_to_cash"] is True
        # Day 2: both gates confirm -> fires.
        force2 = cro.decide(state, date(2020, 4, 1), _SENTINEL_EPISODE, peak,
                             base_would_pick_nasdaq=False, hyg_confirmed=True, nasdaq_rising=True,
                             logger=_logger(), save_state=_NOOP_SAVE)
        assert force2 is True
        assert state["crash_override"]["fired"] is True
        assert state["crash_override"]["active"] is True

    def test_keeps_forcing_until_base_catches_up(self):
        state = _state()
        peak = date(2020, 2, 19)
        cro.decide(state, date(2020, 3, 23), _SENTINEL_EPISODE, peak, False, False, False, _logger(), save_state=_NOOP_SAVE)
        cro.decide(state, date(2020, 4, 1), _SENTINEL_EPISODE, peak, False, True, True, _logger(), save_state=_NOOP_SAVE)
        # Base still says cash the next day -> override keeps holding Nasdaq.
        force = cro.decide(state, date(2020, 4, 2), _SENTINEL_EPISODE, peak, False, True, True, _logger(), save_state=_NOOP_SAVE)
        assert force is True

    def test_hands_back_once_base_agrees_then_never_refires(self):
        state = _state()
        peak = date(2020, 2, 19)
        cro.decide(state, date(2020, 3, 23), _SENTINEL_EPISODE, peak, False, False, False, _logger(), save_state=_NOOP_SAVE)
        cro.decide(state, date(2020, 4, 1), _SENTINEL_EPISODE, peak, False, True, True, _logger(), save_state=_NOOP_SAVE)
        # Base catches up -> hand back.
        force = cro.decide(state, date(2020, 4, 5), _SENTINEL_EPISODE, peak, True, True, True, _logger(), save_state=_NOOP_SAVE)
        assert force is False
        assert state["crash_override"]["resolved"] is True
        # Base dips to cash again later in the same episode -> must NOT re-arm.
        force_again = cro.decide(state, date(2020, 6, 1), _SENTINEL_EPISODE, peak, False, True, True, _logger(), save_state=_NOOP_SAVE)
        assert force_again is False

    def test_resolved_with_no_op_when_base_naturally_beats_the_signal(self):
        state = _state()
        peak = date(2020, 2, 19)
        cro.decide(state, date(2020, 3, 23), _SENTINEL_EPISODE, peak, False, False, False, _logger(), save_state=_NOOP_SAVE)
        # Base re-enters naturally before HYG ever confirms.
        force = cro.decide(state, date(2020, 4, 10), _SENTINEL_EPISODE, peak, True, False, False, _logger(), save_state=_NOOP_SAVE)
        assert force is False
        assert state["crash_override"]["resolved"] is True
        assert state["crash_override"]["fired"] is False

    def test_new_episode_peak_date_resets_state(self):
        """A later, separate crash episode must get its own fresh one-shot, not inherit a
        prior episode's resolved/fired flags."""
        state = _state()
        peak1 = date(2020, 2, 19)
        cro.decide(state, date(2020, 3, 23), _SENTINEL_EPISODE, peak1, False, False, False, _logger(), save_state=_NOOP_SAVE)
        cro.decide(state, date(2020, 4, 1), _SENTINEL_EPISODE, peak1, False, True, True, _logger(), save_state=_NOOP_SAVE)
        assert state["crash_override"]["fired"] is True

        peak2 = date(2022, 1, 3)
        force = cro.decide(state, date(2022, 1, 4), _SENTINEL_EPISODE, peak2, True, False, False, _logger(), save_state=_NOOP_SAVE)
        assert force is False
        assert state["crash_override"]["fired"] is False
        assert state["crash_override"]["episode_peak_date"] == peak2.isoformat()

    def test_episode_ending_clears_state(self):
        """No episode currently active (find_episodes window closed) must clear state so a
        fresh episode later starts clean, not resolved-by-default."""
        state = _state()
        peak = date(2020, 2, 19)
        cro.decide(state, date(2020, 3, 23), _SENTINEL_EPISODE, peak, False, False, False, _logger(), save_state=_NOOP_SAVE)
        cro.decide(state, None, None, None, False, False, False, _logger(), save_state=_NOOP_SAVE)
        assert state["crash_override"] == {}

    def test_episode_boundary_does_not_reset_hyg_streak_state(self):
        """The HYG confirm/exit streaks and sticky-armed flag track a continuous daily
        series (same as the batch script's `_sticky_confirmed`) and must survive an
        episode starting or ending unchanged — only the episode-tracking keys reset."""
        state = _state()
        state["crash_override"] = {"hyg_confirm_streak": 4, "hyg_sticky_armed": True, "hyg_ratio_last": 0.93}

        peak = date(2020, 2, 19)
        cro.decide(state, date(2020, 3, 23), _SENTINEL_EPISODE, peak, False, False, False, _logger(), save_state=_NOOP_SAVE)
        assert state["crash_override"]["hyg_confirm_streak"] == 4
        assert state["crash_override"]["hyg_sticky_armed"] is True

        # Episode ends.
        cro.decide(state, None, None, None, False, False, False, _logger(), save_state=_NOOP_SAVE)
        assert state["crash_override"]["hyg_confirm_streak"] == 4
        assert state["crash_override"]["hyg_sticky_armed"] is True
        assert "episode_peak_date" not in state["crash_override"]

        # A later, separate episode starts.
        peak2 = date(2022, 1, 3)
        cro.decide(state, date(2022, 1, 4), _SENTINEL_EPISODE, peak2, False, False, False, _logger(), save_state=_NOOP_SAVE)
        assert state["crash_override"]["hyg_confirm_streak"] == 4
        assert state["crash_override"]["hyg_sticky_armed"] is True

    def test_restart_mid_episode_does_not_rearm(self):
        """The doc's explicit requirement: a daemon restart between the fire day and the
        hand-back day must not reset `fired`/`active` and re-arm the trigger."""
        state = _state()
        peak = date(2020, 2, 19)
        cro.decide(state, date(2020, 3, 23), _SENTINEL_EPISODE, peak, False, False, False, _logger(), save_state=_NOOP_SAVE)
        cro.decide(state, date(2020, 4, 1), _SENTINEL_EPISODE, peak, False, True, True, _logger(), save_state=_NOOP_SAVE)
        assert state["crash_override"]["fired"] is True

        # Simulate a restart: state is plain JSON-serializable, round-trip it.
        import json
        reloaded = json.loads(json.dumps(state))

        # New day, HYG/price gates no longer confirming (e.g. a transient dip) — must still
        # force, because `active` persisted True across the "restart".
        force = cro.decide(reloaded, date(2020, 4, 2), _SENTINEL_EPISODE, peak, False, False, False, _logger(), save_state=_NOOP_SAVE)
        assert force is True
        assert reloaded["crash_override"]["fired"] is True  # not re-armed/reset


class TestHygRecoveryConfirmedToday:
    def _series(self, values: list[float]) -> pd.Series:
        idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
        return pd.Series(values, index=idx, dtype=float)

    def _cfg(self, **overrides) -> dict:
        cfg = cro.load_config(None)
        cfg.update(overrides)
        return cfg

    def test_insufficient_history_stays_unarmed(self):
        state = _state()
        series = self._series([100.0] * 10)
        assert cro.hyg_recovery_confirmed_today(series, state, self._cfg()) is False

    def test_arms_after_confirm_days_consecutive_qualifying_closes(self):
        state = _state()
        cfg = self._cfg(confirm_days=3, exit_confirm_days=3)
        base = [100.0] * 252  # rolling 252d high = 100
        for i, close in enumerate([100.0, 100.0, 95.0]):  # 0.95 >= 0.90 threshold every day here
            series = self._series(base + [100.0] * i + [close])
            armed = cro.hyg_recovery_confirmed_today(series, state, cfg)
        assert armed is True

    def test_does_not_disarm_on_a_single_bad_day(self):
        state = _state()
        cfg = self._cfg(confirm_days=2, exit_confirm_days=3)
        base = [100.0] * 252
        # Arm it: two qualifying days.
        cro.hyg_recovery_confirmed_today(self._series(base + [95.0]), state, cfg)
        cro.hyg_recovery_confirmed_today(self._series(base + [95.0, 95.0]), state, cfg)
        assert state["crash_override"]["hyg_sticky_armed"] is True
        # One disqualifying day (ratio below threshold) must not disarm (exit_confirm_days=3).
        armed = cro.hyg_recovery_confirmed_today(self._series(base + [95.0, 95.0, 50.0]), state, cfg)
        assert armed is True

    def test_disarms_after_exit_confirm_days_consecutive_bad_closes(self):
        state = _state()
        cfg = self._cfg(confirm_days=2, exit_confirm_days=2)
        base = [100.0] * 252
        cro.hyg_recovery_confirmed_today(self._series(base + [95.0]), state, cfg)
        cro.hyg_recovery_confirmed_today(self._series(base + [95.0, 95.0]), state, cfg)
        assert state["crash_override"]["hyg_sticky_armed"] is True
        cro.hyg_recovery_confirmed_today(self._series(base + [95.0, 95.0, 50.0]), state, cfg)
        armed = cro.hyg_recovery_confirmed_today(self._series(base + [95.0, 95.0, 50.0, 50.0]), state, cfg)
        assert armed is False

    def test_same_session_counted_exactly_once_across_hourly_cycles(self):
        """The decision cycle runs ~7x/day on the SAME settled daily close; without the session
        cursor the streak would hit confirm_days within one afternoon instead of N days."""
        state = _state()
        cfg = self._cfg(confirm_days=3, exit_confirm_days=3)
        series = self._series([100.0] * 252 + [95.0])  # qualifying close, one session
        for _ in range(7):  # one trading day's worth of hourly cycles
            armed = cro.hyg_recovery_confirmed_today(series, state, cfg)
        assert state["crash_override"]["hyg_confirm_streak"] == 1
        assert armed is False
        assert state["crash_override"]["hyg_streak_session_date"] == series.index[-1].date().isoformat()

    def test_streak_advances_only_when_a_new_session_arrives(self):
        state = _state()
        cfg = self._cfg(confirm_days=3, exit_confirm_days=3)
        base = [100.0] * 252
        for extra in ([95.0], [95.0, 95.0], [95.0, 95.0, 95.0]):
            series = self._series(base + extra)
            cro.hyg_recovery_confirmed_today(series, state, cfg)  # counted
            armed = cro.hyg_recovery_confirmed_today(series, state, cfg)  # same session: no-op
        assert state["crash_override"]["hyg_confirm_streak"] == 3
        assert armed is True

    def test_pre_cursor_state_is_reset_not_trusted(self):
        """State written before per-session counting was per-cycle-inflated — streaks AND the
        armed flag derived from them must reset and re-confirm from scratch (fails toward
        'signal not confirmed')."""
        state = _state()
        state["crash_override"] = {"hyg_confirm_streak": 35, "hyg_exit_streak": 0, "hyg_sticky_armed": True}
        cfg = self._cfg(confirm_days=3, exit_confirm_days=3)
        armed = cro.hyg_recovery_confirmed_today(self._series([100.0] * 252 + [95.0]), state, cfg)
        assert armed is False
        assert state["crash_override"]["hyg_confirm_streak"] == 1
        assert state["crash_override"]["hyg_sticky_armed"] is False


class TestPriceRisingToday:
    def _series(self, values: list[float]) -> pd.Series:
        idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
        return pd.Series(values, index=idx, dtype=float)

    def test_rise_days_zero_is_always_true(self):
        assert cro.price_rising_today(self._series([100.0, 90.0]), rise_days=0) is True

    def test_true_when_price_higher_than_n_days_ago(self):
        series = self._series([100.0] + [90.0] * 9 + [105.0])
        assert cro.price_rising_today(series, rise_days=10) is True

    def test_false_when_price_lower_than_n_days_ago(self):
        series = self._series([100.0] + [90.0] * 9 + [95.0])
        assert cro.price_rising_today(series, rise_days=10) is False

    def test_none_series_is_false(self):
        assert cro.price_rising_today(None, rise_days=10) is False


class TestCurrentEpisode:
    def test_finds_a_real_crash_episode(self):
        """Flat at 100 (peak), crash to 70 (30% drawdown, clears min_depth=0.20), flat
        through the horizon. `today` inside the window must resolve to that episode."""
        n = 300
        prices = [100.0] * 50 + [70.0] * (n - 50)
        idx = pd.bdate_range("2020-01-02", periods=n)
        series = pd.Series(prices, index=idx)
        today = idx[55].date()
        ep = cro.current_episode(series, today, min_depth=0.20)
        assert ep is not None
        assert idx[ep.peak_idx].date() <= today <= idx[ep.end_idx].date()

    def test_none_outside_any_episode(self):
        n = 300
        prices = [100.0] * n  # flat, no drawdown at all
        idx = pd.bdate_range("2020-01-02", periods=n)
        series = pd.Series(prices, index=idx)
        assert cro.current_episode(series, idx[100].date(), min_depth=0.20) is None

    def test_none_for_empty_or_missing_series(self):
        assert cro.current_episode(None, date(2020, 1, 1), min_depth=0.20) is None
        assert cro.current_episode(pd.Series(dtype=float), date(2020, 1, 1), min_depth=0.20) is None

    def test_tz_aware_index_does_not_raise(self):
        """Live series from IBKRDataClient.fetch_daily are UTC tz-aware (see
        ibkr_data.py's `out.index = pd.to_datetime(out.index, utc=True)`) — `today`
        (a plain date) must not raise 'Cannot compare tz-naive and tz-aware timestamps'
        (prod incident 2026-11-04, first live run after enabling the override)."""
        n = 300
        prices = [100.0] * 50 + [70.0] * (n - 50)
        idx = pd.bdate_range("2020-01-02", periods=n, tz="UTC")
        series = pd.Series(prices, index=idx)
        today = idx[55].date()
        ep = cro.current_episode(series, today, min_depth=0.20)
        assert ep is not None


class _FakeDailyClient:
    """Stands in for IBKRDataClient: returns canned frames, records calls."""

    def __init__(self, frames: dict, calls: list):
        self._frames = frames
        self._calls = calls

    def fetch_daily(self, symbol, period="max", historical_only=False):
        self._calls.append((symbol, historical_only))
        return self._frames[symbol]

    def disconnect(self):
        pass


def _daily_frame(last_session: str, n: int = 5) -> pd.DataFrame:
    idx = pd.bdate_range(end=last_session, periods=n, tz="UTC")
    return pd.DataFrame({"Close": np.linspace(100.0, 101.0, n)}, index=idx)


class TestFetchEqqqHygDaily:
    def setup_method(self):
        cro._clear_series_cache()

    def _cfg(self):
        return cro.load_config({"enabled": True})

    def test_cursor_stamped_from_fetched_data_not_wall_clock(self):
        """18:00 ET Wednesday: session 2026-11-11 should be settled, but the feed only has
        data through Tuesday. The cursor must say Tuesday (so the 900s retry keeps trying),
        not Wednesday (which would silently serve stale data for ~24h)."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        state = {}
        frames = {"EQQQ.L": _daily_frame("2026-11-10"), "HYG": _daily_frame("2026-11-10")}
        calls = []
        now = datetime(2026, 11, 11, 18, 0, tzinfo=ZoneInfo("America/New_York"))
        eqqq, hyg = cro.fetch_eqqq_hyg_daily(
            state, self._cfg(), _logger(),
            client_factory=lambda cid: _FakeDailyClient(frames, calls),
            save_state=_NOOP_SAVE, now=now,
        )
        assert eqqq is not None and hyg is not None
        assert state["crash_override"]["last_fetch_session"] == "2026-11-10"

    def test_cache_only_readback_served_from_memory_not_reparsed(self):
        """Cycles between fetches must not re-read the full-history CSVs: once the series for
        the current cursor session are in memory, the client is not constructed at all."""
        from datetime import datetime, timezone
        # 2026-11-13 is the Friday before the Saturday `now` below: the latest settled session.
        state = {"crash_override": {"last_fetch_session": "2026-11-13"}}
        frames = {"EQQQ.L": _daily_frame("2026-11-13"), "HYG": _daily_frame("2026-11-13")}
        calls = []
        factory_calls = []

        def factory(cid):
            factory_calls.append(cid)
            return _FakeDailyClient(frames, calls)

        # Saturday: cursor already at the latest settled session, so not due — cache-only path.
        now = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)
        first = cro.fetch_eqqq_hyg_daily(state, self._cfg(), _logger(),
                                         client_factory=factory, save_state=_NOOP_SAVE, now=now)
        assert len(factory_calls) == 1  # disk readback once
        second = cro.fetch_eqqq_hyg_daily(state, self._cfg(), _logger(),
                                          client_factory=factory, save_state=_NOOP_SAVE, now=now)
        assert len(factory_calls) == 1  # served from memory, no new client
        assert second[0] is first[0] and second[1] is first[1]


class TestAppStatusDict:
    def test_shape_with_no_state_yet(self):
        cfg = cro.load_config({"enabled": True})
        out = cro.app_status_dict({}, cfg)
        assert out["enabled"] is True
        assert out["episode_active"] is False
        assert out["thresholds"]["hyg_recovery_threshold"] == 0.90

    def test_reflects_persisted_state(self):
        cfg = cro.load_config({"enabled": True})
        state = {"crash_override": {"episode_peak_date": "2020-02-19", "fired": True, "active": True}}
        out = cro.app_status_dict(state, cfg)
        assert out["episode_active"] is True
        assert out["fired"] is True
        assert out["active"] is True
