"""Live wiring of the one-shot crash-recovery override (see ONE_SHOT_CRASH_RECOVERY_OVERRIDE.md).

Ported from `scripts/tier_analysis/crash_recovery_override_sweep.py`'s validated
`hyg_recovery(threshold=0.90, confirm_days=5, rise_days=10)` one-shot mechanism — the
single config that passed the project's double criterion (beats both the plain VXN
23/24 reference and the deployed VVIX-banded reference, train AND test). Almost the
entire edge is 2020 alone; `min_depth=0.20` was found to be knife-edge, not robust (see
the writeup). This is a wrap-around override on the live tier-1 decision
(`allocation_manager.py`'s VXN/VVIX-banded signal), not a replacement of it — outside an
active crash episode, or whenever the signal never confirms, behaviour is unchanged.

Episode detection and the Nasdaq/credit series both use EQQQ (GBP, UNHEDGED) — the exact
series `research/crash_phases.py`'s `min_depth=0.20` was calibrated against and the one
the whole 26yr backtest stack's P&L is measured on (`data_synthetic/hourly_spliced`,
built from EQQQ.csv). EQGB (what live actually holds) is GBP-HEDGED and would not match
either calibration — see todo.md for the parked question of re-validating the whole tier
strategy on EQGB instead.

State is persisted under `daemon_state["crash_override"]` so a daemon restart mid-episode
cannot re-arm the one-shot trigger (the doc's explicit requirement).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from .cboe_intraday import DAILY_SETTLED_TIME
from ..research.crash_phases import Episode, find_episodes

FETCH_RETRY_SECONDS = 900  # same cooldown idea as live_daemon.update_vvix_band's VVIX_RETRY_SECONDS

_CONFIG_DEFAULTS = {
    "enabled": False,
    "hyg_recovery_threshold": 0.90,
    "confirm_days": 5,
    "exit_confirm_days": 5,
    "rise_days": 10,
    "min_depth": 0.20,
    "ibkr_client_id": 3,
}
_CONFIG_KEYS = frozenset(_CONFIG_DEFAULTS)


def load_config(section: dict | None) -> dict:
    """Validate+fill the `crash_recovery_override` config section. An unknown key raises
    (same allow-list posture as allocation_manager._CONFIG_KEYS) so a typo cannot silently
    leave a threshold at its default. A missing section is just disabled."""
    if not section:
        return dict(_CONFIG_DEFAULTS)
    unknown = set(section) - _CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown crash_recovery_override keys {sorted(unknown)}; allowed: {sorted(_CONFIG_KEYS)}")
    cfg = dict(_CONFIG_DEFAULTS)
    cfg.update(section)
    return cfg


def _latest_settled_session(now_utc: datetime) -> date:
    """Newest weekday whose daily EQQQ/HYG close should be settled by `now_utc`.

    Same shape as live_daemon.py's `_latest_settled_session` (used for VVIX) — duplicated
    rather than imported because that helper is private to live_daemon.py and this module
    sits below it in the import graph (live_daemon.py imports this module, not the other
    way round). Holidays are not modelled: on one, the expected session never settles and
    the fetch just keeps returning the previous close, harmlessly re-attempted next cycle.
    """
    et = now_utc.astimezone(ZoneInfo("America/New_York"))
    hh, mm = (int(x) for x in DAILY_SETTLED_TIME.split(":"))
    day = et.date()
    if et.weekday() >= 5 or (et.hour, et.minute) < (hh, mm):
        day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


_EPISODE_STATE_KEYS = ("episode_peak_date", "exited_to_cash", "fired", "active", "resolved")


def _reset_episode_state(state: dict) -> None:
    """Clear only the per-episode keys, never the HYG streak/sticky-armed counters or the
    fetch cursor — those track a continuous daily series and must survive an episode
    boundary unchanged (same as the batch script's `_sticky_confirmed`, which runs over
    the whole series regardless of the crash-episode mask)."""
    for key in _EPISODE_STATE_KEYS:
        state.pop(key, None)


def _default_client_factory(client_id: int):
    from ..broker.ibkr_data import IBKRDataClient
    return IBKRDataClient(client_id=client_id)


def fetch_eqqq_hyg_daily(
    daemon_state: dict,
    cfg: dict,
    logger: logging.Logger,
    *,
    client_factory=None,
    save_state=None,
    now: datetime | None = None,
) -> tuple[pd.Series | None, pd.Series | None]:
    """Daily EQQQ and HYG closes, as Series indexed by date, via IBKR.

    Fetches at most once per settled session (same due/retry-cooldown shape as
    live_daemon.update_vvix_band), reading back from `IBKRDataClient.fetch_daily`'s own
    on-disk cache (data/cache/ibkr_daily/) either way, so a skipped or failed fetch still
    returns the last cached series rather than None — missing/stale data falls through to
    "signal not confirmed", never raises, per the doc's own missing-data default.
    """
    if client_factory is None:
        client_factory = _default_client_factory
    if save_state is None:
        from ..markov_cli.live_daemon import save_daemon_state as save_state
    if now is None:
        now = datetime.now(timezone.utc)

    state = daemon_state.setdefault("crash_override", {})
    last_session_str = state.get("last_fetch_session")
    last_session = date.fromisoformat(last_session_str) if last_session_str else None
    last_attempt = state.get("last_fetch_attempt_at")
    due = last_session is None or last_session < _latest_settled_session(now)
    if due and last_attempt is not None and (now - datetime.fromisoformat(last_attempt)).total_seconds() < FETCH_RETRY_SECONDS:
        due = False

    eqqq = hyg = None
    if due:
        state["last_fetch_attempt_at"] = now.isoformat()
        client = client_factory(cfg["ibkr_client_id"])
        try:
            eqqq = client.fetch_daily("EQQQ.L", period="max")
            hyg = client.fetch_daily("HYG", period="max")
        except Exception as e:
            logger.warning(f"Crash-recovery override: EQQQ/HYG fetch failed ({e})")
            eqqq = hyg = None
        finally:
            client.disconnect()
        if eqqq is not None and not eqqq.empty and hyg is not None and not hyg.empty:
            state["last_fetch_session"] = _latest_settled_session(now).isoformat()
            save_state(daemon_state)
            logger.info(f"Crash-recovery override: EQQQ/HYG refreshed through {eqqq.index[-1].date()}/{hyg.index[-1].date()}")
        else:
            logger.debug("Crash-recovery override: EQQQ/HYG fetch returned no new data this attempt")
            eqqq = hyg = None  # fall through to a cache-only readback below

    if eqqq is None or hyg is None:
        # Not due this cycle, or the due fetch above failed — serve whatever is cached
        # (historical_only=True never opens a connection when the cache already has data).
        client = client_factory(cfg["ibkr_client_id"])
        try:
            eqqq = client.fetch_daily("EQQQ.L", period="max", historical_only=True)
            hyg = client.fetch_daily("HYG", period="max", historical_only=True)
        finally:
            client.disconnect()
    eqqq_close = eqqq["Close"] if eqqq is not None and not eqqq.empty else None
    hyg_close = hyg["Close"] if hyg is not None and not hyg.empty else None
    return eqqq_close, hyg_close


def _streak(flag: bool, prev: int) -> int:
    """Scalar version of the batch script's vectorized `_streak`: consecutive-True run
    length ending today."""
    return prev + 1 if flag else 0


def hyg_recovery_confirmed_today(
    hyg_close: pd.Series | None,
    daemon_state: dict,
    cfg: dict,
) -> bool:
    """Today's HYG / its own trailing-252d high vs `hyg_recovery_threshold`, confirmed
    sticky the same way as the batch script's `_sticky_confirmed`: arms after
    `confirm_days` consecutive qualifying days, stays armed through blips, disarms only
    after `exit_confirm_days` consecutive non-qualifying days. Persists the two streak
    counters + armed flag in daemon_state so the deadband survives a restart.
    """
    state = daemon_state.setdefault("crash_override", {})
    if hyg_close is None or len(hyg_close) < 252:
        return bool(state.get("hyg_sticky_armed", False))

    rolling_high = hyg_close.rolling(252, min_periods=252).max()
    ratio = float(hyg_close.iloc[-1] / rolling_high.iloc[-1])
    state["hyg_ratio_last"] = ratio
    qualifies = ratio >= cfg["hyg_recovery_threshold"]

    confirm_streak = _streak(qualifies, state.get("hyg_confirm_streak", 0))
    exit_streak = _streak(not qualifies, state.get("hyg_exit_streak", 0))
    state["hyg_confirm_streak"] = confirm_streak
    state["hyg_exit_streak"] = exit_streak

    armed = state.get("hyg_sticky_armed", False)
    if not armed and confirm_streak >= cfg["confirm_days"]:
        armed = True
    elif armed and exit_streak >= cfg["exit_confirm_days"]:
        armed = False
    state["hyg_sticky_armed"] = armed
    return armed


def price_rising_today(nasdaq_close: pd.Series | None, rise_days: int) -> bool:
    """Today's EQQQ close vs its close `rise_days` trading days ago. `rise_days <= 0` is a
    no-op (always True), matching the batch script."""
    if rise_days <= 0:
        return True
    if nasdaq_close is None or len(nasdaq_close) <= rise_days:
        return False
    return bool(nasdaq_close.iloc[-1] > nasdaq_close.iloc[-1 - rise_days])


def current_episode(nasdaq_close: pd.Series | None, today: date, min_depth: float) -> Episode | None:
    """The crash episode (research.crash_phases.find_episodes) whose [peak, end] window
    contains `today`, if any. Recomputed fresh each call — cheap (thousands of rows, not
    millions) — so there is no separate streaming peak/trough tracker to keep in sync."""
    if nasdaq_close is None or nasdaq_close.empty:
        return None
    episodes = find_episodes(nasdaq_close, min_depth=min_depth)
    idx = nasdaq_close.index
    today_ts = pd.Timestamp(today)
    for ep in episodes:
        if idx[ep.peak_idx] <= today_ts <= idx[ep.end_idx]:
            return ep
    return None


def decide(
    daemon_state: dict,
    today: date,
    episode: Episode | None,
    episode_peak_date: date | None,
    base_would_pick_nasdaq: bool,
    hyg_confirmed: bool,
    nasdaq_rising: bool,
    logger: logging.Logger,
    *,
    save_state=None,
) -> bool:
    """The one-shot fire/hand-back state machine. Returns whether to force tier 1 today.

    Mirrors `one_shot_override_tiers` from the batch research script, but online: instead
    of scanning a known full history for "the gap between base's own exit and its own next
    natural re-entry", it tracks the same three facts incrementally as they happen live.
    """
    if save_state is None:
        from ..markov_cli.live_daemon import save_daemon_state as save_state

    state = daemon_state.setdefault("crash_override", {})

    if episode is None:
        if state.get("episode_peak_date") is not None:
            _reset_episode_state(state)
            save_state(daemon_state)
        return False

    if state.get("episode_peak_date") != episode_peak_date.isoformat():
        _reset_episode_state(state)
        state["episode_peak_date"] = episode_peak_date.isoformat()
        state["exited_to_cash"] = False
        state["fired"] = False
        state["active"] = False
        state["resolved"] = False

    if state.get("resolved"):
        return False

    if not state.get("exited_to_cash") and not base_would_pick_nasdaq:
        state["exited_to_cash"] = True
        logger.info(f"Crash-recovery override: base exited to cash on {today} within episode {episode_peak_date} — armed to watch for early re-entry signal")

    exited = state.get("exited_to_cash", False)
    force = False
    if not state.get("fired"):
        if exited and base_would_pick_nasdaq:
            state["resolved"] = True
            logger.info(f"Crash-recovery override: base naturally re-entered Nasdaq on {today} before the signal ever confirmed — episode resolved, no override")
        elif exited and hyg_confirmed and nasdaq_rising:
            state["fired"] = True
            state["active"] = True
            force = True
            logger.info(f"Crash-recovery override: FIRED on {today} (episode peak {episode_peak_date}) — forcing Nasdaq re-entry ahead of the base rule")
    else:
        if base_would_pick_nasdaq:
            state["active"] = False
            state["resolved"] = True
            logger.info(f"Crash-recovery override: base caught up on {today} — handing control back")
        else:
            force = state.get("active", False)

    save_state(daemon_state)
    return force


def update_and_decide(
    daemon_state: dict,
    today: date,
    cfg: dict,
    base_would_pick_nasdaq: bool,
    logger: logging.Logger,
    *,
    client_factory=None,
    save_state=None,
    now: datetime | None = None,
) -> bool:
    """Top-level entry point for live_daemon.py's decision cycle: fetch today's series
    (gated to once/settled-session), evaluate the gates, run the one-shot state machine,
    return whether to force tier 1. Disabled config is the caller's job to check first."""
    eqqq_close, hyg_close = fetch_eqqq_hyg_daily(daemon_state, cfg, logger, client_factory=client_factory, save_state=save_state, now=now)
    episode = current_episode(eqqq_close, today, cfg["min_depth"])
    hyg_confirmed = hyg_recovery_confirmed_today(hyg_close, daemon_state, cfg)
    nasdaq_rising = price_rising_today(eqqq_close, cfg["rise_days"])
    daemon_state.setdefault("crash_override", {})["nasdaq_rising_last"] = nasdaq_rising
    episode_peak_date = eqqq_close.index[episode.peak_idx].date() if episode is not None else None
    return decide(
        daemon_state, today, episode, episode_peak_date,
        base_would_pick_nasdaq, hyg_confirmed, nasdaq_rising, logger,
        save_state=save_state,
    )


def app_status_dict(daemon_state: dict, cfg: dict) -> dict:
    """UI block for app_status.json."""
    state = daemon_state.get("crash_override", {})
    return {
        "enabled": cfg.get("enabled", False),
        "episode_active": state.get("episode_peak_date") is not None,
        "episode_peak_date": state.get("episode_peak_date"),
        "exited_to_cash": state.get("exited_to_cash", False),
        "hyg_ratio": state.get("hyg_ratio_last"),
        "hyg_confirm_streak": state.get("hyg_confirm_streak", 0),
        "hyg_exit_streak": state.get("hyg_exit_streak", 0),
        "hyg_sticky_armed": state.get("hyg_sticky_armed", False),
        "nasdaq_rising": state.get("nasdaq_rising_last"),
        "fired": state.get("fired", False),
        "active": state.get("active", False),
        "resolved": state.get("resolved", False),
        "as_of_date": daemon_state.get("crash_override", {}).get("last_fetch_session"),
        "thresholds": {
            "hyg_recovery_threshold": cfg.get("hyg_recovery_threshold"),
            "confirm_days": cfg.get("confirm_days"),
            "exit_confirm_days": cfg.get("exit_confirm_days"),
            "rise_days": cfg.get("rise_days"),
            "min_depth": cfg.get("min_depth"),
        },
    }
