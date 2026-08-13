"""Tests for net_retry.py — hard timeout + retry wrapper for flaky network calls."""

from __future__ import annotations

import time

from Strategy_Auto_Trader.core.net_retry import call_with_timeout_retry


def test_returns_result_on_success():
    result = call_with_timeout_retry(lambda: 42)
    assert result == 42


def test_returns_none_when_fn_always_raises():
    def _boom():
        raise ValueError("network is down")

    result = call_with_timeout_retry(_boom, retries=1, backoff=0.01)
    assert result is None


def test_retries_then_succeeds():
    calls = {"n": 0}

    def _flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ConnectionError("transient")
        return "ok"

    result = call_with_timeout_retry(_flaky, retries=2, backoff=0.01)
    assert result == "ok"
    assert calls["n"] == 2


def test_timeout_returns_none_without_blocking_on_slow_call():
    """A call that hangs well past the timeout must not make the wrapper
    itself block that long — the abandoned thread is not waited on."""
    def _hangs():
        time.sleep(5)
        return "too late"

    start = time.time()
    result = call_with_timeout_retry(_hangs, timeout=0.1, retries=0, backoff=0.01)
    elapsed = time.time() - start

    assert result is None
    assert elapsed < 2.0


def test_exhausts_all_retries_before_giving_up():
    calls = {"n": 0}

    def _always_fails():
        calls["n"] += 1
        raise RuntimeError("nope")

    call_with_timeout_retry(_always_fails, retries=2, backoff=0.01)
    assert calls["n"] == 3
