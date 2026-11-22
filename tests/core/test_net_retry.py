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


def test_is_empty_retries_then_succeeds():
    calls = {"n": 0}

    def _empty_then_full():
        calls["n"] += 1
        return [] if calls["n"] < 2 else [1, 2, 3]

    result = call_with_timeout_retry(
        _empty_then_full, retries=2, backoff=0.01, is_empty=lambda r: not r
    )
    assert result == [1, 2, 3]
    assert calls["n"] == 2


def test_is_empty_returns_last_empty_result_after_exhausting_retries():
    calls = {"n": 0}

    def _always_empty():
        calls["n"] += 1
        return []

    result = call_with_timeout_retry(
        _always_empty, retries=2, backoff=0.01, is_empty=lambda r: not r
    )
    assert result == []
    assert calls["n"] == 3


def test_is_empty_none_does_not_retry_on_falsy_result():
    """Without is_empty, a falsy-but-valid result is returned as-is, unchanged behavior."""
    calls = {"n": 0}

    def _once():
        calls["n"] += 1
        return []

    result = call_with_timeout_retry(_once, retries=2, backoff=0.01)
    assert result == []
    assert calls["n"] == 1


def test_label_is_prefixed_on_retry_warning(caplog):
    calls = {"n": 0}

    def _empty_then_full():
        calls["n"] += 1
        return [] if calls["n"] < 2 else [1]

    with caplog.at_level("WARNING"):
        call_with_timeout_retry(
            _empty_then_full, retries=1, backoff=0.01, is_empty=lambda r: not r, label="^VIX"
        )
    assert any(r.message.startswith("^VIX: ") for r in caplog.records)


def test_label_is_prefixed_on_final_empty_giveup(caplog):
    with caplog.at_level("WARNING"):
        result = call_with_timeout_retry(
            lambda: [], retries=1, backoff=0.01, is_empty=lambda r: not r, label="^VXN"
        )
    assert result == []
    assert any("^VXN: " in r.message and "giving up" in r.message for r in caplog.records)


def test_label_is_prefixed_on_exception_warning(caplog):
    def _boom():
        raise ValueError("down")

    with caplog.at_level("WARNING"):
        call_with_timeout_retry(_boom, retries=0, backoff=0.01, label="^VVIX")
    assert any(r.message.startswith("^VVIX: ") for r in caplog.records)
