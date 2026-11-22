"""Hard wall-clock timeout + retry wrapper for flaky network calls (yfinance).

yfinance calls (yf.download, Ticker().info) don't expose a single reliable
timeout: some chain multiple sub-requests (cookie/crumb fetch, then the real
call), each with its own internal timeout that can stack well past any one
request's limit, and Ticker().info takes no timeout kwarg at all. A worker
thread with future.result(timeout=...) is the only way to hard-cap wall time
regardless of what yfinance does internally underneath.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def call_with_timeout_retry(
    fn: Callable[[], T],
    timeout: float = 15.0,
    retries: int = 1,
    backoff: float = 3.0,
    is_empty: Callable[[T], bool] | None = None,
    label: str | None = None,
) -> T | None:
    """Call fn() with a hard per-attempt timeout, retrying on timeout, exception, or empty result.

    The worker thread is never killed on timeout (Python threads can't be
    forcibly killed) — it's abandoned via shutdown(wait=False) so the caller
    doesn't block waiting for it. It dies with the process (each live_daemon
    ticker already runs in its own short-lived multiprocessing child), so an
    abandoned thread outliving one attempt is not a leak.

    yfinance can return an empty-but-valid result (no exception) when Yahoo has no
    data for a symbol/interval right now — logged by yfinance's own logger as
    "possibly delisted", not raised. Without `is_empty`, that result is returned
    as-is on the first attempt, same as any other call. Pass `is_empty` to treat
    that case as retryable too, e.g. `is_empty=lambda df: df.empty`.

    `label` is prefixed to every log line this call emits (e.g. the symbol/ticker being
    fetched) — without it, concurrent callers of a shared fetch helper (e.g. VIX/VXN/VVIX
    all going through the same yfinance wrapper) log indistinguishable "attempt 1/2" lines,
    making a multi-symbol outage look like one ambiguous event instead of three identifiable ones.

    Returns fn()'s result, or None if every attempt times out or raises.
    """
    tag = f"{label}: " if label else ""
    for attempt in range(retries + 1):
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(fn)
        try:
            result = future.result(timeout=timeout)
            pool.shutdown(wait=False)
            if is_empty is not None and is_empty(result):
                if attempt < retries:
                    logger.warning(
                        f"{tag}call_with_timeout_retry: attempt {attempt + 1}/{retries + 1} "
                        f"returned empty result, retrying"
                    )
                    time.sleep(backoff * (attempt + 1))
                    continue
                logger.warning(
                    f"{tag}call_with_timeout_retry: attempt {attempt + 1}/{retries + 1} "
                    f"returned empty result, giving up"
                )
            return result
        except Exception as e:
            pool.shutdown(wait=False)
            logger.warning(
                f"{tag}call_with_timeout_retry: attempt {attempt + 1}/{retries + 1} "
                f"failed ({type(e).__name__}: {e})"
            )
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
    return None
