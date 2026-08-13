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
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def call_with_timeout_retry(
    fn: Callable[[], T],
    timeout: float = 15.0,
    retries: int = 1,
    backoff: float = 3.0,
) -> T | None:
    """Call fn() with a hard per-attempt timeout, retrying on timeout or exception.

    The worker thread is never killed on timeout (Python threads can't be
    forcibly killed) — it's abandoned via shutdown(wait=False) so the caller
    doesn't block waiting for it. It dies with the process (each live_daemon
    ticker already runs in its own short-lived multiprocessing child), so an
    abandoned thread outliving one attempt is not a leak.

    Returns fn()'s result, or None if every attempt times out or raises.
    """
    for attempt in range(retries + 1):
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(fn)
        try:
            result = future.result(timeout=timeout)
            pool.shutdown(wait=False)
            return result
        except Exception:
            pool.shutdown(wait=False)
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
    return None
