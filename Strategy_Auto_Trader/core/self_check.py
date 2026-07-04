"""Startup self-checks — fail loudly at process start, not silently mid-trade.

A half-broken environment can import cleanly yet fail at every model fit:
a partial hmmlearn uninstall (2026-07-04) left `import hmmlearn` working
while its compiled submodules were gone, so every refit silently produced
no model and the daemon would have traded without its regime signal.

run_startup_checks() exercises the real code paths once (including an
actual tiny HMM fit) and raises SelfCheckError listing every failure, so
entry points can refuse to start instead of degrading silently.
"""

from __future__ import annotations

import logging

import numpy as np


class SelfCheckError(RuntimeError):
    """One or more startup self-checks failed; the process must not trade."""


def check_data_stack() -> str:
    """Verify the numeric/data stack imports and performs trivial work."""
    import pandas as pd
    import yfinance

    frame = pd.DataFrame({"x": np.arange(5, dtype=float)})
    if float(frame["x"].mean()) != 2.0:
        raise SelfCheckError("pandas/numpy produced wrong arithmetic result")
    return (f"numpy {np.__version__}, pandas {pd.__version__}, "
            f"yfinance {yfinance.__version__}")


def check_hmm_fit() -> str:
    """Fit a real (tiny) HMM through the production fit path."""
    import hmmlearn
    from hmmlearn import hmm  # noqa: F401 — surfaces broken submodules directly

    from ..quant_hmm.quant_engine import fit_hmm_expanding

    rng = np.random.default_rng(0)
    returns = np.concatenate([
        rng.normal(0.002, 0.005, 120),
        rng.normal(-0.002, 0.02, 120),
    ])
    result = fit_hmm_expanding(returns, n_components=2, n_seeds=2, n_iter=20)
    if result is None:
        raise SelfCheckError(
            "hmmlearn imported but a test HMM fit produced no model")
    model, order = result
    if model is None or len(order) != 2:
        raise SelfCheckError("test HMM fit returned a malformed result")
    return f"hmmlearn {hmmlearn.__version__} test fit OK"


def check_broker_module() -> str:
    """Verify the IBKR client library is importable (connection is checked
    separately at connect time, since TWS may legitimately start later)."""
    import ib_insync
    return f"ib_insync {ib_insync.__version__} importable"


def check_broker_connectivity(
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 97,
    retries: int = 1,
) -> str:
    """Connect to TWS / IB Gateway and query the session's accounts.

    Uses its own client id so it never collides with the trading
    connection. Retries once because the ib_insync handshake is known to
    time out transiently even when TWS is healthy.
    """
    from ..broker.ibkr_adapter import IBKRAdapter

    adapter = IBKRAdapter(host=host, port=port, client_id=client_id)
    last_exc: Exception | None = None
    for _ in range(retries + 1):
        try:
            adapter.connect()
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            adapter.disconnect()
    if last_exc is not None:
        raise SelfCheckError(
            f"cannot connect to TWS/Gateway at {host}:{port} ({last_exc}) — "
            "is it running, logged in, with API enabled on this port?"
        ) from last_exc

    try:
        accounts = adapter.managed_accounts()
    finally:
        adapter.disconnect()
    if not accounts:
        raise SelfCheckError(
            f"connected to {host}:{port} but session reports no accounts")
    return f"connected to {host}:{port}, accounts: {', '.join(accounts)}"


def run_startup_checks(
    *,
    require_hmm: bool = True,
    require_broker: bool = False,
    logger: logging.Logger | None = None,
) -> list[str]:
    """Run applicable checks; raise SelfCheckError listing every failure.

    require_broker adds both the ib_insync import check and a real
    connect-and-query against TWS/Gateway — a process configured to place
    real orders must not start unless the broker is actually reachable.

    Returns the per-check success messages when everything passes.
    """
    checks: list[tuple[str, object]] = [("data stack", check_data_stack)]
    if require_hmm:
        checks.append(("HMM fit", check_hmm_fit))
    if require_broker:
        checks.append(("broker module", check_broker_module))
        checks.append(("broker connectivity", check_broker_connectivity))

    passed: list[str] = []
    failures: list[str] = []
    for name, fn in checks:
        try:
            msg = fn()
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            if logger:
                logger.error(f"Self-check FAILED [{name}]: {exc}")
            continue
        passed.append(f"{name}: {msg}")
        if logger:
            logger.info(f"Self-check OK [{name}]: {msg}")

    if failures:
        raise SelfCheckError(
            "startup self-checks failed, refusing to run — "
            + "; ".join(failures)
        )
    return passed
