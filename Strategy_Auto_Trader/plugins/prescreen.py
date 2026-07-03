"""Ticker prescreen plugin implementations.

VolatilityPrescreen — default, wraps quant_hmm/vol_screen.volatility_profile.
NullPrescreen       — always passes (returns empty dict).

The prescreen is called once per ticker before the engine loop, not per-bar,
so it is not wired into consolidated_engine.py — it is used by batch.py and
quant_run.py.  The plugin exists so batch dispatchers can swap it out via
config without changing their call sites.
"""

from __future__ import annotations


class VolatilityPrescreen:
    """Runs volatility_profile from vol_screen and returns the result.

    Satisfies PrescreenProtocol.

    Parameters
    ----------
    period : str
        yfinance period string for the daily data fetch (default '2y').
    min_efficiency_ratio : float
        Tickers with efficiency ratio below this are considered choppy and
        should be skipped.  The caller (batch.py) enforces this threshold;
        this plugin just provides the raw profile.
    """

    def __init__(self, period: str = "2y") -> None:
        self._period = period

    def __call__(self, ticker: str) -> dict | None:
        from ..quant_hmm.vol_screen import volatility_profile
        return volatility_profile(ticker, period=self._period)


class NullPrescreen:
    """Always passes — returns an empty dict so callers treat every ticker
    as accepted.

    Satisfies PrescreenProtocol.
    """

    def __call__(self, ticker: str) -> dict | None:
        return {}
