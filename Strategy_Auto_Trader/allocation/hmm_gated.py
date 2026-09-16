"""HMM-gated allocation: go defensive only if VIX high AND HMM bearish.

Avoids 2021 false alarm where VIX stayed high but market rallied.
Requires hourly SPY data to compute HMM regime.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..plugins.persistent_hmm import PersistentHMMRegimeModel
from ..quant_hmm.quant_engine import discretize_p_bull
from .rotator import AllocationRotator

_log = logging.getLogger(__name__)

HMM_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "hmm_cache"
HMM_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class HMMGatedAllocator(AllocationRotator):
    """Allocation with HMM P(Bull) gate to filter false VIX signals."""

    def __init__(
        self,
        market_ticker: str = "SPY",
        vix_threshold: float = 20.0,
        pbull_gate: float = 0.4,
        hmm_dates: pd.DatetimeIndex | None = None,
        hmm_closes: np.ndarray | None = None,
    ):
        super().__init__(market_ticker=market_ticker, vix_threshold=vix_threshold)
        self.pbull_gate = pbull_gate  # If P(Bull) >= this, don't go defensive
        self.hmm_model = None

        if hmm_dates is not None and hmm_closes is not None:
            self._build_hmm(hmm_dates, hmm_closes)

    def _build_hmm(self, dates: pd.DatetimeIndex, closes: np.ndarray) -> None:
        """Fit or load HMM from cache."""
        try:
            self.hmm_model = PersistentHMMRegimeModel(
                cache_path=HMM_CACHE_DIR / f"{self.market_ticker}_daily.pkl",
                dates=dates,
                closes=closes,
                refit_bars=500,
                min_train_bars=500,
            )
            _log.info(f"Built HMM regime model for {self.market_ticker}")
        except Exception as e:
            _log.warning(f"Failed to build HMM: {e}; falling back to VIX-only gating")
            self.hmm_model = None

    def signal_with_hmm(
        self,
        date: pd.Timestamp,
        vix: float | None,
        pbull: float | None,
    ) -> tuple[bool, str]:
        """Compute gated allocation signal.

        Returns:
            (market_signal, reason)
            market_signal: True = stay in market, False = go defensive
        """
        if vix is None:
            return True, "No VIX; default bullish"

        vix_bullish = vix <= self.vix_threshold

        # If VIX says bullish, always stay in market
        if vix_bullish:
            return True, f"VIX={vix:.1f} <= {self.vix_threshold} (bullish)"

        # If VIX says bearish, check HMM gate
        if pbull is None:
            # No HMM, trust VIX alone
            return False, f"VIX={vix:.1f} > {self.vix_threshold} (bearish, no HMM)"

        pbull_passes_gate = pbull >= self.pbull_gate
        if pbull_passes_gate:
            # HMM says bull despite VIX: override VIX signal (avoid 2021 false alarm)
            return True, f"VIX={vix:.1f} elevated but P(Bull)={pbull:.2f} >= gate; stay market"

        # Both VIX and HMM say bearish
        return False, f"VIX={vix:.1f} + P(Bull)={pbull:.2f} < gate; go defensive"

    def backtest_with_hmm(
        self,
        market_df: pd.DataFrame,
        defensive_df: pd.DataFrame,
        vix_df: pd.DataFrame | None = None,
        pbull_series: pd.Series | None = None,
        initial_cash: float = 100_000.0,
    ) -> dict:
        """Run allocation backtest with HMM gating.

        Similar to parent backtest() but uses signal_with_hmm instead.
        """
        # Align dates
        dates = market_df.index.intersection(defensive_df.index)
        if vix_df is not None:
            dates = dates.intersection(vix_df.index)
        if pbull_series is not None:
            dates = dates.intersection(pbull_series.index)

        def get_column(df, names):
            for name in names:
                if name in df.columns:
                    return df[name]
            raise KeyError(f"None of {names} found in columns {df.columns.tolist()}")

        market_prices = get_column(market_df, ["Close", "CLOSE", "close"]).loc[dates].values
        defensive_prices = get_column(defensive_df, ["Close", "CLOSE", "close"]).loc[dates].values
        vix_closes = (
            get_column(vix_df, ["Close", "CLOSE", "close"]).loc[dates].values if vix_df is not None else [None] * len(dates)
        )
        pbull_values = pbull_series.loc[dates].values if pbull_series is not None else [None] * len(dates)

        nav = initial_cash
        navs = [nav]
        allocations = []
        reasons = []
        returns = [0.0]

        prev_market_price = market_prices[0]
        prev_defensive_price = defensive_prices[0]

        for i, date in enumerate(dates):
            market_signal, reason = self.signal_with_hmm(date, vix_closes[i], pbull_values[i])
            reasons.append((date, reason))

            allocation = 1.0 if market_signal else 0.0
            allocations.append(allocation)

            market_return = (market_prices[i] - prev_market_price) / prev_market_price
            defensive_return = (defensive_prices[i] - prev_defensive_price) / prev_defensive_price

            blended_return = allocation * market_return + (1 - allocation) * defensive_return
            nav = nav * (1 + blended_return)
            returns.append(blended_return * 100)

            prev_market_price = market_prices[i]
            prev_defensive_price = defensive_prices[i]

        daily_nav = pd.DataFrame({
            "date": dates,
            "nav": navs[1:],
            "allocation_pct": [a * 100 for a in allocations],
            "daily_return_pct": returns[1:],
        })
        daily_nav["cumulative_return_pct"] = (daily_nav["nav"] / initial_cash - 1) * 100

        rets = np.array(returns[1:]) / 100
        summary = self._compute_summary(rets, allocations, initial_cash, navs[-1])
        summary["vix_threshold"] = self.vix_threshold
        summary["pbull_gate"] = self.pbull_gate

        return {
            "daily_nav": daily_nav,
            "reasons": reasons,
            "summary": summary,
        }

    @staticmethod
    def _compute_summary(returns: np.ndarray, allocations: list[float], initial_cash: float, final_nav: float) -> dict:
        """Compute Sharpe, Sortino, max drawdown, % time invested."""
        if len(returns) == 0:
            return {
                "n_days": 0,
                "total_return_pct": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_drawdown_pct": 0.0,
                "pct_time_in_market": 0.0,
            }

        total_ret = (final_nav - initial_cash) / initial_cash
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        sharpe = (mean_ret * 252 / std_ret) if std_ret > 0 else 0.0

        down_rets = returns[returns < 0]
        downside_std = np.std(down_rets, ddof=1) if len(down_rets) > 1 else 0.0
        sortino = (mean_ret * 252 / downside_std) if downside_std > 0 else 0.0

        cumulative = np.cumprod(1 + returns) * initial_cash
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = np.min(drawdowns)

        pct_in_market = np.mean(allocations) * 100

        return {
            "n_days": len(returns),
            "total_return_pct": total_ret * 100,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown_pct": max_dd * 100,
            "pct_time_in_market": pct_in_market,
        }
