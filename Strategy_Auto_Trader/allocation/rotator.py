"""VIX+HMM allocation rotator — daily signal to select market or defensive asset."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


@dataclass
class AllocationSignal:
    """Daily allocation decision: which asset to hold."""
    date: pd.Timestamp
    vix: float | None  # Daily VIX close (None if unavailable)
    pbull: float | None  # HMM P(Bull) or daily regime state
    vix_bullish: bool  # True if VIX <= vix_threshold
    pbull_bullish: bool  # True if pbull >= pbull_threshold
    both_bullish: bool  # True if both signals agree on risk-on
    market_signal: bool  # True = stay in market, False = go defensive
    reason: str  # Explanation of signal


class AllocationRotator:
    """Daily allocation: market vs defensive based on VIX + HMM P(Bull).

    Supports two modes:
    - binary: 100% in / 100% out (cliff at threshold)
    - graduated: scale allocation from vix_min to vix_max
    """

    def __init__(
        self,
        market_ticker: str = "SPY",
        vix_threshold: float = 20.0,
        pbull_threshold: float = 0.5,
        mode: str = "binary",
        vix_min: float = 10.0,
        vix_max: float = 30.0,
    ):
        self.market_ticker = market_ticker
        self.vix_threshold = vix_threshold
        self.pbull_threshold = pbull_threshold
        self.mode = mode  # "binary" or "graduated"
        self.vix_min = vix_min
        self.vix_max = vix_max

    def signal(
        self,
        date: pd.Timestamp,
        vix: float | None,
        pbull: float | None,
    ) -> AllocationSignal:
        """Compute daily allocation signal.

        Args:
            date: Date of signal
            vix: Daily VIX close (None if unavailable)
            pbull: HMM P(Bull) or daily regime state (None if unavailable)

        Returns:
            AllocationSignal with decision and reasoning.
        """
        vix_bullish = False
        pbull_bullish = False

        if self.mode == "binary":
            vix_bullish = vix is not None and vix <= self.vix_threshold
            pbull_bullish = pbull is not None and pbull >= self.pbull_threshold
            both_bullish = (
                vix_bullish and pbull_bullish
                if (vix is not None and pbull is not None)
                else (vix_bullish if vix is not None else pbull_bullish)
            )
        elif self.mode == "graduated":
            # Scale allocation based on VIX: vix_min=100%, vix_max=0%, linear between
            if vix is not None:
                alloc_pct = max(0, min(100, 100 - (vix - self.vix_min) / (self.vix_max - self.vix_min) * 100))
                vix_bullish = alloc_pct >= 50
                both_bullish = vix_bullish
            else:
                both_bullish = True  # Default bullish if no VIX
                alloc_pct = 100
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        reason_parts = []
        if vix is not None:
            if self.mode == "binary":
                status = "bullish" if vix_bullish else "bearish"
                reason_parts.append(f"VIX={vix:.1f} ({status})")
            else:
                alloc_pct = max(0, min(100, 100 - (vix - self.vix_min) / (self.vix_max - self.vix_min) * 100))
                reason_parts.append(f"VIX={vix:.1f} → {alloc_pct:.0f}% allocated")
        if pbull is not None:
            pbull_bullish = pbull >= self.pbull_threshold
            status = "bullish" if pbull_bullish else "bearish"
            reason_parts.append(f"P(Bull)={pbull:.2f} ({status})")

        reason = " + ".join(reason_parts) if reason_parts else "no signal"

        return AllocationSignal(
            date=date,
            vix=vix,
            pbull=pbull,
            vix_bullish=vix_bullish,
            pbull_bullish=pbull_bullish,
            both_bullish=both_bullish,
            market_signal=both_bullish,
            reason=reason,
        )

    def backtest(
        self,
        market_df: pd.DataFrame,
        defensive_df: pd.DataFrame,
        vix_df: pd.DataFrame | None = None,
        pbull_series: pd.Series | None = None,
        initial_cash: float = 100_000.0,
    ) -> dict:
        """Run allocation backtest: 100% market or 100% defensive daily.

        Args:
            market_df: Daily OHLCV for market asset (SPY/FTSE100), index=date
            defensive_df: Daily OHLCV for defensive asset, index=date
            vix_df: Daily VIX, index=date, use "Close" column (case-insensitive)
            pbull_series: Daily HMM P(Bull), index=date
            initial_cash: Starting capital

        Returns:
            Dict with:
                - daily_nav: pd.DataFrame with date, nav, pct_return, allocation%
                - signals: list of AllocationSignal per day
                - summary: dict with Sharpe, Sortino, max_dd, pct_invested
        """
        # Align all series to market dates
        dates = market_df.index.intersection(defensive_df.index)
        if vix_df is not None:
            dates = dates.intersection(vix_df.index)
        if pbull_series is not None:
            dates = dates.intersection(pbull_series.index)

        # Find close column (case-insensitive)
        def get_column(df, names):
            for name in names:
                if name in df.columns:
                    return df[name]
            raise KeyError(f"None of {names} found in columns {df.columns.tolist()}")

        market_prices = get_column(market_df, ["Close", "CLOSE", "close"]).loc[dates].values
        defensive_prices = get_column(defensive_df, ["Close", "CLOSE", "close"]).loc[dates].values
        if vix_df is not None:
            vix_closes = get_column(vix_df, ["Close", "CLOSE", "close"]).loc[dates].values
        else:
            vix_closes = [None] * len(dates)
        pbull_values = pbull_series.loc[dates].values if pbull_series is not None else [None] * len(dates)

        nav = initial_cash
        navs = [nav]
        allocations = []  # % in market (100 - this = % in defensive)
        signals = []
        returns = [0.0]  # Daily %return

        prev_market_price = market_prices[0]
        prev_defensive_price = defensive_prices[0]
        prev_allocation = 0.5  # Start neutral

        for i, date in enumerate(dates):
            signal = self.signal(date, vix_closes[i], pbull_values[i])
            signals.append(signal)

            # Determine allocation based on mode
            if self.mode == "binary":
                allocation = 1.0 if signal.market_signal else 0.0
            elif self.mode == "graduated":
                # Compute graduated allocation from VIX
                if vix_closes[i] is not None:
                    allocation = max(0, min(1, (self.vix_max - vix_closes[i]) / (self.vix_max - self.vix_min)))
                else:
                    allocation = 1.0  # Default to market if no VIX
            else:
                allocation = 1.0  # Fallback

            allocations.append(allocation)

            # Compute daily return
            market_return = (market_prices[i] - prev_market_price) / prev_market_price
            defensive_return = (defensive_prices[i] - prev_defensive_price) / prev_defensive_price

            # Blended return: previous allocation + current prices
            blended_return = allocation * market_return + (1 - allocation) * defensive_return
            nav = nav * (1 + blended_return)
            returns.append(blended_return * 100)  # Store as %

            navs.append(nav)
            prev_market_price = market_prices[i]
            prev_defensive_price = defensive_prices[i]
            prev_allocation = allocation

        daily_nav = pd.DataFrame({
            "date": dates,
            "nav": navs[1:],  # Skip initial
            "allocation_pct": [a * 100 for a in allocations],  # % in market
            "daily_return_pct": returns[1:],  # Skip initial 0
        })
        daily_nav["cumulative_return_pct"] = (daily_nav["nav"] / initial_cash - 1) * 100

        # Compute summary stats
        rets = np.array(returns[1:]) / 100  # Back to decimal
        summary = self._compute_summary(rets, allocations, initial_cash, navs[-1])
        summary["vix_threshold"] = self.vix_threshold
        summary["pbull_threshold"] = self.pbull_threshold
        summary["market_ticker"] = self.market_ticker

        return {
            "daily_nav": daily_nav,
            "signals": signals,
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

        # Total return
        total_ret = (final_nav - initial_cash) / initial_cash

        # Sharpe (annualized, assume 252 trading days)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

        # Sortino (only downside volatility)
        down_rets = returns[returns < 0]
        downside_std = np.std(down_rets, ddof=1) if len(down_rets) > 1 else 0.0
        sortino = (mean_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

        # Max drawdown
        cumulative = np.cumprod(1 + returns) * initial_cash
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = np.min(drawdowns)

        # % time in market
        pct_in_market = np.mean(allocations) * 100

        return {
            "n_days": len(returns),
            "total_return_pct": total_ret * 100,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown_pct": max_dd * 100,
            "pct_time_in_market": pct_in_market,
        }
