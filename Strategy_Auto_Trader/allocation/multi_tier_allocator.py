"""Multi-tier allocation: pick best market based on VIX tier.

Tier 1 (VIX ≤ 15): SPY
Tier 2 (15 < VIX ≤ 20): ISF.L
Tier 3 (VIX > 20): CSH2.L (defensive/money-market)

Daily rebalance based on VIX close. Outputs portfolio composition and yearly breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


@dataclass
class MultiTierSignal:
    """Daily allocation decision: which tier (market) to hold."""
    date: pd.Timestamp
    vix: float | None
    tier: int  # 1=SPY, 2=ISF.L, 3=CSH2.L
    asset: str  # "SPY", "ISF.L", or "CSH2.L"
    reason: str


class MultiTierAllocator:
    """Multi-tier allocation: dynamically select market based on VIX tiers.

    Tier 1 (VIX ≤ 15): 100% SPY
    Tier 2 (15 < VIX ≤ 20): 100% ISF.L
    Tier 3 (VIX > 20): 100% CSH2.L (defensive/money-market)
    """

    def __init__(self, vix_tier1: float = 15.0, vix_tier2: float = 20.0):
        """Initialize multi-tier allocator.

        Args:
            vix_tier1: VIX threshold for Tier 1 (SPY). VIX ≤ this → Tier 1
            vix_tier2: VIX threshold for Tier 2 (ISF.L). vix_tier1 < VIX ≤ this → Tier 2
                       VIX > this → Tier 3 (CSH2.L)
        """
        self.vix_tier1 = vix_tier1
        self.vix_tier2 = vix_tier2

    def signal(self, date: pd.Timestamp, vix: float | None) -> MultiTierSignal:
        """Compute daily tier and asset allocation.

        Args:
            date: Date of signal
            vix: Daily VIX close (None if unavailable)

        Returns:
            MultiTierSignal with tier, asset, and reasoning
        """
        if vix is None:
            tier = 1  # Default to SPY if no VIX
            asset = "SPY"
            reason = "no VIX data, default to SPY"
        elif vix <= self.vix_tier1:
            tier = 1
            asset = "SPY"
            reason = f"VIX={vix:.1f} ≤ {self.vix_tier1} → Tier 1 (SPY)"
        elif vix <= self.vix_tier2:
            tier = 2
            asset = "ISF.L"
            reason = f"VIX={vix:.1f} ∈ ({self.vix_tier1}, {self.vix_tier2}] → Tier 2 (ISF.L)"
        else:
            tier = 3
            asset = "CSH2.L"
            reason = f"VIX={vix:.1f} > {self.vix_tier2} → Tier 3 (CSH2.L defensive)"

        return MultiTierSignal(
            date=date,
            vix=vix,
            tier=tier,
            asset=asset,
            reason=reason,
        )

    def backtest(
        self,
        spy_df: pd.DataFrame,
        isfl_df: pd.DataFrame,
        shv_df: pd.DataFrame,
        vix_df: pd.DataFrame | None = None,
        initial_cash: float = 100_000.0,
    ) -> dict:
        """Run multi-tier allocation backtest: pick SPY, ISF.L, or CSH2.L daily.

        Args:
            spy_df: Daily OHLCV for SPY, index=date
            isfl_df: Daily OHLCV for ISF.L, index=date
            shv_df: Daily OHLCV for tier-3 asset (CSH2.L), index=date
            vix_df: Daily VIX, index=date, use "Close" column (case-insensitive)
            initial_cash: Starting capital

        Returns:
            Dict with:
                - daily_nav: pd.DataFrame with date, nav, asset held, tier
                - signals: list of MultiTierSignal per day
                - summary: dict with Sharpe, Sortino, max_dd, tier breakdown
        """
        # Align all series to common dates
        dates = spy_df.index.intersection(isfl_df.index).intersection(shv_df.index)
        if vix_df is not None:
            dates = dates.intersection(vix_df.index)

        # Get close columns (case-insensitive)
        def get_column(df, names):
            for name in names:
                if name in df.columns:
                    return df[name]
            raise KeyError(f"None of {names} found in columns {df.columns.tolist()}")

        spy_closes = get_column(spy_df, ["Close", "CLOSE", "close"]).loc[dates].values
        isfl_closes = get_column(isfl_df, ["Close", "CLOSE", "close"]).loc[dates].values
        tier3_closes = get_column(shv_df, ["Close", "CLOSE", "close"]).loc[dates].values
        vix_closes = get_column(vix_df, ["Close", "CLOSE", "close"]).loc[dates].values if vix_df is not None else [None] * len(dates)

        nav = initial_cash
        navs = [nav]
        assets = []  # Asset held each day
        signals = []
        returns = [0.0]

        prev_spy_price = spy_closes[0]
        prev_isfl_price = isfl_closes[0]
        prev_tier3_price = tier3_closes[0]
        prev_asset = "SPY"

        for i, date in enumerate(dates):
            signal = self.signal(date, vix_closes[i])
            signals.append(signal)
            assets.append(signal.asset)

            # Compute daily return based on current holding
            if signal.asset == "SPY":
                daily_return = (spy_closes[i] - prev_spy_price) / prev_spy_price if prev_spy_price > 0 else 0
            elif signal.asset == "ISF.L":
                daily_return = (isfl_closes[i] - prev_isfl_price) / prev_isfl_price if prev_isfl_price > 0 else 0
            else:  # CSH2.L (tier 3)
                daily_return = (tier3_closes[i] - prev_tier3_price) / prev_tier3_price if prev_tier3_price > 0 else 0

            nav = nav * (1 + daily_return)
            returns.append(daily_return * 100)

            navs.append(nav)
            prev_spy_price = spy_closes[i]
            prev_isfl_price = isfl_closes[i]
            prev_tier3_price = tier3_closes[i]
            prev_asset = signal.asset

        daily_nav = pd.DataFrame({
            "date": dates,
            "nav": navs[1:],
            "asset": assets,
            "tier": [s.tier for s in signals],
            "daily_return_pct": returns[1:],
        })
        daily_nav["cumulative_return_pct"] = (daily_nav["nav"] / initial_cash - 1) * 100

        # Compute summary stats
        rets = np.array(returns[1:]) / 100
        summary = self._compute_summary(rets, initial_cash, navs[-1])

        # Tier breakdown
        summary["days_tier1_spy"] = (daily_nav["tier"] == 1).sum()
        summary["days_tier2_isfl"] = (daily_nav["tier"] == 2).sum()
        summary["days_tier3_shv"] = (daily_nav["tier"] == 3).sum()
        summary["pct_tier1"] = summary["days_tier1_spy"] / len(daily_nav) * 100 if len(daily_nav) > 0 else 0
        summary["pct_tier2"] = summary["days_tier2_isfl"] / len(daily_nav) * 100 if len(daily_nav) > 0 else 0
        summary["pct_tier3"] = summary["days_tier3_shv"] / len(daily_nav) * 100 if len(daily_nav) > 0 else 0

        return {
            "daily_nav": daily_nav,
            "signals": signals,
            "summary": summary,
        }

    @staticmethod
    def _compute_summary(returns: np.ndarray, initial_cash: float, final_nav: float) -> dict:
        """Compute Sharpe, Sortino, max drawdown."""
        if len(returns) == 0:
            return {
                "n_days": 0,
                "total_return_pct": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_drawdown_pct": 0.0,
            }

        total_ret = (final_nav - initial_cash) / initial_cash * 100

        # Sharpe (annualized, 252 trading days)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

        # Sortino (downside vol only)
        down_rets = returns[returns < 0]
        downside_std = np.std(down_rets, ddof=1) if len(down_rets) > 1 else 0.0
        sortino = (mean_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

        # Max drawdown
        navs = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(navs)
        drawdown = (navs - running_max) / running_max * 100
        max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0

        return {
            "n_days": len(returns),
            "total_return_pct": total_ret,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown_pct": max_dd,
        }
