"""4-tier allocation: Nasdaq/SPY/ISF.L/CSH2.L with VXN + VIX gating.

Tier 1: Nasdaq (EQGB.L) — gated by VXN threshold
Tier 2: SPY — gated by VIX ≤ vix_tier1
Tier 3: ISF.L — gated by vix_tier1 < VIX ≤ vix_tier2
Tier 4: CSH2.L — defensive fallback
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


@dataclass
class MultiTierSignal:
    """Daily allocation decision."""
    date: pd.Timestamp
    vix: float | None
    vxn: float | None
    tier: int  # 1=Nasdaq, 2=SPY, 3=ISF.L, 4=CSH2.L
    asset: str
    reason: str


class MultiTierAllocator4Tier:
    """4-tier allocation with VXN + VIX gating.

    Tier 1: Nasdaq (VXN ≤ vxn_threshold)
    Tier 2: SPY (VIX ≤ vix_tier1)
    Tier 3: ISF.L (vix_tier1 < VIX ≤ vix_tier2)
    Tier 4: CSH2.L (defensive, always available)
    """

    def __init__(
        self,
        vxn_threshold: float = 18.0,
        vix_tier1: float = 15.0,
        vix_tier2: float = 17.5,
    ):
        self.vxn_threshold = vxn_threshold
        self.vix_tier1 = vix_tier1
        self.vix_tier2 = vix_tier2

    def signal(
        self, date: pd.Timestamp, vxn: float | None, vix: float | None
    ) -> MultiTierSignal:
        """Compute daily tier allocation.

        Args:
            date: Date of signal
            vxn: Daily VXN close (None if unavailable)
            vix: Daily VIX close (None if unavailable)

        Returns:
            MultiTierSignal with tier, asset, and reasoning
        """
        # Tier 1: Check VXN (Nasdaq)
        if vxn is not None and vxn <= self.vxn_threshold:
            tier = 1
            asset = "Nasdaq"
            reason = f"VXN={vxn:.1f} ≤ {self.vxn_threshold} → Tier 1 (Nasdaq)"
        # Tier 2-4: VIX-based fallback
        elif vix is None:
            tier = 4
            asset = "CSH2.L"
            reason = "no VIX data, fallback to CSH2.L"
        elif vix <= self.vix_tier1:
            tier = 2
            asset = "SPY"
            reason = f"VIX={vix:.1f} ≤ {self.vix_tier1} → Tier 2 (SPY)"
        elif vix <= self.vix_tier2:
            tier = 3
            asset = "ISF.L"
            reason = f"VIX={vix:.1f} ∈ ({self.vix_tier1}, {self.vix_tier2}] → Tier 3 (ISF.L)"
        else:
            tier = 4
            asset = "CSH2.L"
            reason = f"VIX={vix:.1f} > {self.vix_tier2} → Tier 4 (CSH2.L)"

        return MultiTierSignal(
            date=date,
            vix=vix,
            vxn=vxn,
            tier=tier,
            asset=asset,
            reason=reason,
        )

    def backtest(
        self,
        nasdaq_df: pd.DataFrame,
        spy_df: pd.DataFrame,
        isfl_df: pd.DataFrame,
        csh2_df: pd.DataFrame,
        vxn_df: pd.DataFrame | None = None,
        vix_df: pd.DataFrame | None = None,
        initial_cash: float = 100_000.0,
    ) -> dict:
        """Run 4-tier allocation backtest.

        Args:
            nasdaq_df: Daily OHLCV for Nasdaq (EQGB.L proxy), index=date
            spy_df: Daily OHLCV for SPY, index=date
            isfl_df: Daily OHLCV for ISF.L, index=date
            csh2_df: Daily OHLCV for CSH2.L, index=date
            vxn_df: Daily VXN, index=date
            vix_df: Daily VIX, index=date
            initial_cash: Starting capital

        Returns:
            Dict with daily_nav, signals, summary
        """
        # Align all series to common dates
        dates = (
            nasdaq_df.index.intersection(spy_df.index)
            .intersection(isfl_df.index)
            .intersection(csh2_df.index)
        )
        if vxn_df is not None:
            dates = dates.intersection(vxn_df.index)
        if vix_df is not None:
            dates = dates.intersection(vix_df.index)

        # Get close columns (case-insensitive)
        def get_column(df, names):
            for name in names:
                if name in df.columns:
                    return df[name]
            raise KeyError(f"None of {names} found in columns {df.columns.tolist()}")

        nasdaq_closes = get_column(nasdaq_df, ["Close", "CLOSE", "close"]).loc[dates].values
        spy_closes = get_column(spy_df, ["Close", "CLOSE", "close"]).loc[dates].values
        isfl_closes = get_column(isfl_df, ["Close", "CLOSE", "close"]).loc[dates].values
        csh2_closes = get_column(csh2_df, ["Close", "CLOSE", "close"]).loc[dates].values
        vxn_closes = (
            get_column(vxn_df, ["Close", "CLOSE", "close"]).loc[dates].values
            if vxn_df is not None
            else [None] * len(dates)
        )
        vix_closes = (
            get_column(vix_df, ["Close", "CLOSE", "close"]).loc[dates].values
            if vix_df is not None
            else [None] * len(dates)
        )

        nav = initial_cash
        navs = [nav]
        assets = []
        signals = []
        returns = [0.0]

        prev_nasdaq = nasdaq_closes[0]
        prev_spy = spy_closes[0]
        prev_isfl = isfl_closes[0]
        prev_csh2 = csh2_closes[0]

        for i, date in enumerate(dates):
            signal = self.signal(date, vxn_closes[i], vix_closes[i])
            signals.append(signal)
            assets.append(signal.asset)

            # Compute daily return based on current holding
            if signal.asset == "Nasdaq":
                daily_return = (
                    (nasdaq_closes[i] - prev_nasdaq) / prev_nasdaq
                    if prev_nasdaq > 0
                    else 0
                )
            elif signal.asset == "SPY":
                daily_return = (
                    (spy_closes[i] - prev_spy) / prev_spy if prev_spy > 0 else 0
                )
            elif signal.asset == "ISF.L":
                daily_return = (
                    (isfl_closes[i] - prev_isfl) / prev_isfl
                    if prev_isfl > 0
                    else 0
                )
            else:  # CSH2.L
                daily_return = (
                    (csh2_closes[i] - prev_csh2) / prev_csh2
                    if prev_csh2 > 0
                    else 0
                )

            nav = nav * (1 + daily_return)
            returns.append(daily_return * 100)

            navs.append(nav)
            prev_nasdaq = nasdaq_closes[i]
            prev_spy = spy_closes[i]
            prev_isfl = isfl_closes[i]
            prev_csh2 = csh2_closes[i]

        daily_nav = pd.DataFrame(
            {
                "date": dates,
                "nav": navs[1:],
                "asset": assets,
                "tier": [s.tier for s in signals],
                "daily_return_pct": returns[1:],
            }
        )
        daily_nav["cumulative_return_pct"] = (daily_nav["nav"] / initial_cash - 1) * 100

        # Compute summary stats
        rets = np.array(returns[1:]) / 100
        summary = self._compute_summary(rets, initial_cash, navs[-1])

        # Tier breakdown
        for tier_num in range(1, 5):
            tier_name = {1: "Nasdaq", 2: "SPY", 3: "ISF.L", 4: "CSH2.L"}[tier_num]
            days = (daily_nav["tier"] == tier_num).sum()
            summary[f"days_tier{tier_num}_{tier_name}"] = days
            summary[f"pct_tier{tier_num}"] = (
                days / len(daily_nav) * 100 if len(daily_nav) > 0 else 0
            )

        return {
            "daily_nav": daily_nav,
            "signals": signals,
            "summary": summary,
        }

    @staticmethod
    def _compute_summary(
        returns: np.ndarray, initial_cash: float, final_nav: float
    ) -> dict:
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
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        sharpe = (mean_ret * 252 / std_ret) if std_ret > 0 else 0.0

        down_rets = returns[returns < 0]
        downside_std = np.std(down_rets, ddof=1) if len(down_rets) > 1 else 0.0
        sortino = (mean_ret * 252 / downside_std) if downside_std > 0 else 0.0

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
