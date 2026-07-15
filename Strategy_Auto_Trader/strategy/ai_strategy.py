"""Simple 'AI' ensemble strategy — lightweight heuristic ensemble.

This is not a machine-learning model; rather a small ensemble-style
heuristic that combines the HMM regime signal, RSI and short-term
price displacement into a single normalized score and uses thresholds to
decide entries. It is intended as a flexible hybrid that can be tuned
or replaced by a learned model later.
"""

from __future__ import annotations

from math import tanh

from ..plugins.exit_rules import StandardExitRules
from ..plugins.types import BarData, EntryDecision, ExitResult, RegimeState, TradeState


class AiEntry:
    """Heuristic ensemble entry that normalizes inputs and uses a tanh scoring."""

    buy_threshold: float = 0.4
    sell_threshold: float = -0.4

    def __init__(self, vol_filter_ok: bool = True) -> None:
        self._vol_filter_ok = vol_filter_ok

    def _normalize(self, regime: RegimeState, mom: dict) -> float:
        # regime.regime_signal ~ [-1,1] ideally; fall back to 0
        reg = float(regime.regime_signal or 0.0)
        # RSI normalized to [-1,1] around 50
        rsi = float(mom.get("cur_rsi", 50.0))
        rsi_n = (rsi - 50.0) / 50.0
        # short-term displacement (pct_from_sma20), clamp to [-0.1, 0.1]
        disp = float(mom.get("pct_from_sma20", 0.0) or 0.0)
        disp_n = max(-0.1, min(0.1, disp)) / 0.1
        # volume ratio centered at 1 -> small contribution
        vol = float(mom.get("volume_ratio", 1.0) or 1.0)
        vol_n = max(0.0, min(2.0, vol)) - 1.0

        # weighted linear combination
        raw = 0.45 * reg + 0.35 * rsi_n + 0.15 * disp_n + 0.05 * vol_n
        return raw

    def evaluate(self, regime: RegimeState, mom: dict, _volume_ratio: float, currently_in: bool = False) -> EntryDecision:
        if not self._vol_filter_ok:
            return EntryDecision(flag="HOLD", raw_flag="HOLD", score=0.0, reason="vol_filter: unsuitable (choppy/mean-reverting)")

        raw = self._normalize(regime, mom)
        # pass through tanh for smoothness
        score = tanh(raw * 2.0)
        flag = "BUY" if score >= self.buy_threshold else ("SELL" if score <= self.sell_threshold else "HOLD")

        return EntryDecision(flag=flag, raw_flag=flag, score=float(round(score, 3)), reason="ensemble_tanh")


class AiExit:
    _stop: float = 0.05
    _target: float = 0.18

    def __init__(self) -> None:
        self._impl = StandardExitRules(
            stop_loss_pct=self._stop,
            trailing_stop=0.0,
            vol_stop_mult=1.0,
            vol_stop_window=20,
            profit_stop_scale=0.35,
            min_stop_pct=0.04,
            max_hold_days=0,
            exit_on_macd_cross=True,
            exit_on_rsi_reversal=True,
            exit_on_consolidation=True,
            use_sar_stop=False,
        )

    @property
    def stop_loss_pct(self) -> float:
        return self._stop

    @property
    def take_profit_pct(self) -> float:
        return self._target

    def check(self, trade: TradeState, bar: BarData) -> ExitResult:
        return self._impl.check(trade, bar)
