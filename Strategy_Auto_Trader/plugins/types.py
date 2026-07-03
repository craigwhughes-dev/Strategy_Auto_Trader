"""Shared typed data containers passed between plugin slots."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegimeState:
    """Output of one forward step of the HMM regime model."""
    p_bull: float
    p_bear: float
    p_bull_smooth: float
    regime_signal: float    # p_bull_smooth - p_bear
    hmm_vote: int           # 0=Bear, 1=Sideways, 2=Bull


@dataclass
class TradeState:
    """Snapshot of the current open-trade variables for one bar."""
    position: float
    entry_price: float
    stop_level: float
    target_level: float
    peak_price_since_entry: float
    days_in_trade: int
    entry_bar: int


@dataclass
class BarData:
    """Per-bar context needed by exit-rule plugins."""
    t: int
    cur_close: float
    daily_vol_t: float      # rolling-volatility value (for vol-stop computation)
    use_sar_stop: bool
    sar_val: float | None
    need_exit: bool         # True if any exit-indicator flag is enabled
    macd_bc: bool           # MACD bearish cross on this bar
    rsi_ob: bool            # RSI dropped below 70 on this bar
    rsi_ml: bool            # RSI momentum loss on this bar
    consol: bool            # Consolidation (BB + ATR squeeze) on this bar


@dataclass
class ExitResult:
    """Output of an exit-rule check for one bar."""
    exit_hit: bool
    sell_reason: str
    peak_price_since_entry: float
    days_in_trade: int


@dataclass
class EntryDecision:
    """Output of an entry-strategy evaluation for one bar."""
    flag: str        # "BUY" | "HOLD" | "SELL" — final decision after gate
    raw_flag: str    # pre-gate composite signal (used by transition guard)
    score: float     # weighted vote score
    reason: str      # gate reason if overridden, else empty
