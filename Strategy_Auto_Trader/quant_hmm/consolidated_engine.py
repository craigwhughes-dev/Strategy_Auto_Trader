"""Consolidated walk-forward backtest engine.

Combines the hourly HMM regime-probability chassis from quant_engine with the
daily engine's vote-based composite signal, quality gate, and full exit set.

Key design decisions
---------------------
- Observable Markov model is dropped.  The Gaussian HMM's smoothed P(Bull) is
  discretized into a 3-way bull/sideways/bear vote (via discretize_p_bull) and
  fed into composite_signal()'s hmm_state slot.  The 'markov' weight is set to
  zero so it does not double-count.
- Votes (RSI, trend/SMA20+50, SMA200, volume) are recomputed on every hourly
  bar from the precomputed indicator series — no daily-cadence broadcast.
- The quality gate (_apply_quality_gate) runs after composite_signal on every
  BUY bar and on every in-trade bar, using the same mom dict as the vote.
- The regime_signal passed to the quality gate is p_bull_smooth - p_bear (the
  continuous analogue of P(Bull)-P(Bear)), matching the gate's -1..1 thresholds.
- Exit priority: hard stop-loss/take-profit > trailing/vol-stop > max-hold >
  SAR > MACD/RSI/consolidation.  The stop-loss/take-profit are mapped to the
  _check_exit_conditions rr_stop_level/rr_target_level mechanism (rr_ratio=1).
- Kelly position sizing is unchanged (driven by trailing trade P&L).
- Each algorithmic concern is now behind a typing.Protocol plugin slot so
  individual components can be swapped out independently (see plugins/).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.momentum import (
    compute_rsi, compute_sma, composite_signal,
    compute_parabolic_sar, compute_macd, compute_bollinger, compute_atr,
)
from ..core.quality_gate import _apply_quality_gate
from ..core.exits import _effective_stop_for_bar, _check_exit_conditions
from .quant_engine import (
    fit_hmm_expanding, _forward_step_incremental,
    kelly_fraction, discretize_p_bull,
    _compute_effective_thresholds, _compute_volume_ratio,
    _sharpe, _max_dd, _simulate_portfolio_value, _build_quant_backtest_stats,
    _empty_result,
)
from ..plugins.types import BarData, EntryDecision, ExitResult, RegimeState, TradeState
from ..plugins.protocols import (
    ContextAdjusterProtocol,
    ExitRulesProtocol,
    PositionSizerProtocol,
    QualityGateProtocol,
    RegimeModelProtocol,
    SignalGeneratorProtocol,
)
from ..strategy.base import EntryStrategyProtocol, ExitStrategyProtocol

# Weight vector: markov slot zeroed — HMM carries that role via hmm_state.
_CONSOLIDATED_WEIGHTS = {
    "markov": 0.0,
    "rsi":    1.5,
    "trend":  1.0,
    "sma200": 2.0,
    "volume": 1.0,
    "hmm":    1.5,  # upweighted since it now carries both regime signals
}


def _precompute_hourly_vote_series(
    close: pd.Series,
    volume: np.ndarray | None,
    *,
    rsi_period: int = 14,
    ma_fast: int = 20,
    ma_slow: int = 50,
    ma_trend: int = 200,
    vol_stop_window: int = 20,
    exit_on_macd_cross: bool = False,
    exit_on_rsi_reversal: bool = True,
    exit_on_consolidation: bool = False,
    use_sar_stop: bool = False,
    sar_af_start: float = 0.02,
    sar_af_step: float = 0.02,
    sar_af_max: float = 0.20,
) -> dict:
    """Precompute all causal indicator series needed by the consolidated per-bar loop."""
    close_s = pd.Series(close) if not isinstance(close, pd.Series) else close

    rsi_full   = compute_rsi(close_s, rsi_period)
    sma20_full = compute_sma(close_s, ma_fast)
    sma50_full = compute_sma(close_s, ma_slow)
    sma200_full = compute_sma(close_s, ma_trend)

    log_rets = np.log(close_s / close_s.shift(1))
    rolling_vol_full = log_rets.rolling(vol_stop_window).std()

    rsi_x_above_50 = (rsi_full >= 50) & (rsi_full.shift(1) < 50)
    rsi_x_below_40 = (rsi_full < 40) & (rsi_full.shift(1) >= 40)

    vol_ratio_full = None
    if volume is not None and len(volume) > 20:
        vol_s = pd.Series(volume)
        vol_avg = vol_s.rolling(20).mean()
        vol_ratio_full = (vol_s / vol_avg).values

    need_exit = exit_on_macd_cross or exit_on_rsi_reversal or exit_on_consolidation
    macd_bear_x = rsi_ob_exit = rsi_mom_loss = consol_full = None
    if need_exit:
        macd_line_full, macd_sig_full, _ = compute_macd(close_s)
        macd_bear_x = (macd_line_full < macd_sig_full) & (macd_line_full.shift(1) >= macd_sig_full.shift(1))
        rsi_ob_exit = (rsi_full < 70) & (rsi_full.shift(1) >= 70)
        rsi_mom_loss = ((rsi_full < 50) & (rsi_full.shift(1) >= 50)
                        & (rsi_full.rolling(6).max().shift(1) >= 60))
        bb_mid, bb_up, bb_lo = compute_bollinger(close_s)
        bb_w = (bb_up - bb_lo) / bb_mid
        bb_w_avg = bb_w.rolling(20).mean()
        bb_sq = bb_w < bb_w_avg
        atr_s = compute_atr(close_s, window=14)
        atr_avg_s = atr_s.rolling(20).mean()
        atr_r_s = atr_s / atr_avg_s
        consol_full = bb_sq & (atr_r_s < 0.75)

    sar_full = None
    if use_sar_stop:
        sar_full = compute_parabolic_sar(close_s, af_start=sar_af_start,
                                         af_step=sar_af_step, af_max=sar_af_max)

    return {
        "rsi_full": rsi_full,
        "sma20_full": sma20_full,
        "sma50_full": sma50_full,
        "sma200_full": sma200_full,
        "rolling_vol_full": rolling_vol_full,
        "rsi_x_above_50": rsi_x_above_50,
        "rsi_x_below_40": rsi_x_below_40,
        "vol_ratio_full": vol_ratio_full,
        "need_exit": need_exit,
        "macd_bear_x": macd_bear_x,
        "rsi_ob_exit": rsi_ob_exit,
        "rsi_mom_loss": rsi_mom_loss,
        "consol_full": consol_full,
        "sar_full": sar_full,
    }


def _build_mom_snap(
    t: int,
    cur_close: float,
    pre: dict,
    rsi_cross_lookback: int = 4,
) -> dict:
    """Snapshot the momentum dict for bar t (matches composite_signal / quality_gate input)."""
    rsi_full    = pre["rsi_full"]
    sma20_full  = pre["sma20_full"]
    sma50_full  = pre["sma50_full"]
    sma200_full = pre["sma200_full"]

    cur_rsi   = float(rsi_full.iloc[t])
    cur_sma20 = float(sma20_full.iloc[t])
    cur_sma50 = float(sma50_full.iloc[t])

    recent_above_50 = bool(pre["rsi_x_above_50"].iloc[max(0, t - rsi_cross_lookback): t + 1].any())
    recent_below_40 = bool(pre["rsi_x_below_40"].iloc[max(0, t - rsi_cross_lookback): t + 1].any())

    snap: dict = {
        "cur_rsi": cur_rsi,
        "recent_cross_above_50": recent_above_50,
        "recent_cross_below_40": recent_below_40,
        "above_sma20": cur_close > cur_sma20,
        "above_sma50": cur_close > cur_sma50,
    }

    if pre["vol_ratio_full"] is not None and t < len(pre["vol_ratio_full"]):
        vr = pre["vol_ratio_full"][t]
        snap["volume_ratio"] = float(vr) if np.isfinite(vr) else None

    sma200_s = pre["sma200_full"]
    if sma200_s is not None and t < len(sma200_s):
        sv = sma200_s.iloc[t]
        if pd.notna(sv):
            snap["above_sma200"] = cur_close > float(sv)

    return snap


class _PluginEntryAdapter:
    """Wraps (signal_generator + quality_gate) plugins into EntryStrategyProtocol.

    Used when no entry_strategy is explicitly provided so the engine loop can
    always call a single evaluate() regardless of which API is in use.
    """

    def __init__(self, signal_plugin: SignalGeneratorProtocol, gate_plugin: QualityGateProtocol) -> None:
        self._signal = signal_plugin
        self._gate = gate_plugin

    def evaluate(
        self,
        regime: RegimeState,
        mom: dict,
        volume_ratio: float,
        currently_in: bool = False,
    ) -> EntryDecision:
        raw = self._signal.generate(regime, mom)
        gated = self._gate.apply(raw, mom, regime.regime_signal, currently_in=currently_in)
        return EntryDecision(
            flag=gated["flag"],
            raw_flag=raw["flag"],
            score=float(raw.get("score", 0.0)),
            reason=gated.get("reason", ""),
        )


class _PluginExitAdapter:
    """Wraps exit_rules plugin into ExitStrategyProtocol.

    Exposes the engine-level stop_loss_pct and take_profit_pct as properties
    so the loop can read them uniformly regardless of which API is in use.
    """

    def __init__(self, exit_plugin: ExitRulesProtocol, stop_loss_pct: float, take_profit_pct: float) -> None:
        self._plugin = exit_plugin
        self._stop = stop_loss_pct
        self._target = take_profit_pct

    @property
    def stop_loss_pct(self) -> float:
        return self._stop

    @property
    def take_profit_pct(self) -> float:
        return self._target

    def check(self, trade: TradeState, bar: BarData) -> ExitResult:
        return self._plugin.check(trade, bar)


def consolidated_backtest(
    df: pd.DataFrame,
    *,
    entry_prob: float = 0.65,
    exit_prob: float = 0.40,
    stop_loss_pct: float = 0.05,
    take_profit_pct: float = 0.15,
    volume_min_ratio: float = 1.0,
    min_train_bars: int = 500,
    hmm_refit_bars: int = 500,
    initial_cash: float = 20_000.0,
    trade_cost: float = 10.0,
    use_kelly: bool = True,
    kelly_lookback: int = 20,
    sentiment_score: float = 0.0,
    vix_signal: int = 0,
    regime_smooth: int = 24,
    min_hold_bars: int = 48,
    # vote + quality-gate parameters
    rsi_period: int = 14,
    ma_fast: int = 20,
    ma_slow: int = 50,
    ma_trend: int = 200,
    buy_threshold: float = 3.0,
    sell_threshold: float = -3.0,
    weights: dict | None = None,
    # trailing/vol-stop parameters (mirroring daily engine)
    trailing_stop: float = 0.0,
    vol_stop_mult: float = 0.0,
    vol_stop_window: int = 20,
    profit_stop_scale: float = 0.0,
    min_stop_pct: float = 0.05,
    max_hold_days: int = 0,
    # optional exit indicator flags
    exit_on_macd_cross: bool = False,
    exit_on_rsi_reversal: bool = False,
    exit_on_consolidation: bool = False,
    use_sar_stop: bool = False,
    sar_af_start: float = 0.02,
    sar_af_step: float = 0.02,
    sar_af_max: float = 0.20,
    # plugin injection points (None → use defaults matching original behaviour)
    regime_model: RegimeModelProtocol | None = None,
    signal_generator: SignalGeneratorProtocol | None = None,
    quality_gate: QualityGateProtocol | None = None,
    exit_rules: ExitRulesProtocol | None = None,
    position_sizer: PositionSizerProtocol | None = None,
    context_adjuster: ContextAdjusterProtocol | None = None,
    # strategy injection (supersedes signal_generator + quality_gate + exit_rules when set)
    entry_strategy: EntryStrategyProtocol | None = None,
    exit_strategy: ExitStrategyProtocol | None = None,
) -> dict:
    """Walk-forward consolidated backtest on hourly data.

    Entry logic
    -----------
    Uses composite_signal() (vote-based, same as the daily engine) where the
    HMM's discretized P(Bull) fills the hmm_state slot.  A BUY signal is only
    acted on if:
      1. composite_signal returns BUY (weighted score >= buy_threshold)
      2. _apply_quality_gate does not veto it
      3. volume >= volume_min_ratio × average

    Exit logic (in priority order)
    -----
    1. Hard stop-loss (stop_loss_pct from entry price)
    2. Hard take-profit (take_profit_pct from entry price)
    3. Trailing / vol-scaled stop (trailing_stop / vol_stop_mult)
    4. Max-hold-bars
    5. Parabolic SAR stop (optional)
    6. MACD bearish cross / RSI reversal / consolidation (optional)
    7. composite_signal SELL + quality-gate adverse-exit → SELL (after min_hold_bars)

    Position sizing
    ---------------
    Kelly fraction based on trailing realised trade P&L (unchanged from
    quant_backtest), capped at 25%, floored at 2%.

    Plugin injection
    ----------------
    Pass any of the six plugin kwargs to override the default behaviour.
    All kwargs default to None which resolves to the original inline logic.
    """
    # ------------------------------------------------------------------
    # Resolve plugins (None → default matching original inline behaviour)
    # ------------------------------------------------------------------
    from ..plugins.context_adjuster import SentimentAdjuster
    from ..plugins.exit_rules import StandardExitRules
    from ..plugins.hmm_regime import HMMRegimeModel
    from ..plugins.kelly_sizer import KellySizer
    from ..plugins.quality_gate import QualityGatePlugin
    from ..plugins.vote_signal import CompositeSignalGenerator

    # Context adjuster runs first — adjusts entry/exit prob thresholds and stop.
    # If a strategy is set, use its stop_loss_pct as the base; otherwise the kwarg.
    _base_stop = exit_strategy.stop_loss_pct if exit_strategy is not None else stop_loss_pct
    adj = context_adjuster or SentimentAdjuster()
    effective_entry_prob, effective_exit_prob, effective_stop_adj = adj.adjust(
        entry_prob, exit_prob, _base_stop, sentiment_score, vix_signal,
    )

    effective_weights = {**_CONSOLIDATED_WEIGHTS, **(weights or {})}

    regime_plugin: RegimeModelProtocol = regime_model or HMMRegimeModel(
        min_train_bars=min_train_bars,
        refit_bars=hmm_refit_bars,
        regime_smooth=regime_smooth,
        bull_edge=effective_entry_prob,
        bear_edge=effective_exit_prob,
    )

    # Build low-level plugin instances (needed for adapters even when a strategy is set)
    signal_plugin: SignalGeneratorProtocol = signal_generator or CompositeSignalGenerator(
        weights=effective_weights,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )
    gate_plugin: QualityGateProtocol = quality_gate or QualityGatePlugin()
    exit_plugin: ExitRulesProtocol = exit_rules or StandardExitRules(
        stop_loss_pct=stop_loss_pct,
        trailing_stop=trailing_stop,
        vol_stop_mult=vol_stop_mult,
        vol_stop_window=vol_stop_window,
        profit_stop_scale=profit_stop_scale,
        min_stop_pct=min_stop_pct,
        max_hold_days=max_hold_days,
        exit_on_macd_cross=exit_on_macd_cross,
        exit_on_rsi_reversal=exit_on_rsi_reversal,
        exit_on_consolidation=exit_on_consolidation,
        use_sar_stop=use_sar_stop,
    )
    sizer_plugin: PositionSizerProtocol = position_sizer or KellySizer(
        use_kelly=use_kelly,
        lookback=kelly_lookback,
    )

    # Unified entry/exit interfaces — strategy supersedes plugin-level adapters when set.
    _entry: EntryStrategyProtocol = (
        entry_strategy
        if entry_strategy is not None
        else _PluginEntryAdapter(signal_plugin, gate_plugin)
    )
    _exit: ExitStrategyProtocol = (
        exit_strategy
        if exit_strategy is not None
        else _PluginExitAdapter(exit_plugin, effective_stop_adj, take_profit_pct)
    )
    # When a strategy provides its own stop/take-profit, context adjuster still
    # adjusts the stop but the take-profit comes straight from the strategy.
    _effective_stop = effective_stop_adj   # after context-adjuster nudge
    _effective_target = _exit.take_profit_pct

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    close = df["Close"].values.astype(float)
    volume = df["Volume"].values.astype(float) if "Volume" in df.columns else None
    dates = df.index
    returns = np.diff(np.log(close))
    n = len(close)

    vol_ratio_arr = _compute_volume_ratio(volume, n)

    close_s = pd.Series(close, index=dates)
    pre = _precompute_hourly_vote_series(
        close_s, volume,
        rsi_period=rsi_period, ma_fast=ma_fast, ma_slow=ma_slow, ma_trend=ma_trend,
        vol_stop_window=vol_stop_window,
        exit_on_macd_cross=exit_on_macd_cross, exit_on_rsi_reversal=exit_on_rsi_reversal,
        exit_on_consolidation=exit_on_consolidation,
        use_sar_stop=use_sar_stop, sar_af_start=sar_af_start,
        sar_af_step=sar_af_step, sar_af_max=sar_af_max,
    )

    # ------------------------------------------------------------------
    # Walk-forward per-bar loop
    # ------------------------------------------------------------------
    rows = []
    position = 0.0
    entry_price = 0.0
    stop_level = 0.0
    target_level = 0.0
    entry_bar = 0
    peak_price_since_entry = 0.0
    days_in_trade = 0
    prev_signal_flag: str | None = None

    for t in range(min_train_bars, n):
        # 1. Periodic HMM refit on expanding window
        if regime_plugin.needs_refit(t):
            regime_plugin.refit(returns[:t])

        # 2. HMM forward step → RegimeState (or skip bar if not yet fitted)
        regime_state = regime_plugin.step(returns, t)
        if regime_state is None:
            continue

        # 3. Momentum snapshot for this bar
        mom_snap = _build_mom_snap(t, float(close[t]), pre)

        # 4. Per-bar scalars
        cur_vol_ratio = float(vol_ratio_arr[t]) if t < len(vol_ratio_arr) else 1.0
        cur_close = float(close[t])
        prev_close = float(close[t - 1])
        bar_return = (cur_close - prev_close) / prev_close if prev_close > 0 else 0.0

        sell_reason = ""
        trade_event = ""
        bars_held = t - entry_bar if position > 0 else 0

        # 5. Precompute exit-indicator booleans for this bar
        need_exit = pre["need_exit"]
        macd_bc = bool(pre["macd_bear_x"].iloc[t]) if (need_exit and pre["macd_bear_x"] is not None) else False
        rsi_ob  = bool(pre["rsi_ob_exit"].iloc[t]) if (need_exit and pre["rsi_ob_exit"] is not None) else False
        rsi_ml  = bool(pre["rsi_mom_loss"].iloc[t]) if (need_exit and pre["rsi_mom_loss"] is not None) else False
        consol  = bool(pre["consol_full"].iloc[t]) if (need_exit and pre["consol_full"] is not None) else False
        sar_val = float(pre["sar_full"].iloc[t]) if (use_sar_stop and pre["sar_full"] is not None) else None

        # 6. Exit checks (only while in a position)
        exit_hit = False
        if position > 0:
            trade_state = TradeState(
                position=position,
                entry_price=entry_price,
                stop_level=stop_level,
                target_level=target_level,
                peak_price_since_entry=peak_price_since_entry,
                days_in_trade=days_in_trade,
                entry_bar=entry_bar,
            )
            bar_data = BarData(
                t=t,
                cur_close=cur_close,
                daily_vol_t=float(pre["rolling_vol_full"].iloc[t]),
                use_sar_stop=use_sar_stop,
                sar_val=sar_val,
                need_exit=need_exit,
                macd_bc=macd_bc,
                rsi_ob=rsi_ob,
                rsi_ml=rsi_ml,
                consol=consol,
            )
            exit_result = _exit.check(trade_state, bar_data)
            exit_hit = exit_result.exit_hit
            sell_reason = exit_result.sell_reason
            peak_price_since_entry = exit_result.peak_price_since_entry
            days_in_trade = exit_result.days_in_trade

            if exit_hit:
                trade_event = "SELL"
            elif bars_held >= min_hold_bars:
                # After minimum hold, also allow entry-strategy SELL signal
                sig_in = _entry.evaluate(regime_state, mom_snap, cur_vol_ratio, currently_in=True)
                if sig_in.flag == "SELL":
                    trade_event = "SELL"
                    sell_reason = sig_in.reason or "signal"
                    exit_hit = True

            if exit_hit:
                trade_pl = (cur_close - entry_price) / entry_price
                sizer_plugin.record(trade_pl)
                position = 0.0
                peak_price_since_entry = 0.0
                days_in_trade = 0

        if position == 0 and not exit_hit:
            # Entry: BUY requires entry_strategy BUY + volume check
            vol_ok = cur_vol_ratio >= volume_min_ratio
            if vol_ok:
                decision = _entry.evaluate(regime_state, mom_snap, cur_vol_ratio, currently_in=False)

                # Transition guard: only enter on a non-BUY → BUY flip
                raw_flag = decision.raw_flag
                if raw_flag == "BUY" and (prev_signal_flag is None or prev_signal_flag == "BUY"):
                    raw_flag = "HOLD"
                prev_signal_flag = decision.raw_flag

                if raw_flag == "BUY" and decision.flag == "BUY":
                    position = sizer_plugin.position
                    entry_price = cur_close
                    stop_level  = cur_close * (1 - _effective_stop)
                    target_level = cur_close * (1 + _effective_target)
                    entry_bar  = t
                    peak_price_since_entry = cur_close
                    days_in_trade = 0
                    trade_event = "BUY"
            else:
                # Volume too low: still evaluate for transition guard
                decision = _entry.evaluate(regime_state, mom_snap, cur_vol_ratio, currently_in=False)
                if prev_signal_flag != "BUY" or decision.raw_flag != "BUY":
                    prev_signal_flag = decision.raw_flag

        strategy_return = position * bar_return if position > 0 else 0.0

        # Generate the entry-strategy snapshot for logging (diagnostic, not decision)
        log_decision = _entry.evaluate(regime_state, mom_snap, cur_vol_ratio,
                                       currently_in=(position > 0))

        rows.append({
            # --- Market data ---
            "date": dates[t],
            "close": round(cur_close, 4),
            # --- HMM regime ---
            "p_bull": round(regime_state.p_bull, 4),
            "p_bull_smooth": round(regime_state.p_bull_smooth, 4),
            "p_bear": round(regime_state.p_bear, 4),
            "regime_signal": round(regime_state.regime_signal, 4),
            "hmm_vote": regime_state.hmm_vote,
            # --- Momentum indicators ---
            "rsi": round(mom_snap.get("cur_rsi", float("nan")), 2),
            "above_sma20": bool(mom_snap.get("above_sma20", False)),
            "above_sma50": bool(mom_snap.get("above_sma50", False)),
            "above_sma200": mom_snap.get("above_sma200"),
            "rsi_x_above_50": bool(mom_snap.get("recent_cross_above_50", False)),
            "rsi_x_below_40": bool(mom_snap.get("recent_cross_below_40", False)),
            # --- Volume ---
            "volume_ratio": round(cur_vol_ratio, 2),
            # --- Signal & gate output ---
            "signal_flag": log_decision.raw_flag,
            "signal_score": round(log_decision.score, 3),
            "gate_flag": log_decision.flag,
            "gate_reason": log_decision.reason,
            # --- Trade state ---
            "position": round(position, 4),
            "trade_event": trade_event,
            "sell_reason": sell_reason,
            "entry_price": round(entry_price, 4) if position > 0 else None,
            "stop_level": round(stop_level, 4) if position > 0 else None,
            "target_level": round(target_level, 4) if position > 0 else None,
            "days_in_trade": days_in_trade if position > 0 else 0,
            "peak_since_entry": round(peak_price_since_entry, 4) if position > 0 else None,
            # --- Sizing ---
            "kelly_fraction": round(sizer_plugin.current_kelly, 4),
            # --- Returns ---
            "bar_return": round(bar_return, 6),
            "strategy_return": round(strategy_return, 6),
        })

    if not rows:
        return _empty_result(initial_cash)
    detail = pd.DataFrame(rows).set_index("date")
    if detail.empty:
        return _empty_result(initial_cash)

    strat_ret  = detail["strategy_return"].values
    strat_equity = (1 + strat_ret).cumprod()
    bh_ret   = detail["bar_return"].values
    bh_equity  = (1 + bh_ret).cumprod()
    detail["strategy_equity"] = strat_equity
    detail["bh_equity"]       = bh_equity

    portfolio_values = _simulate_portfolio_value(detail, initial_cash, trade_cost)
    detail["portfolio_value"] = portfolio_values

    return _build_quant_backtest_stats(
        detail, strat_ret, bh_ret, strat_equity, bh_equity, initial_cash,
        portfolio_values, sizer_plugin.trade_results, sizer_plugin.current_kelly,
    )
