"""Momentum indicators: RSI re-entry cross and moving-average trend filters."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's smoothed RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def compute_ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


_DEFAULT_WEIGHTS = {
    "markov": 1.0,
    "rsi":    1.5,
    "trend":  1.0,   # merged SMA20/SMA50 (above both=+1, below both=-1, mixed=0)
    "sma200": 2.0,   # long-term trend gate — higher weight as a filter
    "volume": 1.0,
    "hmm":    1.0,
}


def composite_signal(
    markov_signal: float,
    mom: dict,
    sell_threshold: float = -3.0,
    hmm_state: int | None = None,
    buy_threshold: float = 3.0,
    weights: dict[str, float] | None = None,
) -> dict:
    """Combine Markov, momentum, volume and HMM into a weighted BUY / HOLD / SELL.

    Each indicator casts a raw vote (+1, -1, or 0) which is then multiplied
    by its weight. The weighted sum is compared to buy/sell thresholds.

    Key changes from original equal-weight system:
      - SMA20 and SMA50 merged into a single 'trend' vote (fixes correlation)
      - SMA200 gets 2× weight (strong trend filter)
      - RSI gets 1.5× weight (better at timing)
      - Volume confirmation added (above-average volume = +1)
    """
    w = {**_DEFAULT_WEIGHTS, **(weights or {})}
    votes: dict[str, int] = {}
    vote_weights: dict[str, float] = {}

    # Markov: signal = P(Bull|current) - P(Bear|current)
    if markov_signal > 0.20:
        votes["markov"] = 1
    elif markov_signal < -0.20:
        votes["markov"] = -1
    else:
        votes["markov"] = 0
    vote_weights["markov"] = w["markov"]

    # RSI momentum (skipped when upstream disabled the RSI indicator)
    if "cur_rsi" in mom:
        rsi = mom["cur_rsi"]
        if rsi >= 50 or mom["recent_cross_above_50"]:
            votes["rsi"] = 1
        elif rsi < 40 or mom["recent_cross_below_40"]:
            votes["rsi"] = -1
        else:
            votes["rsi"] = 0
        vote_weights["rsi"] = w["rsi"]

    # Merged short-term trend: SMA20 + SMA50
    # Both above = bullish, both below = bearish, mixed = neutral
    above20 = mom.get("above_sma20", False)
    above50 = mom.get("above_sma50", False)
    if above20 and above50:
        votes["trend"] = 1
    elif not above20 and not above50:
        votes["trend"] = -1
    else:
        votes["trend"] = 0
    vote_weights["trend"] = w["trend"]

    # SMA200 long-term trend gate (higher weight)
    if "above_sma200" in mom and mom["above_sma200"] is not None:
        votes["sma200"] = 1 if mom["above_sma200"] else -1
        vote_weights["sma200"] = w["sma200"]

    # Volume confirmation (above 20-day average volume = bullish confirmation)
    if "volume_ratio" in mom and mom["volume_ratio"] is not None:
        vr = mom["volume_ratio"]
        if vr > 1.2:
            votes["volume"] = 1
        elif vr < 0.7:
            votes["volume"] = -1
        else:
            votes["volume"] = 0
        vote_weights["volume"] = w["volume"]

    # HMM regime (Bear=-1, Sideways=0, Bull=+1)
    if hmm_state is not None:
        votes["hmm"] = {0: -1, 1: 0, 2: 1}.get(hmm_state, 0)
        vote_weights["hmm"] = w["hmm"]

    max_score = sum(vote_weights.values())
    score = sum(votes[k] * vote_weights[k] for k in votes)

    if score >= buy_threshold:
        flag = "BUY"
    elif score <= sell_threshold:
        flag = "SELL"
    else:
        flag = "HOLD"

    return {"flag": flag, "score": round(score, 1), "votes": votes,
            "weights": vote_weights, "max_score": round(max_score, 1)}


def compute_parabolic_sar(
    close: pd.Series,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.20,
) -> pd.Series:
    """Parabolic SAR (Stop and Reverse).

    Returns a Series of SAR values. In an uptrend the SAR trails below price;
    when price drops below SAR, the trend reverses.
    """
    n = len(close)
    sar = np.zeros(n)
    trend = np.ones(n, dtype=int)  # +1 = uptrend, -1 = downtrend
    af = af_start
    ep = float(close.iloc[0])     # extreme point
    sar[0] = float(close.iloc[0])

    if n < 2:
        return pd.Series(sar, index=close.index)

    # Seed: assume uptrend if second close > first, else downtrend
    if float(close.iloc[1]) >= float(close.iloc[0]):
        trend[0] = 1
        sar[0] = float(close.iloc[0])
        ep = float(close.iloc[1])
    else:
        trend[0] = -1
        sar[0] = float(close.iloc[0])
        ep = float(close.iloc[1])

    for i in range(1, n):
        prev_sar = sar[i - 1]
        prev_trend = trend[i - 1] if i > 0 else 1
        cur = float(close.iloc[i])

        # Compute new SAR
        new_sar = prev_sar + af * (ep - prev_sar)

        if prev_trend == 1:  # uptrend
            new_sar = min(new_sar, float(close.iloc[i - 1]))
            if i >= 2:
                new_sar = min(new_sar, float(close.iloc[i - 2]))

            if cur < new_sar:  # reversal to downtrend
                trend[i] = -1
                new_sar = ep
                ep = cur
                af = af_start
            else:
                trend[i] = 1
                if cur > ep:
                    ep = cur
                    af = min(af + af_step, af_max)
        else:  # downtrend
            new_sar = max(new_sar, float(close.iloc[i - 1]))
            if i >= 2:
                new_sar = max(new_sar, float(close.iloc[i - 2]))

            if cur > new_sar:  # reversal to uptrend
                trend[i] = 1
                new_sar = ep
                ep = cur
                af = af_start
            else:
                trend[i] = -1
                if cur < ep:
                    ep = cur
                    af = min(af + af_step, af_max)

        sar[i] = new_sar

    return pd.Series(sar, index=close.index)


def compute_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(
    close: pd.Series, window: int = 20, n_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: middle, upper, lower."""
    middle = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = middle + n_std * std
    lower = middle - n_std * std
    return middle, upper, lower


def compute_atr(close: pd.Series, high: pd.Series | None = None,
                low: pd.Series | None = None, window: int = 14) -> pd.Series:
    """Average True Range. Falls back to close-based range if no high/low."""
    if high is not None and low is not None:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    else:
        tr = (close - close.shift(1)).abs()
    return tr.rolling(window).mean()


def exit_indicators(
    close: pd.Series,
    rsi: pd.Series | None = None,
    rsi_period: int = 14,
    lookback: int = 5,
) -> dict:
    """Compute exit-warning indicators (reported only, not used in buy/sell).

    Returns a dict of current-day scalars and a 'detail' DataFrame with
    full time series for all indicators.
    """
    if rsi is None:
        rsi = compute_rsi(close, rsi_period)

    # ── MACD ──────────────────────────────────────────────────────────────
    macd_line, macd_signal, macd_hist = compute_macd(close)
    macd_bearish_cross = (macd_line < macd_signal) & (macd_line.shift(1) >= macd_signal.shift(1))
    macd_bullish_cross = (macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1))

    # ── RSI reversals ─────────────────────────────────────────────────────
    rsi_exit_overbought = (rsi < 70) & (rsi.shift(1) >= 70)
    rsi_momentum_loss = (rsi < 50) & (rsi.shift(1) >= 50) & (rsi.rolling(lookback + 1).max().shift(1) >= 60)

    # ── Consolidation / stall ─────────────────────────────────────────────
    bb_mid, bb_upper, bb_lower = compute_bollinger(close)
    bb_width = (bb_upper - bb_lower) / bb_mid
    bb_width_avg = bb_width.rolling(20).mean()
    bb_squeeze = bb_width < bb_width_avg

    atr = compute_atr(close, window=14)
    atr_avg = atr.rolling(20).mean()
    atr_ratio = atr / atr_avg
    consolidating = bb_squeeze & (atr_ratio < 0.75)

    # ── Current-day snapshots ─────────────────────────────────────────────
    hist_val = float(macd_hist.iloc[-1])
    hist_prev = float(macd_hist.iloc[-2]) if len(macd_hist) >= 2 else 0.0

    # MACD histogram as % of price — comparable across stocks
    cur_price = float(close.iloc[-1])
    macd_hist_pct = hist_val / cur_price * 100 if cur_price > 0 else 0.0

    # MACD momentum label based on histogram direction
    if hist_val > 0 and hist_val > hist_prev:
        macd_label = "Bullish + strengthening"
        macd_status = "green"
    elif hist_val > 0 and hist_val <= hist_prev:
        macd_label = "Bullish but fading"
        macd_status = "amber"
    elif hist_val < 0 and hist_val < hist_prev:
        macd_label = "Bearish + strengthening"
        macd_status = "red"
    elif hist_val < 0 and hist_val >= hist_prev:
        macd_label = "Bearish but improving"
        macd_status = "amber"
    else:
        macd_label = "Neutral"
        macd_status = "grey"

    # RSI label
    cur_rsi_val = float(rsi.iloc[-1])
    if cur_rsi_val >= 70:
        rsi_status_label = "Overbought — watch for reversal"
        rsi_status = "amber"
    elif cur_rsi_val >= 50:
        rsi_status_label = "Healthy momentum"
        rsi_status = "green"
    elif cur_rsi_val >= 40:
        rsi_status_label = "Weakening — watch for breakdown"
        rsi_status = "amber"
    else:
        rsi_status_label = "Bearish"
        rsi_status = "red"

    # Bollinger label
    bb_w_val = float(bb_width.iloc[-1])
    bb_avg_val = float(bb_width_avg.iloc[-1]) if pd.notna(bb_width_avg.iloc[-1]) else None
    if bb_avg_val and bb_w_val < bb_avg_val * 0.8:
        bb_label = "Tight squeeze — breakout likely"
        bb_status = "amber"
    elif bb_avg_val and bb_w_val < bb_avg_val:
        bb_label = "Narrowing — volatility contracting"
        bb_status = "amber"
    elif bb_avg_val and bb_w_val > bb_avg_val * 1.2:
        bb_label = "Wide — strong trend in progress"
        bb_status = "green"
    else:
        bb_label = "Normal range"
        bb_status = "grey"

    # ATR label
    atr_r_val = float(atr_ratio.iloc[-1]) if pd.notna(atr_ratio.iloc[-1]) else None
    if atr_r_val is None:
        atr_label = "Insufficient data"
        atr_status = "grey"
    elif atr_r_val > 1.3:
        atr_label = "High volatility — trending"
        atr_status = "green"
    elif atr_r_val > 0.8:
        atr_label = "Normal"
        atr_status = "grey"
    elif atr_r_val > 0.5:
        atr_label = "Low — momentum stalling"
        atr_status = "amber"
    else:
        atr_label = "Very low — stalled / consolidating"
        atr_status = "red"

    cur = {
        "macd_line": float(macd_line.iloc[-1]),
        "macd_signal": float(macd_signal.iloc[-1]),
        "macd_histogram": hist_val,
        "macd_hist_pct": macd_hist_pct,
        "macd_hist_prev": hist_prev,
        "macd_bearish_cross": bool(macd_bearish_cross.iloc[-lookback:].any()),
        "macd_bullish_cross": bool(macd_bullish_cross.iloc[-lookback:].any()),
        "macd_trend": "bullish" if hist_val > 0 else "bearish",
        "macd_label": macd_label,
        "macd_status": macd_status,
        "rsi_exit_overbought": bool(rsi_exit_overbought.iloc[-lookback:].any()),
        "rsi_momentum_loss": bool(rsi_momentum_loss.iloc[-lookback:].any()),
        "rsi_status_label": rsi_status_label,
        "rsi_status": rsi_status,
        "bb_width": bb_w_val,
        "bb_width_avg": bb_avg_val,
        "bb_squeeze": bool(bb_squeeze.iloc[-1]),
        "bb_label": bb_label,
        "bb_status": bb_status,
        "atr": float(atr.iloc[-1]),
        "atr_ratio": atr_r_val,
        "atr_label": atr_label,
        "atr_status": atr_status,
        "consolidating": bool(consolidating.iloc[-1]),
    }

    # Build warnings list
    warnings = []
    if cur["macd_bearish_cross"]:
        warnings.append("MACD bearish crossover (last 5d)")
    if macd_status == "red":
        warnings.append(f"MACD {macd_label}")
    if cur["rsi_exit_overbought"]:
        warnings.append("RSI dropped below 70 (overbought exit)")
    if cur["rsi_momentum_loss"]:
        warnings.append("RSI momentum loss (dropped below 50 from 60+)")
    if cur["consolidating"]:
        warnings.append("Consolidation (BB squeeze + low ATR)")
    elif cur["bb_squeeze"]:
        warnings.append("Bollinger squeeze (narrowing volatility)")
    if cur["macd_bullish_cross"]:
        warnings.append("MACD bullish crossover (last 5d)")
    cur["warnings"] = warnings

    detail = pd.DataFrame({
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_histogram": macd_hist,
        "macd_bearish_cross": macd_bearish_cross,
        "macd_bullish_cross": macd_bullish_cross,
        "rsi_exit_overbought": rsi_exit_overbought,
        "rsi_momentum_loss": rsi_momentum_loss,
        "bb_width": bb_width,
        "bb_squeeze": bb_squeeze,
        "atr": atr,
        "atr_ratio": atr_ratio,
        "consolidating": consolidating,
    })

    cur["detail"] = detail
    return cur


def momentum_signals(
    close: pd.Series,
    volume: pd.Series | None = None,
    rsi_period: int = 14,
    ma_fast: int = 20,
    ma_slow: int = 50,
    ma_trend: int = 200,
    lookback_cross: int = 5,
) -> dict:
    """Compute RSI, MA, and volume signals.

    Returns a summary dict (scalars for printing) and a 'detail' DataFrame
    with the full time series for saving to CSV.
    """
    rsi = compute_rsi(close, rsi_period)
    sma20 = compute_sma(close, ma_fast)
    sma50 = compute_sma(close, ma_slow)
    sma200 = compute_sma(close, ma_trend)
    ema20 = compute_ema(close, ma_fast)

    rsi_cross_above_40 = (rsi >= 40) & (rsi.shift(1) < 40)
    rsi_cross_above_50 = (rsi >= 50) & (rsi.shift(1) < 50)
    rsi_cross_below_40 = (rsi < 40) & (rsi.shift(1) >= 40)

    cur_rsi = float(rsi.iloc[-1])
    cur_close = float(close.iloc[-1])
    cur_sma20 = float(sma20.iloc[-1])
    cur_sma50 = float(sma50.iloc[-1])
    cur_ema20 = float(ema20.iloc[-1])

    _sma200_raw = sma200.iloc[-1]
    cur_sma200 = float(_sma200_raw) if pd.notna(_sma200_raw) else None

    above_sma20 = cur_close > cur_sma20
    above_sma50 = cur_close > cur_sma50
    above_sma200 = (cur_close > cur_sma200) if cur_sma200 is not None else None
    pct_from_sma20 = (cur_close - cur_sma20) / cur_sma20
    pct_from_sma50 = (cur_close - cur_sma50) / cur_sma50
    pct_from_sma200 = ((cur_close - cur_sma200) / cur_sma200) if cur_sma200 is not None else None

    recent_cross_above_40 = bool(rsi_cross_above_40.iloc[-lookback_cross:].any())
    recent_cross_above_50 = bool(rsi_cross_above_50.iloc[-lookback_cross:].any())
    recent_cross_below_40 = bool(rsi_cross_below_40.iloc[-lookback_cross:].any())

    if cur_rsi >= 70:
        rsi_label = "Overbought"
    elif cur_rsi >= 50:
        rsi_label = "Bullish momentum"
    elif cur_rsi >= 40:
        rsi_label = "Recovering (watch for cross above 50)"
    elif cur_rsi >= 30:
        rsi_label = "Bearish"
    else:
        rsi_label = "Oversold"

    detail = pd.DataFrame({
        "close": close,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "ema20": ema20,
        "rsi": rsi,
        "above_sma20": close > sma20,
        "above_sma50": close > sma50,
        "above_sma200": close > sma200,
        "pct_from_sma20": (close - sma20) / sma20,
        "rsi_cross_above_40": rsi_cross_above_40,
        "rsi_cross_above_50": rsi_cross_above_50,
        "rsi_cross_below_40": rsi_cross_below_40,
    })

    volume_ratio = cur_volume = avg_volume = None
    if volume is not None and len(volume) > 20:
        vol_avg = volume.rolling(20).mean()
        vol_ratio = volume / vol_avg
        cur_volume = float(volume.iloc[-1])
        avg_volume = float(vol_avg.iloc[-1]) if pd.notna(vol_avg.iloc[-1]) else None
        volume_ratio = float(vol_ratio.iloc[-1]) if pd.notna(vol_ratio.iloc[-1]) else None

        detail["volume"] = volume
        detail["volume_avg20"] = vol_avg
        detail["volume_ratio"] = vol_ratio

    return {
        "cur_rsi": cur_rsi,
        "rsi_label": rsi_label,
        "recent_cross_above_40": recent_cross_above_40,
        "recent_cross_above_50": recent_cross_above_50,
        "recent_cross_below_40": recent_cross_below_40,
        "cur_close": cur_close,
        "cur_sma20": cur_sma20,
        "cur_sma50": cur_sma50,
        "cur_ema20": cur_ema20,
        "cur_sma200": cur_sma200,
        "above_sma20": above_sma20,
        "above_sma50": above_sma50,
        "above_sma200": above_sma200,
        "pct_from_sma20": pct_from_sma20,
        "pct_from_sma50": pct_from_sma50,
        "pct_from_sma200": pct_from_sma200,
        "volume_ratio": volume_ratio,
        "cur_volume": cur_volume,
        "avg_volume": avg_volume,
        "detail": detail,
    }
