"""Alternative data signals: options IV, put/call ratio, VIX regime, insider activity.

These feed into the quant engine as additional confirmation/warning signals
alongside the core HMM regime probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Options-derived signals
# ---------------------------------------------------------------------------

def options_signals(ticker: str) -> dict:
    """Fetch options-derived signals from yfinance.

    Returns dict with:
      - iv_rank: current IV percentile vs 1-year range (0-100)
      - iv_current: current average implied volatility
      - put_call_ratio: total put OI / total call OI
      - put_call_signal: +1 (contrarian bullish, high P/C), -1 (bearish, low P/C), 0 (neutral)
      - iv_signal: +1 (low IV, cheap entry), -1 (high IV, risky), 0 (neutral)
      - skew: OTM put IV - OTM call IV (positive = fear)
    """
    import yfinance as yf

    result = {
        "iv_rank": None, "iv_current": None,
        "put_call_ratio": None, "put_call_signal": 0,
        "iv_signal": 0, "skew": None,
    }

    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return result

        # Use nearest expiry for current sentiment, and ~30-day for skew
        near_exp = expirations[0]
        mid_exp = expirations[min(2, len(expirations) - 1)]

        calls_near = tk.option_chain(near_exp).calls
        puts_near = tk.option_chain(near_exp).puts

        if calls_near.empty or puts_near.empty:
            return result

        # Put/Call ratio by open interest
        total_call_oi = calls_near["openInterest"].sum()
        total_put_oi = puts_near["openInterest"].sum()

        if total_call_oi > 0:
            pc_ratio = total_put_oi / total_call_oi
            result["put_call_ratio"] = round(float(pc_ratio), 3)

            # Contrarian signal: extreme P/C > 1.2 is bullish (too much fear),
            # P/C < 0.5 is bearish (too much complacency)
            if pc_ratio > 1.2:
                result["put_call_signal"] = 1
            elif pc_ratio < 0.5:
                result["put_call_signal"] = -1

        # Average IV across near-expiry options
        all_ivs = pd.concat([
            calls_near["impliedVolatility"].dropna(),
            puts_near["impliedVolatility"].dropna(),
        ])
        if not all_ivs.empty:
            current_iv = float(all_ivs.median())
            result["iv_current"] = round(current_iv, 4)

            # IV rank: compare current IV to historical price volatility
            hist = tk.history(period="1y")
            if not hist.empty:
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)
                hist_vol = float(hist["Close"].pct_change().std() * np.sqrt(252))
                if hist_vol > 0:
                    iv_ratio = current_iv / hist_vol
                    iv_rank = min(100, max(0, (iv_ratio - 0.5) / 1.5 * 100))
                    result["iv_rank"] = round(iv_rank, 1)

                    # Low IV = cheap options = good entry; high IV = expensive/risky
                    if iv_rank < 25:
                        result["iv_signal"] = 1
                    elif iv_rank > 75:
                        result["iv_signal"] = -1

        # Skew: compare OTM put IV to OTM call IV
        try:
            info = tk.info
            spot = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        except Exception:
            spot = 0

        if spot and spot > 0:
            otm_puts = puts_near[puts_near["strike"] < spot * 0.95]
            otm_calls = calls_near[calls_near["strike"] > spot * 1.05]
            if not otm_puts.empty and not otm_calls.empty:
                put_iv = otm_puts["impliedVolatility"].median()
                call_iv = otm_calls["impliedVolatility"].median()
                if pd.notna(put_iv) and pd.notna(call_iv) and call_iv > 0:
                    skew = float(put_iv - call_iv)
                    result["skew"] = round(skew, 4)

    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# VIX regime signal
# ---------------------------------------------------------------------------

def vix_regime() -> dict:
    """Fetch VIX data and compute regime signal.

    Returns dict with:
      - vix_current: current VIX level
      - vix_sma20: 20-day SMA of VIX
      - vix_regime: "low_vol" (<15), "normal" (15-25), "high_vol" (25-35), "crisis" (>35)
      - vix_signal: +1 (low vol, safe to enter), -1 (high vol, caution), 0 (neutral)
      - vix_term_structure: "contango" (normal, calm) or "backwardation" (fear, near > far)
    """
    import yfinance as yf

    result = {
        "vix_current": None, "vix_sma20": None,
        "vix_regime": None, "vix_signal": 0,
        "vix_term_structure": None,
    }

    try:
        vix = yf.download("^VIX", period="60d", progress=False, auto_adjust=True)
        if vix.empty:
            return result

        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)

        close = vix["Close"].dropna()
        if close.empty:
            return result

        current = float(close.iloc[-1])
        sma20 = float(close.tail(20).mean())
        result["vix_current"] = round(current, 2)
        result["vix_sma20"] = round(sma20, 2)

        # Regime classification
        if current < 15:
            result["vix_regime"] = "low_vol"
            result["vix_signal"] = 1
        elif current < 25:
            result["vix_regime"] = "normal"
            result["vix_signal"] = 0
        elif current < 35:
            result["vix_regime"] = "high_vol"
            result["vix_signal"] = -1
        else:
            result["vix_regime"] = "crisis"
            result["vix_signal"] = -1

        # VIX term structure: compare VIX to VIX3M (3-month)
        try:
            vix3m = yf.download("^VIX3M", period="5d", progress=False, auto_adjust=True)
            if not vix3m.empty:
                if isinstance(vix3m.columns, pd.MultiIndex):
                    vix3m.columns = vix3m.columns.get_level_values(0)
                vix3m_val = float(vix3m["Close"].dropna().iloc[-1])
                if vix3m_val > 0:
                    result["vix_term_structure"] = "contango" if current < vix3m_val else "backwardation"
        except Exception:
            pass

    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# VIX as HMM observable (for 2D HMM)
# ---------------------------------------------------------------------------

def fetch_vix_hourly(period: str = "730d") -> pd.Series | None:
    """Fetch hourly VIX data for use as second HMM observable."""
    import yfinance as yf

    try:
        df = yf.download("^VIX", period=period, interval="1h",
                         progress=False, auto_adjust=True)
    except Exception:
        return None

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df["Close"].dropna()


# ---------------------------------------------------------------------------
# Insider transaction signals
# ---------------------------------------------------------------------------

def insider_signals(ticker: str) -> dict:
    """Check recent insider transactions for buy/sell bias.

    Returns dict with:
      - insider_buys_90d: number of insider buys in last 90 days
      - insider_sells_90d: number of insider sells in last 90 days
      - insider_net: buys - sells
      - insider_signal: +1 (net buying), -1 (net selling), 0 (neutral/no data)
      - insider_total_value: approximate total value of recent transactions
    """
    import yfinance as yf

    result = {
        "insider_buys_90d": 0, "insider_sells_90d": 0,
        "insider_net": 0, "insider_signal": 0,
        "insider_total_value": 0,
    }

    try:
        tk = yf.Ticker(ticker)

        # yfinance provides insider transactions
        try:
            insiders = tk.insider_transactions
        except Exception:
            insiders = None

        if insiders is None or (hasattr(insiders, 'empty') and insiders.empty):
            return result

        if isinstance(insiders, pd.DataFrame) and not insiders.empty:
            # Filter to last 90 days
            now = pd.Timestamp.now()
            cutoff = now - pd.Timedelta(days=90)

            # The column names vary; try common ones
            date_col = None
            for col in ["Start Date", "Date", "startDate"]:
                if col in insiders.columns:
                    date_col = col
                    break

            if date_col:
                insiders[date_col] = pd.to_datetime(insiders[date_col], errors="coerce")
                recent = insiders[insiders[date_col] >= cutoff]
            else:
                recent = insiders

            # Count buys vs sells
            text_col = None
            for col in ["Text", "Transaction", "transaction"]:
                if col in insiders.columns:
                    text_col = col
                    break

            if text_col and not recent.empty:
                texts = recent[text_col].astype(str).str.lower()
                n_buys = texts.str.contains("purchase|buy|acquisition", na=False).sum()
                n_sells = texts.str.contains("sale|sell|disposition", na=False).sum()

                result["insider_buys_90d"] = int(n_buys)
                result["insider_sells_90d"] = int(n_sells)
                result["insider_net"] = int(n_buys - n_sells)

                # Signal: net buying is bullish, net selling is bearish
                if n_buys > n_sells and n_buys >= 2:
                    result["insider_signal"] = 1
                elif n_sells > n_buys and n_sells >= 3:
                    result["insider_signal"] = -1

                # Total value
                value_col = None
                for col in ["Value", "value", "Shares", "shares"]:
                    if col in recent.columns:
                        value_col = col
                        break
                if value_col:
                    result["insider_total_value"] = float(
                        pd.to_numeric(recent[value_col], errors="coerce").sum()
                    )

    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Short interest signal
# ---------------------------------------------------------------------------

def short_interest_signal(ticker: str) -> dict:
    """Estimate short interest from available data.

    Returns dict with:
      - short_pct_float: short interest as % of float (if available)
      - short_ratio: days to cover
      - short_signal: +1 (high short interest = squeeze potential),
                      -1 (rising short interest = bearish pressure), 0 (neutral)
    """
    import yfinance as yf

    result = {
        "short_pct_float": None, "short_ratio": None, "short_signal": 0,
    }

    try:
        info = yf.Ticker(ticker).info
        short_pct = info.get("shortPercentOfFloat")
        short_ratio = info.get("shortRatio")

        if short_pct is not None:
            result["short_pct_float"] = round(float(short_pct) * 100, 2)
        if short_ratio is not None:
            result["short_ratio"] = round(float(short_ratio), 2)

        # High short interest (>10%) with our model saying BUY = squeeze potential
        # Very high short interest (>20%) is a warning either way
        if short_pct is not None:
            pct = float(short_pct) * 100
            if pct > 20:
                result["short_signal"] = 1  # squeeze potential
            elif pct > 10:
                result["short_signal"] = 1  # moderate squeeze potential
            elif pct < 2:
                result["short_signal"] = 0  # no significant short presence

    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Composite sentiment score
# ---------------------------------------------------------------------------

def composite_sentiment(
    ticker: str,
    *,
    include_options: bool = True,
    include_vix: bool = True,
    include_insider: bool = True,
    include_short: bool = True,
) -> dict:
    """Compute a composite sentiment score from all available alternative data.

    Returns dict with all individual signals plus:
      - sentiment_score: weighted composite (-1 to +1)
      - sentiment_label: "bullish", "bearish", or "neutral"
      - confidence: how many data sources contributed (0-4)
    """
    signals = []
    weights = []
    all_data = {}
    confidence = 0

    if include_options:
        opts = options_signals(ticker)
        all_data["options"] = opts
        if opts["put_call_ratio"] is not None:
            signals.append(opts["put_call_signal"])
            weights.append(1.5)  # Options flow is a strong signal
            confidence += 1
        if opts["iv_signal"] != 0:
            signals.append(opts["iv_signal"])
            weights.append(1.0)

    if include_vix:
        vix = vix_regime()
        all_data["vix"] = vix
        if vix["vix_current"] is not None:
            signals.append(vix["vix_signal"])
            weights.append(1.0)
            confidence += 1
            # Backwardation is an extra warning
            if vix["vix_term_structure"] == "backwardation":
                signals.append(-1)
                weights.append(0.5)

    if include_insider:
        ins = insider_signals(ticker)
        all_data["insider"] = ins
        if ins["insider_net"] != 0:
            signals.append(ins["insider_signal"])
            weights.append(2.0)  # Insider buying is one of the strongest signals
            confidence += 1

    if include_short:
        si = short_interest_signal(ticker)
        all_data["short_interest"] = si
        if si["short_pct_float"] is not None:
            signals.append(si["short_signal"])
            weights.append(0.5)
            confidence += 1

    # Compute weighted score
    if signals and weights:
        total_weight = sum(weights)
        score = sum(s * w for s, w in zip(signals, weights)) / total_weight
    else:
        score = 0.0

    if score > 0.3:
        label = "bullish"
    elif score < -0.3:
        label = "bearish"
    else:
        label = "neutral"

    all_data["sentiment_score"] = round(score, 3)
    all_data["sentiment_label"] = label
    all_data["confidence"] = confidence

    return all_data
