"""HMM regime-probability quant signal engine + volume + Kelly sizing.

Core principles:
  1. HMM regime probabilities as continuous signals (not binary votes)
  2. Hourly data for precise stop execution (no overnight gap slippage)
  3. Volume confirmation (only enter on above-average volume)
  4. Single regime-based exit (P(Bull) drops below threshold)
  5. Kelly criterion position sizing
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fetch_hourly(ticker: str, period: str = "730d") -> pd.DataFrame | None:
    """Fetch hourly OHLCV data from yfinance."""
    import yfinance as yf
    try:
        df = yf.download(ticker, period=period, interval="1h",
                         progress=False, auto_adjust=True)
    except Exception:
        return None
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fit_hmm_expanding(
    returns: np.ndarray,
    n_components: int = 3,
    n_seeds: int = 5,
    n_iter: int = 100,
) -> tuple | None:
    """Fit a Gaussian HMM and return (model, state_order)."""
    try:
        from hmmlearn import hmm
    except ImportError:
        return None

    X = returns.reshape(-1, 1)
    best_model = None
    best_score = -np.inf

    for seed in range(n_seeds):
        model = hmm.GaussianHMM(
            n_components=n_components, covariance_type="diag",
            n_iter=n_iter, random_state=seed,
        )
        try:
            model.fit(X)
            score = float(model.score(X))
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        return None

    means = np.array([best_model.means_[k][0] for k in range(n_components)])
    order = np.argsort(means)  # 0=Bear, 1=Sideways, 2=Bull
    return best_model, order


def hmm_regime_probabilities(
    model, order: np.ndarray, obs: np.ndarray,
) -> np.ndarray:
    """Forward-only regime probabilities (causal, no lookahead).

    Returns array of shape (n, 3) with columns [P(Bear), P(Sideways), P(Bull)].
    """
    from scipy.stats import norm

    K = model.n_components
    n = len(obs)
    X = obs.reshape(-1, 1)

    log_emit = np.zeros((n, K))
    for k in range(K):
        log_emit[:, k] = norm.logpdf(
            X[:, 0], model.means_[k, 0], np.sqrt(model.covars_[k].flatten()[0])
        )

    log_transmat = np.log(model.transmat_ + 1e-300)
    probs = np.zeros((n, 3))

    log_alpha = np.log(model.startprob_ + 1e-300) + log_emit[0]
    log_alpha -= np.logaddexp.reduce(log_alpha)
    for j in range(3):
        probs[0, j] = np.exp(log_alpha[order[j]])

    for t in range(1, n):
        log_alpha_new = np.empty(K)
        for k in range(K):
            log_alpha_new[k] = np.logaddexp.reduce(log_alpha + log_transmat[:, k]) + log_emit[t, k]
        log_alpha = log_alpha_new - np.logaddexp.reduce(log_alpha_new)
        for j in range(3):
            probs[t, j] = np.exp(log_alpha[order[j]])

    return probs


def _forward_step_incremental(
    model, order: np.ndarray, returns: np.ndarray, t: int,
    log_alpha: np.ndarray | None,
) -> tuple[float, float, np.ndarray]:
    """Single incremental forward step. Returns (p_bull, p_bear, new_log_alpha).

    If log_alpha is None, initialises from scratch up to time t.
    Otherwise extends by one observation — O(K²) instead of O(t×K²).
    """
    from scipy.stats import norm

    K = model.n_components
    log_transmat = np.log(model.transmat_ + 1e-300)

    def _log_emit(obs_idx):
        x = returns[obs_idx]
        return np.array([
            norm.logpdf(x, model.means_[k, 0], np.sqrt(model.covars_[k].flatten()[0]))
            for k in range(K)
        ])

    if log_alpha is None:
        # Bootstrap from the start up to t
        log_alpha = np.log(model.startprob_ + 1e-300) + _log_emit(0)
        log_alpha -= np.logaddexp.reduce(log_alpha)
        for s in range(1, min(t, len(returns))):
            le = _log_emit(s)
            la_new = np.empty(K)
            for k in range(K):
                la_new[k] = np.logaddexp.reduce(log_alpha + log_transmat[:, k]) + le[k]
            log_alpha = la_new - np.logaddexp.reduce(la_new)
    else:
        # Single step extension
        obs_idx = t - 1  # returns[t-1] is the return from close[t-1] to close[t]
        if obs_idx < len(returns):
            le = _log_emit(obs_idx)
            la_new = np.empty(K)
            for k in range(K):
                la_new[k] = np.logaddexp.reduce(log_alpha + log_transmat[:, k]) + le[k]
            log_alpha = la_new - np.logaddexp.reduce(la_new)

    # Extract sorted probabilities
    probs = np.exp(log_alpha)
    p_bear = float(probs[order[0]])
    p_bull = float(probs[order[2]])

    return p_bull, p_bear, log_alpha


def discretize_p_bull(p_bull_smooth: float, bull_edge: float = 0.65, bear_edge: float = 0.40) -> int:
    """Map a smoothed P(Bull) probability to the 3-way vote used by composite_signal().

    0 = Bear, 1 = Sideways, 2 = Bull — matching the hmm_state convention in
    composite_signal() so the HMM carries the same role as the Markov regime
    vote in the daily engine.

    bull_edge / bear_edge default to the quant_backtest entry / exit thresholds
    so the discretized vote is consistent with the engine's entry/exit logic.
    """
    if p_bull_smooth >= bull_edge:
        return 2  # Bull
    if p_bull_smooth <= bear_edge:
        return 0  # Bear
    return 1  # Sideways


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Kelly criterion: optimal fraction of capital to risk.

    Returns a fraction between 0 and 0.25 (capped for safety).
    """
    if avg_loss == 0 or win_rate <= 0:
        return 0.0
    b = abs(avg_win / avg_loss)
    q = 1 - win_rate
    kelly = (win_rate * b - q) / b
    return max(0.0, min(0.25, kelly))


def _compute_effective_thresholds(
    entry_prob: float, exit_prob: float, stop_loss_pct: float,
    sentiment_score: float, vix_signal: int,
) -> tuple[float, float, float]:
    """Sentiment- and VIX-adjusted entry/exit/stop thresholds.

    Bullish sentiment (score > 0) lowers the entry barrier, bearish raises it.
    VIX in a high-vol regime (-1) tightens the exit (exit sooner) and widens
    the stop (allow more room). Returns (entry_prob, exit_prob, stop_pct).
    """
    entry_adj = max(-0.10, min(0.10, -sentiment_score * 0.05))
    effective_entry_prob = entry_prob + entry_adj

    effective_exit_prob = exit_prob
    effective_stop = stop_loss_pct
    if vix_signal == -1:
        effective_exit_prob = min(0.50, exit_prob + 0.05)
        effective_stop = min(0.08, stop_loss_pct + 0.02)

    return effective_entry_prob, effective_exit_prob, effective_stop


def _compute_volume_ratio(volume: np.ndarray | None, n: int, lookback: int = 100) -> np.ndarray:
    """Current / rolling-average volume ratio, defaulting to 1.0 where unavailable."""
    if volume is None:
        return np.ones(n)
    vol_avg = pd.Series(volume).rolling(lookback).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(vol_avg > 0, volume / vol_avg, 1.0)


def _sharpe(r: np.ndarray) -> float:
    std = np.std(r, ddof=1)
    if std == 0 or not np.isfinite(std):
        return float("nan")
    # Annualise: assume ~1700 trading hours/year
    return float(np.mean(r) / std * np.sqrt(1700))


def _max_dd(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min()) if len(eq) else float("nan")


def _simulate_portfolio_value(detail: pd.DataFrame, initial_cash: float, trade_cost: float) -> list:
    """Cash P&L simulation: deduct trade_cost on each BUY/SELL, then compound
    through that bar's strategy_return."""
    cash = initial_cash
    portfolio_values = []
    for _, row in detail.iterrows():
        if row["trade_event"] in ("BUY", "SELL"):
            cash -= trade_cost
        cash *= (1 + float(row["strategy_return"]))
        portfolio_values.append(round(cash, 2))
    return portfolio_values


def _build_quant_backtest_stats(
    detail: pd.DataFrame,
    strat_ret: np.ndarray,
    bh_ret: np.ndarray,
    strat_equity: np.ndarray,
    bh_equity: np.ndarray,
    initial_cash: float,
    portfolio_values: list,
    trade_results: list,
    current_kelly: float,
) -> dict:
    n_buys = (detail["trade_event"] == "BUY").sum()
    n_sells = (detail["trade_event"] == "SELL").sum()
    final_portfolio = portfolio_values[-1] if portfolio_values else initial_cash

    return {
        "sharpe_strategy": _sharpe(strat_ret),
        "sharpe_bh": _sharpe(bh_ret),
        "total_return_strategy": float(strat_equity[-1] - 1) if len(strat_equity) else 0,
        "total_return_bh": float(bh_equity[-1] - 1) if len(bh_equity) else 0,
        "max_drawdown_strategy": _max_dd(strat_equity),
        "max_drawdown_bh": _max_dd(bh_equity),
        "initial_cash": initial_cash,
        "final_portfolio": final_portfolio,
        "total_pl": final_portfolio - initial_cash,
        "n_buys": int(n_buys),
        "n_sells": int(n_sells),
        "trade_results": trade_results,
        "final_kelly": current_kelly,
        "detail": detail,
        "n_bars": len(detail),
    }


def quant_backtest(
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
) -> dict:
    """Walk-forward backtest using HMM regime probabilities on hourly data.

    Entry: P(Bull) > entry_prob AND volume > volume_min_ratio × avg
    Exit:  P(Bull) < exit_prob OR hard stop-loss OR take-profit
    Sizing: Kelly fraction based on recent trade performance (capped at 25%)

    regime_smooth: rolling window (bars) for smoothing P(Bull) on exit decisions.
        Raw hourly probabilities oscillate too fast; smoothing prevents premature exits.
    min_hold_bars: minimum bars before regime exit can fire.
        Stop-loss and take-profit always fire immediately.

    Sentiment integration:
      - sentiment_score adjusts entry threshold: bullish lowers it, bearish raises it
      - vix_signal: -1 in high-vol regime tightens exit and widens stop
    """
    close = df["Close"].values.astype(float)
    volume = df["Volume"].values.astype(float) if "Volume" in df.columns else None
    dates = df.index
    returns = np.diff(np.log(close))
    n = len(close)

    effective_entry_prob, effective_exit_prob, effective_stop = _compute_effective_thresholds(
        entry_prob, exit_prob, stop_loss_pct, sentiment_score, vix_signal,
    )
    vol_ratio = _compute_volume_ratio(volume, n)

    # Walk-forward
    rows = []
    current_model = None
    current_order = None
    position = 0.0
    entry_price = 0.0
    stop_level = 0.0
    target_level = 0.0
    entry_bar = 0
    trade_results = []  # recent trade P&L for Kelly
    current_kelly = 0.10  # start conservative

    # Cache for incremental forward filter
    _log_alpha = None
    _model_id = None

    # Buffer for smoothing regime probabilities
    p_bull_history = []

    for t in range(min_train_bars, n):
        # Refit HMM on expanding window periodically
        if current_model is None or (t - min_train_bars) % hmm_refit_bars == 0:
            result = fit_hmm_expanding(returns[:t], n_seeds=3, n_iter=50)
            if result is not None:
                current_model, current_order = result
                _log_alpha = None  # reset cache on refit
                _model_id = id(current_model)

        if current_model is None:
            continue

        # Incremental forward filter: extend by one observation instead of
        # recomputing from scratch. Reset on model refit.
        p_bull, p_bear, _log_alpha = _forward_step_incremental(
            current_model, current_order, returns, t, _log_alpha,
        )
        p_bull_history.append(p_bull)

        # Smoothed P(Bull) for exit decisions — raw hourly is too noisy
        if len(p_bull_history) >= regime_smooth:
            p_bull_smooth = float(np.mean(p_bull_history[-regime_smooth:]))
        else:
            p_bull_smooth = p_bull

        cur_vol_ratio = float(vol_ratio[t]) if t < len(vol_ratio) else 1.0
        cur_close = float(close[t])
        cur_date = dates[t]

        # Daily return for P&L
        prev_close = float(close[t - 1])
        bar_return = (cur_close - prev_close) / prev_close if prev_close > 0 else 0.0

        sell_reason = ""
        trade_event = ""
        bars_held = t - entry_bar if position > 0 else 0

        if position > 0:
            # Stop-loss and take-profit fire immediately (no hold restriction)
            if cur_close <= stop_level:
                sell_reason = f"stop_loss({(entry_price - cur_close) / entry_price * 100:.1f}%)"
                trade_event = "SELL"
            elif cur_close >= target_level:
                sell_reason = f"take_profit({(cur_close - entry_price) / entry_price * 100:.1f}%)"
                trade_event = "SELL"
            elif bars_held >= min_hold_bars and p_bull_smooth < effective_exit_prob:
                sell_reason = f"regime_exit(P_bull_smooth={p_bull_smooth:.2f}<{effective_exit_prob})"
                trade_event = "SELL"

            if trade_event == "SELL":
                trade_pl = (cur_close - entry_price) / entry_price
                trade_results.append(trade_pl)
                position = 0.0

                # Update Kelly from recent trades
                if use_kelly and len(trade_results) >= kelly_lookback:
                    recent = trade_results[-kelly_lookback:]
                    wins = [r for r in recent if r > 0]
                    losses = [r for r in recent if r < 0]
                    wr = len(wins) / len(recent) if recent else 0
                    aw = np.mean(wins) if wins else 0
                    al = np.mean(losses) if losses else -0.05
                    current_kelly = kelly_fraction(wr, aw, al)

        elif position == 0:
            # Check entry — use smoothed P(Bull) to filter out noise spikes
            vol_ok = cur_vol_ratio >= volume_min_ratio
            if p_bull_smooth >= effective_entry_prob and vol_ok:
                position = current_kelly if use_kelly else 0.10
                position = max(0.02, position)  # minimum 2% allocation
                entry_price = cur_close
                stop_level = cur_close * (1 - effective_stop)
                target_level = cur_close * (1 + take_profit_pct)
                entry_bar = t
                trade_event = "BUY"

        strategy_return = position * bar_return if position > 0 else 0.0

        rows.append({
            "date": cur_date,
            "close": round(cur_close, 4),
            "p_bull": round(p_bull, 4),
            "p_bull_smooth": round(p_bull_smooth, 4),
            "p_bear": round(p_bear, 4),
            "volume_ratio": round(cur_vol_ratio, 2),
            "position": round(position, 4),
            "trade_event": trade_event,
            "sell_reason": sell_reason,
            "entry_price": round(entry_price, 4) if position > 0 else None,
            "stop_level": round(stop_level, 4) if position > 0 else None,
            "target_level": round(target_level, 4) if position > 0 else None,
            "kelly_fraction": round(current_kelly, 4),
            "bar_return": round(bar_return, 6),
            "strategy_return": round(strategy_return, 6),
        })

    if not rows:
        return _empty_result(initial_cash)
    detail = pd.DataFrame(rows).set_index("date")
    if detail.empty:
        return _empty_result(initial_cash)

    # Equity curves
    strat_ret = detail["strategy_return"].values
    strat_equity = (1 + strat_ret).cumprod()
    bh_ret = detail["bar_return"].values
    bh_equity = (1 + bh_ret).cumprod()
    detail["strategy_equity"] = strat_equity
    detail["bh_equity"] = bh_equity

    # Portfolio simulation
    portfolio_values = _simulate_portfolio_value(detail, initial_cash, trade_cost)
    detail["portfolio_value"] = portfolio_values

    return _build_quant_backtest_stats(
        detail, strat_ret, bh_ret, strat_equity, bh_equity, initial_cash,
        portfolio_values, trade_results, current_kelly,
    )


def _empty_result(initial_cash):
    return {
        "sharpe_strategy": float("nan"), "sharpe_bh": float("nan"),
        "total_return_strategy": 0, "total_return_bh": 0,
        "max_drawdown_strategy": float("nan"), "max_drawdown_bh": float("nan"),
        "initial_cash": initial_cash, "final_portfolio": initial_cash, "total_pl": 0,
        "n_buys": 0, "n_sells": 0, "trade_results": [], "final_kelly": 0,
        "detail": pd.DataFrame(), "n_bars": 0,
    }
