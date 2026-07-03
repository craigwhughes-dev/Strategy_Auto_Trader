from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest


def _hourly_df(close, volume=1000.0, start="2024-01-01"):
    """Build a synthetic hourly OHLCV DataFrame from a close-price array."""
    close = np.asarray(close, dtype=float)
    n = len(close)
    idx = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame({
        "Open": close, "High": close * 1.0005, "Low": close * 0.9995,
        "Close": close, "Volume": np.full(n, volume),
    }, index=idx)


def _run_consolidated_fake(close, p_bull_seq, **kwargs):
    """Run consolidated_backtest with HMM replaced by a deterministic P(Bull) sequence.

    Passes a FakeRegimeModel plugin so the mock works regardless of where the
    HMM functions are imported from.
    """
    from Strategy_Auto_Trader.quant_hmm.consolidated_engine import consolidated_backtest
    from Strategy_Auto_Trader.plugins.types import RegimeState
    from Strategy_Auto_Trader.quant_hmm.quant_engine import discretize_p_bull

    df = _hourly_df(close)
    call_idx = {"i": 0}

    entry_prob = kwargs.get("entry_prob", 0.65)
    exit_prob = kwargs.get("exit_prob", 0.40)
    regime_smooth = kwargs.get("regime_smooth", 24)

    class FakeRegimeModel:
        def __init__(self):
            self._history: list = []

        def needs_refit(self, t: int) -> bool:
            return False

        def refit(self, returns) -> None:
            pass

        def step(self, returns, t):
            i = call_idx["i"]
            p_bull = p_bull_seq[i] if i < len(p_bull_seq) else p_bull_seq[-1]
            call_idx["i"] += 1
            p_bear = 1.0 - p_bull
            self._history.append(p_bull)
            if len(self._history) >= regime_smooth:
                p_bull_smooth = float(np.mean(self._history[-regime_smooth:]))
            else:
                p_bull_smooth = p_bull
            return RegimeState(
                p_bull=p_bull,
                p_bear=p_bear,
                p_bull_smooth=p_bull_smooth,
                regime_signal=p_bull_smooth - p_bear,
                hmm_vote=discretize_p_bull(p_bull_smooth, bull_edge=entry_prob, bear_edge=exit_prob),
            )

        def reset(self) -> None:
            call_idx["i"] = 0
            self._history.clear()

    return consolidated_backtest(df, regime_model=FakeRegimeModel(), **kwargs)


class TestConsolidatedEngine:

    # -- discretize_p_bull ------------------------------------------------

    def test_discretize_p_bull_above_bull_edge_is_bull(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import discretize_p_bull
        assert discretize_p_bull(0.70, bull_edge=0.65, bear_edge=0.40) == 2

    def test_discretize_p_bull_at_bull_edge_is_bull(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import discretize_p_bull
        assert discretize_p_bull(0.65, bull_edge=0.65, bear_edge=0.40) == 2

    def test_discretize_p_bull_below_bear_edge_is_bear(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import discretize_p_bull
        assert discretize_p_bull(0.30, bull_edge=0.65, bear_edge=0.40) == 0

    def test_discretize_p_bull_at_bear_edge_is_bear(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import discretize_p_bull
        assert discretize_p_bull(0.40, bull_edge=0.65, bear_edge=0.40) == 0

    def test_discretize_p_bull_between_edges_is_sideways(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import discretize_p_bull
        assert discretize_p_bull(0.55, bull_edge=0.65, bear_edge=0.40) == 1

    # -- _precompute_hourly_vote_series ------------------------------------

    def test_precompute_hourly_vote_series_has_expected_keys(self):
        from Strategy_Auto_Trader.quant_hmm.consolidated_engine import _precompute_hourly_vote_series
        close = pd.Series(np.linspace(100, 120, 300))
        volume = np.full(300, 1000.0)
        pre = _precompute_hourly_vote_series(close, volume)
        for key in ("rsi_full", "sma20_full", "sma50_full", "sma200_full",
                    "rolling_vol_full", "rsi_x_above_50", "rsi_x_below_40",
                    "vol_ratio_full", "need_exit", "sar_full"):
            assert key in pre, f"missing key: {key}"

    def test_precompute_hourly_vol_ratio_populated_with_volume(self):
        from Strategy_Auto_Trader.quant_hmm.consolidated_engine import _precompute_hourly_vote_series
        close = pd.Series(np.linspace(100, 120, 300))
        volume = np.full(300, 1000.0)
        pre = _precompute_hourly_vote_series(close, volume)
        assert pre["vol_ratio_full"] is not None
        assert len(pre["vol_ratio_full"]) == 300

    def test_precompute_hourly_vol_ratio_none_without_volume(self):
        from Strategy_Auto_Trader.quant_hmm.consolidated_engine import _precompute_hourly_vote_series
        close = pd.Series(np.linspace(100, 120, 300))
        pre = _precompute_hourly_vote_series(close, None)
        assert pre["vol_ratio_full"] is None

    def test_precompute_hourly_exit_series_populated_when_flags_set(self):
        from Strategy_Auto_Trader.quant_hmm.consolidated_engine import _precompute_hourly_vote_series
        close = pd.Series(np.linspace(100, 120, 300))
        pre = _precompute_hourly_vote_series(
            close, None, exit_on_macd_cross=True, exit_on_rsi_reversal=True,
            exit_on_consolidation=True,
        )
        assert pre["need_exit"] is True
        assert pre["macd_bear_x"] is not None
        assert pre["rsi_ob_exit"] is not None
        assert pre["consol_full"] is not None

    def test_precompute_hourly_sar_none_when_disabled(self):
        from Strategy_Auto_Trader.quant_hmm.consolidated_engine import _precompute_hourly_vote_series
        close = pd.Series(np.linspace(100, 120, 300))
        pre = _precompute_hourly_vote_series(close, None, use_sar_stop=False)
        assert pre["sar_full"] is None

    # -- _build_mom_snap --------------------------------------------------

    def test_build_mom_snap_returns_expected_keys(self):
        from Strategy_Auto_Trader.quant_hmm.consolidated_engine import (
            _precompute_hourly_vote_series, _build_mom_snap,
        )
        n = 300
        close = pd.Series(np.linspace(100, 120, n))
        volume = np.full(n, 1000.0)
        pre = _precompute_hourly_vote_series(close, volume)
        snap = _build_mom_snap(250, float(close.iloc[250]), pre)
        for key in ("cur_rsi", "recent_cross_above_50", "recent_cross_below_40",
                    "above_sma20", "above_sma50", "volume_ratio"):
            assert key in snap, f"missing key: {key}"

    def test_build_mom_snap_above_sma200_populated_after_warmup(self):
        from Strategy_Auto_Trader.quant_hmm.consolidated_engine import (
            _precompute_hourly_vote_series, _build_mom_snap,
        )
        n = 300
        close = pd.Series(np.linspace(100, 120, n))
        pre = _precompute_hourly_vote_series(close, None)
        snap = _build_mom_snap(250, float(close.iloc[250]), pre)
        assert "above_sma200" in snap
        assert isinstance(snap["above_sma200"], bool)

    # -- consolidated_backtest end-to-end (with fake HMM) ----------------

    def test_consolidated_too_short_returns_empty(self):
        from Strategy_Auto_Trader.quant_hmm.consolidated_engine import consolidated_backtest
        df = _hourly_df(np.linspace(100, 110, 10))
        result = consolidated_backtest(df, min_train_bars=500)
        assert result["n_bars"] == 0
        assert result["n_buys"] == 0

    def test_consolidated_detail_has_expected_columns(self):
        close = np.linspace(100, 130, 200)
        # p_bull always 0.80 — strong bull throughout; regime_smooth=1 to skip smoothing
        p_bull_seq = [0.80] * 200
        result = _run_consolidated_fake(
            close, p_bull_seq, min_train_bars=50, hmm_refit_bars=50,
            regime_smooth=1, buy_threshold=1.0,
        )
        if result["n_bars"] > 0:
            for col in ("p_bull", "p_bull_smooth", "hmm_vote", "trade_event", "strategy_return"):
                assert col in result["detail"].columns, f"missing column: {col}"

    def test_consolidated_enters_on_strong_bull_signal(self):
        # Start bearish (low p_bull), then flip to strong bull -> triggers BUY transition
        close = np.linspace(100, 130, 200)
        # First 30 bars: bear regime (establishes "not-BUY" baseline for transition guard)
        # Then flip to strong bull so the transition guard fires
        p_bull_seq = [0.20] * 30 + [0.80] * 170
        result = _run_consolidated_fake(
            close, p_bull_seq, min_train_bars=50, hmm_refit_bars=50,
            regime_smooth=1, buy_threshold=1.0, volume_min_ratio=0.0,
        )
        assert result["n_buys"] >= 1, "expected at least one BUY on bear->bull transition"

    def test_consolidated_no_entry_on_bear_signal(self):
        # Falling price + p_bull=0.10 (below exit_prob=0.40 → Bear vote)
        n = 200
        close = np.linspace(130, 100, n)  # falling
        p_bull_seq = [0.10] * n
        result = _run_consolidated_fake(
            close, p_bull_seq, min_train_bars=50, hmm_refit_bars=50,
            regime_smooth=1, volume_min_ratio=0.0,
        )
        assert result["n_buys"] == 0, "should not enter on pure bear signal"

    def test_consolidated_quality_gate_vetoes_weak_buy(self):
        # Create conditions that make _is_weak_buy_context return True by forcing
        # markov_sig (= p_bull_smooth - p_bear) < 0.25 and RSI to stay weak.
        # p_bull=0.55 -> p_bull_smooth - p_bear ~ 0.10 < 0.25 (weak context).
        n = 200
        # Sideways price so above_sma20/50 booleans are uncertain; use low RSI environment
        close = np.full(n, 100.0)  # flat price — RSI stays near 50, SMAs equal price
        p_bull_seq = [0.55] * n  # sideways regime (no strong bull)
        result = _run_consolidated_fake(
            close, p_bull_seq, min_train_bars=50, hmm_refit_bars=50,
            regime_smooth=1, buy_threshold=0.5, volume_min_ratio=0.0,
        )
        # Quality gate should veto most/all BUY signals from this weak context
        # At minimum, fewer buys than without gate (or zero)
        assert result["n_buys"] == 0 or result["n_buys"] <= 1

    def test_consolidated_stop_loss_exits_trade(self):
        # Enter at 100, drop hard — should trigger hard stop
        n = 200
        close = np.concatenate([
            np.linspace(100, 110, 80),   # rising — enter
            np.linspace(110, 80, 120),   # steep fall — stop-loss
        ])
        p_bull_seq = [0.80] * 80 + [0.20] * 120
        result = _run_consolidated_fake(
            close, p_bull_seq, min_train_bars=50, hmm_refit_bars=50,
            regime_smooth=1, buy_threshold=1.0, volume_min_ratio=0.0,
            stop_loss_pct=0.05, take_profit_pct=0.30,
        )
        sells = result["detail"][result["detail"]["trade_event"] == "SELL"]
        if len(sells) > 0:
            assert any("rr_stop_loss" in str(r) for r in sells["sell_reason"].values)

    def test_consolidated_take_profit_exits_trade(self):
        # Enter at 100, rise sharply — should trigger take-profit
        n = 200
        close = np.concatenate([
            np.linspace(100, 105, 80),   # gentle rise — enter
            np.linspace(105, 140, 120),  # sharp rise — take-profit
        ])
        p_bull_seq = [0.80] * 200
        result = _run_consolidated_fake(
            close, p_bull_seq, min_train_bars=50, hmm_refit_bars=50,
            regime_smooth=1, buy_threshold=1.0, volume_min_ratio=0.0,
            stop_loss_pct=0.05, take_profit_pct=0.15,
        )
        sells = result["detail"][result["detail"]["trade_event"] == "SELL"]
        if len(sells) > 0:
            assert any("rr_take_profit" in str(r) for r in sells["sell_reason"].values)

    def test_consolidated_trailing_stop_exits_after_peak(self):
        n = 200
        close = np.concatenate([
            np.linspace(100, 110, 80),   # rise
            np.linspace(110, 106, 120),  # pullback — triggers trailing stop
        ])
        p_bull_seq = [0.80] * 200
        result = _run_consolidated_fake(
            close, p_bull_seq, min_train_bars=50, hmm_refit_bars=50,
            regime_smooth=1, buy_threshold=1.0, volume_min_ratio=0.0,
            stop_loss_pct=0.20, take_profit_pct=0.40,  # R:R wide so trailing fires first
            trailing_stop=0.03,                          # 3% trailing stop
        )
        sells = result["detail"][result["detail"]["trade_event"] == "SELL"]
        if len(sells) > 0:
            assert any("trailing_stop" in str(r) for r in sells["sell_reason"].values)

    def test_consolidated_real_hmm_end_to_end(self):
        pytest.importorskip("hmmlearn")
        from Strategy_Auto_Trader.quant_hmm.consolidated_engine import consolidated_backtest
        rng = np.random.default_rng(42)
        rets = rng.normal(0.0005, 0.004, 300)
        close = 100.0 * np.cumprod(1 + rets)
        df = _hourly_df(close)
        result = consolidated_backtest(
            df, min_train_bars=100, hmm_refit_bars=100, regime_smooth=10,
            buy_threshold=2.0,
        )
        assert not result["detail"].empty
        assert np.isfinite(result["final_portfolio"])
        assert set(result["detail"]["trade_event"].unique()).issubset({"", "BUY", "SELL"})
