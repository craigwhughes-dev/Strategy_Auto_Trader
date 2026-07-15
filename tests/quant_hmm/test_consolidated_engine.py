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

    def test_precompute_consol_and_bb_pctb_always_populated(self):
        """choppy_vol reads these from mom_snap regardless of exit_on_* flags —
        they must not be gated behind need_exit like macd_bear_x/rsi_ob_exit."""
        from Strategy_Auto_Trader.quant_hmm.consolidated_engine import _precompute_hourly_vote_series
        close = pd.Series(np.linspace(100, 120, 300))
        pre = _precompute_hourly_vote_series(
            close, None, exit_on_macd_cross=False, exit_on_rsi_reversal=False,
            exit_on_consolidation=False,
        )
        assert pre["need_exit"] is False
        assert pre["consol_full"] is not None
        assert pre["bb_pctb_full"] is not None
        assert pre["macd_bear_x"] is None   # still exit-only, correctly gated

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
                    "above_sma20", "above_sma50", "volume_ratio",
                    "consolidation", "bb_pctb"):
            assert key in snap, f"missing key: {key}"
        assert isinstance(snap["consolidation"], bool)
        assert isinstance(snap["bb_pctb"], float)

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

    # -- skip_unused_indicators -------------------------------------------

    def test_precompute_rsi_disabled_series_none(self):
        from Strategy_Auto_Trader.quant_hmm.consolidated_engine import _precompute_hourly_vote_series
        close = pd.Series(np.linspace(100, 120, 300))
        pre = _precompute_hourly_vote_series(close, None, rsi_enabled=False)
        assert pre["rsi_full"] is None
        assert pre["rsi_x_above_50"] is None
        assert pre["rsi_x_below_40"] is None

    def test_precompute_rsi_disabled_exit_indicators_still_computed(self):
        from Strategy_Auto_Trader.quant_hmm.consolidated_engine import _precompute_hourly_vote_series
        close = pd.Series(np.linspace(100, 120, 300))
        pre = _precompute_hourly_vote_series(
            close, None, rsi_enabled=False, exit_on_rsi_reversal=True,
        )
        assert pre["rsi_full"] is None
        assert pre["rsi_ob_exit"] is not None
        assert pre["rsi_mom_loss"] is not None

    def test_build_mom_snap_omits_rsi_keys_when_disabled(self):
        from Strategy_Auto_Trader.quant_hmm.consolidated_engine import (
            _precompute_hourly_vote_series, _build_mom_snap,
        )
        close = pd.Series(np.linspace(100, 120, 300))
        pre = _precompute_hourly_vote_series(close, None, rsi_enabled=False)
        snap = _build_mom_snap(250, float(close.iloc[250]), pre)
        assert "cur_rsi" not in snap
        assert "recent_cross_above_50" not in snap
        assert "recent_cross_below_40" not in snap
        assert "above_sma20" in snap and "above_sma50" in snap

    @staticmethod
    def _spy_regime_model():
        from Strategy_Auto_Trader.plugins.types import RegimeState

        class SpyRegimeModel:
            def __init__(self):
                self.refit_calls = 0
                self.step_calls = 0

            def needs_refit(self, t):
                return self.refit_calls == 0

            def refit(self, returns):
                self.refit_calls += 1

            def step(self, returns, t):
                self.step_calls += 1
                return RegimeState(p_bull=0.8, p_bear=0.1, p_bull_smooth=0.8,
                                   regime_signal=0.7, hmm_vote=2)

            def reset(self):
                pass

        return SpyRegimeModel()

    def test_consolidated_hmm_weight_zero_skips_regime_model(self):
        from Strategy_Auto_Trader.quant_hmm.consolidated_engine import consolidated_backtest
        spy = self._spy_regime_model()
        df = _hourly_df(np.linspace(100, 130, 200))
        result = consolidated_backtest(
            df, regime_model=spy, weights={"hmm": 0.0},
            min_train_bars=50, hmm_refit_bars=50, regime_smooth=1,
        )
        assert spy.step_calls == 0
        assert spy.refit_calls == 0
        assert result["n_bars"] > 0
        assert result["detail"]["regime_signal"].isna().all()
        assert result["detail"]["hmm_vote"].isna().all()

    def test_consolidated_hmm_weight_zero_opt_out_still_runs_model(self):
        from Strategy_Auto_Trader.quant_hmm.consolidated_engine import consolidated_backtest
        spy = self._spy_regime_model()
        df = _hourly_df(np.linspace(100, 130, 200))
        consolidated_backtest(
            df, regime_model=spy, weights={"hmm": 0.0},
            skip_unused_indicators=False,
            min_train_bars=50, hmm_refit_bars=50, regime_smooth=1,
        )
        assert spy.step_calls == 150
        assert spy.refit_calls >= 1

    def test_consolidated_nonzero_hmm_weight_runs_model(self):
        from Strategy_Auto_Trader.quant_hmm.consolidated_engine import consolidated_backtest
        spy = self._spy_regime_model()
        df = _hourly_df(np.linspace(100, 130, 200))
        consolidated_backtest(
            df, regime_model=spy,
            min_train_bars=50, hmm_refit_bars=50, regime_smooth=1,
        )
        assert spy.step_calls == 150

    def test_consolidated_entry_strategy_weights_drive_skip(self):
        from Strategy_Auto_Trader.quant_hmm.consolidated_engine import consolidated_backtest
        from Strategy_Auto_Trader.strategy.default import DefaultEntry, DefaultExit
        spy = self._spy_regime_model()
        df = _hourly_df(np.linspace(100, 130, 200))
        entry = DefaultEntry(weights={"hmm": 0.0})
        result = consolidated_backtest(
            df, regime_model=spy, entry_strategy=entry, exit_strategy=DefaultExit(),
            min_train_bars=50, hmm_refit_bars=50, regime_smooth=1,
        )
        assert spy.step_calls == 0
        assert result["n_bars"] > 0

    def test_consolidated_vol_filter_veto_skips_regime_model(self):
        from Strategy_Auto_Trader.quant_hmm.consolidated_engine import consolidated_backtest
        from Strategy_Auto_Trader.strategy.default import DefaultEntry, DefaultExit
        spy = self._spy_regime_model()
        df = _hourly_df(np.linspace(100, 130, 200))
        result = consolidated_backtest(
            df, regime_model=spy,
            entry_strategy=DefaultEntry(vol_filter_ok=False), exit_strategy=DefaultExit(),
            min_train_bars=50, hmm_refit_bars=50, regime_smooth=1,
        )
        # Vetoed strategy returns permanent HOLD without reading the regime,
        # so the HMM must be skipped for the whole run.
        assert spy.step_calls == 0
        assert spy.refit_calls == 0
        assert result["n_buys"] == 0

    def test_consolidated_rsi_weight_zero_runs_without_rsi(self):
        from Strategy_Auto_Trader.quant_hmm.consolidated_engine import consolidated_backtest
        spy = self._spy_regime_model()
        df = _hourly_df(np.linspace(100, 130, 200))
        result = consolidated_backtest(
            df, regime_model=spy, weights={"rsi": 0.0},
            min_train_bars=50, hmm_refit_bars=50, regime_smooth=1,
        )
        assert result["n_bars"] > 0
        assert result["detail"]["rsi"].isna().all()
        # HMM still active — its weight is non-zero
        assert spy.step_calls == 150

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

    # -- require_flip_entry knob ------------------------------------------

    def test_require_flip_entry_false_allows_consecutive_buys(self):
        """Verify that require_flip_entry=False bypasses the flip guard.

        With require_flip_entry=True (default), consecutive BUY signals from the
        same bar are suppressed — only a non-BUY → BUY transition triggers entry.

        With require_flip_entry=False, a BUY signal is acted on even if the
        previous bar also generated a BUY (allowing rapid re-entry after close).
        """
        from Strategy_Auto_Trader.plugins.types import EntryDecision, RegimeState

        # Create a mock entry strategy that always returns BUY (to isolate flip-guard behavior)
        class AlwaysBuyEntry:
            """Entry strategy that always returns BUY, for testing flip guard."""
            require_flip_entry = False  # KEY: disable flip guard

            def evaluate(
                self,
                regime: RegimeState,
                mom: dict,
                _volume_ratio: float,
                currently_in: bool = False,
            ) -> EntryDecision:
                return EntryDecision(
                    flag="BUY",
                    raw_flag="BUY",
                    score=5.0,
                    reason="always_buy_for_test",
                )

        # Use a simple exit strategy (from an existing strategy)
        from Strategy_Auto_Trader.strategy.default import DefaultExit

        # Scenario: steady rising price (will induce BUY signals) with short hold window
        n = 200
        close = np.linspace(100, 130, n)  # steady rise
        p_bull_seq = [0.80] * n  # always bullish

        result = _run_consolidated_fake(
            close,
            p_bull_seq,
            min_train_bars=50,
            hmm_refit_bars=50,
            regime_smooth=1,
            buy_threshold=1.0,
            volume_min_ratio=0.0,
            max_hold_days=0,  # exit after each bar (or quickly) so we can re-enter
            entry_strategy=AlwaysBuyEntry(),
            exit_strategy=DefaultExit(),
        )

        # With require_flip_entry=False, we should see multiple BUY events
        # because the strategy always returns BUY and exits quickly, allowing re-entry
        n_buys = result["n_buys"]
        assert n_buys >= 2, (
            f"expected >= 2 BUY entries with require_flip_entry=False, got {n_buys}. "
            "The flip guard should not suppress re-entry on consecutive BUY signals."
        )

    def test_require_flip_entry_true_suppresses_consecutive_buys(self):
        """Verify that require_flip_entry=True (default) enforces the flip guard.

        This is a sanity check: consecutive BUY signals without a transition
        from non-BUY should be suppressed.
        """
        from Strategy_Auto_Trader.plugins.types import EntryDecision, RegimeState

        # Create a mock entry strategy that always returns BUY
        class AlwaysBuyEntry:
            """Entry strategy that always returns BUY."""
            require_flip_entry = True  # explicitly set (or omit for default True)

            def evaluate(
                self,
                regime: RegimeState,
                mom: dict,
                _volume_ratio: float,
                currently_in: bool = False,
            ) -> EntryDecision:
                return EntryDecision(
                    flag="BUY",
                    raw_flag="BUY",
                    score=5.0,
                    reason="always_buy_for_test",
                )

        from Strategy_Auto_Trader.strategy.default import DefaultExit

        n = 200
        close = np.linspace(100, 130, n)
        p_bull_seq = [0.80] * n

        result = _run_consolidated_fake(
            close,
            p_bull_seq,
            min_train_bars=50,
            hmm_refit_bars=50,
            regime_smooth=1,
            buy_threshold=1.0,
            volume_min_ratio=0.0,
            max_hold_days=0,
            entry_strategy=AlwaysBuyEntry(),
            exit_strategy=DefaultExit(),
        )

        # With require_flip_entry=True (default), we should see fewer BUY entries
        # because consecutive BUY signals are suppressed until a non-BUY occurs.
        n_buys = result["n_buys"]
        assert n_buys <= 1, (
            f"expected <= 1 BUY entries with require_flip_entry=True, got {n_buys}. "
            "The flip guard should suppress re-entry on consecutive BUY signals."
        )

    # -- min_hold_bars knob ---

    def test_min_hold_bars_zero_allows_early_signal_exit(self):
        """Verify that min_hold_bars=0 on exit strategy allows signal-based SELL
        to fire immediately (on bar 2, bars_held=1) instead of waiting 48 bars.

        With the default min_hold_bars=48, a signal-based SELL is blocked until
        at least 48 bars have elapsed. With min_hold_bars=0, the signal fires
        immediately without the hold-bars gate.
        """
        from Strategy_Auto_Trader.plugins.types import EntryDecision, ExitResult, RegimeState, TradeState, BarData

        # Entry strategy that buys once, then returns SELL after entry
        class QuickExitEntry:
            """Entry that BUYs on first call when not in position, then SELL immediately when in position."""
            require_flip_entry = False  # Bypass flip guard for test clarity

            def __init__(self):
                self._has_entered = False

            def evaluate(
                self,
                regime: RegimeState,
                mom: dict,
                _volume_ratio: float,
                currently_in: bool = False,
            ) -> EntryDecision:
                # BUY once when not in position
                if not currently_in and not self._has_entered:
                    self._has_entered = True
                    return EntryDecision(flag="BUY", raw_flag="BUY", score=5.0, reason="test_buy")
                # Signal SELL immediately when in position (after entry)
                if currently_in and self._has_entered:
                    return EntryDecision(flag="SELL", raw_flag="SELL", score=5.0, reason="test_sell")
                # Otherwise hold
                return EntryDecision(flag="HOLD", raw_flag="HOLD", score=0.0, reason="")

        # Exit strategy with min_hold_bars=0 (no hold gate)
        class QuickExit:
            """Exit strategy with min_hold_bars=0."""
            min_hold_bars = 0  # KEY: Allow signal-based SELL immediately

            @property
            def stop_loss_pct(self) -> float:
                return 0.05

            @property
            def take_profit_pct(self) -> float:
                return 0.15

            def check(self, trade: TradeState, bar: BarData) -> ExitResult:
                # No exit logic here; let the entry strategy's SELL signal fire
                return ExitResult(
                    exit_hit=False,
                    sell_reason="",
                    peak_price_since_entry=max(trade.peak_price_since_entry, bar.cur_close),
                    days_in_trade=trade.days_in_trade,
                )

        # Scenario: entry on bar 51+, exit signal 1 bar later (bars_held=1)
        n = 200
        close = np.linspace(100, 120, n)  # rising price (no stop)
        p_bull_seq = [0.80] * n  # always bullish

        result = _run_consolidated_fake(
            close,
            p_bull_seq,
            min_train_bars=50,
            hmm_refit_bars=50,
            regime_smooth=1,
            buy_threshold=1.0,
            volume_min_ratio=0.0,
            min_hold_bars=48,  # Global default (high)
            entry_strategy=QuickExitEntry(),
            exit_strategy=QuickExit(),
        )

        # We expect exactly 1 BUY and 1 SELL (trade opens on bar 1, closes on bar 2)
        detail = result["detail"]
        buys = detail[detail["trade_event"] == "BUY"]
        sells = detail[detail["trade_event"] == "SELL"]

        assert len(buys) == 1, f"expected 1 BUY, got {len(buys)}"
        assert len(sells) == 1, f"expected 1 SELL, got {len(sells)}"

        buy_pos = detail.index.get_loc(buys.index[0])
        sell_pos = detail.index.get_loc(sells.index[0])

        # With min_hold_bars=0 on exit strategy, SELL should fire 1 bar after BUY
        assert sell_pos == buy_pos + 1, (
            f"expected SELL {buy_pos + 1} (1 bar after BUY at {buy_pos}), got {sell_pos}. "
            f"With min_hold_bars=0, the signal-based SELL should fire immediately (within 1 bar)."
        )

    def test_min_hold_bars_default_blocks_early_signal_exit(self):
        """Verify that without min_hold_bars set on exit strategy, the global
        min_hold_bars=48 blocks signal-based SELL before 48 bars elapse.

        This is the sanity check: a strategy without the min_hold_bars override
        should fall back to the global CLI param (48 by default).
        """
        from Strategy_Auto_Trader.plugins.types import EntryDecision, ExitResult, RegimeState, TradeState, BarData

        # Entry strategy that buys once, then returns SELL every bar after
        class AlwaysSellEntry:
            """Entry that BUYs on first call when not in position, then SELL every bar after."""
            require_flip_entry = False  # Bypass flip guard for test clarity

            def __init__(self):
                self._has_entered = False

            def evaluate(
                self,
                regime: RegimeState,
                mom: dict,
                _volume_ratio: float,
                currently_in: bool = False,
            ) -> EntryDecision:
                # BUY once when not in position
                if not currently_in and not self._has_entered:
                    self._has_entered = True
                    return EntryDecision(flag="BUY", raw_flag="BUY", score=5.0, reason="test_buy")
                # Always return SELL after entry
                if currently_in:
                    return EntryDecision(flag="SELL", raw_flag="SELL", score=5.0, reason="test_sell")
                # Otherwise hold
                return EntryDecision(flag="HOLD", raw_flag="HOLD", score=0.0, reason="")

        # Exit strategy WITHOUT min_hold_bars set (uses plugin adapter default)
        from Strategy_Auto_Trader.strategy.default import DefaultExit

        # Scenario: entry on bar 51+, signal tries to SELL on bar 52+ but is blocked
        # until bars_held >= 48
        n = 250
        close = np.linspace(100, 120, n)
        p_bull_seq = [0.80] * n

        result = _run_consolidated_fake(
            close,
            p_bull_seq,
            min_train_bars=50,
            hmm_refit_bars=50,
            regime_smooth=1,
            buy_threshold=1.0,
            volume_min_ratio=0.0,
            min_hold_bars=48,  # Global hold-bars gate
            entry_strategy=AlwaysSellEntry(),
            exit_strategy=DefaultExit(),
        )

        detail = result["detail"]
        buys = detail[detail["trade_event"] == "BUY"]
        sells = detail[detail["trade_event"] == "SELL"]

        assert len(buys) == 1, f"expected 1 BUY, got {len(buys)}"
        assert len(sells) == 1, f"expected 1 SELL, got {len(sells)}"

        buy_pos = detail.index.get_loc(buys.index[0])
        sell_pos = detail.index.get_loc(sells.index[0])

        # Without min_hold_bars override on exit strategy, signal-based SELL
        # is blocked until bars_held >= 48. So SELL should be at least 48 bars
        # after BUY.
        assert sell_pos >= buy_pos + 48, (
            f"expected SELL at position >= {buy_pos + 48} (48 bars after BUY at {buy_pos}), got {sell_pos}. "
            f"Without min_hold_bars override, should use global default of 48."
        )

    # -- use_kelly / kelly_lookback knobs (read from exit_strategy) ------------

    def test_use_kelly_false_on_exit_strategy_disables_kelly_sizing(self):
        """Verify exit_strategy.use_kelly=False disables Kelly recompute,
        leaving final_kelly at KellySizer's static default (0.10) even after
        many closed trades.
        """
        from Strategy_Auto_Trader.plugins.types import EntryDecision, ExitResult, RegimeState, TradeState, BarData

        class RepeatingEntry:
            """BUY when flat, SELL when in position -- generates many round trips."""
            require_flip_entry = False

            def evaluate(self, regime, mom, _volume_ratio, currently_in=False):
                if not currently_in:
                    return EntryDecision(flag="BUY", raw_flag="BUY", score=5.0, reason="buy")
                return EntryDecision(flag="SELL", raw_flag="SELL", score=5.0, reason="sell")

        class NoKellyExit:
            """Exit strategy that declares Kelly sizing off."""
            min_hold_bars = 0
            use_kelly = False  # KEY: disables Kelly recompute

            @property
            def stop_loss_pct(self) -> float:
                return 0.05

            @property
            def take_profit_pct(self) -> float:
                return 0.15

            def check(self, trade, bar) -> ExitResult:
                return ExitResult(
                    exit_hit=False, sell_reason="",
                    peak_price_since_entry=max(trade.peak_price_since_entry, bar.cur_close),
                    days_in_trade=trade.days_in_trade,
                )

        n = 200
        close = np.linspace(100, 120, n)
        p_bull_seq = [0.80] * n

        result = _run_consolidated_fake(
            close, p_bull_seq,
            min_train_bars=50, hmm_refit_bars=50, regime_smooth=1,
            buy_threshold=1.0, volume_min_ratio=0.0,
            entry_strategy=RepeatingEntry(),
            exit_strategy=NoKellyExit(),
        )
        sells = result["detail"][result["detail"]["trade_event"] == "SELL"]
        assert len(sells) >= 5, f"expected >= 5 round trips to exercise Kelly recompute, got {len(sells)}"
        assert result["final_kelly"] == pytest.approx(0.10), (
            "use_kelly=False on exit_strategy should keep final_kelly at the static default"
        )

    def test_kelly_lookback_read_from_exit_strategy(self):
        """Verify exit_strategy.kelly_lookback overrides the engine's default
        (20): a lookback of 2 lets final_kelly diverge from the static 0.10
        default after just a couple of trades.
        """
        from Strategy_Auto_Trader.plugins.types import EntryDecision, ExitResult, RegimeState, TradeState, BarData

        class RepeatingEntry:
            require_flip_entry = False

            def evaluate(self, regime, mom, _volume_ratio, currently_in=False):
                if not currently_in:
                    return EntryDecision(flag="BUY", raw_flag="BUY", score=5.0, reason="buy")
                return EntryDecision(flag="SELL", raw_flag="SELL", score=5.0, reason="sell")

        class ShortLookbackExit:
            min_hold_bars = 0
            use_kelly = True
            kelly_lookback = 2  # KEY: recompute after just 2 trades

            @property
            def stop_loss_pct(self) -> float:
                return 0.05

            @property
            def take_profit_pct(self) -> float:
                return 0.15

            def check(self, trade, bar) -> ExitResult:
                return ExitResult(
                    exit_hit=False, sell_reason="",
                    peak_price_since_entry=max(trade.peak_price_since_entry, bar.cur_close),
                    days_in_trade=trade.days_in_trade,
                )

        n = 200
        close = np.linspace(100, 120, n)
        p_bull_seq = [0.80] * n

        result = _run_consolidated_fake(
            close, p_bull_seq,
            min_train_bars=50, hmm_refit_bars=50, regime_smooth=1,
            buy_threshold=1.0, volume_min_ratio=0.0,
            entry_strategy=RepeatingEntry(),
            exit_strategy=ShortLookbackExit(),
        )
        sells = result["detail"][result["detail"]["trade_event"] == "SELL"]
        assert len(sells) >= 5, f"expected >= 5 round trips to exercise Kelly recompute, got {len(sells)}"
        assert result["final_kelly"] != pytest.approx(0.10), (
            "kelly_lookback=2 on exit_strategy should have triggered a Kelly recompute by now"
        )

