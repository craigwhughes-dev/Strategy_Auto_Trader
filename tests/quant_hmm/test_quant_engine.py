from __future__ import annotations

import math
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


def _run_with_fake_regime(close, p_bull_seq, **kwargs):
    """Run quant_backtest with the HMM fitting/inference replaced by a
    deterministic, caller-controlled P(Bull) sequence.

    This isolates the engine's entry/exit/stop/sentiment logic from the
    stochastic HMM fit, so behaviour can be tested precisely and fast.
    """
    from Strategy_Auto_Trader.quant_hmm import quant_engine as qe

    df = _hourly_df(close)
    call_idx = {"i": 0}

    def fake_fit(returns, n_seeds=3, n_iter=50):
        return object(), np.array([0, 1, 2])

    def fake_step(model, order, returns, t, log_alpha):
        i = call_idx["i"]
        p_bull = p_bull_seq[i] if i < len(p_bull_seq) else p_bull_seq[-1]
        call_idx["i"] += 1
        return p_bull, 1 - p_bull, np.zeros(3)

    with mock.patch.object(qe, "fit_hmm_expanding", fake_fit), \
         mock.patch.object(qe, "_forward_step_incremental", fake_step):
        return qe.quant_backtest(df, **kwargs)


class TestQuantEngine:

    # -- kelly_fraction -----------------------------------------------------

    def test_kelly_fraction_zero_avg_loss(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import kelly_fraction
        assert kelly_fraction(0.6, 0.10, 0.0) == 0.0

    def test_kelly_fraction_nonpositive_win_rate(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import kelly_fraction
        assert kelly_fraction(0.0, 0.10, -0.05) == 0.0

    def test_kelly_fraction_capped_at_25pct(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import kelly_fraction
        # Very favourable edge should still be capped
        assert kelly_fraction(0.9, 0.30, -0.05) == 0.25

    def test_kelly_fraction_floored_at_zero(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import kelly_fraction
        # Poor win rate / payoff -> negative raw Kelly, floored to 0
        assert kelly_fraction(0.2, 0.05, -0.10) == 0.0

    def test_kelly_fraction_known_value(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import kelly_fraction
        # b=2, win_rate=0.5 -> kelly = (0.5*2 - 0.5)/2 = 0.25 -> capped to 0.25
        result = kelly_fraction(0.5, 0.10, -0.05)
        assert abs(result - 0.25) < 1e-10

    # -- fetch_hourly (mocked) -----------------------------------------------

    def test_fetch_hourly_returns_dataframe(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import fetch_hourly
        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        fake_df = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0,
                                 "Close": np.linspace(100, 110, 10), "Volume": 1000.0}, index=idx)
        with mock.patch("yfinance.download", return_value=fake_df):
            df = fetch_hourly("TEST")
        assert df is not None
        assert len(df) == 10
        assert "Close" in df.columns

    def test_fetch_hourly_empty_returns_none(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import fetch_hourly
        with mock.patch("yfinance.download", return_value=pd.DataFrame()):
            assert fetch_hourly("TEST") is None

    def test_fetch_hourly_exception_returns_none(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import fetch_hourly
        with mock.patch("yfinance.download", side_effect=Exception("network error")):
            assert fetch_hourly("TEST") is None

    def test_fetch_hourly_flattens_multiindex_columns(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import fetch_hourly
        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        cols = pd.MultiIndex.from_product([["Open", "Close"], ["TEST"]])
        fake_df = pd.DataFrame(np.ones((5, 2)), index=idx, columns=cols)
        with mock.patch("yfinance.download", return_value=fake_df):
            df = fetch_hourly("TEST")
        assert not isinstance(df.columns, pd.MultiIndex)

    # -- hmm_regime_probabilities / _forward_step_incremental ---------------

    def test_hmm_regime_probabilities_shape_and_bounds(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import hmm_regime_probabilities

        class MockModel:
            n_components = 3
            means_ = np.array([[-0.01], [0.0], [0.01]])
            covars_ = np.array([[[0.001]], [[0.001]], [[0.001]]])
            transmat_ = np.array([[0.8, 0.1, 0.1],
                                  [0.1, 0.8, 0.1],
                                  [0.1, 0.1, 0.8]])
            startprob_ = np.array([0.34, 0.33, 0.33])

        model = MockModel()
        order = np.array([0, 1, 2])  # already sorted Bear, Sideways, Bull
        obs = np.random.default_rng(0).normal(0, 0.01, 50)
        probs = hmm_regime_probabilities(model, order, obs)
        assert probs.shape == (50, 3)
        # Each row of [P(Bear), P(Sideways), P(Bull)] should sum to ~1
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(50), atol=1e-6)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_forward_step_incremental_matches_batch(self):
        """Incremental stepping should agree with the batch forward filter."""
        from Strategy_Auto_Trader.quant_hmm.quant_engine import (
            hmm_regime_probabilities, _forward_step_incremental,
        )

        class MockModel:
            n_components = 3
            means_ = np.array([[-0.02], [0.0], [0.02]])
            covars_ = np.array([[[0.001]], [[0.001]], [[0.001]]])
            transmat_ = np.array([[0.7, 0.2, 0.1],
                                  [0.1, 0.7, 0.2],
                                  [0.2, 0.1, 0.7]])
            startprob_ = np.array([0.34, 0.33, 0.33])

        model = MockModel()
        order = np.array([0, 1, 2])
        returns = np.random.default_rng(1).normal(0, 0.01, 30)
        batch_probs = hmm_regime_probabilities(model, order, returns)

        log_alpha = None
        for t in range(1, len(returns) + 1):
            p_bull, p_bear, log_alpha = _forward_step_incremental(model, order, returns, t, log_alpha)
            assert abs(p_bull - batch_probs[t - 1, 2]) < 1e-8
            assert abs(p_bear - batch_probs[t - 1, 0]) < 1e-8

    # -- quant_backtest: structural / empty-input behaviour ------------------

    def test_quant_backtest_too_short_returns_empty(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import quant_backtest
        df = _hourly_df(np.full(100, 100.0))
        result = quant_backtest(df, min_train_bars=500)
        assert result["detail"].empty
        assert result["n_buys"] == 0
        assert result["n_sells"] == 0
        assert result["total_pl"] == 0
        assert math.isnan(result["sharpe_strategy"])

    def test_quant_backtest_detail_has_expected_columns(self):
        close = np.full(30, 100.0)
        result = _run_with_fake_regime(close, [0.30] * 20, min_train_bars=10)
        detail = result["detail"]
        assert not detail.empty
        for col in ["close", "p_bull", "p_bull_smooth", "p_bear", "volume_ratio",
                     "position", "trade_event", "sell_reason", "kelly_fraction",
                     "portfolio_value"]:
            assert col in detail.columns

    # -- quant_backtest: entry/exit logic (deterministic, via fake regime) --

    def test_quant_backtest_enters_on_smoothed_threshold_cross(self):
        close = np.full(60, 100.0)
        p_bull_seq = [0.30] * 10 + [0.90] * 15 + [0.10] * 15
        result = _run_with_fake_regime(
            close, p_bull_seq, min_train_bars=20, regime_smooth=5,
            min_hold_bars=5, entry_prob=0.65, exit_prob=0.40,
        )
        detail = result["detail"]
        buys = detail[detail["trade_event"] == "BUY"]
        sells = detail[detail["trade_event"] == "SELL"]
        assert len(buys) >= 1
        assert len(sells) >= 1
        # Sell must come after buy and be a regime exit (price is flat, no stop/target hit)
        assert sells.iloc[0]["sell_reason"].startswith("regime_exit")
        assert detail.index.get_loc(sells.index[0]) > detail.index.get_loc(buys.index[0])

    def test_quant_backtest_no_entry_below_threshold(self):
        close = np.full(30, 100.0)
        result = _run_with_fake_regime(
            close, [0.60] * 20, min_train_bars=10, regime_smooth=5, entry_prob=0.65,
        )
        assert result["n_buys"] == 0

    def test_quant_backtest_min_hold_blocks_early_regime_exit(self):
        close = np.full(50, 100.0)
        p_bull_seq = [0.90] * 10 + [0.05] * 30
        result = _run_with_fake_regime(
            close, p_bull_seq, min_train_bars=10, regime_smooth=5,
            min_hold_bars=15, entry_prob=0.65, exit_prob=0.40,
        )
        detail = result["detail"]
        buys = detail[detail["trade_event"] == "BUY"]
        sells = detail[detail["trade_event"] == "SELL"]
        assert len(buys) == 1 and len(sells) == 1
        entry_t = detail.index.get_loc(buys.index[0])
        exit_t = detail.index.get_loc(sells.index[0])
        # Smoothed P(Bull) drops below exit_prob well before min_hold_bars elapses,
        # but the exit must not fire until the hold period is satisfied.
        assert exit_t - entry_t == 15
        assert sells.iloc[0]["sell_reason"].startswith("regime_exit")

    def test_quant_backtest_stop_loss_ignores_min_hold(self):
        # Flat price through the entry bar (t=min_train_bars=10), then an
        # immediate sharp drop past the stop level on the very next bar.
        close = np.array([100.0] * 11 + [90.0] * 29)
        result = _run_with_fake_regime(
            close, [0.90] * 30, min_train_bars=10, regime_smooth=5,
            min_hold_bars=50, stop_loss_pct=0.05,
        )
        detail = result["detail"]
        sells = detail[detail["trade_event"] == "SELL"]
        assert len(sells) >= 1
        first_sell = sells.iloc[0]
        assert first_sell["sell_reason"].startswith("stop_loss")
        buys = detail[detail["trade_event"] == "BUY"]
        # Stop-loss fires the very next bar after entry, despite min_hold_bars=50
        assert detail.index.get_loc(sells.index[0]) - detail.index.get_loc(buys.index[0]) == 1

    def test_quant_backtest_take_profit_ignores_min_hold(self):
        close = np.array([100.0] * 15 + [120.0] * 25)
        result = _run_with_fake_regime(
            close, [0.90] * 30, min_train_bars=10, regime_smooth=5,
            min_hold_bars=50, take_profit_pct=0.15,
        )
        detail = result["detail"]
        sells = detail[detail["trade_event"] == "SELL"]
        assert len(sells) >= 1
        assert sells.iloc[0]["sell_reason"].startswith("take_profit")

    def test_quant_backtest_volume_gate_blocks_entry(self):
        from Strategy_Auto_Trader.quant_hmm import quant_engine as qe
        close = np.full(30, 100.0)
        df = _hourly_df(close, volume=1.0)  # constant low volume -> ratio == 1.0
        call_idx = {"i": 0}
        p_bull_seq = [0.90] * 20

        def fake_fit(returns, n_seeds=3, n_iter=50):
            return object(), np.array([0, 1, 2])

        def fake_step(model, order, returns, t, log_alpha):
            i = call_idx["i"]
            call_idx["i"] += 1
            return p_bull_seq[min(i, len(p_bull_seq) - 1)], 0.1, np.zeros(3)

        with mock.patch.object(qe, "fit_hmm_expanding", fake_fit), \
             mock.patch.object(qe, "_forward_step_incremental", fake_step):
            result = qe.quant_backtest(df, min_train_bars=10, regime_smooth=5,
                                        volume_min_ratio=2.0)  # requires 2x avg volume
        # Volume never reaches 2x its own rolling average (it's constant) -> no entries
        assert result["n_buys"] == 0

    # -- quant_backtest: sentiment / VIX threshold adjustments ---------------

    def test_quant_backtest_bullish_sentiment_lowers_entry_threshold(self):
        close = np.full(30, 100.0)
        p_bull_seq = [0.60] * 20  # below baseline entry_prob=0.65

        baseline = _run_with_fake_regime(
            close, p_bull_seq, min_train_bars=10, regime_smooth=5,
            entry_prob=0.65, sentiment_score=0.0,
        )
        bullish = _run_with_fake_regime(
            close, p_bull_seq, min_train_bars=10, regime_smooth=5,
            entry_prob=0.65, sentiment_score=1.0,  # entry_adj = -0.05 -> effective 0.60
        )
        assert baseline["n_buys"] == 0
        assert bullish["n_buys"] >= 1

    def test_quant_backtest_vix_high_vol_widens_stop(self):
        # 6% drop after entry: triggers a 5% stop but not a 7% (vix-widened) stop
        close = np.array([100.0] * 15 + [94.0] * 25)

        baseline = _run_with_fake_regime(
            close, [0.90] * 30, min_train_bars=10, regime_smooth=5,
            min_hold_bars=50, stop_loss_pct=0.05, vix_signal=0,
        )
        high_vix = _run_with_fake_regime(
            close, [0.90] * 30, min_train_bars=10, regime_smooth=5,
            min_hold_bars=50, stop_loss_pct=0.05, vix_signal=-1,  # effective_stop -> 0.07
        )
        baseline_sells = baseline["detail"][baseline["detail"]["trade_event"] == "SELL"]
        high_vix_sells = high_vix["detail"][high_vix["detail"]["trade_event"] == "SELL"]
        assert len(baseline_sells) >= 1
        assert baseline_sells.iloc[0]["sell_reason"].startswith("stop_loss")
        # With the wider stop, the 6% drop shouldn't trigger a stop-loss exit
        assert not any(r.startswith("stop_loss") for r in high_vix_sells["sell_reason"])

    # -- quant_backtest: real HMM end-to-end smoke test -----------------------

    def test_quant_backtest_real_hmm_end_to_end(self):
        pytest.importorskip("hmmlearn")
        from Strategy_Auto_Trader.quant_hmm.quant_engine import quant_backtest
        rng = np.random.default_rng(42)
        rets = rng.normal(0.0005, 0.004, 300)
        close = 100.0 * np.cumprod(1 + rets)
        df = _hourly_df(close)
        result = quant_backtest(df, min_train_bars=100, hmm_refit_bars=100, regime_smooth=10)
        assert not result["detail"].empty
        assert np.isfinite(result["final_portfolio"])
        assert set(result["detail"]["trade_event"].unique()).issubset({"", "BUY", "SELL"})

    # -- _compute_effective_thresholds ----------------------------------------

    def test_compute_effective_thresholds_neutral_sentiment_no_vix(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _compute_effective_thresholds
        entry, exit_, stop = _compute_effective_thresholds(0.65, 0.40, 0.05, 0.0, 0)
        assert entry == 0.65
        assert exit_ == 0.40
        assert stop == 0.05

    def test_compute_effective_thresholds_bullish_lowers_entry(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _compute_effective_thresholds
        entry, _, _ = _compute_effective_thresholds(0.65, 0.40, 0.05, 1.0, 0)
        assert entry < 0.65

    def test_compute_effective_thresholds_bearish_raises_entry(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _compute_effective_thresholds
        entry, _, _ = _compute_effective_thresholds(0.65, 0.40, 0.05, -1.0, 0)
        assert entry > 0.65

    def test_compute_effective_thresholds_entry_adjustment_capped(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _compute_effective_thresholds
        entry, _, _ = _compute_effective_thresholds(0.65, 0.40, 0.05, 100.0, 0)
        assert entry == 0.65 - 0.10  # capped at -0.10

    def test_compute_effective_thresholds_high_vix_widens_stop_and_tightens_exit(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _compute_effective_thresholds
        _, exit_, stop = _compute_effective_thresholds(0.65, 0.40, 0.05, 0.0, -1)
        assert exit_ == 0.45
        assert stop == 0.07

    # -- fit_hmm_expanding error handling ---------------------------------------

    def test_fit_hmm_expanding_raises_on_broken_hmmlearn(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import fit_hmm_expanding
        with mock.patch.dict(
                "sys.modules", {"hmmlearn": None, "hmmlearn.hmm": None}):
            with pytest.raises(RuntimeError, match="missing or broken"):
                fit_hmm_expanding(np.random.default_rng(0).normal(size=200))

    def test_fit_hmm_expanding_warns_when_all_seeds_fail(self, caplog):
        pytest.importorskip("hmmlearn")
        import logging
        from Strategy_Auto_Trader.quant_hmm import quant_engine as qe

        class ExplodingHMM:
            def __init__(self, **kwargs):
                pass

            def fit(self, X):
                raise ValueError("degenerate data")

        with mock.patch("hmmlearn.hmm.GaussianHMM", ExplodingHMM), \
             caplog.at_level(logging.WARNING,
                             logger="Strategy_Auto_Trader.quant_hmm.quant_engine"):
            result = qe.fit_hmm_expanding(np.zeros(200), n_seeds=2)
        assert result is None
        assert any("all 2 seeds" in r.message for r in caplog.records)

    # -- _compute_volume_ratio -------------------------------------------------

    def test_compute_volume_ratio_none_returns_ones(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _compute_volume_ratio
        ratio = _compute_volume_ratio(None, 10)
        assert (ratio == 1.0).all()
        assert len(ratio) == 10

    def test_compute_volume_ratio_above_average(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _compute_volume_ratio
        # rolling(100) window at the last bar covers 99 prior bars of 1000 + itself (2000)
        volume = np.concatenate([np.full(100, 1000.0), np.array([2000.0])])
        ratio = _compute_volume_ratio(volume, len(volume), lookback=100)
        expected = 2000.0 / ((99 * 1000.0 + 2000.0) / 100)
        assert abs(ratio[-1] - expected) < 1e-9
        assert ratio[-1] > 1.5  # clearly above-average volume

    # -- _sharpe / _max_dd -------------------------------------------------------

    def test_sharpe_zero_std_returns_nan(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _sharpe
        assert math.isnan(_sharpe(np.array([0.01, 0.01, 0.01])))

    def test_max_dd_empty_returns_nan(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _max_dd
        assert math.isnan(_max_dd(np.array([])))

    def test_max_dd_known_value(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _max_dd
        eq = np.array([1.0, 1.2, 0.9])
        assert abs(_max_dd(eq) - (-0.25)) < 1e-10

    # -- _sortino ------------------------------------------------------------

    def test_sortino_empty_returns_nan(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _sortino
        assert math.isnan(_sortino(np.array([])))

    def test_sortino_all_positive_returns_nan(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _sortino
        assert math.isnan(_sortino(np.full(100, 0.01)))

    def test_sortino_below_min_downside_bars_returns_nan(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import (
            _sortino, _MIN_DOWNSIDE_BARS,
        )
        r = np.full(200, 0.01)
        r[:_MIN_DOWNSIDE_BARS - 1] = -0.01
        assert math.isnan(_sortino(r))

    def test_sortino_at_min_downside_bars_is_finite(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import (
            _sortino, _MIN_DOWNSIDE_BARS,
        )
        r = np.full(200, 0.01)
        r[:_MIN_DOWNSIDE_BARS] = -0.01
        assert np.isfinite(_sortino(r))

    def test_sortino_known_value_full_sample_convention(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import (
            _sortino, _HOURS_PER_YEAR, _MIN_DOWNSIDE_BARS,
        )
        r = np.concatenate([np.full(_MIN_DOWNSIDE_BARS, -0.02), np.full(80, 0.01)])
        downside_dev = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
        expected = np.mean(r) / downside_dev * np.sqrt(_HOURS_PER_YEAR)
        assert abs(_sortino(r) - expected) < 1e-12

    def test_sortino_exceeds_sharpe_for_positive_skew(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _sharpe, _sortino
        rng = np.random.default_rng(42)
        r = np.where(rng.random(2000) < 0.3, -0.005, 0.02)
        assert _sortino(r) > _sharpe(r) > 0

    # -- _calmar ------------------------------------------------------------

    def test_calmar_empty_returns_nan(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _calmar
        assert math.isnan(_calmar(np.array([])))

    def test_calmar_no_drawdown_returns_nan(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _calmar
        assert math.isnan(_calmar(np.array([1.0, 1.1, 1.2])))

    def test_calmar_nonpositive_final_equity_returns_nan(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _calmar
        assert math.isnan(_calmar(np.array([1.0, 0.5, 0.0])))

    def test_calmar_known_value(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import (
            _calmar, _HOURS_PER_YEAR,
        )
        eq = np.array([1.0, 1.2, 0.9, 1.1])
        years = len(eq) / _HOURS_PER_YEAR
        expected = (1.1 ** (1 / years) - 1) / 0.25
        assert math.isclose(_calmar(eq), expected, rel_tol=1e-12)

    def test_calmar_negative_for_losing_strategy(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _calmar
        eq = np.linspace(1.0, 0.8, 100)
        assert _calmar(eq) < 0

    # -- _simulate_portfolio_value ---------------------------------------------

    def test_simulate_portfolio_value_deducts_cost_on_trade_events(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _simulate_portfolio_value
        detail = pd.DataFrame({
            "trade_event": ["BUY", "", "SELL"],
            "strategy_return": [0.0, 0.10, 0.0],
        })
        values = _simulate_portfolio_value(detail, initial_cash=1000.0, trade_cost=10.0)
        assert values[0] == 990.0
        assert values[1] == 1089.0
        assert values[2] == 1079.0

    # -- _build_quant_backtest_stats ---------------------------------------------

    def test_build_quant_backtest_stats_assembles_dict(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _build_quant_backtest_stats
        detail = pd.DataFrame({"trade_event": ["BUY", "", "SELL"]})
        strat_ret = np.array([0.0, 0.01, -0.005])
        bh_ret = np.array([0.0, 0.005, 0.005])
        strat_equity = (1 + strat_ret).cumprod()
        bh_equity = (1 + bh_ret).cumprod()
        result = _build_quant_backtest_stats(
            detail, strat_ret, bh_ret, strat_equity, bh_equity,
            initial_cash=1000.0, portfolio_values=[990.0, 1089.0, 1079.0],
            trade_results=[0.05], current_kelly=0.12,
        )
        assert result["n_buys"] == 1
        assert result["n_sells"] == 1
        assert result["final_portfolio"] == 1079.0
        assert result["total_pl"] == 79.0
        assert result["final_kelly"] == 0.12
        for key in ("sortino_strategy", "sortino_bh", "calmar_strategy", "calmar_bh"):
            assert key in result


class TestInformationRatio:

    def _ir(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _information_ratio
        return _information_ratio

    def test_empty_arrays_nan(self):
        import math
        assert math.isnan(self._ir()(np.array([]), np.array([])))

    def test_mismatched_lengths_nan(self):
        import math
        assert math.isnan(self._ir()(np.array([0.01, 0.02]), np.array([0.01])))

    def test_identical_returns_nan(self):
        import math
        r = np.array([0.01, -0.02, 0.03, 0.0])
        assert math.isnan(self._ir()(r, r))

    def test_consistent_outperformance_positive(self):
        rng = np.random.default_rng(42)
        bh = rng.normal(0.0, 0.01, 500)
        strat = bh + rng.normal(0.001, 0.002, 500)
        assert self._ir()(strat, bh) > 0

    def test_consistent_underperformance_negative(self):
        rng = np.random.default_rng(42)
        bh = rng.normal(0.0, 0.01, 500)
        strat = bh + rng.normal(-0.001, 0.002, 500)
        assert self._ir()(strat, bh) < 0

    def test_annualisation_scale(self):
        # active = [+0.01, -0.01, ...]: mean 0 exactly -> IR 0
        strat = np.array([0.02, 0.00] * 50)
        bh = np.array([0.01, 0.01] * 50)
        assert abs(self._ir()(strat, bh)) < 1e-12


class TestCaptureRatio:

    def _cr(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _capture_ratio
        return _capture_ratio

    def test_empty_arrays_nan(self):
        import math
        assert math.isnan(self._cr()(np.array([]), np.array([]), up=True))

    def test_full_replication_capture_one(self):
        r = np.array([0.01, -0.02, 0.03, -0.01])
        assert abs(self._cr()(r, r, up=True) - 1.0) < 1e-12
        assert abs(self._cr()(r, r, up=False) - 1.0) < 1e-12

    def test_flat_strategy_zero_capture(self):
        strat = np.zeros(4)
        bh = np.array([0.01, -0.02, 0.03, -0.01])
        assert self._cr()(strat, bh, up=True) == 0.0
        assert self._cr()(strat, bh, up=False) == 0.0

    def test_no_down_bars_nan(self):
        import math
        strat = np.array([0.01, 0.02])
        bh = np.array([0.01, 0.02])
        assert math.isnan(self._cr()(strat, bh, up=False))

    def test_no_up_bars_nan(self):
        import math
        strat = np.array([-0.01, -0.02])
        bh = np.array([-0.01, -0.02])
        assert math.isnan(self._cr()(strat, bh, up=True))

    def test_half_position_half_capture(self):
        bh = np.array([0.02, -0.02])
        strat = bh * 0.5
        up = self._cr()(strat, bh, up=True)
        down = self._cr()(strat, bh, up=False)
        assert abs(up - 0.5) < 0.01
        assert abs(down - 0.5) < 0.01

    def test_stats_and_empty_result_carry_new_keys(self):
        from Strategy_Auto_Trader.quant_hmm.quant_engine import _empty_result
        empty = _empty_result(1000.0)
        for key in ("information_ratio", "up_capture", "down_capture"):
            assert key in empty
