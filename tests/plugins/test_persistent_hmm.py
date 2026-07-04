from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hmmlearn")


def _hourly_df(close, start="2024-01-01"):
    close = np.asarray(close, dtype=float)
    idx = pd.date_range(start, periods=len(close), freq="h")
    return pd.DataFrame({
        "Open": close, "High": close * 1.0005, "Low": close * 0.9995,
        "Close": close, "Volume": np.full(len(close), 1_000_000.0),
    }, index=idx)


def _close_series(n, seed=42):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.004, n)
    return 100.0 * np.cumprod(1 + rets)


def _make_model(cache_path, df, **overrides):
    from Strategy_Auto_Trader.plugins.persistent_hmm import PersistentHMMRegimeModel
    kwargs = dict(min_train_bars=100, refit_bars=100, regime_smooth=10)
    kwargs.update(overrides)
    return PersistentHMMRegimeModel(
        cache_path, dates=df.index, closes=df["Close"].values, **kwargs)


def _run(df, model):
    from Strategy_Auto_Trader.quant_hmm.consolidated_engine import consolidated_backtest
    return consolidated_backtest(
        df, regime_model=model, min_train_bars=100, hmm_refit_bars=100,
        regime_smooth=10,
    )


class TestPersistentHMM:

    def test_cold_run_builds_cache(self, tmp_path):
        df = _hourly_df(_close_series(300))
        cache = tmp_path / "TEST.pkl"
        model = _make_model(cache, df)
        result = _run(df, model)
        assert model.computed_steps == 200   # bars 100..299
        assert model.cache_hits == 0
        assert result["n_bars"] == 200
        model.save()
        assert cache.exists()

    def test_warm_rerun_serves_from_cache_with_identical_results(self, tmp_path):
        df = _hourly_df(_close_series(300))
        cache = tmp_path / "TEST.pkl"
        model1 = _make_model(cache, df)
        result1 = _run(df, model1)
        model1.save()

        model2 = _make_model(cache, df)
        result2 = _run(df, model2)
        assert model2.computed_steps == 0
        assert model2.cache_hits == 200
        pd.testing.assert_frame_equal(result1["detail"], result2["detail"])

    def test_sliding_window_only_steps_new_bars(self, tmp_path):
        full = _close_series(310)
        df1 = _hourly_df(full)[:300]
        df2 = _hourly_df(full)[10:310]
        cache = tmp_path / "TEST.pkl"

        model1 = _make_model(cache, df1)
        result1 = _run(df1, model1)
        model1.save()

        model2 = _make_model(cache, df2)
        result2 = _run(df2, model2)
        # df2's bars 100..289 map to df1's stepped bars 110..299 (cached);
        # bars 290..299 are genuinely new.
        assert model2.cache_hits == 190
        assert model2.computed_steps == 10

        # Overlapping timestamps must agree between the two runs.
        d1, d2 = result1["detail"], result2["detail"]
        common = d1.index.intersection(d2.index)
        assert len(common) == 190
        np.testing.assert_allclose(
            d1.loc[common, "p_bull"].values, d2.loc[common, "p_bull"].values)

    def test_warm_save_does_not_overwrite_cache(self, tmp_path):
        df = _hourly_df(_close_series(300))
        cache = tmp_path / "TEST.pkl"
        model1 = _make_model(cache, df)
        _run(df, model1)
        model1.save()
        original_bytes = cache.read_bytes()

        model2 = _make_model(cache, df)
        _run(df, model2)
        assert model2.computed_steps == 0
        model2.save()   # nothing new computed -> must be a no-op
        assert cache.read_bytes() == original_bytes

    def test_data_revision_invalidates_cache(self, tmp_path):
        close = _close_series(300)
        df = _hourly_df(close)
        cache = tmp_path / "TEST.pkl"
        model1 = _make_model(cache, df)
        _run(df, model1)
        model1.save()

        revised = close.copy()
        revised[150] *= 1.01
        df_rev = _hourly_df(revised)
        model2 = _make_model(cache, df_rev)
        assert model2._cached_through == -1
        _run(df_rev, model2)
        assert model2.computed_steps == 200

    def test_param_change_invalidates_cache(self, tmp_path):
        df = _hourly_df(_close_series(300))
        cache = tmp_path / "TEST.pkl"
        model1 = _make_model(cache, df)
        _run(df, model1)
        model1.save()

        model2 = _make_model(cache, df, regime_smooth=24)
        assert model2._cached_through == -1

    def test_window_starting_before_cache_invalidates(self, tmp_path):
        full = _close_series(310)
        df1 = _hourly_df(full)[5:305]
        df2 = _hourly_df(full)[:300]
        cache = tmp_path / "TEST.pkl"
        model1 = _make_model(cache, df1)
        _run(df1, model1)
        model1.save()

        # df2 starts 5 bars earlier than the cached window: the persisted
        # filter state cannot serve it, so the cache must be ignored.
        model2 = _make_model(cache, df2)
        assert model2._cached_through == -1

    def test_corrupt_cache_file_recomputes(self, tmp_path):
        df = _hourly_df(_close_series(300))
        cache = tmp_path / "TEST.pkl"
        cache.write_bytes(b"not a pickle")
        model = _make_model(cache, df)
        assert model._cached_through == -1
        _run(df, model)
        assert model.computed_steps == 200

    def test_chained_runs_resave_and_resume_correctly(self, tmp_path):
        full = _close_series(320)
        cache = tmp_path / "TEST.pkl"

        model1 = _make_model(cache, _hourly_df(full)[:300])
        result1 = _run(_hourly_df(full)[:300], model1)
        model1.save()

        df2 = _hourly_df(full)[10:310]
        model2 = _make_model(cache, df2)
        result2 = _run(df2, model2)
        assert model2.cache_hits == 190
        assert model2.computed_steps == 10
        model2.save()

        # Run 3 must resume from run 2's re-saved merged cache, which mixes
        # values copied from run 1's cache with run 2's newly computed bars.
        df3 = _hourly_df(full)[20:320]
        model3 = _make_model(cache, df3)
        result3 = _run(df3, model3)
        assert model3.cache_hits == 190
        assert model3.computed_steps == 10

        d1, d2, d3 = result1["detail"], result2["detail"], result3["detail"]
        common23 = d2.index.intersection(d3.index)
        assert len(common23) == 190
        np.testing.assert_allclose(
            d2.loc[common23, "p_bull"].values, d3.loc[common23, "p_bull"].values)
        common13 = d1.index.intersection(d3.index)
        assert len(common13) == 180
        np.testing.assert_allclose(
            d1.loc[common13, "p_bull"].values, d3.loc[common13, "p_bull"].values)

    def test_refit_cadence_persists_across_runs(self, tmp_path):
        from Strategy_Auto_Trader.plugins.persistent_hmm import PersistentHMMRegimeModel

        class _RefitSpy(PersistentHMMRegimeModel):
            def __init__(self, *args, **kwargs):
                self.refit_bars_seen: list[int] = []
                super().__init__(*args, **kwargs)

            def refit(self, returns):
                # consolidated_backtest calls refit(returns[:t]), so the
                # length of the training slice is the bar index of the refit.
                self.refit_bars_seen.append(len(returns))
                super().refit(returns)

        full = _close_series(310)
        cache = tmp_path / "TEST.pkl"

        # Cold run over 250 bars: initial fit at t=100, cadence refit at
        # t=200, leaving bars_since_refit=50 to be persisted.
        df1 = _hourly_df(full)[:250]
        model1 = _RefitSpy(
            cache, dates=df1.index, closes=df1["Close"].values,
            min_train_bars=100, refit_bars=100, regime_smooth=10)
        _run(df1, model1)
        assert model1.refit_bars_seen == [100, 200]
        assert model1._bars_since_refit == 50
        model1.save()

        # Resumed run extends to 310 bars. With the persisted counter at 50,
        # the next refit must fire after 50 more stepped bars, at t=300 —
        # not at t=350 as a reset counter would schedule.
        df2 = _hourly_df(full)
        model2 = _RefitSpy(
            cache, dates=df2.index, closes=df2["Close"].values,
            min_train_bars=100, refit_bars=100, regime_smooth=10)
        _run(df2, model2)
        assert model2.cache_hits == 150
        assert model2.computed_steps == 60
        assert model2.refit_bars_seen == [300]

    def test_window_entirely_after_cache_recomputes(self, tmp_path):
        full = _close_series(700)
        cache = tmp_path / "TEST.pkl"
        df1 = _hourly_df(full)[:300]
        model1 = _make_model(cache, df1)
        _run(df1, model1)
        model1.save()

        # Downtime so long that the new window starts after the cached window
        # ends: no overlap, so the cache must be ignored and fully recomputed.
        df2 = _hourly_df(full)[350:650]
        model2 = _make_model(cache, df2)
        assert model2._cached_through == -1
        _run(df2, model2)
        assert model2.cache_hits == 0
        assert model2.computed_steps == 200

    def test_mismatched_lengths_raise(self, tmp_path):
        from Strategy_Auto_Trader.plugins.persistent_hmm import PersistentHMMRegimeModel
        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        with pytest.raises(ValueError):
            PersistentHMMRegimeModel(tmp_path / "x.pkl", dates=idx, closes=np.ones(9))
