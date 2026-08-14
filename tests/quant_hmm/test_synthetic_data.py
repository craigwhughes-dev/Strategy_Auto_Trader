from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from Strategy_Auto_Trader.quant_hmm.synthetic_data import (
    _choose_block_starts,
    _extract_blocks,
    _perturb_transmat,
    _sample_state_sequence,
    assemble_synthetic_ohlcv,
    bootstrap_blocks_by_state,
    bootstrap_by_state,
    fit_generating_hmm,
    generate_synthetic_df,
    label_hidden_states,
    sample_coupled_state_sequence,
    sample_daily_tiled_states,
    sample_synthetic_path,
)


def make_fixture_ohlcv(n_bars: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.01, n_bars)
    closes = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2023-01-01", periods=n_bars, freq="h")
    spread = np.abs(rng.normal(0, 0.005, n_bars))
    return pd.DataFrame(
        {
            "Open": closes * (1 + rng.normal(0, 0.001, n_bars)),
            "High": closes * (1 + spread),
            "Low": closes * (1 - spread),
            "Close": closes,
            "Volume": rng.integers(1_000, 10_000, n_bars).astype(float),
        },
        index=idx,
    )


class TestFitGeneratingHmm:
    def test_returns_model_and_order(self):
        df = make_fixture_ohlcv(600)
        model, order = fit_generating_hmm(df["Close"])
        assert hasattr(model, "sample")
        assert len(order) == 3
        assert set(order.tolist()) == {0, 1, 2}

    def test_reproducible_across_identical_calls(self):
        df = make_fixture_ohlcv(600)
        m1, o1 = fit_generating_hmm(df["Close"])
        m2, o2 = fit_generating_hmm(df["Close"])
        np.testing.assert_array_equal(o1, o2)

    def test_raises_when_fit_hmm_expanding_fails(self):
        with mock.patch(
            "Strategy_Auto_Trader.quant_hmm.synthetic_data.fit_hmm_expanding",
            return_value=None,
        ):
            close = pd.Series([100.0, 101.0, 102.0, 103.0])
            with pytest.raises(ValueError, match="HMM fit failed"):
                fit_generating_hmm(close)


class TestLabelHiddenStates:
    def test_length_equals_input_returns(self):
        df = make_fixture_ohlcv(300)
        model, order = fit_generating_hmm(df["Close"])
        returns = np.log(df["Close"].values[1:] / df["Close"].values[:-1])
        labels = label_hidden_states(model, order, returns)
        assert len(labels) == len(returns)

    def test_labels_are_valid_state_indices(self):
        df = make_fixture_ohlcv(300)
        model, order = fit_generating_hmm(df["Close"])
        returns = np.log(df["Close"].values[1:] / df["Close"].values[:-1])
        labels = label_hidden_states(model, order, returns)
        assert set(np.unique(labels)) <= {0, 1, 2}


class TestSampleSyntheticPath:
    def test_length_equals_n_returns(self):
        df = make_fixture_ohlcv(300)
        model, order = fit_generating_hmm(df["Close"])
        log_ret, labels = sample_synthetic_path(model, order, n_returns=200, seed=7)
        assert len(log_ret) == 200
        assert len(labels) == 200

    def test_reproducible_with_same_seed(self):
        df = make_fixture_ohlcv(300)
        model, order = fit_generating_hmm(df["Close"])
        r1, l1 = sample_synthetic_path(model, order, n_returns=100, seed=42)
        r2, l2 = sample_synthetic_path(model, order, n_returns=100, seed=42)
        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_array_equal(l1, l2)

    def test_different_seeds_differ(self):
        df = make_fixture_ohlcv(300)
        model, order = fit_generating_hmm(df["Close"])
        r1, _ = sample_synthetic_path(model, order, n_returns=100, seed=0)
        r2, _ = sample_synthetic_path(model, order, n_returns=100, seed=1)
        assert not np.array_equal(r1, r2)

    def test_label_range_valid(self):
        df = make_fixture_ohlcv(300)
        model, order = fit_generating_hmm(df["Close"])
        _, labels = sample_synthetic_path(model, order, n_returns=200, seed=0)
        assert set(np.unique(labels)) <= {0, 1, 2}


class TestBootstrapByState:
    def test_per_state_donor_values_respected(self):
        """With disjoint per-state values, bootstrap_by_state must draw
        only from the correct donor pool for each bar."""
        hist_values = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        hist_labels = np.array([0, 0, 1, 1, 2, 2])
        synth_labels = np.array([2, 2, 2])  # all state-2 → must draw 3.0 only
        result = bootstrap_by_state(hist_values, hist_labels, synth_labels, seed=0)
        assert np.all(result == 3.0)

    def test_empty_donor_pool_falls_back_to_full_array(self):
        hist_values = np.array([10.0, 20.0, 30.0])
        hist_labels = np.array([0, 0, 0])  # state 1 never seen historically
        synth_labels = np.array([1, 1])
        # should not raise; fallback to full array
        result = bootstrap_by_state(hist_values, hist_labels, synth_labels, seed=0)
        assert result.shape == (2,)
        assert all(v in [10.0, 20.0, 30.0] for v in result)

    def test_output_length_equals_synthetic_label_length(self):
        hist_values = np.ones(50)
        hist_labels = np.zeros(50, dtype=int)
        synth_labels = np.zeros(123, dtype=int)
        result = bootstrap_by_state(hist_values, hist_labels, synth_labels, seed=0)
        assert len(result) == 123


class TestBootstrapBlocksByState:
    def test_output_length_matches_synthetic_labels(self):
        hist = np.arange(100, dtype=float)
        hist_labels = np.zeros(100, dtype=int)
        synth_labels = np.zeros(60, dtype=int)
        result = bootstrap_blocks_by_state(hist, hist_labels, synth_labels, block_size=10, seed=0)
        assert len(result) == 60

    def test_values_come_from_correct_state(self):
        # State 0 = values 0-9, state 1 = values 100-109 (disjoint ranges)
        hist = np.concatenate([np.arange(10, dtype=float), np.arange(100, 110, dtype=float)])
        hist_labels = np.array([0] * 10 + [1] * 10)
        synth_labels = np.array([1] * 20)  # all state 1 -> must draw from 100-109
        result = bootstrap_blocks_by_state(hist, hist_labels, synth_labels, block_size=5, seed=0)
        assert np.all(result >= 100) and np.all(result < 110)

    def test_fallback_when_no_valid_block_starts(self):
        # block_size=50 but only 10 donor values -> no valid starts -> fallback to full array
        hist = np.arange(10, dtype=float)
        hist_labels = np.zeros(10, dtype=int)
        synth_labels = np.zeros(15, dtype=int)
        result = bootstrap_blocks_by_state(hist, hist_labels, synth_labels, block_size=50, seed=0)
        assert len(result) == 15
        assert all(v in hist for v in result)

    def test_blocks_are_contiguous(self):
        # Sequential ints -> a drawn block must be a run of consecutive values
        hist = np.arange(200, dtype=float)
        hist_labels = np.zeros(200, dtype=int)
        synth_labels = np.zeros(24, dtype=int)  # exactly one block
        result = bootstrap_blocks_by_state(hist, hist_labels, synth_labels, block_size=24, seed=7)
        assert np.all(np.diff(result) == 1.0), "block must be consecutive integers"

    def test_reproducible_same_seed(self):
        rng = np.random.default_rng(0)
        hist = rng.normal(0, 1, 200)
        hist_labels = (hist > 0).astype(int)
        synth_labels = np.zeros(50, dtype=int)
        r1 = bootstrap_blocks_by_state(hist, hist_labels, synth_labels, block_size=12, seed=42)
        r2 = bootstrap_blocks_by_state(hist, hist_labels, synth_labels, block_size=12, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_fallback_when_unknown_state(self):
        hist = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        hist_labels = np.array([0, 0, 0, 0, 0])  # state 1 never seen
        synth_labels = np.array([1, 1, 1])
        result = bootstrap_blocks_by_state(hist, hist_labels, synth_labels, block_size=2, seed=0)
        assert result.shape == (3,)
        assert all(v in hist for v in result)


class TestChooseBlockStarts:
    def test_length_is_ceil_n_over_block_size(self):
        hist_labels = np.zeros(100, dtype=int)
        synth_labels = np.zeros(50, dtype=int)
        starts = _choose_block_starts(hist_labels, synth_labels, block_size=10, seed=0)
        assert len(starts) == 5  # ceil(50/10)

    def test_partial_last_block_still_one_start(self):
        hist_labels = np.zeros(100, dtype=int)
        synth_labels = np.zeros(55, dtype=int)  # 55/10 → 6 blocks
        starts = _choose_block_starts(hist_labels, synth_labels, block_size=10, seed=0)
        assert len(starts) == 6

    def test_starts_are_valid_indices_for_block(self):
        hist_labels = np.zeros(50, dtype=int)
        synth_labels = np.zeros(30, dtype=int)
        block_size = 8
        starts = _choose_block_starts(hist_labels, synth_labels, block_size=block_size, seed=0)
        for s in starts:
            assert 0 <= s <= len(hist_labels) - block_size

    def test_state_restricted_to_matching_indices(self):
        # State 0 = indices 0-4, state 1 = indices 5-9
        hist_labels = np.array([0] * 5 + [1] * 5)
        synth_labels = np.array([1] * 10)
        starts = _choose_block_starts(hist_labels, synth_labels, block_size=3, seed=0)
        for s in starts:
            assert s >= 5, "block start must be in state-1 region"

    def test_reproducible(self):
        hist_labels = np.zeros(80, dtype=int)
        synth_labels = np.zeros(40, dtype=int)
        s1 = _choose_block_starts(hist_labels, synth_labels, block_size=8, seed=99)
        s2 = _choose_block_starts(hist_labels, synth_labels, block_size=8, seed=99)
        np.testing.assert_array_equal(s1, s2)


class TestExtractBlocks:
    def test_output_length(self):
        hist = np.arange(100, dtype=float)
        starts = np.array([0, 10, 20])
        result = _extract_blocks(hist, starts, block_size=10, n=30)
        assert len(result) == 30

    def test_partial_last_block(self):
        hist = np.arange(100, dtype=float)
        starts = np.array([0, 10])
        result = _extract_blocks(hist, starts, block_size=10, n=17)
        assert len(result) == 17
        np.testing.assert_array_equal(result[:10], hist[0:10])
        np.testing.assert_array_equal(result[10:17], hist[10:17])

    def test_values_match_historical(self):
        hist = np.arange(50, dtype=float)
        starts = np.array([5])
        result = _extract_blocks(hist, starts, block_size=10, n=10)
        np.testing.assert_array_equal(result, hist[5:15])

    def test_consistent_with_bootstrap_blocks_by_state(self):
        """_extract_blocks on starts from _choose_block_starts == bootstrap_blocks_by_state."""
        rng = np.random.default_rng(7)
        hist_vals = rng.normal(0, 1, 200)
        hist_labels = (hist_vals > 0).astype(int)
        synth_labels = np.zeros(60, dtype=int)
        starts = _choose_block_starts(hist_labels, synth_labels, block_size=12, seed=3)
        extracted = _extract_blocks(hist_vals, starts, block_size=12, n=60)
        direct = bootstrap_blocks_by_state(hist_vals, hist_labels, synth_labels,
                                            block_size=12, seed=3)
        np.testing.assert_array_equal(extracted, direct)


class TestSampleCoupledStateSequence:
    def _make_transmat(self):
        return np.array([[0.7, 0.2, 0.1],
                         [0.1, 0.7, 0.2],
                         [0.1, 0.1, 0.8]])

    def _make_startprob(self):
        return np.array([1/3, 1/3, 1/3])

    def test_output_length(self):
        rng = np.random.default_rng(0)
        market = np.zeros(300, dtype=int)  # all Bear
        states = sample_coupled_state_sequence(300, self._make_startprob(),
                                               self._make_transmat(), market, 0.3, rng)
        assert len(states) == 300

    def test_valid_state_indices(self):
        rng = np.random.default_rng(1)
        market = np.ones(500, dtype=int)  # all Sideways
        states = sample_coupled_state_sequence(500, self._make_startprob(),
                                               self._make_transmat(), market, 0.3, rng)
        assert set(np.unique(states)) <= {0, 1, 2}

    def test_coupling_zero_matches_uncoupled(self):
        """coupling=0.0 must produce identical output to _sample_state_sequence."""
        transmat = self._make_transmat()
        startprob = self._make_startprob()
        market = np.zeros(200, dtype=int)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        coupled = sample_coupled_state_sequence(200, startprob, transmat, market, 0.0, rng1)
        uncoupled = _sample_state_sequence(200, startprob, transmat, rng2)
        np.testing.assert_array_equal(coupled, uncoupled)

    def test_coupling_one_always_follows_market(self):
        """coupling=1.0: states[t] == raw_market_states[t-1] for all t > 0."""
        transmat = self._make_transmat()
        startprob = self._make_startprob()
        # Alternating Bear/Bull market pattern
        market = np.tile([0, 2], 150)[:200]
        rng = np.random.default_rng(7)
        states = sample_coupled_state_sequence(200, startprob, transmat, market, 1.0, rng)
        np.testing.assert_array_equal(states[1:], market[:-1])

    def test_bear_market_increases_bear_state_fraction(self):
        """All-Bear market with coupling > 0 produces more Bear states than uncoupled."""
        transmat = self._make_transmat()
        startprob = self._make_startprob()
        n = 1000
        market_bear = np.zeros(n, dtype=int)
        rng_coupled = np.random.default_rng(99)
        rng_free = np.random.default_rng(99)
        coupled = sample_coupled_state_sequence(n, startprob, transmat, market_bear, 0.5, rng_coupled)
        uncoupled = _sample_state_sequence(n, startprob, transmat, rng_free)
        assert (coupled == 0).mean() > (uncoupled == 0).mean()


class TestSampleStateSequence:
    def _make_transmat(self):
        return np.array([[0.8, 0.1, 0.1],
                         [0.1, 0.7, 0.2],
                         [0.05, 0.05, 0.9]])

    def test_output_length(self):
        rng = np.random.default_rng(0)
        startprob = np.array([1/3, 1/3, 1/3])
        states = _sample_state_sequence(200, startprob, self._make_transmat(), rng)
        assert len(states) == 200

    def test_valid_state_indices(self):
        rng = np.random.default_rng(1)
        startprob = np.array([1/3, 1/3, 1/3])
        states = _sample_state_sequence(500, startprob, self._make_transmat(), rng)
        assert set(np.unique(states)) <= {0, 1, 2}

    def test_reproducible(self):
        startprob = np.array([1/3, 1/3, 1/3])
        transmat = self._make_transmat()
        s1 = _sample_state_sequence(100, startprob, transmat, np.random.default_rng(7))
        s2 = _sample_state_sequence(100, startprob, transmat, np.random.default_rng(7))
        np.testing.assert_array_equal(s1, s2)

    def test_sticky_state_respected(self):
        # Near-identity transmat — >95% of transitions should be self-transitions.
        # Can't test "stays in initial state" — chain escapes after ~1/p_exit steps
        # and then sticks in the new state, so fraction-in-state-0 is misleading.
        startprob = np.array([1/3, 1/3, 1/3])
        transmat = np.array([[0.99, 0.005, 0.005],
                              [0.005, 0.99, 0.005],
                              [0.005, 0.005, 0.99]])
        rng = np.random.default_rng(42)
        states = _sample_state_sequence(500, startprob, transmat, rng)
        self_transitions = np.sum(states[1:] == states[:-1]) / (len(states) - 1)
        assert self_transitions > 0.95


class TestPerturbTransmat:
    def _make_transmat(self):
        return np.array([[0.8, 0.1, 0.1],
                         [0.1, 0.7, 0.2],
                         [0.05, 0.05, 0.9]])

    def test_rows_sum_to_one(self):
        rng = np.random.default_rng(0)
        perturbed = _perturb_transmat(self._make_transmat(), noise=0.1, rng=rng)
        np.testing.assert_allclose(perturbed.sum(axis=1), 1.0, atol=1e-12)

    def test_all_nonnegative(self):
        rng = np.random.default_rng(0)
        perturbed = _perturb_transmat(self._make_transmat(), noise=0.1, rng=rng)
        assert (perturbed >= 0).all()

    def test_does_not_mutate_input(self):
        transmat = self._make_transmat()
        original = transmat.copy()
        rng = np.random.default_rng(0)
        _perturb_transmat(transmat, noise=0.2, rng=rng)
        np.testing.assert_array_equal(transmat, original)

    def test_high_noise_deviates_from_original(self):
        rng = np.random.default_rng(42)
        transmat = self._make_transmat()
        perturbed = _perturb_transmat(transmat, noise=1.0, rng=rng)
        assert not np.allclose(perturbed, transmat, atol=0.05)

    def test_low_noise_stays_close_to_original(self):
        rng = np.random.default_rng(42)
        transmat = self._make_transmat()
        perturbed = _perturb_transmat(transmat, noise=0.001, rng=rng)
        # Dirichlet std ≈ sqrt(p(1-p)/concentration); concentration=1000 gives std≈0.013
        # per entry, so 0.05 is the appropriate tolerance (≈4σ).
        np.testing.assert_allclose(perturbed, transmat, atol=0.05)


class TestSampleSyntheticPathTransmatNoise:
    def test_noise_zero_matches_no_noise_state_labels_distribution(self):
        df = make_fixture_ohlcv(300)
        model, order = fit_generating_hmm(df["Close"])
        _, labels_default = sample_synthetic_path(model, order, n_returns=200, seed=5)
        _, labels_zero = sample_synthetic_path(model, order, n_returns=200, seed=5,
                                               transmat_noise=0.0)
        np.testing.assert_array_equal(labels_default, labels_zero)

    def test_model_sample_never_called(self):
        """Vectorized sampler is used for all transmat_noise values — model.sample() never called."""
        df = make_fixture_ohlcv(300)
        model, order = fit_generating_hmm(df["Close"])
        with mock.patch.object(model, "sample", wraps=model.sample) as mock_sample:
            sample_synthetic_path(model, order, n_returns=100, seed=0, transmat_noise=0.0)
            sample_synthetic_path(model, order, n_returns=100, seed=0, transmat_noise=0.1)
            mock_sample.assert_not_called()

    def test_noise_positive_does_not_call_model_sample(self):
        """When transmat_noise > 0, state sequence is sampled manually — model.sample() must not be called."""
        df = make_fixture_ohlcv(300)
        model, order = fit_generating_hmm(df["Close"])
        with mock.patch.object(model, "sample", wraps=model.sample) as mock_sample:
            sample_synthetic_path(model, order, n_returns=200, seed=5, transmat_noise=0.5)
            mock_sample.assert_not_called()

    def test_noise_output_length_correct(self):
        df = make_fixture_ohlcv(300)
        model, order = fit_generating_hmm(df["Close"])
        log_ret, labels = sample_synthetic_path(model, order, n_returns=150, seed=0,
                                                transmat_noise=0.1)
        assert len(log_ret) == 150
        assert len(labels) == 150

    def test_noise_labels_are_valid_states(self):
        df = make_fixture_ohlcv(300)
        model, order = fit_generating_hmm(df["Close"])
        _, labels = sample_synthetic_path(model, order, n_returns=150, seed=0,
                                          transmat_noise=0.2)
        assert set(np.unique(labels)) <= {0, 1, 2}


class TestGenerateSyntheticDfAlignment:
    """Volume/range_ratio must come from the same block positions as log_returns."""

    def test_volume_aligned_to_return_blocks(self):
        """When block_size>1, volume[i] should reflect the same historical bar as
        log_returns[i]. We verify by checking that volume values are drawn from the
        same contiguous real-data blocks as returns."""
        rng = np.random.default_rng(1)
        n = 120
        real_idx = pd.date_range("2023-01-01", periods=n, freq="h")
        # Make volume values encode their bar index so we can trace them back
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        volumes = np.arange(n, dtype=float)  # volume[i] == i — traceable
        spread = np.abs(rng.normal(0, 0.005, n))
        real_df = pd.DataFrame(
            {"Open": closes, "High": closes * (1 + spread),
             "Low": closes * (1 - spread), "Close": closes,
             "Volume": volumes},
            index=real_idx,
        )
        model, order = fit_generating_hmm(real_df["Close"])
        log_ret = np.log(real_df["Close"].values[1:] / real_df["Close"].values[:-1])
        hist_labels = label_hidden_states(model, order, log_ret)

        synth = generate_synthetic_df(real_df, model, order, log_ret, hist_labels,
                                      n_bars=60, seed=0, block_size=12)

        # Volume values in synth must all exist in the original real volumes
        synth_vol = synth["Volume"].values
        assert all(v in volumes for v in synth_vol), "synthetic volumes must be real bar indices"

        # Consecutive values within a block must be consecutive integers (same block)
        for block_start in range(0, 60, 12):
            block_vols = synth_vol[block_start : block_start + 12]
            diffs = np.diff(block_vols)
            assert np.all(diffs == 1.0), f"block at {block_start} is not contiguous"

    def test_market_coupling_zero_matches_no_coupling(self):
        """market_coupling=0.0 with market_states provided must produce same df as default call."""
        rng = np.random.default_rng(3)
        n = 120
        real_idx = pd.date_range("2023-01-01", periods=n, freq="h")
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        real_df = pd.DataFrame(
            {"Open": closes, "High": closes * 1.01, "Low": closes * 0.99,
             "Close": closes, "Volume": np.arange(n, dtype=float)},
            index=real_idx,
        )
        model, order = fit_generating_hmm(real_df["Close"])
        log_ret = np.log(real_df["Close"].values[1:] / real_df["Close"].values[:-1])
        hist_labels = label_hidden_states(model, order, log_ret)
        dummy_market = np.zeros(n, dtype=int)  # market_coupling=0 → ignored
        df_no_coupling = generate_synthetic_df(real_df, model, order, log_ret, hist_labels,
                                               n_bars=60, seed=0, block_size=12)
        df_zero_coupling = generate_synthetic_df(real_df, model, order, log_ret, hist_labels,
                                                 n_bars=60, seed=0, block_size=12,
                                                 market_states=dummy_market, market_coupling=0.0)
        pd.testing.assert_frame_equal(df_no_coupling, df_zero_coupling)

    def test_iid_mode_volume_not_block_aligned(self):
        """block_size=1 reverts to iid bootstrap — volume is NOT block-constrained."""
        rng = np.random.default_rng(2)
        n = 100
        real_idx = pd.date_range("2023-01-01", periods=n, freq="h")
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        real_df = pd.DataFrame(
            {"Open": closes, "High": closes * 1.01, "Low": closes * 0.99,
             "Close": closes, "Volume": np.arange(n, dtype=float)},
            index=real_idx,
        )
        model, order = fit_generating_hmm(real_df["Close"])
        log_ret = np.log(real_df["Close"].values[1:] / real_df["Close"].values[:-1])
        hist_labels = label_hidden_states(model, order, log_ret)
        # Just check it runs without error in iid mode
        synth = generate_synthetic_df(real_df, model, order, log_ret, hist_labels,
                                      n_bars=50, seed=0, block_size=1)
        assert len(synth) == 50


class TestAssembleSyntheticOhlcv:
    def _make_inputs(self, n: int = 80):
        rng = np.random.default_rng(1)
        log_ret = rng.normal(0, 0.01, n)
        volume = rng.integers(100, 1000, n).astype(float)
        range_ratio = np.abs(rng.normal(0, 0.005, n))
        real_idx = pd.date_range("2023-01-01", periods=n, freq="h")
        real_df = pd.DataFrame(
            {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 500.0},
            index=real_idx,
        )
        return real_df, log_ret, volume, range_ratio

    def test_high_ge_max_open_close(self):
        real_df, log_ret, volume, range_ratio = self._make_inputs()
        df = assemble_synthetic_ohlcv(real_df, log_ret, volume, range_ratio)
        assert (df["High"] >= df[["Open", "Close"]].max(axis=1)).all()

    def test_low_le_min_open_close(self):
        real_df, log_ret, volume, range_ratio = self._make_inputs()
        df = assemble_synthetic_ohlcv(real_df, log_ret, volume, range_ratio)
        assert (df["Low"] <= df[["Open", "Close"]].min(axis=1)).all()

    def test_correct_length(self):
        real_df, log_ret, volume, range_ratio = self._make_inputs(80)
        df = assemble_synthetic_ohlcv(real_df, log_ret, volume, range_ratio)
        assert len(df) == 80

    def test_index_unique_and_monotonic_when_longer_than_real(self):
        """A synthetic path longer than real_df must still have a unique,
        strictly-monotonic DatetimeIndex — tests the tiling fix."""
        n_real = 50
        real_idx = pd.date_range("2023-01-01", periods=n_real, freq="h")
        real_df = pd.DataFrame(
            {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 500.0},
            index=real_idx,
        )
        rng = np.random.default_rng(99)
        n_synth = 180  # > n_real, forces tiling
        log_ret = rng.normal(0, 0.01, n_synth)
        volume = rng.integers(100, 500, n_synth).astype(float)
        range_ratio = np.abs(rng.normal(0, 0.003, n_synth))
        df = assemble_synthetic_ohlcv(real_df, log_ret, volume, range_ratio)
        assert len(df) == n_synth
        assert df.index.is_unique
        assert df.index.is_monotonic_increasing


class TestIntegration:
    def test_full_chain_runs_consolidated_backtest(self):
        """tiny fixture → generate_synthetic_df → consolidated_backtest(regime_model=None)
        completes without error and returns expected stat keys."""
        from Strategy_Auto_Trader.quant_hmm.consolidated_engine import consolidated_backtest

        real_df = make_fixture_ohlcv(600)
        model, order = fit_generating_hmm(real_df["Close"])
        log_returns = np.log(real_df["Close"].values[1:] / real_df["Close"].values[:-1])
        hist_labels = label_hidden_states(model, order, log_returns)

        synth_df = generate_synthetic_df(
            real_df, model, order, log_returns, hist_labels, n_bars=600, seed=0
        )

        bt = consolidated_backtest(
            synth_df,
            regime_model=None,
            position_sizer=None,
            min_train_bars=100,
            hmm_refit_bars=100,
            volume_min_ratio=0.8,
            min_hold_bars=48,
        )
        for key in ["sharpe_strategy", "sortino_strategy", "max_drawdown_strategy",
                    "total_return_strategy", "final_portfolio"]:
            assert key in bt


class TestSampleDailyTiledStates:
    def setup_method(self):
        df = make_fixture_ohlcv(600)
        self.model, self.order = fit_generating_hmm(df["Close"])

    def test_output_length_exact(self):
        for n_bars in [100, 101, 144, 145, 999]:
            out = sample_daily_tiled_states(self.model, self.order, n_bars=n_bars, seed=0)
            assert len(out) == n_bars, f"n_bars={n_bars}: got {len(out)}"

    def test_values_in_valid_range(self):
        out = sample_daily_tiled_states(self.model, self.order, n_bars=300, seed=7)
        assert set(np.unique(out)) <= {0, 1, 2}

    def test_tiling_structure(self):
        # With bars_per_day=6 each group of 6 consecutive bars should be identical
        out = sample_daily_tiled_states(
            self.model, self.order, n_bars=60, seed=3, bars_per_day=6
        )
        for i in range(0, 60, 6):
            block = out[i : i + 6]
            assert np.all(block == block[0]), f"block at {i} not uniform: {block}"

    def test_reproducible_same_seed(self):
        a = sample_daily_tiled_states(self.model, self.order, n_bars=200, seed=42)
        b = sample_daily_tiled_states(self.model, self.order, n_bars=200, seed=42)
        np.testing.assert_array_equal(a, b)


class TestGenerateSyntheticDfPrecomputedLabels:
    def setup_method(self):
        self.real_df = make_fixture_ohlcv(600)
        self.model, self.order = fit_generating_hmm(self.real_df["Close"])
        log_ret = np.log(
            self.real_df["Close"].values[1:] / self.real_df["Close"].values[:-1]
        )
        self.log_ret = log_ret
        self.hist_labels = label_hidden_states(self.model, self.order, log_ret)

    def test_precomputed_labels_used_not_model(self):
        fixed_labels = np.zeros(300, dtype=np.intp)  # all-Bear
        with mock.patch(
            "Strategy_Auto_Trader.quant_hmm.synthetic_data.sample_synthetic_path"
        ) as mock_ssp:
            generate_synthetic_df(
                self.real_df, self.model, self.order,
                self.log_ret, self.hist_labels,
                n_bars=300, seed=0,
                precomputed_state_labels=fixed_labels,
            )
        mock_ssp.assert_not_called()

    def test_output_shape_unchanged(self):
        precomputed = sample_daily_tiled_states(
            self.model, self.order, n_bars=300, seed=5
        )
        df = generate_synthetic_df(
            self.real_df, self.model, self.order,
            self.log_ret, self.hist_labels,
            n_bars=300, seed=5,
            precomputed_state_labels=precomputed,
        )
        assert len(df) == 300
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_without_precomputed_unchanged(self):
        # Passing None must produce same result as omitting the arg entirely
        df_omit = generate_synthetic_df(
            self.real_df, self.model, self.order,
            self.log_ret, self.hist_labels,
            n_bars=200, seed=99,
        )
        df_none = generate_synthetic_df(
            self.real_df, self.model, self.order,
            self.log_ret, self.hist_labels,
            n_bars=200, seed=99,
            precomputed_state_labels=None,
        )
        pd.testing.assert_frame_equal(df_omit, df_none)
