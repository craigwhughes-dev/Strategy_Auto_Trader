from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from tests.conftest import _dates


class TestHMMExtension:

    def test_forward_filter_returns_correct_length(self):
        """Test _forward_filter with a mock model object."""
        from Strategy_Auto_Trader.extensions.hmm_extension import _forward_filter

        class MockModel:
            n_components = 3
            means_ = np.array([[0.01], [-0.01], [0.0]])
            covars_ = np.array([[[0.001]], [[0.001]], [[0.001]]])
            transmat_ = np.array([[0.7, 0.2, 0.1],
                                  [0.1, 0.7, 0.2],
                                  [0.2, 0.1, 0.7]])
            startprob_ = np.array([0.33, 0.34, 0.33])

        model = MockModel()
        X = np.random.randn(100, 1) * 0.01
        states = _forward_filter(model, X)
        assert len(states) == 100
        assert all(0 <= s < 3 for s in states)

    def test_forward_filter_deterministic(self):
        """Same input -> same output."""
        from Strategy_Auto_Trader.extensions.hmm_extension import _forward_filter

        class MockModel:
            n_components = 2
            means_ = np.array([[0.05], [-0.05]])
            covars_ = np.array([[[0.01]], [[0.01]]])
            transmat_ = np.array([[0.8, 0.2], [0.3, 0.7]])
            startprob_ = np.array([0.5, 0.5])

        model = MockModel()
        np.random.seed(42)
        X = np.random.randn(50, 1) * 0.03
        s1 = _forward_filter(model, X)
        s2 = _forward_filter(model, X)
        np.testing.assert_array_equal(s1, s2)

    def test_fit_hmm_no_hmmlearn(self):
        """If hmmlearn is not available, fit_hmm returns (None, None)."""
        from Strategy_Auto_Trader.extensions import hmm_extension
        with mock.patch.dict("sys.modules", {"hmmlearn": None, "hmmlearn.hmm": None}):
            # Force reimport of the function to test the ImportError path
            import importlib
            importlib.reload(hmm_extension)
            result = hmm_extension.fit_hmm(pd.Series(np.random.randn(100)))
            assert result == (None, None)
        # Reload again to restore
        importlib.reload(hmm_extension)

    def test_fit_hmm_with_hmmlearn(self):
        """Test fit_hmm when hmmlearn IS available."""
        pytest.importorskip("hmmlearn")
        from Strategy_Auto_Trader.extensions.hmm_extension import fit_hmm

        np.random.seed(42)
        returns = pd.Series(
            np.concatenate([
                np.random.normal(0.01, 0.02, 100),
                np.random.normal(-0.01, 0.03, 100),
                np.random.normal(0.0, 0.01, 100),
            ]),
            index=_dates(300),
        )
        result = fit_hmm(returns, n_components=3, n_seeds=3, n_iter=50)
        if result == (None, None):
            pytest.skip("hmmlearn not installed")
        assert isinstance(result, dict)
        assert "forward_states" in result
        assert "regime_names" in result
        assert result["regime_names"] == ["Bear", "Sideways", "Bull"]
        assert len(result["forward_states"]) == 300
        assert "transition_matrix" in result
        assert result["transition_matrix"].shape == (3, 3)
        # Stationary distribution sums to 1
        assert abs(result["stationary_distribution"].sum() - 1.0) < 1e-6
